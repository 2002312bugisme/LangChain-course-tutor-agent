"""FastAPI 入口：Web 化（阶段 4）。

接口：
- GET  /health       健康检查（含当前模型信息）
- POST /chat         完整回复（ainvoke）：返回 thinking + reply + 工具摘要
- POST /chat/stream  SSE 流式：reasoning / token / tool / done / error 事件

SSE 事件设计（对应需求⑦，预研结论落地）：
- chunk.additional_kwargs["reasoning_content"] 有值 → reasoning 事件
- chunk.content 有值 → token 事件
- updates 模式下的工具执行 → tool 事件
"""
import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.agent import ensure_agent
from app.config import settings
from app.memory import memory_ctx
from app.model import get_model
from app.schemas import LearningPlan

# 记忆连接生命周期：app 运行期间保持（阶段 6）
_memory = {}

# 后台任务引用集合：防止 asyncio.create_task 的任务被垃圾回收
# （task 无引用会被 GC，导致标题生成"时灵时不灵"——用户实测发现）
_background_tasks: set[asyncio.Task] = set()


async def _generate_thread_title(tid: str) -> None:
    """异步生成会话标题（仅首次）：用 LLM 总结会话内容，存 Store。

    namespace=("threads", tid) + key="title"——与会话数据隔离。
    由调用方 create_task 后台执行，不阻塞 SSE 响应。
    """
    try:
        store = _memory["store"]
        ns = ("threads", tid)
        if await store.aget(ns, "title"):
            return  # 已有标题，跳过（避免重复调用 LLM 浪费成本）

        # 读会话前几条消息作为总结素材
        snap = await _memory["agent"].aget_state({"configurable": {"thread_id": tid}})
        msgs = (snap.values.get("messages", []) if snap else []) or []
        texts = [
            str(m.content)[:80]
            for m in msgs
            if m.type in ("human", "ai") and m.content
        ][:6]
        if not texts:
            return

        prompt = (
            "你是会话标题生成器。根据对话内容生成一个简短标题。\n"
            "要求：不超过 12 个汉字；提炼核心主题；不要标点、引号、解释；直接输出标题。\n"
            "示例：\n"
            "对话：'推荐一门 Python 入门课' → 标题：Python入门课程推荐\n"
            "对话：'我现在想学习vue搭配python规划学习路线' → 标题：Vue+Python学习路线规划\n"
            "对话内容：\n"
            + "\n".join(texts)
        )
        resp = await get_model().ainvoke(prompt)
        title = resp.content.strip().split("\n")[0].strip()[:18] or "新会话"
        await store.aput(ns, "title", title)
    except Exception:
        pass  # 标题生成失败不影响主流程


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时创建记忆连接 + Agent 单例，关闭时释放。"""
    async with memory_ctx() as (cp, st):
        _memory["checkpointer"] = cp
        _memory["store"] = st
        _memory["agent"] = await ensure_agent(cp, st)
        yield


app = FastAPI(title="课栈 - 编程学习助手", version="0.6.0", lifespan=lifespan)

# CORS：开发阶段允许 Vite 前端跨域（阶段 4 起前端在 5173 端口）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None  # 阶段 6：会话 ID（checkpointer 记忆键）


def _thread_config(thread_id: str | None) -> dict:
    """构造运行 config：带 thread_id 则启用会话记忆。

    config 里 configurable.thread_id 是 checkpointer 的检索键：
    同一 thread_id 自动恢复历史（续聊），不同 thread_id 互相隔离。
    """
    if not thread_id:
        return {}
    return {"configurable": {"thread_id": thread_id}}


def _tool_summary(messages: list) -> list[dict]:
    """从执行结果中提取工具调用摘要（供 /chat 返回）。"""
    summary = []
    for m in messages:
        if m.type == "tool":
            summary.append({"name": m.name, "result": str(m.content)[:80]})
    return summary


@app.get("/health")
async def health():
    model = get_model()
    return {
        "status": "ok",
        "model": model.model_name,
        "provider": "deepseek",
    }


@app.get("/threads")
async def list_threads():
    """会话列表：按最近对话时间（checkpoint.ts）降序 + Store 标题。"""
    cp = _memory["checkpointer"]
    store = _memory["store"]
    threads: dict[str, dict] = {}
    async for item in cp.alist(None):
        cfg = item.config or {}
        tid = cfg.get("configurable", {}).get("thread_id")
        if not tid:
            continue
        ck = item.checkpoint or {}
        ts = ck.get("ts", "")   # ISO 时间戳：会话最近活动时间
        step = (item.metadata or {}).get("step", 0)
        # 同一会话保留最新 checkpoint（按 ts 比较）
        if tid not in threads or ts > threads[tid]["ts"]:
            threads[tid] = {"thread_id": tid, "step": step, "ts": ts}
    # 最近对话时间降序
    result = sorted(threads.values(), key=lambda x: x["ts"], reverse=True)
    for t in result:
        item = await store.aget(("threads", t["thread_id"]), "title")
        t["title"] = (item.value if item else None) or "新会话"
    return result


@app.delete("/threads/{tid}")
async def delete_thread(tid: str):
    """删除会话：清 checkpointer 全部 checkpoint + Store 标题。

    注意：用户学习进度（跨会话数据）在独立 namespace，不受影响。
    """
    await _memory["checkpointer"].adelete_thread(tid)
    await _memory["store"].adelete(("threads", tid), "title")
    return {"ok": True, "thread_id": tid}


@app.post("/threads/{tid}/rename")
async def rename_thread(tid: str, payload: dict):
    """重命名会话：更新 Store 中的标题。"""
    title = (payload.get("title") or "").strip()[:30]
    if not title:
        return {"ok": False, "error": "标题不能为空"}
    await _memory["store"].aput(("threads", tid), "title", title)
    return {"ok": True, "thread_id": tid, "title": title}


@app.get("/threads/{tid}/messages")
async def thread_messages(tid: str):
    """读取某会话的完整消息历史（checkpointer state）。

    AI 消息附带 thinking（reasoning_content）——用户需求：
    切换会话后思考过程要保留展示（工具调用列表不需要）。
    """
    snap = await _memory["agent"].aget_state({"configurable": {"thread_id": tid}})
    msgs = (snap.values.get("messages", []) if snap else []) or []
    out = []
    for m in msgs:
        if not m.content or m.type not in ("human", "ai"):
            continue
        item = {"role": m.type, "content": str(m.content)}
        if m.type == "ai":
            item["thinking"] = m.additional_kwargs.get("reasoning_content", "")
        out.append(item)
    return {"thread_id": tid, "messages": out}


@app.post("/plan")
async def create_plan(req: ChatRequest):
    """学习计划生成（阶段 7：结构化输出演示）。

    用 with_structured_output(LearningPlan)：模型直接返回符合 schema 的
    Pydantic 对象（JSON），前端无需解析文本即可渲染卡片。

    ⚠️ deepseek-v4-flash 推理模型实验结论（2026-08 实测）：
    - thinking 模式下不支持 structured output（tool_choice 被拒）
    - 必须 extra_body={"thinking": {"type": "disabled"}} 关闭思考
    - method="function_calling" 可用；json_schema 不可用（unavailable）
    """
    model = ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        timeout=settings.timeout,
        max_retries=settings.max_retries,
        extra_body={"thinking": {"type": "disabled"}},
    )
    structured_model = model.with_structured_output(LearningPlan, method="function_calling")
    result = await structured_model.ainvoke(f"请为用户规划编程学习路线：{req.message}")
    return result.model_dump() if hasattr(result, "model_dump") else result


@app.get("/threads/{tid}/export")
async def export_thread(tid: str):
    """导出会话为 Markdown（阶段 7 需求）。

    格式：
    # 会话标题
    > 导出时间 / 会话 ID
    ## 👤 用户 / ## 🤖 助手（思考过程用引用块）
    """
    snap = await _memory["agent"].aget_state({"configurable": {"thread_id": tid}})
    msgs = (snap.values.get("messages", []) if snap else []) or []

    # 标题（Store 中的 LLM 总结或手动重命名）
    item = await _memory["store"].aget(("threads", tid), "title")
    title = (item.value if item else None) or "会话"

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [f"# {title}", "", f"> 导出时间：{now}", f"> 会话 ID：`{tid}`", ""]
    for m in msgs:
        if m.type == "human" and m.content:
            lines += ["---", "", f"## 👤 用户", "", str(m.content), ""]
        elif m.type == "ai" and m.content:
            lines += ["---", "", "## 🤖 助手", ""]
            reasoning = m.additional_kwargs.get("reasoning_content")
            if reasoning:
                lines += [
                    "> **思考过程**",
                    "",
                    *[f"> {line}" for line in str(reasoning).splitlines()],
                    "",
                ]
            lines += [str(m.content), ""]

    return {"title": title, "markdown": "\n".join(lines)}


@app.post("/chat")
async def chat(req: ChatRequest):
    """完整版（非流式）：等待 Agent 全部执行完，返回思考+回复+工具链。"""
    result = await _memory["agent"].ainvoke(
        {"messages": [("user", req.message)]},
        _thread_config(req.thread_id),
    )

    # 最后一条 AI 消息 = 最终回复（含思考过程）
    last_ai = next(m for m in reversed(result["messages"]) if m.type == "ai")
    return {
        "thinking": last_ai.additional_kwargs.get("reasoning_content", ""),
        "reply": last_ai.content,
        "tools": _tool_summary(result["messages"]),
        "usage": last_ai.usage_metadata,
    }


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE 流式：思考/回复/工具调用分段推送。"""

    async def event_gen():
        # 混合流模式：messages 拿 token 级 chunk，updates 拿节点级更新
        async for mode, data in _memory["agent"].astream(
            {"messages": [("user", req.message)]},
            _thread_config(req.thread_id),
            stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                chunk, metadata = data
                reasoning = chunk.additional_kwargs.get("reasoning_content")
                if reasoning:
                    yield _sse("reasoning", {"content": reasoning})
                if chunk.content:
                    yield _sse("token", {"content": chunk.content})
            elif mode == "updates":
                for node, update in data.items():
                    if node == "tools" and update:
                        # 工具执行结果到达 → 通知前端展示
                        for msg in update.get("messages", []):
                            if msg.type == "tool":
                                yield _sse("tool", {"name": msg.name, "result": str(msg.content)[:60]})
        yield _sse("done", {"message": "完成"})

    if req.thread_id:
        # 后台生成会话标题（不阻塞 SSE，仅首次）
        # ★ 必须保存 task 引用：无引用的 task 会被 GC，标题生成会"随机失败"
        task = asyncio.create_task(_generate_thread_title(req.thread_id))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


def _sse(event_type: str, data: dict) -> str:
    """格式化一条 SSE 消息：data: {json}\n\n"""
    return f"data: {json.dumps({'type': event_type, **data}, ensure_ascii=False)}\n\n"
