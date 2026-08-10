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
from pydantic import BaseModel

from app.agent import ensure_agent
from app.memory import memory_ctx
from app.model import get_model

# 记忆连接生命周期：app 运行期间保持（阶段 6）
_memory = {}


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
            "根据以下对话内容，用不超过 12 个字总结一个会话标题。"
            "只输出标题本身，不要标点、引号或解释：\n"
            + "\n".join(texts)
        )
        resp = await get_model().ainvoke(prompt)
        title = resp.content.strip().split("\n")[0][:15] or "新会话"
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
    """会话列表：枚举 thread_id + 从 Store 合并 LLM 生成的标题。"""
    cp = _memory["checkpointer"]
    store = _memory["store"]
    threads: dict[str, dict] = {}
    async for item in cp.alist(None):
        cfg = item.config or {}
        tid = cfg.get("configurable", {}).get("thread_id")
        if not tid:
            continue
        step = (item.metadata or {}).get("step", 0)
        if tid not in threads or step > threads[tid]["step"]:
            threads[tid] = {"thread_id": tid, "step": step}
    # 最新会话在前 + 合并标题
    result = sorted(threads.values(), key=lambda x: -x["step"])
    for t in result:
        item = await store.aget(("threads", t["thread_id"]), "title")
        t["title"] = (item.value if item else None) or "新会话"
    return result


@app.get("/threads/{tid}/messages")
async def thread_messages(tid: str):
    """读取某会话的完整消息历史（checkpointer state）。"""
    snap = await _memory["agent"].aget_state({"configurable": {"thread_id": tid}})
    msgs = (snap.values.get("messages", []) if snap else []) or []
    return {
        "thread_id": tid,
        "messages": [
            {"role": m.type, "content": str(m.content)}
            for m in msgs
            if m.type in ("human", "ai") and m.content
        ],
    }


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
        asyncio.create_task(_generate_thread_title(req.thread_id))

    return StreamingResponse(event_gen(), media_type="text/event-stream")


def _sse(event_type: str, data: dict) -> str:
    """格式化一条 SSE 消息：data: {json}\n\n"""
    return f"data: {json.dumps({'type': event_type, **data}, ensure_ascii=False)}\n\n"
