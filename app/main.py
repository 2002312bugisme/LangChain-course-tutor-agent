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
import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agent import get_agent
from app.model import get_model

app = FastAPI(title="课栈 - 编程学习助手", version="0.4.0")

# CORS：开发阶段允许 Vite 前端跨域（阶段 4 起前端在 5173 端口）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None  # 阶段 6 引入 checkpointer 后启用


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


@app.post("/chat")
async def chat(req: ChatRequest):
    """完整版（非流式）：等待 Agent 全部执行完，返回思考+回复+工具链。"""
    result = await get_agent().ainvoke({"messages": [("user", req.message)]})

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
        async for mode, data in get_agent().astream(
            {"messages": [("user", req.message)]},
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

    return StreamingResponse(event_gen(), media_type="text/event-stream")


def _sse(event_type: str, data: dict) -> str:
    """格式化一条 SSE 消息：data: {json}\n\n"""
    return f"data: {json.dumps({'type': event_type, **data}, ensure_ascii=False)}\n\n"
