"""记忆工具（阶段 6）：InjectedStore 读写演示。

知识点覆盖（agent_api_reference.md 3.5/8.2）：
- InjectedStore: LangGraph 注入的跨会话存储对象（namespace+key 层级）
- 工具不需要模型提供这些参数（不在 schema 中，模型看不到）

⚠️ 工具必须是 async def + store.aput/aget：
AsyncSqliteStore 的同步方法（put/get）在主事件循环调用会抛
InvalidStateError（阶段 6 实测踩坑）——与 wrap 中间件同款"异步环境必须用异步接口"。
"""
from typing import Annotated

from langchain.tools import InjectedStore, tool
from langgraph.store.base import BaseStore

from app.memory import DEFAULT_USER_NS


@tool
async def save_progress(
    topic: str,
    level: str,
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """记录用户已完成的学习主题和掌握程度（跨会话保存）。

    Args:
        topic: 已学主题，如 "Python 基础语法"、"Vue 组件化"。
        level: 掌握程度，取值 "入门"、"进阶"、"高级"。
    """
    # namespace=("users","default","progress") + key="topics"：层级键值存储
    ns = (*DEFAULT_USER_NS, "progress")
    item = await store.aget(ns, "topics")
    current = item.value if item else {}
    current[topic] = level
    await store.aput(ns, "topics", current)
    return f"已记录：{topic}（{level}），当前共掌握 {len(current)} 个主题"


@tool
async def get_progress(
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """查询用户跨会话保存的学习进度。"""
    ns = (*DEFAULT_USER_NS, "progress")
    item = await store.aget(ns, "topics")
    data = item.value if item else {}
    if not data:
        return "还没有学习进度记录。可以让我记录：'我学完了 Python 基础，入门水平'"
    lines = [f"已掌握 {len(data)} 个主题："]
    for topic, level in data.items():
        lines.append(f"- {topic}（{level}）")
    return "\n".join(lines)
