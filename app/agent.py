"""Agent 工厂：create_agent 组装（阶段 3 + 5 + 6）。

阶段 6：checkpointer（会话续聊）+ store（跨会话记忆）。
⚠️ Agent 实例需要 checkpointer/store（async with 内创建），
所以工厂改为 async 惰性单例（ensure_agent），在 memory_ctx 内初始化。
"""
from langchain.agents import create_agent

from app.middleware import MIDDLEWARES
from app.model import get_model
from app.prompts import AGENT_SYSTEM_PROMPT
from app.tools.course_tools import get_course_detail, record_search_log, search_courses
from app.tools.memory_tools import get_progress, save_progress

_agent = None


async def ensure_agent(checkpointer, store):
    """惰性创建 Agent 单例（进程内只建一次）。

    checkpointer/store 由调用方在 memory_ctx 内传入（连接生命周期归 ctx 管）。
    """
    global _agent
    if _agent is None:
        _agent = create_agent(
            model=get_model(),
            tools=[
                search_courses,
                get_course_detail,
                record_search_log,
                save_progress,
                get_progress,
            ],
            system_prompt=AGENT_SYSTEM_PROMPT,
            middleware=MIDDLEWARES,
            checkpointer=checkpointer,
            store=store,
        )
    return _agent
