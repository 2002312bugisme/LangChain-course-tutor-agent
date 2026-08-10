"""Agent 工厂：create_agent 组装（阶段 3 + 阶段 5 中间件）。

知识点覆盖（对应 agent_api_reference.md 阶段 4/7）：
- create_agent(model, tools, system_prompt, middleware)
- 工具注册：内部自动 bind_tools + 构建 ToolNode 执行器
- 全局单例（与 get_model 一致，生产推荐）
"""
from functools import lru_cache

from langchain.agents import create_agent

from app.middleware import MIDDLEWARES
from app.model import get_model
from app.prompts import AGENT_SYSTEM_PROMPT
from app.tools.course_tools import get_course_detail, record_search_log, search_courses


@lru_cache(maxsize=1)
def get_agent():
    """创建（并缓存）Agent 实例。

    阶段 5 起挂载中间件：日志观察 + 道别拦截 + 动态提示词 + 思考语言守卫。
    """
    return create_agent(
        model=get_model(),
        tools=[search_courses, get_course_detail, record_search_log],
        system_prompt=AGENT_SYSTEM_PROMPT,
        middleware=MIDDLEWARES,
    )
