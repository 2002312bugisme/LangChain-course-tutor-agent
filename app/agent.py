"""Agent 工厂：create_agent 组装（阶段 3）。

知识点覆盖（对应 agent_api_reference.md 阶段 4）：
- create_agent(model, tools, system_prompt)
- 工具注册：内部自动 bind_tools + 构建 ToolNode 执行器
- 全局单例（与 get_model 一致，生产推荐）
"""
from functools import lru_cache

from langchain.agents import create_agent

from app.model import get_model
from app.tools.course_tools import get_course_detail, record_search_log, search_courses

# Agent 级系统提示：约束角色 + 工具使用指引（阶段 6 升级为动态 prompt）
AGENT_SYSTEM_PROMPT = """你是编程学习助手"课栈"，帮助用户查找课程、规划学习。

## 思考语言规定（重要）
- 你的整个思考过程（reasoning）必须使用中文，与用户输入语言保持一致

## 工具使用指引
- 用户问"有没有/推荐/找 XX 课"时，先调用 search_courses 查询课程库
- 用户要看某门课详情时，用 get_course_detail（ID 从搜索结果的编号列获取）
- 用户咨询课程问题时，同时调用 record_search_log 记录日志
- 查询结果直接整理给用户，不要编造课程信息

## 行为准则
- 回答简洁，用中文
- 不知道的就说不知道，不要编造"""


@lru_cache(maxsize=1)
def get_agent():
    """创建（并缓存）Agent 实例。

    阶段 4 起被 FastAPI 复用；lru_cache 保证进程内单例。
    """
    return create_agent(
        model=get_model(),
        tools=[search_courses, get_course_detail, record_search_log],
        system_prompt=AGENT_SYSTEM_PROMPT,
    )
