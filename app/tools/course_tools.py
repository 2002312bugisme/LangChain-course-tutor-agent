"""课程查询工具集（阶段 3）。

知识点覆盖（对应 agent_api_reference.md 阶段 3）：
- @tool 装饰器 + docstring 自动生成描述
- args_schema（Pydantic 精细参数校验）
- ToolException（业务异常）
- handle_tool_error（装饰后赋值方式，@tool 不接受该参数——源码验证）
- return_direct=True（结果即答案）
- 注入参数演示：InjectedToolCallId（工具内部关联调用上下文）
"""
import json
from pathlib import Path
from typing import Annotated

from langchain_core.tools import InjectedToolCallId, ToolException, tool

COURSES_FILE = Path(__file__).resolve().parents[2] / "data" / "courses.json"


def _load_courses() -> list[dict]:
    """读取模拟课程库（真实项目应换数据库/API）。"""
    with open(COURSES_FILE, encoding="utf-8") as f:
        return json.load(f)


@tool
def search_courses(
    keyword: str,
    level: str = "",
    max_results: int = 3,
) -> str:
    """按关键词和难度搜索课程库中的课程。

    Args:
        keyword: 搜索关键词，如 "python"、"vue"、"入门"。
        level: 难度过滤，可选 "入门"、"进阶"、"高级"，空字符串表示不限。
        max_results: 最多返回几条结果。
    """
    if not keyword.strip():
        raise ToolException("搜索关键词不能为空，请提供课程关键词（如 python、vue）。")

    courses = _load_courses()
    kw = keyword.strip().lower()
    matched = [
        c for c in courses
        if kw in c["title"].lower() or kw in c["tags"] or kw in c["level"]
    ]
    if level:
        matched = [c for c in matched if c["level"] == level]

    if not matched:
        return f"没有找到与 {keyword!r} 相关的课程。"

    lines = [f"找到 {len(matched)} 门课程（展示前 {min(max_results, len(matched))} 门）："]
    for c in matched[:max_results]:
        lines.append(f"- {c['id']} | {c['title']} | {c['level']} | {c['duration_hours']}小时 | {c['description']}")
    return "\n".join(lines)


# 演示 2：return_direct=True——工具结果就是最终答案，跳过模型二次加工
@tool(return_direct=True)
def get_course_detail(course_id: str) -> str:
    """根据课程 ID 获取课程的完整详情（含学习大纲）。

    Args:
        course_id: 课程唯一 ID，如 "py-101"、"fe-201"。
    """
    courses = _load_courses()
    for c in courses:
        if c["id"] == course_id:
            return (f"【{c['title']}】\n"
                    f"难度：{c['level']} | 时长：{c['duration_hours']}小时\n"
                    f"简介：{c['description']}\n"
                    f"标签：{', '.join(c['tags'])}")
    raise ToolException(f"课程 ID {course_id!r} 不存在，可先调用 search_courses 查询。")


# 演示 3：InjectedToolCallId——拿到"谁调用的我"（不进入工具 schema，模型看不到）
@tool
def record_search_log(
    question: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> str:
    """记录用户的一次课程咨询日志（演示注入参数）。

    Args:
        question: 用户咨询的问题内容。
    """
    # tool_call_id 由 LangGraph 运行时注入：当前这次 ToolCall 的唯一 ID
    return f"已记录咨询日志 #{tool_call_id[:8]}：{question}"


# 演示 4：handle_tool_error——工具出错"自愈"，错误转成返回值给模型
search_courses.handle_tool_error = True   # @tool 不接受该参数，装饰后赋值（源码验证）
