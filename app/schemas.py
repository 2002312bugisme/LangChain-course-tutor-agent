"""Pydantic 模型（阶段 7）：结构化输出 schema。"""
from typing import Literal

from pydantic import BaseModel, Field


class Topic(BaseModel):
    """学习计划中的单个知识点。"""

    name: str = Field(description="知识点名称，如 'Python 基础语法'")
    order: int = Field(description="学习顺序，从 1 开始")
    minutes: int = Field(description="建议学习时长（分钟）")


class LearningPlan(BaseModel):
    """学习计划（结构化输出的核心 schema）。

    前端拿到这个对象直接渲染卡片，无需解析文本。
    """

    goal: str = Field(description="学习目标概述，一句话")
    level: Literal["入门", "进阶", "高级"] = Field(description="难度级别")
    total_hours: float = Field(description="总时长（小时）")
    topics: list[Topic] = Field(description="按学习顺序排列的知识点列表")
