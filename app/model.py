"""模型工厂：全项目唯一的模型创建入口。

设计要点：
1. 使用 init_chat_model() 统一工厂（provider 推断 + 支持未来切 ConfigurableModel）
2. 模块级单例：进程内只创建一次模型实例（Agent 复用实例是官方推荐）
3. 参数全部来自 settings（.env 驱动），不硬编码
"""
from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from app.config import settings


@lru_cache(maxsize=1)
def get_model() -> BaseChatModel:
    """创建（并缓存）模型实例。

    用 lru_cache 实现单例：多次调用返回同一个实例，避免重复初始化开销。
    阶段 9 将升级为 ConfigurableModel（运行时切换 deepseek-v4-flash/pro）。
    """
    return init_chat_model(
        # 注意：model 不能带 "deepseek:" 前缀——显式传 model_provider 时
        # _parse_model 不会剥离前缀，前缀会原样传给 API 导致 400 错误。
        # （实测：API 报 "you passed deepseek:deepseek-v4-flash"）
        model=settings.deepseek_model,
        model_provider="deepseek",
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        timeout=settings.timeout,
        max_retries=settings.max_retries,
    )
