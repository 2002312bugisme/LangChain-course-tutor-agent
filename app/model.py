"""模型工厂：全项目唯一的模型创建入口。

设计要点：
1. 使用 init_chat_model() 统一工厂（provider 推断）
2. 阶段 9：升级为 ConfigurableModel——运行时通过
   config["configurable"]["chat_model_model"] 切换模型（flash/pro），
   切换后模型键前缀为 chat_model（config_prefix="chat_model"）
3. 模块级单例：进程内只创建一次（ConfigurableModel 实例本身可切换）
4. 参数全部来自 settings（.env 驱动），不硬编码
"""
from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from app.config import settings

# 可用模型列表（DeepSeek API 实测支持的两个名字，阶段 9）
MODEL_OPTIONS = ["deepseek-v4-flash", "deepseek-v4-pro"]


@lru_cache(maxsize=1)
def get_model() -> BaseChatModel:
    """创建（并缓存）ConfigurableModel 实例。

    注意：model 不能带 "deepseek:" 前缀——显式传 model_provider 时
    _parse_model 不会剥离前缀，前缀会原样传给 API 导致 400 错误。
    """
    return init_chat_model(
        model=settings.deepseek_model,
        model_provider="deepseek",
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        timeout=settings.timeout,
        max_retries=settings.max_retries,
        configurable_fields=["model"],  # 阶段 9：允许运行时切换模型
        config_prefix="chat_model",     # config 键：configurable.chat_model_model
    )
