"""集中配置：加载 .env，向全项目提供统一的模型参数。"""
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# 关键：把 .env 加载进 os.environ。
# pydantic-settings 自己会读 .env 文件，但 LangChain 的 api_key 默认值
# （from_env 机制）读的是【进程环境变量】——不 load_dotenv() 就报
# "DEEPSEEK_API_KEY must be set"。这行让两套机制都拿到配置。
load_dotenv()


class Settings(BaseSettings):
    """所有配置项集中定义，字段名与 .env 键一一对应。"""

    model_config = SettingsConfigDict(
        env_file=".env",       # 从项目根目录 .env 读取
        env_file_encoding="utf-8",
        extra="ignore",        # .env 里多余键忽略，不报错
    )

    # 模型连接
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"

    # 模型调用参数
    temperature: float = 0.3
    max_tokens: int = 200
    timeout: float = 30
    max_retries: int = 2

    # LangSmith（阶段 9 启用）
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "langchain_learning_agent"


settings = Settings()
