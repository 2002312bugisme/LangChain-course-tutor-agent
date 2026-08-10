# 阶段 0：项目骨架 + 配置（说明文档）

> 状态：✅ 已完成并验证
> 对应设计文档：DESIGN.md 阶段 0
> 验证时间：2026-08

## 1. 目标

建立项目骨架：目录结构、敏感配置管理、依赖清单。本阶段不写任何业务逻辑。

## 2. 搭建步骤

1. 安装 FastAPI 全家桶：`pip install fastapi uvicorn`（检查确认 python-dotenv / pydantic-settings 已随 langchain 环境存在）
2. 创建目录：`app/`（后端包）、`app/tools/`、`app/rag/`、`data/knowledge/`、`docs/`
3. 创建 `.env`（敏感配置）、`.gitignore`（防泄漏）、`requirements.txt`（依赖清单）、`app/config.py`（配置加载）

## 3. 目录结构（本阶段变更）

```
D:\Code\LangChain_1.2\
├── .env                    # 新增：敏感配置（已 gitignore）
├── .gitignore              # 新增
├── requirements.txt        # 新增：依赖清单
├── app/
│   ├── __init__.py         # 新增：包标记
│   ├── config.py           # 新增：集中配置
│   ├── tools/__init__.py   # 新增：工具包（阶段 3 用）
│   └── rag/__init__.py     # 新增：RAG 包（阶段 8 用）
├── data/knowledge/         # 新增：RAG 知识源目录（阶段 8 用）
└── docs/                   # 新增：阶段说明文档
```

## 4. 新增文件详解

### 4.1 `.env` —— 敏感配置

```ini
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
TEMPERATURE=0.3
MAX_TOKENS=200
TIMEOUT=30
MAX_RETRIES=2
```

**为什么这么做**：密钥和可调参数不进代码——代码要提交 git/分享，配置不用。改参数（如 temperature）只改 .env，不碰代码。

### 4.2 `.gitignore` —— 防泄漏清单

忽略 `.env`（密钥）、`__pycache__`、`.venv`、`node_modules`（前端）、`*.db`（阶段 6 数据库）等。

**为什么这么做**：git 提交时误带密钥 = 安全事故。宁可提前全忽略。

### 4.3 `requirements.txt` —— 依赖清单

显式列出 langchain / langchain-openai / langgraph / fastapi / uvicorn / pydantic-settings；阶段 6/8 的包（sqlite-saver、jieba、rank-bm25）以注释形式预留。

**为什么这么做**：换机器/换环境一条命令复现环境。预留注释让"下一步装什么"可见。

### 4.4 `app/config.py` —— 集中配置（本阶段核心）

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    temperature: float = 0.3
    max_tokens: int = 200
    timeout: float = 30
    max_retries: int = 2
    ...

settings = Settings()   # 模块级单例
```

**逻辑**：`pydantic-settings` 的 `BaseSettings` 自动读 `.env`，字段名与键名映射（`deepseek_api_key` ↔ `DEEPSEEK_API_KEY`，大小写不敏感）；类型自动转换（`TEMPERATURE=0.3` → float）；模块级单例让全项目 `from app.config import settings` 拿到同一份配置。

**方案对比（为什么选 pydantic-settings）**：

| 方案 | 优点 | 缺点 | 结论 |
|---|---|---|---|
| `os.getenv()` 散落各处 | 零依赖 | 无类型检查、无默认值管理、散落难维护 | ❌ |
| `python-dotenv` + 手动解析 | 轻量 | 类型转换/默认值/校验都要手写 | ❌ |
| **`pydantic-settings`** | **类型自动转换、默认值、校验、IDE 补全、单一配置类** | 多一个依赖（但 langchain 生态本就用 pydantic） | ✅ 选用 |

## 5. 验证记录

```bash
python -c "from app.config import settings; print(settings.deepseek_model, settings.temperature)"
# 输出：deepseek-v4-flash 0.3   ✅ 配置加载正常
```

## 6. 踩坑与备注

- `extra="ignore"` 很重要：.env 里将来加了未声明键（如 LANGCHAIN_API_KEY 提前声明了），不会因多余键报错
- 本机 fastapi/uvicorn 未预装，已 pip 安装（0.141.1 / 0.52.1）
- `pydantic-settings` 2.6.1 已随环境存在，无需安装
