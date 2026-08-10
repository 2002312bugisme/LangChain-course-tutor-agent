# 阶段 1：最小模型调用（说明文档）

> 状态：✅ 已完成并验证（真实 API 调用）
> 对应设计文档：DESIGN.md 阶段 1
> 验证时间：2026-08

## 1. 目标

跑通"模型调用"最小闭环：`init_chat_model` 创建模型 → `invoke()` 一次性问答 → `stream()` 流式输出。为阶段 2（消息层）打基础。

## 2. 搭建步骤

1. `pip install langchain-deepseek`——`init_chat_model` 的 deepseek provider 需要此集成包（底层实例化 `ChatDeepSeek`）
2. 新建 `app/model.py`：模型工厂（单例）
3. 新建 `app/cli.py`：CLI 验证脚本（invoke / stream 两个子命令）
4. 修复 `app/config.py`：补 `load_dotenv()`

## 3. 目录变更

```
app/
├── config.py    # 修改：顶部加 load_dotenv()
├── model.py     # 新增：get_model() 模型工厂
└── cli.py       # 新增：验证 CLI
```

## 4. 新增文件详解

### 4.1 `app/model.py` —— 模型工厂

```python
@lru_cache(maxsize=1)
def get_model() -> BaseChatModel:
    return init_chat_model(
        model=settings.deepseek_model,      # "deepseek-v4-flash"（无前缀！）
        model_provider="deepseek",          # 显式指定提供商
        temperature=settings.temperature,   # 0.3
        max_tokens=settings.max_tokens,     # 200
        timeout=settings.timeout,           # 30
        max_retries=settings.max_retries,   # 2
    )
```

**逻辑**：
- `init_chat_model` 按 `model_provider` 找到 `ChatDeepSeek` 类并实例化，参数（temperature 等）全部透传
- `lru_cache` 实现**进程级单例**：多次调用返回同一实例（Agent 复用实例是官方推荐，见 DESIGN.md）
- 参数全部来自 settings（.env），改参数不动代码

### 4.2 `app/cli.py` —— 验证脚本

- `demo_invoke`：`model.invoke(prompt)` 返回完整 `AIMessage`，打印 content + 元数据（model_name / finish_reason）+ token 用量（usage_metadata）
- `demo_stream`：`model.stream(prompt)` 返回 `AIMessageChunk` 迭代器，逐块打印（打字机效果）
- 入口：`python -m app.cli invoke "..."` / `python -m app.cli stream "..."`

## 5. 两个实战踩坑（重点学习素材）

### 坑 1：`load_dotenv()` 缺失 → "DEEPSEEK_API_KEY must be set"

**现象**：init_chat_model 报 `Value error, If using default api base, DEEPSEEK_API_KEY must be set`

**原因**：LangChain 的 `api_key` 字段默认值用 `from_env` 机制——读的是**进程环境变量**（`os.environ`），而 `.env` 文件本身不会自动进环境变量。pydantic-settings 读 .env 只是"自己知道配置"，**不会写 os.environ**。

**修复**：`app/config.py` 顶部加 `load_dotenv()`，把 .env 内容注入 os.environ。

**启示**：.env 文件有两个消费方（pydantic-settings 直接读文件、LangChain 读环境变量），必须同时满足。这也解释了为什么很多项目用 `load_dotenv()` 开道。

### 坑 2：模型名前缀 → API 400 "you passed deepseek:deepseek-v4-flash"

**现象**：`init_chat_model(model="deepseek:deepseek-v4-flash", model_provider="deepseek")` 请求 400

**原因（源码验证）**：`_parse_model` 的剥离逻辑是——

```python
model, model_provider = _parse_model(model, model_provider)
# 实测：
# _parse_model("deepseek:deepseek-v4-flash", "deepseek")
#   → ("deepseek:deepseek-v4-flash", "deepseek")   # 前缀没被剥离！
# _parse_model("deepseek-v4-flash", "deepseek")
#   → ("deepseek-v4-flash", "deepseek")            # 无前缀，正常
```

**显式传 model_provider 时，`provider:` 前缀不会被剥离**，原样传给 ChatDeepSeek 再发给 API → 400。

**正确姿势**（二选一）：
```python
# 方式 A（本项目采用）：model 不带前缀 + 显式 model_provider
init_chat_model(model="deepseek-v4-flash", model_provider="deepseek")

# 方式 B：只靠前缀推断，不传 model_provider
init_chat_model(model="deepseek:deepseek-v4-flash")
```

**方案对比（为什么选 A）**：A 显式声明 provider，不依赖推断规则；B 依赖 `_infer_provider` 的猜测（qwen 等推断不出）。A 更稳。但要注意**别把前缀和 model_provider 同时传**——这正是坑的根源。

## 6. 验证记录

```bash
# invoke：一次性完整回复
python -m app.cli invoke "你好，用一句话介绍你自己"
# ✅ 回复: 你好，我是DeepSeek... / model=deepseek-v4-flash / finish_reason=stop / token: 输入89 输出77 总计166

# stream：逐块打字机
python -m app.cli stream "用一句话介绍 FastAPI"
# ✅ 输出: FastAPI 是一个基于 Python 类型提示、高性能、易用的现代 Web 框架...
```

## 7. 知识点映射

| 手册条目 | 落地 |
|---|---|
| 1.1 init_chat_model | model.py 工厂（model_provider 显式指定） |
| 1.1 注意（kwargs 透传） | temperature/max_tokens/timeout/max_retries 全部透传生效 |
| 1.3 推理类参数 | 参数来自 .env 设计值 |
| 1.9 运行方法 invoke/stream | cli.py 两个 demo |
| 2.7 usage_metadata | 打印 token 用量 |
