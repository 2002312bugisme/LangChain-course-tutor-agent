# 阶段 9：高级收尾（LangSmith + 模型切换 + 错误分层 + batch）

> 上一阶段：阶段 8（RAG 知识库）✅
> 本阶段新增：ConfigurableModel 运行时切换、`/models`、`/batch`、错误分层处理、CLI batch、前端模型下拉、README 补全

---

## 1. 本阶段做什么

设计文档阶段 9 的四件事全部落地：

| 任务 | 实现 |
|---|---|
| LangSmith 开关 | config 已有字段；README 补使用说明（无 key 不追踪，有 key 自动上报 trace） |
| ConfigurableModel 模型切换 | `get_model()` 升级为 `_ConfigurableModel`，`/models` 接口 + 前端下拉 |
| 错误处理（SDK 异常分层） | FastAPI exception handlers：400/429/504/502/500 统一 JSON 结构 |
| batch 接口演示 | `POST /batch` + CLI `python -m app.cli batch "q1" "q2"` |

## 2. 关键实现与知识点

### 2.1 ConfigurableModel（运行时切换模型）

```python
get_model() = init_chat_model(
    model=settings.deepseek_model,
    model_provider="deepseek",
    ...,
    configurable_fields=["model"],  # 允许运行时切换
    config_prefix="chat_model",     # config 键前缀
)
```

- 返回 `_ConfigurableModel`：实例本身单例，但每次调用时根据
  `config["configurable"]["chat_model_model"]` 解析实际模型
- 请求侧传 `{"configurable": {"chat_model_model": "deepseek-v4-pro"}}` 即可切换
- 实测：flash/pro 切换生效（回复风格可辨），**模型名是 `deepseek-v4-pro`（不是 deepseek-v4）**
- 可用模型列表：`MODEL_OPTIONS = ["deepseek-v4-flash", "deepseek-v4-pro"]`（API 实测）

### 2.2 ⚠️ 挂 checkpointer 的 agent，config 必须显式带 thread_id

两个实测报错（同一根源）：
- `_thread_config(None, model)` 返回 `{"configurable": {"chat_model_model": ...}}`
  → `KeyError: 'thread_id'`（checkpointer 直接取键，**不合并默认值**）
- `abatch(inputs)` 不传 config → `Checkpointer requires ... thread_id`

**修复**：`_thread_config` 总是返回 `{"configurable": {"thread_id": ...}}`，
无会话时用 `"default"`；CLI batch 为每个请求分配独立 `batch-{i}` thread。

### 2.3 错误分层（SDK 异常体系）

```
BadRequestError（400 参数/模型名错）  → 400 {"code": "bad_request", ...}
RateLimitError（429 限流）           → 429 {"code": "rate_limit"}
APITimeoutError（超时）              → 504 {"code": "timeout"}
APIConnectionError（网络）           → 502 {"code": "connection"}
Exception（兜底）                    → 500 {"code": "internal"}
```

- 统一错误结构 `{"code", "message"}`，前端可据此区分错误类型
- SSE 流内错误不中断连接：`event_gen` try/except → 发 `error` 事件（前端已有处理）

### 2.4 batch（批量并发）

- `POST /batch {messages: [...]}`：`agent.abatch()` 并发处理，总耗时 ≈ 单个请求耗时
- 实测：3 个请求 3.5s（串行约 10s+）
- **batch 无状态**：不接 thread_id（每次独立），不写会话历史
- CLI：`python -m app.cli batch "1+1=?" "你好"`（为每个请求分配独立 thread）

### 2.5 前端模型切换

- header 新增模型下拉（`fetch /models` 填充，localStorage 记忆选择）
- 请求体带 `model` 字段（`currentModel`），切换即时生效
- 显示短名：`deepseek-v4-flash → flash`、`deepseek-v4-pro → pro`

## 3. 测试用例

| 编号 | 操作 | 预期 |
|---|---|---|
| TC-57 | header 模型下拉选择 pro → 发送问题 | 回复风格与 flash 不同（pro 更详细） |
| TC-58 | 选择 flash → 发送 | 恢复 flash 回复；选择持久化（刷新后保留） |
| TC-59 | `python -m app.cli batch "1+1=?" "你好"` | 并发执行，总耗时 ≈ 单请求 |
| TC-60 | 直接调 `POST /chat` 传非法 model | 400 `{"code": "bad_request"}`（错误分层） |
| TC-61 | 正常聊天/计划/知识库问答回归 | 全部正常（model 缺省走默认模型） |
| TC-62 | 无 LangSmith key 启动 | 正常（不追踪不报错） |

## 4. 验证记录

- `/models` → `['deepseek-v4-flash', 'deepseek-v4-pro']` ✅
- flash / pro 切换回复可辨 ✅
- `/batch` 3 请求 3.5s（并发）✅；CLI batch 2 请求 3.2s ✅
- 非法模型名 → 400 `{"code": "bad_request"}` ✅
- 修复过程：`_thread_config` KeyError / abatch thread_id 校验 —— 已记录于 2.2

## 5. 提交

- `git commit: feat(phase9): 模型切换 + 错误分层 + batch + LangSmith 说明`

---

## 6. 对话补充：验收过程记录

### 6.1 模型下拉不显示（Vite 代理第三次坑）

- 用户反馈：header 没有模型下拉
- 根因：`/models` 新增后未加代理白名单 → 前端拿到 index.html → JSON 解析失败 → 列表空
- 修复：proxy 加 `'/models'`、`'/batch'` + 重启 Vite
- **教训**：这是第三次（/threads、/plan、/models）——README 已固化"新增后端接口先加代理"

### 6.2 CLI 运行目录坑

- 用户从 `docs/` 目录跑 `python -m app.cli batch` → `ModuleNotFoundError: No module named 'app'`
- 根因：`python -m` 以**当前工作目录**为模块搜索起点
- 修复：README 醒目注明"必须在项目根目录运行"

### 6.3 LangSmith 配置与验证全过程

- **第一次 key 401**：`LangSmithAuthError: 401 Invalid token`（key 无效）
- **第二次 key 验证**：显式 `Client(api_key=...)` → list_runs 成功 ✅
- **验证陷阱**：脚本进程不加载 .env（环境变量无 key）→ 查询必须显式传 key；后端进程必须重启才读新配置
- **最终确认**：后端跑一次"推荐一门 Python 入门课"→ LangSmith 出现 5 条 runs：
  `[chain] logging.after_agent → [tool] get_course_detail → [chain] tools → [chain] logging.after_model → [llm] ChatDeepSeek 1.6s`
- **后续使用**：所有调用自动上报；LangSmith 控制台可看时间线/LLM 调用/token 用量/工具参数/中间件钩子
- **安全**：key 只存 .env（gitignore），Git 历史无明文
