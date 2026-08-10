# Agent 代码导读（只关注 Agent 逻辑，不看 FastAPI/Vue）

> 适用：只想理解"Agent 是怎么想、怎么调工具、怎么记忆"的代码逻辑，
> 不关心 Web 展示层。总代码约 1500 行（app/ 业务包），Agent 核心约 900 行。

---

## 0. 一句话架构

```
用户消息 → create_agent 组装好的图（LangGraph）
         → 模型（ConfigurableModel / deepseek）  ←→ 工具（6 个）
         → checkpointer（短期记忆）+ Store（长期记忆）
         → 回复消息
中间件（middleware）挂在模型调用前后，RAG 以工具形式存在
```

**Agent 本体 = `app/agent.py` 里的 `create_agent(...)` 一行组装**，其余都是被它引用的"零件"。

---

## 1. 推荐阅读顺序（自底向上：先看零件，再看组装）

| 顺序 | 文件 | 行数 | 看什么 |
|---|---|---|---|
| 1 | `app/config.py` | ~40 | 所有配置从哪来（.env） |
| 2 | `app/model.py` | ~40 | 模型怎么创建、怎么切换（ConfigurableModel） |
| 3 | `app/prompts.py` | ~25 | Agent 的"人设"（system_prompt + 工具使用指引） |
| 4 | `app/tools/course_tools.py` | ~90 | 工具怎么定义（@tool / InjectedToolCallId / return_direct） |
| 5 | `app/tools/memory_tools.py` | ~55 | 跨会话记忆工具（InjectedStore 注入） |
| 6 | `app/tools/knowledge_tools.py` | ~15 | RAG 检索工具（把检索结果喂给模型） |
| 7 | `app/middleware.py` | ~140 | 钩子中间件：日志 / 动态提示 / 重试 / 跳转 |
| 8 | `app/memory.py` | ~45 | 记忆连接（SQLite checkpointer + Store） |
| 9 | `app/rag/` | ~180 | RAG：BM25 / 切片 / 索引 |
| 10 | `app/schemas.py` | ~30 | 结构化输出契约（LearningPlan） |
| 11 | **`app/agent.py`** | ~45 | **组装：create_agent 一行把零件拼起来** |
| 12 | `app/cli.py` | ~230 | 不依赖前端的验证入口（agent / agent-stream / batch） |

**跳过**：`app/main.py`（FastAPI 接口层）、`app/messages.py`（CLI 消息演示）、前端全部。

---

## 2. 核心阅读路径：一次对话发生了什么

以 `python -m app.cli agent "推荐一门 Python 入门课"` 为线索：

```
app/cli.py: demo_agent()
  └─ app/memory.py: memory_ctx()           # 打开 SQLite 连接（checkpointer/store）
      └─ app/agent.py: ensure_agent()      # 首次调用才组装（单例）
          └─ create_agent(
                 model=get_model(),        # → app/model.py（ConfigurableModel）
                 tools=[search_courses, get_course_detail, record_search_log,
                        save_progress, get_progress, search_knowledge],
                 system_prompt=AGENT_SYSTEM_PROMPT,   # → app/prompts.py
                 middleware=MIDDLEWARES,   # → app/middleware.py
                 checkpointer=..., store=...,          # → app/memory.py
             )
  └─ agent.astream({messages}, config)     # 跑图：模型 ↔ 工具 循环
      ├─ 模型节点：看提示词 + 消息历史 → 决定调哪个工具
      ├─ 工具节点：执行工具（course_tools / memory_tools / knowledge_tools）
      ├─ 循环直到模型不再调工具
      └─ checkpointer 每步保存状态（thread_id 维度）
```

**关键理解**：Agent 的"智能"不在代码里，而在**提示词（prompts.py）+ 工具定义（tools/）+ 中间件（middleware.py）** 三者的配合——模型每轮看 system_prompt 决定"要不要调工具、调哪个"。

---

## 3. 每个文件"为什么这么写"（Agent 视角）

### app/model.py —— 一切从模型开始
- `get_model()` 用 `init_chat_model` 统一工厂，`lru_cache` 单例
- `configurable_fields=["model"]`：同一实例运行时切 flash/pro（config 传 `chat_model_model`）
- 注意注释里的坑：**model 不能带 `deepseek:` 前缀**（显式 provider 时不剥离）

### app/prompts.py —— Agent 的行为准则
- 三段式：思考语言规定（中文思考）、工具使用指引（**什么时候调什么工具**）、行为准则
- RAG 集成就是在这里加一句"知识性问题先调 search_knowledge"

### app/tools/course_tools.py —— 工具定义的完整样本
- `@tool` 装饰器：docstring 自动成为工具描述（模型靠它决定何时调用）
- `record_search_log(question, tool_call_id: Annotated[str, InjectedToolCallId])`：参数自动注入
- `get_course_detail(..., return_direct=True)`：**直接返回不送模型加工**（注意 all() 语义：本轮全部 return_direct 才退出）

### app/tools/memory_tools.py —— 长期记忆怎么读写
- `save_progress` / `get_progress`：async 工具，`InjectedStore` 注入
- namespace=("users", user_id) 隔离不同用户；存 JSON 字符串

### app/tools/knowledge_tools.py —— RAG 入口（仅 10 行）
- 检索器 `get_retriever().format(query, top_k=3)` → 拼好带出处的文本给模型

### app/middleware.py —— 钩子机制（阶段 5 重点）
- `LoggingMiddleware`：before/after 全流程日志（最易读，先看它）
- `goodbye_filter`：`@before_model` + `can_jump_to=["end"]`——识别"再见"直接跳 end 省一次调用
- `DynamicPromptMiddleware` / `GuardRetryMiddleware`：`AgentMiddleware` 子类，**必须同时实现 wrap_model_call + awrap_model_call**（sync/async 双版本，否则报 NotImplementedError）
- `MIDDLEWARES` 列表：顺序即执行顺序

### app/memory.py —— 记忆的连接层
- `AsyncSqliteSaver` / `AsyncSqliteStore`：**必须 Async 版**（Web 场景）
- `from_conn_string` + `async with`：连接生命周期归上下文管

### app/rag/ —— 检索知识（阶段 8）
- `bm25.py`：自实现 BM25（词频饱和/长度归一化/IDF）——纯算法，无 LangChain 依赖
- `chunker.py`：切片参数 800/120 + 章节元数据
- `ingest.py`：`Retriever` 抽象（预留向量库升级）

### app/schemas.py —— 结构化输出契约
- `LearningPlan` / `Topic`：模型按这个 schema 输出 JSON，前端直接渲染卡片
- 与 `/plan` 接口配套（deepseek 需关 thinking + function_calling 方式）

---

## 4. 不动 Web 的验证手段（CLI）

```bash
# 项目根目录执行
python -m app.cli agent "推荐一门 Python 入门课"      # Agent 全流程（工具调用可见）
python -m app.cli agent-stream "推荐一门 Python 入门课" # 流式 + 思考过程
python -m app.cli batch "1+1=?" "你好"                 # 批量并发
python -m app.cli chat                                  # 多轮对话（记忆演示）
```

想改代码验证：改 `prompts.py`（加一句指引）→ 重跑 agent → 对比行为差异。
**最快的学习路径**：改一行 → 跑一次 → 看 LangSmith trace（模型实际怎么想怎么做）。

---

## 5. 分析技巧

1. **从日志读流程**：`LoggingMiddleware` 打印 `── Agent 开始/结束/模型调用完成`——一次调用从头读到尾就是整个执行链
2. **从 LangSmith 读决策**：每轮模型调用都能看到"模型为什么调这个工具"（输入输出完整）
3. **从测试用例反推**：`docs/phase3.md`、`docs/phase5.md` 的 TC 表就是行为规范
4. **改验证三步**：改 prompts/tools/middleware → `python -m app.cli agent "测试问题"` → 看日志 + LangSmith
