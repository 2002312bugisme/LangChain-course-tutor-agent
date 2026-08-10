# 学习 Agent 项目设计文档（v0.2 定稿）

> 目标：从最小 Agent 起步，分阶段构建一个 Web 端 Agent，最大程度覆盖 LangChain.md / agent_api_reference.md 中的知识点，最终包含 RAG + 切片。
> **决策记录（2026-08 用户确认）**：① 业务=编程学习助手；② RAG 知识源=笔记+官方文档双源；③ Embedding=BM25 关键词检索（零向量依赖，中文需 jieba 分词）；④ 前端=Vue 3 搭建；⑤ 后端=FastAPI（轻量，异步+SSE 原生友好）；⑥ 允许按需 pip 安装新包。

---

## 1. 业务方向（待确认）

**推荐：编程学习助手（"菜鸟教程"风格）**

理由：
- 与笔记中所有示例（课程查询、学习进度、路径推荐）天然一致
- 能自然承载全部功能点：工具（课程查询）、记忆（进度）、结构化输出（学习计划）、RAG（文档问答）
- **RAG 知识源素材现成**：用你自己的 `LangChain.md` 学习笔记当知识库——问它"LangChain 里 temperature 是什么"很有实战感

核心能力（最终形态）：
1. **对话问答**：编程/LangChain 知识咨询
2. **课程查询工具**：按关键词/难度查"课程库"（本地 JSON 模拟）
3. **学习进度记忆**：记住用户学过的主题，跨会话保持
4. **学习路径推荐**：结构化输出（推荐计划 JSON）
5. **知识库 RAG**：对 LangChain.md 等文档切片 → 向量检索 → 带引用回答

---

## 2. 技术栈

| 组件 | 选型 | 说明 |
|---|---|---|
| 后端 | **FastAPI** + uvicorn | 轻量、异步原生、SSE 流式简单（不用 Django，重且同步） |
| 模型 | DeepSeek（`deepseek-v4-flash`）via `init_chat_model` | 已有 key；OpenAI 兼容 |
| Agent | `create_agent`（langchain 1.3.14） | 已装 |
| 前端 | **Vue 3 + Vite**（`frontend/` 独立工程） | 用户指定；fetch 读 SSE 流实现打字机 |
| 记忆 | InMemorySaver → SqliteSaver（`langgraph-checkpoint-sqlite`） | 阶段 6 引入 |
| RAG 检索 | **BM25**（零向量依赖）| 中文需 jieba 分词预处理；chunk 级检索 |
| 切片 | `RecursiveCharacterTextSplitter` | 阶段 8 引入 |
| 嵌入 | **不用**（BM25 方案） | 后续可平滑升级到向量库 |
| 观测 | LangSmith（可选，环境变量开关） | 已有笔记 |

---

## 3. 目录结构

```
D:\Code\LangChain_1.2\
├── LangChain.md                  # 学习笔记（兼作 RAG 知识源①）
├── DESIGN.md                     # 本设计文档
├── .env                          # DEEPSEEK_API_KEY 等（gitignore）
├── app/                          # 后端（FastAPI）
│   ├── main.py                   # FastAPI 入口 + 路由
│   ├── config.py                 # 加载 .env，集中配置
│   ├── model.py                  # init_chat_model 工厂 + ConfigurableModel
│   ├── tools/                    # 工具定义
│   │   ├── course_tools.py       # 课程查询（含注入/异常/return_direct 演示）
│   │   └── knowledge_tools.py    # RAG 检索工具（阶段 8）
│   ├── memory.py                 # checkpointer / store（阶段 6）
│   ├── middleware.py             # 中间件（日志/重试/动态 prompt）
│   ├── schemas.py                # Pydantic 模型（结构化输出 + 请求/响应）
│   └── rag/                      # RAG 模块（阶段 8，BM25）
│       ├── ingest.py             # 切片 + 入库
│       └── retriever.py          # BM25 检索
├── data/
│   ├── courses.json              # 模拟课程库
│   └── knowledge/                # RAG 文档源（笔记副本 + 官方文档抓取）
├── frontend/                     # 前端（Vue 3 + Vite，阶段 4 起）
└── requirements.txt
```

---

## 4. 分阶段路线图（每阶段=增量+验收）

### 阶段 0：项目骨架 + 配置
- **做**：目录结构、`.env`、`config.py`（dotenv 加载 DEEPSEEK_API_KEY/MODEL/BASE_URL）、requirements.txt
- **覆盖知识点**：0.1 .env、模型参数设计（temperature=0.3, max_tokens, timeout=30, max_retries=2）
- **验收**：`python -c "from app.config import settings; print(settings.model)"` 跑通

### 阶段 1：最小模型调用
- **做**：`model.py` 用 `init_chat_model(model="deepseek:deepseek-v4-flash", ...)` 建模型；CLI 测试 `invoke("你好")` 与 `stream()` 打字机
- **覆盖知识点**：init_chat_model、参数、invoke/stream、AIMessage 结构（usage_metadata 打印）
- **验收**：CLI 一次问答 + 一次流式输出成功

### 阶段 2：消息层
- **做**：封装消息构造（HumanMessage/SystemMessage）、多轮消息列表传递、`trim_messages` 演示
- **覆盖知识点**：四大消息、快捷构造、trim_messages
- **验收**：手动传 2 轮对话历史，模型能引用前文

### 阶段 3：最小 Agent（两个工具 + create_agent）
- **做**：
  - 工具 1：`search_courses(keyword, level)` 查 data/courses.json —— 普通工具（带 args_schema、ToolException、handle_tool_error 演示）
  - 工具 2：`get_course_detail(course_id)` 设 `return_direct=True` —— 演示"结果即答案"
  - `create_agent(model, tools, system_prompt)` + CLI 跑通
- **覆盖知识点**：@tool 全套、工具注册、注入（InjectedToolCallId 演示）、异常处理、return_direct（**附带复测混调 all() 语义**）
- **验收**：问"有没有 Python 入门课"→ 模型自动调工具回答

### 阶段 4：Web 化（FastAPI + 流式）
- **做**：`main.py` 三个接口：`GET /health`、`POST /chat`（ainvoke）、`POST /chat/stream`（astream + SSE）；`static/index.html` 打字机页面
- **覆盖知识点**：ainvoke/astream、SSE、事件循环不阻塞、config（run_name/tags/metadata）
- **验收**：浏览器打开页面，多轮对话 + 流式输出正常

### 阶段 5：中间件
- **做**：日志中间件（before_agent/after_model 打印统计）、`@wrap_model_call` 重试/降级演示、`@dynamic_prompt`（注入当前时间/用户等级）
- **覆盖知识点**：六钩子、返回值语义、can_jump_to/jump_to（"再见"提前结束）、wrap 洋葱
- **验收**：控制台可见钩子触发顺序；说"再见"直接结束

### 阶段 6：记忆
- **做**：checkpointer（SqliteSaver）+ thread_id 会话管理；Store 存用户等级/已学主题；工具注入 InjectedStore 读写
- **覆盖知识点**：checkpointer vs Store、thread_id、get_state/update_state
- **验收**：刷新页面/重启服务后同一 thread_id 能续聊；跨会话记住用户水平

### 阶段 7：结构化输出
- **做**：`LearningPlan` Pydantic（嵌套：goal/level/topics[]），`response_format=`；顺带演示 ToolStrategy 与 AutoStrategy 对比
- **覆盖知识点**：response_format、三策略、structured_response 退出机制、嵌套 schema
- **验收**：问"帮我规划学 LangChain 的路径"→ 返回结构化 JSON，前端渲染成卡片

### 阶段 8：RAG + 切片（重点阶段，BM25 方案）
- **做**：
  - 知识源双份：① `LangChain.md` 副本；② 抓取 2~3 篇 LangChain 官方文档（如 init_chat_model / create_agent 页面）存为 md
  - `ingest.py`：读文档 → `RecursiveCharacterTextSplitter`（chunk_size/overlap 调参）→ 存 JSON（含来源/章节/序号元数据）
  - `retriever.py`：BM25 检索器（jieba 分词，中文专用）→ 按分数取 top-k chunk
  - `knowledge_tools.py`：`search_knowledge(query)` 检索工具注入 Agent，system_prompt 指示"知识库问题先检索再回答，标注出处"
- **覆盖知识点**：切片参数、BM25 检索原理、检索+工具协同、上下文注入控制
- **验收**：问"init_chat_model 和 ChatOpenAI 有什么区别"→ 回答来自知识库且能引出处

### 阶段 9：高级收尾
- **做**：LangSmith 开关（tracing env）、ConfigurableModel 模型切换接口（flash/pro 动态选）、错误处理完善（SDK 异常分层）、batch 接口演示
- **覆盖知识点**：tracing、configurable_fields、异常体系、batch
- **验收**：LangSmith 里能看到完整 trace；接口能切模型

---

## 5. 接口设计（草案）

```
GET  /health                    → {"status": "ok", "model": ...}
POST /chat                      {thread_id?, message}   → {reply, thread_id, usage}
POST /chat/stream               {thread_id?, message}   → SSE: {type: "token"|"tool"|"done"|"error", data}
POST /plan                      {goal, level}           → LearningPlan JSON（结构化输出）
POST /knowledge/ingest          {path?}                  → 切片入库（阶段 8）
GET  /threads/{tid}             → 会话状态（get_state）
```

---

## 6. 数据模型

```python
# courses.json（模拟课程库）
{"id": "py-101", "title": "Python 入门", "level": "入门", "tags": [...], "duration_hours": 8}

# 会话（checkpointer）
thread_id → AgentState（messages 历史）

# 用户资料（Store）
namespace ("users", "u_xxx") → key "profile" → {"level": "入门", "completed_topics": [...]}

# 知识库（Chroma collection "langchain_notes"）
chunk 文本 + 元数据（来源文件、章节标题、chunk 序号）
```

---

## 7. 已确认决策（用户拍板）

| 决策项 | 结论 |
|---|---|
| 业务方向 | 编程学习助手 |
| RAG 知识源 | 笔记 + 官方文档双源 |
| Embedding | BM25 关键词检索（jieba 分词），零向量依赖 |
| 后端 | FastAPI（轻量） |
| 前端 | Vue 3 + Vite |
| 装包 | 允许按需安装 |

## 8. 风险与预案

| 风险 | 预案 |
|---|---|
| BM25 中文检索效果一般（同义词/语义不匹配） | ① jieba 分词提升；② 预留升级向量库的接口（Retriever 抽象） |
| 官方文档抓取失败（反爬/改版） | 降级为仅本地笔记；或手动粘贴文本 |
| DeepSeek 接口限流 | max_retries + 中间件重试（阶段 5 正好演示） |
| Vue 工程初始化 npm 超时（本机 npm 超时） | 后端先行，前端用 CDN 版 Vue 3 单文件兜底 |
| SqliteSaver 装包失败 | 先用 InMemorySaver 跑通，再升级 |
