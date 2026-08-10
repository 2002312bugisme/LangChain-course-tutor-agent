# 🎓 课栈 · LangChain 编程学习助手

用 LangChain + LangGraph 从零搭建的完整 Agent 应用：DeepSeek 大脑 + 工具调用 + 会话/长期记忆 + 中间件 + 结构化输出 + RAG 知识库 + Web 聊天界面。**10 个阶段渐进式构建，每阶段有独立说明文档（`docs/phase-N.md`）与测试用例。**

## 功能一览

- 🤖 **Agent**：课程查询 / 学习规划 / 知识库问答（自动选择工具）
- 💬 **Web 聊天**：SSE 流式输出，思考过程实时展示（可折叠）、工具调用可见、Markdown 渲染
- 🧠 **记忆**：短期会话记忆（checkpointer）+ 长期用户进度（Store），SQLite 持久化
- 📊 **结构化输出**：学习计划卡片（`/plan`，with_structured_output）
- 📚 **RAG 知识库**：BM25 + jieba 中文检索（`app/rag/`，自实现）
- 🔄 **模型切换**：运行时 flash / pro 动态切换（ConfigurableModel）
- 📥 **会话导出**：任意会话导出 Markdown

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt
cd frontend && npm install && cd ..

# 2. 配置 .env（复制 .env 模板，填 DEEPSEEK_API_KEY）
cp .env.example .env   # 或手动创建

# 3. 准备知识库（可选）：把要检索的 .md 笔记放进 data/knowledge/

# 4. 启动后端（8000）
python -m uvicorn app.main:app --port 8000

# 5. 启动前端（5173，已配代理）
cd frontend && npm run dev
# 浏览器打开 http://localhost:5173
```

## 环境变量（.env）

| 键 | 说明 | 默认 |
|---|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek API Key（必填） | — |
| `DEEPSEEK_MODEL` | 默认模型 | `deepseek-v4-flash` |
| `TEMPERATURE` | 采样温度 | 0.3 |
| `MAX_TOKENS` | 最大输出 token（推理模型会吃预算） | 1000 |
| `TIMEOUT` / `MAX_RETRIES` | 请求超时 / 重试 | 30 / 2 |
| `LANGCHAIN_TRACING_V2` | LangSmith 追踪开关（阶段 9） | false |
| `LANGCHAIN_API_KEY` | LangSmith API Key（https://smith.langchain.com 获取） | — |
| `LANGCHAIN_PROJECT` | LangSmith 项目名 | `langchain_learning_agent` |

> LangSmith：设置 `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY` 后，所有 Agent 调用自动上报，可在控制台查看完整 trace（模型调用、工具链、token 用量）。不开也不影响功能。

## CLI 演示（不依赖前端）

> ⚠️ 必须在**项目根目录**运行（`python -m` 按当前目录找 `app` 包）

```bash
python -m app.cli invoke "你好"          # 单次调用
python -m app.cli stream "你好"          # 流式输出
python -m app.cli chat                   # 多轮对话（带记忆演示）
python -m app.cli agent "推荐一门 Python 入门课"   # Agent 全流程
python -m app.cli batch "1+1=?" "你好"   # 批量并发（阶段 9）
```

## 主要接口

| 接口 | 说明 |
|---|---|
| `POST /chat/stream` | SSE 流式聊天（`{message, thread_id?, model?}`） |
| `POST /chat` | 非流式聊天（返回 thinking/reply/tools） |
| `POST /plan` | 结构化学习计划（返回 LearningPlan JSON） |
| `POST /batch` | 批量并发处理（无状态） |
| `GET /threads` · `GET /threads/{tid}/messages` | 会话列表 / 历史 |
| `DELETE /threads/{tid}` · `POST /threads/{tid}/rename` | 删除 / 重命名会话 |
| `GET /threads/{tid}/export` | 导出会话 Markdown |
| `GET /models` | 可用模型列表（模型切换） |
| `POST /knowledge/ingest` | 知识库重建索引（force） |

## 项目结构

```
app/
├── config.py          # .env 集中配置
├── model.py           # ConfigurableModel 工厂（运行时切模型）
├── agent.py           # create_agent 工厂（工具/中间件/记忆注入）
├── middleware.py      # 钩子中间件（日志/动态提示/重试/跳转）
├── memory.py          # checkpointer + Store（SQLite）
├── prompts.py         # system_prompt（含思考语言约束）
├── schemas.py         # LearningPlan 结构化输出模型
├── rag/               # RAG：BM25 自实现 + 切片 + 索引
├── tools/             # 课程工具 / 记忆工具 / 知识库检索
└── main.py            # FastAPI（SSE/会话/导出/批量/错误分层）
docs/phase-*.md        # 每阶段说明 + 测试用例 + 踩坑记录
DESIGN.md              # 整体设计 + 10 阶段路线图
```

## 踩坑记录（详见各阶段文档）

- deepseek 推理模型：thinking 模式不支持 structured output 的 tool_choice → 需 `extra_body={"thinking": {"type": "disabled"}}` + `method="function_calling"`
- `@tool(handle_tool_error=...)` 在 langchain 1.3.14 会 TypeError → 装饰后赋值或 `StructuredTool.from_function`
- `stream_mode="messages"` 会把工具节点结果混入 → 按 `metadata.langgraph_node != "model"` 过滤
- 挂 checkpointer 的 agent，`abatch`/`ainvoke` 必须显式传 `configurable.thread_id`
- Vite 代理是白名单：新增后端接口记得加 `frontend/vite.config.js`，改后必须重启 Vite
- 前端 Vue3 响应式：流式修改消息必须从 `messages.value[i]` 取代理引用

## 10 阶段路线

0 项目骨架 · 1 模型接入 · 2 消息体系 · 3 工具与 Agent · 4 Web 端 · 5 中间件 · 6 记忆 · 6.5 UI 优化 · 7 结构化输出 · 8 RAG · 9 高级收尾（LangSmith/模型切换/batch/错误分层）
