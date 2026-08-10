# 阶段 4：Web 化（FastAPI + Vue 3 + SSE 流式）

> 状态：✅ 已完成并验证（后端接口 + 前端代理全链路）
> 对应设计文档：DESIGN.md 阶段 4（含新增需求⑦ Web 落地）
> 验证时间：2026-08

## 1. 目标

把 Agent 搬上 Web：FastAPI 后端 + Vue 3 前端，流式打字机 + **思考过程实时展示**（需求⑦ Web 版）。

## 2. 搭建步骤

1. 后端：新建 `app/main.py`（FastAPI 三个接口），`python -m uvicorn app.main:app --port 8000` 启动
2. 前端：`npm create vite@latest frontend -- --template vue` 建 Vue 3 工程，`npm install` 装依赖
3. 前端代码：`src/App.vue`（聊天界面 + SSE 解析）、`vite.config.js`（开发代理）
4. 验证：/health、/chat、/chat/stream 三接口 + 5173→8000 代理链路

## 3. 目录变更

```
app/main.py          # 新增：FastAPI 入口（3 接口 + CORS）
frontend/            # 新增：Vue 3 + Vite 工程
  ├── vite.config.js # 新增：开发代理 5173 → 8000
  └── src/App.vue    # 新增：聊天界面（思考区/工具区/回复区）
```

## 4. 核心实现详解

### 4.1 后端接口设计（app/main.py）

| 接口 | 方法 | 用途 | 实现 |
|---|---|---|---|
| `/health` | GET | 健康检查 + 模型信息 | 返回 model/provider |
| `/chat` | POST | 完整回复（非流式） | `agent.ainvoke` → thinking + reply + 工具摘要 |
| `/chat/stream` | POST | SSE 流式 | `agent.astream(stream_mode=["messages","updates"])` |

**请求体**：`ChatRequest {message, thread_id?}`——thread_id 预留（阶段 6 checkpointer 用）。

### 4.2 SSE 事件协议（需求⑦落地）

```
data: {"type": "reasoning", "content": "..."}   # 思考块（灰色展示）
data: {"type": "token",     "content": "..."}   # 回复块（打字机）
data: {"type": "tool",      "name": "...", "result": "..."}  # 工具执行
data: {"type": "done",      "message": "完成"}
data: {"type": "error",     "message": "..."}
```

**分流逻辑**（预研结论落地）：
- `stream_mode="messages"` → `(chunk, metadata)`：`chunk.additional_kwargs["reasoning_content"]` 有值发 reasoning；`chunk.content` 有值发 token
- `stream_mode="updates"` → `{node: update}`：tools 节点的 ToolMessage → 发 tool 事件（前端显示"正在调用 XX 工具"）

### 4.3 前端实现（App.vue）

- **消息模型**：`{role, content, thinking, tools[]}`——每条 AI 消息承载思考区+工具区+回复区三段
- **SSE 解析**：`fetch` + `ReadableStream` 逐块读取，按 `\n\n` 分割、`data:` 前缀解析 JSON（浏览器原生流式读取，无需 EventSource——因为要 POST）
- **三段式渲染**：🤔 思考（灰底小字）→ 🔧 工具调用（蓝底小字）→ 💬 回复（正文打字机）
- **vite proxy**：`/chat`、`/health` 转发到 8000，前端零 CORS 配置

### 4.4 为什么这样设计（方案对比）

| 决策点 | 方案 A（采用） | 方案 B | 理由 |
|---|---|---|---|
| SSE 实现 | fetch + ReadableStream | EventSource | EventSource 只支持 GET，我们需要 POST 带 body |
| 前端代理 | Vite dev proxy | 后端 CORS 全开 | 开发零配置，浏览器无跨域问题；CORS 仍保留双保险 |
| 流式模式 | 混合 `["messages","updates"]` | 只 messages | messages 拿 token，updates 拿工具节点信息，互补 |
| 思考展示 | 独立灰色区 | 混入回复流 | 需求⑦明确要分段展示，体验清晰 |

## 5. 验证记录

```bash
# 后端三接口
curl http://127.0.0.1:8000/health
# → {"status":"ok","model":"deepseek-v4-flash","provider":"deepseek"}

POST /chat "推荐一门 Python 入门课"
# → thinking + reply + tools: [search_courses, record_search_log]

POST /chat/stream "有没有 Vue 课程？"
# → reasoning×74 → token×83 → tool×2 → done

# 前端代理全链路
curl http://localhost:5173/chat/stream ...（经 5173 转发）
# → reasoning×152 → token×78 → tool×2 → done ✅
```

## 6. 启动方式（给用户）

```bash
# 终端 1：后端
cd D:\Code\LangChain_1.2
python -m uvicorn app.main:app --port 8000

# 终端 2：前端
cd D:\Code\LangChain_1.2\frontend
npm run dev

# 浏览器打开 http://localhost:5173
```

## 7. 测试用例表（浏览器人工测试）

| 编号 | 操作 | 预期 |
|---|---|---|
| TC-13 | 打开 http://localhost:5173 | 页面标题"课栈·编程学习助手"，空态提示 |
| TC-14 | 发送"推荐一门 Python 入门课" | 🤔 思考区灰色实时滚动 → 🔧 工具调用提示 → 💬 回复逐字打字机 |
| TC-15 | 发送"有没有 Vue 进阶课程？" | 同上，回复含 fe-201 课程信息 |
| TC-16 | 发送"1+1等于几" | 无工具调用，直接回复 2 |
| TC-17 | 连续发送两条问题 | 消息列表正常追加，页面自动滚动到底部 |

## 8. 知识点映射

| 手册条目 | 落地 |
|---|---|
| 4.3 CompiledStateGraph.astream | chat/stream 接口 |
| 9.1 invoke vs stream（异步铁律） | ainvoke / astream 接口 |
| 9.2 stream_mode 组合 | ["messages", "updates"] 混流 |
| 9.3 config（thread_id 预留） | ChatRequest.thread_id |
| 需求⑦ | SSE reasoning/token 事件 + 前端思考区 |
| FastAPI 异步 | 事件循环不阻塞（ainvoke 挂起而非卡死） |
