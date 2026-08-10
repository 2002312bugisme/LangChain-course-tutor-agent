# Agent 构建知识框架（概念协同梳理）

> 依据：LangChain.md 学习笔记 + langchain 1.3.14 源码验证
> 主线：**输入 → 状态流转 → 输出**。一切概念都是这条主线上的一环。

---

## 一、总纲：一条主线

```
用户输入
   │
   ▼
┌─────────────────────────────────────────────────────────┐
│  AgentState（图的黑板：messages / jump_to / structured_response）│
│                                                         │
│  ① 模型节点：读 state → 调模型 → 产出 AIMessage          │
│  ② 路由决策：AIMessage 里有什么？                        │
│     ├─ 有 tool_calls ──▶ ③ 工具节点：执行 → ToolMessage/Command → 回①
│     ├─ 有 structured_response ──▶ 结束                   │
│     └─ 无 tool_calls ──────▶ 结束                        │
│                                                         │
│  全程：中间件钩子围观/干预，checkpointer 按 thread_id 存快照│
└─────────────────────────────────────────────────────────┘
   │
   ▼
最终结果（文本 / 结构化对象 / 流式事件）
```

**记忆口诀**：模型是大脑，消息是语言，state 是黑板，工具是手脚，中间件是监督员，checkpointer 是录像机。

---

## 二、概念地图（按构建流程分七层）

| 层 | 核心概念 | 职责 | 关键 API |
|---|---|---|---|
| ① 模型层 | `init_chat_model`、ChatXxx、ConfigurableModel | 提供"大脑" | `init_chat_model()`、`configurable_fields`、`.bind_tools()` |
| ② 消息层 | 四大消息 + AIMessageChunk + ContentBlock | 模型的语言（也是 state 的内容） | `HumanMessage` 等，支持元组/字典快捷构造 |
| ③ 状态层 | AgentState、reducer、ephemeral | 图的"黑板"：所有节点的共享记忆 | `add_messages`、`EphemeralValue`、`state_schema` |
| ④ 工具层 | @tool、注入、异常、return_direct | 与外部世界的接口 | `@tool`、`InjectedState/Store/CallId`、`ToolException`、`Command` |
| ⑤ 中间件层 | 六钩子 + wrap 系列 | 流程的观察点/控制点，不修改 Agent 代码 | `@before_model`、`@wrap_model_call`、`@dynamic_prompt` |
| ⑥ 记忆层 | checkpointer + Store + trim | 会话恢复 / 跨会话共享 / 上下文控制 | `thread_id`、`InMemorySaver`/`SqliteSaver`、`trim_messages` |
| ⑦ 输出层 | 结构化输出 + 流式 | 结果的可消费形态 | `response_format`、`with_structured_output`、`stream_mode` |

---

## 三、一次完整执行旅程（协同主链）

```
用户输入
  │
  ▼
[before_agent]              ← 只跑一次：权限检查、输入预处理
  │
  ▼
┌── 循环 ──────────────────────────────────────────────┐
│  [before_model]          ← 每次循环：消息预处理、动态上下文   │
│  [wrap_model_call]       ← 包裹模型：重试/降级/缓存          │
│      模型 invoke ──▶ AIMessage（可能带 tool_calls）         │
│  [after_model]           ← 内容审核、响应过滤                │
│  │                                                       │
│  路由函数（model_to_tools）根据 AIMessage 决策：            │
│  │                                                       │
│  ├─ tool_calls 非空 ──▶ [wrap_tool_call] ──▶ ToolNode      │
│  │      执行工具 ──▶ 返回 ToolMessage 或 Command           │
│  │      ToolMessage（默认）→ 回模型节点，继续循环            │
│  │      Command(update/goto) → 改状态/跳转，继续循环        │
│  │                                                       │
│  ├─ structured_response 已写入 ──▶ 退出循环                │
│  └─ tool_calls 为空 ──▶ 退出循环                          │
└──────────────────────────────────────────────────────┘
  │
  ▼
[after_agent]               ← 只跑一次：统计、格式化、清理
  │
  ▼
输出（invoke 全量 / stream 分事件）
```

---

## 四、关键协同点详解

### 4.1 模型节点的产出：AIMessage 是"路由的判决书"

模型每次返回的 AIMessage 决定 Agent 下一步：
- `tool_calls=[]` → 结束（模型认为任务完成）
- `tool_calls=[...]` → 路由到工具节点
- 模型节点顺带写 `structured_response` → 路由检测到即结束

### 4.2 工具节点的两种返回值：ToolMessage vs Command

| | ToolMessage | Command |
|---|---|---|
| 本质 | **结果**（数据） | **指令**（数据 + 控制意图） |
| 内容 | `content` + `tool_call_id`（必须与调用匹配） | `update`（改 state）+ `goto`（跳转）+ `resume`（恢复 interrupt）+ `graph`（目标图） |
| 流程影响 | 无（由路由按默认规则决定：回模型） | 有（主动改状态、改走向） |
| 何时用 | 纯结果交付：查询、计算、取数 | 要顺带"做点什么"：追加消息并跳转、恢复挂起、向父图发令 |

```python
# 纯结果 → ToolMessage（自动生成，绑定 tool_call_id）
return "杭州：晴，25°C"

# 带副作用 → Command
return Command(update={"messages": [ToolMessage(content="...", tool_call_id=...)]}, goto="model")
```

### 4.3 Command 四个字段的分工

```
Command(
    update={"messages": [...]},   # 改状态（走 reducer 合并）
    goto="tools",                 # 跳转目标：节点名 / Send 列表 / PARENT
    resume="用户确认了",           # 配合 interrupt()：从暂停处恢复
    graph=Command.PARENT,         # 发给哪个图（当前图 / 父图）
)
```

层级注意：`goto` 是 LangGraph 节点级原语；中间件里用 `jump_to`（受限替代，见 4.5）。

### 4.4 中间件的两种返回值

| 返回值 | 效果 |
|---|---|
| `None` | 纯观察（日志、监控） |
| `dict` | 合并进 state（走 reducer；messages 用 add_messages 追加） |
| 含 `jump_to` 的 dict | 合并 + 流程跳转（`model`/`tools`/`end`），需 `can_jump_to` 白名单声明 |

### 4.5 流程控制的三个层次（容易混，重点）

| 层次 | 原语 | 谁在用 | 是否受白名单约束 |
|---|---|---|---|
| 节点层（图） | `Command(goto=...)` | 工具节点、图节点 | ❌ 无（图路由函数处理） |
| 中间件层（Agent） | `jump_to` 状态字段 | before/after 钩子 | ✅ `can_jump_to` 白名单 |
| 包裹层（wrap） | `Command(goto=...)` | wrap_model_call | ❌ **当前不支持**（NotImplementedError，用 jump_to 代替） |

### 4.6 记忆与状态的分工

```
checkpointer（会话级）          Store（跨会话）
按 thread_id 存整个 state 快照   按 namespace+key 存业务数据
├─ 同一 thread_id 恢复对话       ├─ 用户偏好、学习进度
├─ SqliteSaver 持久化到数据库    └─ InjectedStore 注入工具读写
└─ 每次调用后自动存
         ↓ 配合
trim_messages / 摘要压缩（控制上下文 token 增长）
```

### 4.7 结构化输出的闭环

```
create_agent(response_format=Schema)
        │ 内部选策略（Auto/Tool/Provider）
        ▼
模型产出 → 解析 → 写入 state["structured_response"]
        ▼
路由检测到该字段 → 退出循环 → result["structured_response"] 可取 Pydantic 对象
```

### 4.8 流式输出与执行过程的对应

```
stream_mode="messages" → token 级（打字机效果，metadata.langgraph_node 标来源节点）
stream_mode="updates"  → 节点级（"正在调用工具"这类过程展示）
stream_mode="values"   → 全量 state 快照（调试）
stream_mode="custom"   → middleware 用 stream_writer 发自定义事件
可组合：stream_mode=["updates", "custom"]
```

---

## 五、决策速查表（开发时对照）

| 我要... | 用... |
|---|---|
| 换模型/多模型切换 | `init_chat_model(configurable_fields=[...])` + 运行时 config |
| 给模型装工具 | `create_agent(tools=[...])`（内部自动 bind + 建 ToolNode） |
| 工具结果直接当答案 | `@tool(return_direct=True)`（注意：混调时不会中断，见实测） |
| 工具要读上下文/存数据 | `InjectedState` / `InjectedStore` / `InjectedToolCallId` 形参注入 |
| 工具出错不崩溃 | `handle_tool_error=True`（单数，工具级） |
| 工具带副作用返回 | `return Command(update=..., goto=...)` |
| 会话续聊 | checkpointer + 相同 `thread_id` |
| 跨会话存用户数据 | `store=` + `InjectedStore` |
| 控制流程/拦截内容 | middleware 钩子 + `jump_to`（记得 `can_jump_to`） |
| 模型调用重试/降级/缓存 | `@wrap_model_call`（handler 回调） |
| 提取结构化数据 | `response_format=`（Agent 内）或 `with_structured_output()`（纯模型） |
| 前端打字机效果 | `stream_mode="messages"` |
| 展示思考过程 | `stream_mode="updates"`（或 messages + metadata 节点名） |

---

## 六、一句话总览

**Agent = 状态机**：`state` 是唯一真相（黑板），`消息` 是写在黑板上的内容，`模型` 负责读写并产出判决（tool_calls / structured_response），`工具` 是执行者（结果回写黑板，或发 Command 改变走向），`中间件` 在黑板读写前后插眼（观察/干预），`checkpointer/store` 让黑板跨请求、跨会话存活，`流式/结构化` 决定结果怎么交付。
