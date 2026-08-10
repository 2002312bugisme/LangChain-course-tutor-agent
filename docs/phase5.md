# 阶段 5：中间件（说明文档）

> 状态：✅ 已完成并验证（CLI + Web 双端）
> 对应设计文档：DESIGN.md 阶段 5
> 验证时间：2026-08

## 1. 目标

掌握 Middleware 六钩子：观察型（日志）、控制型（jump_to）、包裹型（wrap 洋葱），以及 @dynamic_prompt 动态提示词；落地需求备选方案"思考语言强约束"。

## 2. 搭建步骤

1. 新建 `app/prompts.py`：静态 system_prompt 独立成模块（避免 agent.py ↔ middleware.py 循环导入）
2. 新建 `app/middleware.py`：4 个中间件
3. 修改 `app/agent.py`：create_agent 挂载 `middleware=MIDDLEWARES`
4. 重启后端（uvicorn 加载新代码）
5. 源码验证：AgentMiddleware 钩子字段、@dynamic_prompt 底层实现（wrap_model_call）、wrap 洋葱顺序实验

## 3. 目录变更

```
app/prompts.py    # 新增：静态提示词（供 agent.py 和 middleware.py 共用）
app/middleware.py # 新增：4 个中间件
app/agent.py      # 修改：挂载 MIDDLEWARES
```

## 4. 中间件详解

### 4.1 LoggingMiddleware（类继承方式）

```python
class LoggingMiddleware(AgentMiddleware):
    name = "logging"
    def before_agent(self, state, runtime): ...  # 1 次：Agent 开始
    def after_model(self, state, runtime): ...    # 每次：模型调用后
    def after_agent(self, state, runtime): ...    # 1 次：Agent 结束
```

**观察型中间件**：所有钩子返回 None（不改状态，纯日志）。
实测输出：
```
[日志] ── Agent 开始，当前 1 条消息 ──
[日志] 模型调用完成：回复 0 字符，含 2 个工具调用   ← 两次（多步推理）
[日志] ── Agent 结束，共 6 条消息 ──
```

### 4.2 goodbye_filter（装饰器 + can_jump_to 白名单）

```python
@before_model(can_jump_to=["end"])
def goodbye_filter(state, runtime):
    if 最后一条消息是道别语:
        return {"jump_to": "end",
                "messages": [AIMessage(content="再见！期待下次为你服务。")]}
```

**实测**：输入"再见" → 日志显示**没有模型调用**（[日志] 直接从开始到结束）——jump_to="end" 真的跳过了模型节点，返回预设消息。`can_jump_to` 白名单是安全机制：不声明则 jump_to 被忽略。

### 4.3 personalized_prompt（@dynamic_prompt）

```python
@dynamic_prompt
def personalized_prompt(request) -> str:
    # request.state 拿消息数，动态生成
    return f"{AGENT_SYSTEM_PROMPT}\n\n## 本次对话上下文\n- 当前时间：{now}...\n- 这是第 {n} 轮对话..."
```

**关键点（源码验证）**：
- @dynamic_prompt 底层就是 **wrap_model_call**（装饰器生成专用中间件类，字段 `state_schema/tools/wrap_model_call`）
- 每次模型调用前执行，**覆盖**静态 system_prompt
- request 中拿不到原静态 prompt → **必须手动拼接**（所以把 AGENT_SYSTEM_PROMPT 抽到 prompts.py）

### 4.4 guard_and_retry（@wrap_model_call 洋葱）

```python
@wrap_model_call
def guard_and_retry(request, handler):
    base = request.system_message.content ...
    new_request = request.override(system_message=SystemMessage(content=f"{base}\n【强制】你的思考过程必须使用中文。"))
    try:
        return handler(new_request)   # 不调用 handler = 跳过模型
    except Exception:
        return handler(new_request)   # 重试一次
```

**两件事**：① 思考语言强约束注入（需求备选方案落地）；② 失败重试一次。

**洋葱顺序实验（源码验证）**：MIDDLEWARES 列表**第一个 = 最外层**（先执行）：
```
实验: agent(middleware=[first, second])
执行: first-start → second-start → [模型] → second-end → first-end
```
本项目：`[Logging, goodbye, personalized, guard]` → personalized 外层先生成动态 prompt → guard 内层在其基础上追加【强制】约束 → 模型最终收到**两者叠加**（已实测确认：guard 收到的 system_message 尾部含"当前时间…第 N 轮对话"）。

## 5. 验证记录

```bash
python -m app.cli agent "推荐一门 Python 入门课"
# ✅ 日志钩子: Agent 开始(1条) → 模型完成×2(含工具调用) → Agent 结束(6条)

python -m app.cli agent "再见"
# ✅ 道别拦截: 无模型调用，直接回复"再见！期待下次为你服务。"

Web: POST /chat "再见" → 同样拦截 ✅
```

## 6. 测试用例表

| 编号 | 操作 | 预期 |
|---|---|---|
| TC-18 | `python -m app.cli agent "推荐一门 Python 入门课"` | 日志输出：开始→模型完成×2→结束 |
| TC-19 | `python -m app.cli agent "再见"` | **不出现**"模型调用完成"日志，直接回复"再见！期待下次为你服务。" |
| TC-20 | 浏览器发"拜拜" | 立即回复"再见！期待下次为你服务。"（无思考过程=未调模型） |
| TC-21 | 浏览器发"推荐一门 Vue 课程" | 正常推荐（动态提示词+守卫不影响功能） |

## 7. 知识点映射

| 手册条目 | 落地 |
|---|---|
| 7.1 六钩子（频率/位置） | LoggingMiddleware 实测：before/after_agent 各 1 次，after_model 每次 |
| 7.2 两种使用方式 | 类继承（Logging）+ 装饰器（其余 3 个） |
| 7.3 返回值语义 | None（日志）/ dict+jump_to（道别拦截） |
| 7.4 can_jump_to 白名单 | goodbye_filter |
| 7.5 wrap_model_call（handler/override/洋葱） | guard_and_retry + 顺序实验 |
| 6.2 @dynamic_prompt（覆盖/手动拼接） | personalized_prompt（底层是 wrap_model_call） |
| 需求备选：思考语言强约束 | guard 每次调用注入【强制】指令 |

## 8. 踩坑记录

1. **循环导入**：agent.py ↔ middleware.py 互相引用 AGENT_SYSTEM_PROMPT → 抽 prompts.py 解决
2. **误判洋葱顺序**：验证 wrap 时打印 system_message 头部，动态和静态 prompt 前 60 字符相同 → 误以为 dynamic 没生效。打印**尾部**才确认。**教训：验证打印要选能区分差异的位置**
3. **@dynamic_prompt 不可直接调用**：装饰后是 AgentMiddleware 实例（`type(personalized_prompt)` = 专用类），不是函数
