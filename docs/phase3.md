# 阶段 3：最小 Agent（说明文档）

> 状态：✅ 已完成并验证（真实 API 调用）
> 对应设计文档：DESIGN.md 阶段 3 + 新增需求⑦（思考流式展示 CLI 先行验证）
> 验证时间：2026-08

## 1. 目标

搭建最小 Agent：两个课程工具 + `create_agent` 组装，让模型具备"自动决定调工具"的能力；同时验证新增需求"流式展示思考过程"。

## 2. 搭建步骤

1. 新建 `data/courses.json`：模拟课程库（9 门课，覆盖入门/进阶/高级）
2. 新建 `app/tools/course_tools.py`：4 个工具（覆盖 @tool 全套知识点）
3. 新建 `app/agent.py`：create_agent 工厂（单例 + Agent 级 system_prompt）
4. 扩展 `app/cli.py`：`agent`（invoke 版）和 `agent-stream`（流式思考展示）子命令
5. 预研验证 stream 模式下 reasoning_content 的 chunk 行为（为阶段 4 SSE 铺路）

## 3. 目录变更

```
data/courses.json          # 新增：模拟课程库
app/tools/course_tools.py  # 新增：课程工具集
app/agent.py               # 新增：create_agent 工厂
app/cli.py                 # 修改：agent / agent-stream 子命令
DESIGN.md                  # 修改：登记新增需求⑦ + 阶段4设计补充
```

## 4. 新增文件详解

### 4.1 `app/tools/course_tools.py` —— 工具集（4 个知识点演示）

| 工具 | 知识点 | 行为 |
|---|---|---|
| `search_courses(keyword, level, max_results)` | @tool + docstring 描述 + 默认参数 | 普通工具：搜索课程库 |
| `get_course_detail(course_id)` | `return_direct=True` | 工具结果直接作为最终输出 |
| `record_search_log(question, tool_call_id)` | `InjectedToolCallId` 注入 | 记录咨询日志（含调用 ID） |
| `search_courses.handle_tool_error = True` | handle_tool_error（装饰后赋值） | 空关键词时错误转返回值 |

**实现细节**：
- `@tool` 装饰器**不接受** `handle_tool_error` 参数（1.3.14 源码验证会 TypeError）→ 用装饰后赋值：`search_courses.handle_tool_error = True`
- 注入参数写法：`tool_call_id: Annotated[str, InjectedToolCallId]`——**不会进入工具 schema**，模型看不到、不用填，由 LangGraph ToolNode 执行时注入
- `return_direct=True` 工具：结果即答案，跳过模型二次加工（省 token 降时延）

### 4.2 `app/agent.py` —— Agent 工厂

```python
AGENT_SYSTEM_PROMPT = """你是编程学习助手"课栈"...
## 工具使用指引
- 用户问"有没有/推荐/找 XX 课"时，先调用 search_courses
- 用户要看详情时，用 get_course_detail
- 用户咨询课程问题时，同时调用 record_search_log
..."""

@lru_cache(maxsize=1)
def get_agent():
    return create_agent(
        model=get_model(),
        tools=[search_courses, get_course_detail, record_search_log],
        system_prompt=AGENT_SYSTEM_PROMPT,
    )
```

**为什么 system_prompt 要写工具使用指引**：模型靠描述判断"何时调哪个工具"，指引写清楚 → 工具调用准确率大幅提升（这是提示工程的关键点）。

### 4.3 cli.py 两个新命令

- `agent "..."`（invoke）：打印完整执行链——用户 → AI 的 tool_calls → 工具结果 → 最终回复
- `agent-stream "..."`：`stream_mode="messages"` 流式输出，**思考/回复分段展示**（新需求⑦）

## 5. 新增需求⑦实现（思考流式展示）—— 预研结论

**stream 模式下 chunk 分流规则**（实测确认）：

```
chunk.additional_kwargs["reasoning_content"] 有值 → 思考阶段（此时 content 为空）
chunk.content 有值 → 回复阶段（此时 reasoning_content 为空）
首块 chunk 两者皆空 → 忽略
```

**实测 chunk 序列**：`chunk00(空) → chunk01~14(思考) → chunk15~N(回复)`——思考 chunk 先全部到达，回复 chunk 随后，**顺序严格分段不交错**。

**CLI 实现**（app/cli.py `demo_agent_stream`）：状态机 `phase: None → thinking → answer`，按字段分流打印 `🤔 思考` / `💬 回复`。

**阶段 4 落地方案**（已写入 DESIGN.md）：SSE 发两种事件 `{type:"reasoning"}` 和 `{type:"token"}`，前端思考区灰字 + 回复区正常显示。

## 6. 验证记录

```bash
python -m app.cli agent "有没有 Python 入门课程？推荐一下"
# ✅ 完整推理链：search_courses(python, 入门) + record_search_log
#    → get_course_detail(py-101)（return_direct 单工具 → 直接结束）
# ✅ InjectedToolCallId 注入生效：已记录咨询日志 #call_01_...

python -m app.cli agent-stream "有没有 Vue 进阶课程？"
# ✅ 🤔 思考先流式出现 → 💬 回复随后流式出现（分段不交错）

# handle_tool_error 行为：
#   空关键词 → "搜索关键词不能为空..."（异常转返回值，不抛）
#   不存在ID → ToolException 照常抛出（未设 handle_tool_error 的工具）
```

## 7. 测试用例表

| 编号 | 前置条件 | 操作步骤 | 预期结果 |
|---|---|---|---|
| TC-09 | .env 正常 | `python -m app.cli agent "有没有 Python 入门课程？推荐一下"` | 执行链含 search_courses + record_search_log + get_course_detail；日志含 call_ 前缀 ID |
| TC-10 | 同上 | `python -m app.cli agent-stream "有没有 Vue 进阶课程？"` | 🤔 思考先出、💬 回复后出；回复含 Vue 3 前端开发实战 |
| TC-11 | 同上 | `python -m app.cli agent "推荐一门后端课程"` | 调用 search_courses(fastapi 或 后端)，正常返回 web-101 |
| TC-12 | 同上 | `python -m app.cli agent "1+1等于几"` | 不调工具，直接回答（无工具调用路径） |

## 8. 知识点映射

| 手册条目 | 落地 |
|---|---|
| 3.1 @tool + docstring | course_tools.py 全部工具 |
| 3.4 return_direct（all 语义） | get_course_detail（单工具调用实测直接结束） |
| 3.5 InjectedToolCallId | record_search_log |
| 3.6/3.7 ToolException + handle_tool_error | search_courses（赋值方式，1.3.14 实测） |
| 4.1 create_agent 参数 | agent.py（tools/system_prompt） |
| 4.3 运行方法 stream + stream_mode | agent-stream（messages 模式） |
| 9.2 stream_mode 细分 | messages 模式 + metadata.langgraph_node |
| 需求⑦ | 思考/回复分段流式（CLI 版验证，阶段 4 Web 化） |
