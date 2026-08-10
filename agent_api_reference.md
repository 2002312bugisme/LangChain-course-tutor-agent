# Agent 全量 API 参考手册（按执行流程分层次）

> 依据：LangChain.md 学习笔记 + langchain 1.3.14 源码验证 + 真实 API 实测
> 组织方式：按"构建并运行一个 Agent"的执行流程分 13 个阶段，从配置准备到错误处理。
> 每个条目统一格式：**是什么 → 签名 → 参数 → 何时用 → 场景 → 注意**
> **验证记录**：2026-08 对照本机源码逐项检查（inspect.signature / model_fields / 源码文本），全部 `⚠️ 源码验证` 标记为修正或补充项；`✅` 未标记的条目验证通过。

---

# 阶段 0：配置准备（.env 与模型参数总览）

## 0.1 .env 文件 + python-dotenv

- **是什么**：把敏感配置（api_key、模型名、base_url）从代码中剥离，存到 `.env` 文件，`from dotenv import load_dotenv()` 加载后经 `os.getenv()` 读取。
- **何时用**：任何要提交到 git / 分享代码的项目。密钥永远不硬编码进代码。
- **场景**：团队协作、多环境（开发/生产不同 key）。
- **注意**：`.env` 要加进 `.gitignore`。LangChain 的 `api_key` 参数不传时会自动读 `OPENAI_API_KEY`/`DEEPSEEK_API_KEY` 等环境变量（字段的 `from_env` 声明决定读哪个）。

## 0.2 模型调用参数速查（初始化时可传，全部可选）

| 参数 | 作用 | 何时用 | 注意 |
|---|---|---|---|
| `model` | 模型名（必填） | 总是 | 可带前缀 `deepseek:deepseek-v4-flash` |
| `api_key` | 密钥 | 不传则读环境变量 | 优先级：显式传 > 环境变量 |
| `base_url` | API 地址 | 换服务商/代理/本地模型 | DeepSeek 默认 `https://api.deepseek.com` |
| `temperature` | 采样温度 0~2 | LangChain 层默认 None（不传时由厂商 SDK 决定实际温度） | 与 top_p 通常不同时调 |
| `top_p` | 核采样候选范围 | 想要"更稳但保留多样性" | 与 temperature 二选一 |
| `max_tokens` | 输出上限 | 控制成本/延迟 | 服务端生成到上限即停（非截断） |
| `timeout` | 请求超时秒数 | 避免长时间挂起 | None 时用 SDK 默认（连接 5s/读取 600s） |
| `max_retries` | 失败重试次数 | 网络不稳 | 当前版本默认 None → SDK 默认 2 次 |
| `stop` | 停止符列表 | 需要强制终止输出 | 命中即停 |
| `n` | 生成几个候选 | 极少用 | 只取第一个，浪费 token |
| `streaming` | 是否流式 | 通常用 stream() 代替 | 需配合回调消费 |
| `presence_penalty` | 话题新鲜度惩罚 | 减少重复 | 负值鼓励重复 |
| `frequency_penalty` | 频率惩罚 | 减少机械重复 | 越高越不重复 |
| `logit_bias` | 指定 token 概率偏移 | 强制关键词出现 | 需知道 token id，很少用 |
| `model_kwargs` | 透传未建模字段 | 模型 API 支持但 LangChain 未列出的参数 | 裸透传，无 schema 转换 |
| `extra_body` | 透传厂商扩展字段 | 如 DeepSeek 的 `thinking` 开关 | 走 OpenAI SDK 的 extra_body |

---

# 阶段 1：模型层

## 1.1 init_chat_model() —— 统一模型工厂

- **是什么**：根据模型名/提供商自动选择具体模型类的工厂函数。
- **签名**：
```python
init_chat_model(
    model: str,                       # 必需："provider:model" 或裸模型名（推断）
    model_provider: str | None = None,# 显式指定提供商
    temperature=None, max_tokens=None, timeout=None, max_retries=None,
    configurable_fields=None,         # None | "any" | list[str]
    config_prefix=None,               # _ConfigurableModel 的键前缀（默认 None）
    **kwargs,                         # 透传给底层模型类
)
```
- **参数**：
  - `model`：支持前缀写法 `deepseek:deepseek-v4-flash`；裸名则按规则推断（`gpt-`→openai、`deepseek-`→deepseek），推断不出的（如 `qwen-plus`）必须显式 `model_provider`。
  - `model_provider`：内置注册表外的提供商（如 `tongyi`）必须显式指定。
  - `configurable_fields`：见 1.6。
  - `config_prefix`：给 configurable 键加前缀，同一请求挂多个可配置模型时避免参数打架。
- **何时用**：需要配置驱动/多模型切换时。模型固定则直接用具体类更类型安全。
- **注意**：返回 `_ConfigurableModel` 时是"延迟实例化"——真正实例化发生在第一次调用时。
- **⚠️ 源码验证**：`temperature/max_tokens/timeout/max_retries` **不是** init_chat_model 的显式形参（实际形参仅 `model/model_provider/configurable_fields/config_prefix/**kwargs`），全部经 `**kwargs` 透传给底层模型类——功能上等价，写法不受影响。

## 1.2 具体模型类（ChatOpenAI / ChatDeepSeek / ChatTongyi）

- **是什么**：各厂商集成包提供的类。`init_chat_model` 底层实例化的就是它们。
- **区别**：ChatOpenAI 是"通用 OpenAI 协议适配器"（可用 `base_url` 接任意兼容服务）；ChatDeepSeek/ChatTongyi 是厂商特化封装（默认 base_url、默认模型名）。
- **何时用**：模型固定、想要 IDE 补全和静态类型检查时。
- **注意**：使用某类前必须先 `pip install` 对应包（langchain-openai / langchain-deepseek / langchain-community）。

## 1.3 模型类参数详解（初始化时）

### 连接类（决定"怎么连"）
- `api_key`（⚠️ 字段名是 `openai_api_key`，alias `api_key`）：不传读环境变量（DEEPSEEK_API_KEY 等）。
- `base_url`（⚠️ 字段名是 `openai_api_base`）：接口地址。
- `request_timeout`（alias `timeout`）：网络超时。用法等价，但 `model_fields` 中查不到 `api_key`/`base_url`/`timeout` 这三个名字（Pydantic 别名机制）。
- `max_retries`：SDK 层重试次数。
- `http_client` / `http_async_client`：手动传 httpx.Client，用于代理/自定义 TLS。
- `openai_proxy`：代理。
- `default_headers` / `default_query`：每个请求默认携带的 HTTP 头/查询参数。

### 推理类（决定"怎么生成"）
- `model_name`：模型名。
- `temperature` / `top_p` / `max_tokens` / `stop` / `streaming` / `n`。
- `presence_penalty` / `frequency_penalty` / `logit_bias`。
- `reasoning_effort`（部分模型）：思考深度档位（low/medium/high）。
- `store`：是否存平台侧（OpenAI 平台功能，与本地记忆无关）。

### 框架通用类（LangChain 内部用）
- `name`：实例名（LangSmith 显示用）。
- `verbose`：详细日志。
- `callbacks`：回调（LangSmith 集成）。
- `tags` / `metadata`：标记。
- `cache`：结果缓存。
- `rate_limiter`：LangChain 频率限制器。

### 高级扩展类
- `client` / `async_client` / `root_client`：内部 SDK 实例（一般不要手动传）。
- `model_kwargs`：透传给 API 的额外字段（如裸传 `tools`，不推荐，用 bind_tools）。
- `extra_body`：厂商扩展字段（如 DeepSeek `thinking`）。
- `disable_streaming` / `include_response_headers`：功能开关。
- `openai_organization` / `service_tier`：OpenAI 遗留兼容参数。

## 1.4 model.profile —— 模型能力画像

- **是什么**：`BaseChatModel` 上的 `ModelProfile | None` 属性，描述模型上下文窗口、能力（多模态/工具调用/结构化输出等）。
- **来源**：LangChain 官方声明（数据来自 models.dev + 官方加工），**不是模型厂商 API 返回**。
- **何时用**：选型时判断"这模型支持什么"；查 `model.profile` 是否为 None 判断该集成是否声明了画像（DeepSeek 集成未声明）。
- **注意**：Beta 功能，字段可能变化。

## 1.5 model_fields —— 查看完整参数列表

- **是什么**：Pydantic 的类属性，列出该模型类所有可配置字段（自身定义 + 继承）。
- **何时用**：想确认某个参数是否支持、找全参数名（官方文档可能不全）。
- **注意**：字段都有默认值，所以"不传也能初始化"；但缺 api_key 会在**真正发请求时**才报错。

## 1.6 _ConfigurableModel —— 运行时切换模型

- **是什么**：`init_chat_model(configurable_fields=...)` 返回的可配置模型包装器。每次调用时从 `config["configurable"]` 读参数，**运行时覆盖默认值**。
- **签名**：
```python
model = init_chat_model(
    model="deepseek:deepseek-v4-flash",
    configurable_fields=["model", "model_provider", "temperature"],
)
result = model.invoke("你好", config={"configurable": {"model": "deepseek:deepseek-v4-pro"}})
```
- **configurable_fields 取值**：
  - `None`：不可配置，返回普通 BaseChatModel
  - `"any"`：所有参数可配置（⚠️ 安全风险：api_key 也能被改）
  - `["model", "temperature"]`：白名单，只允许列出的字段被覆盖（**推荐**）
- **何时用**：多模型 A/B 测试、按用户/任务路由不同模型、配置驱动应用。
- **注意**：① 不在白名单的 configurable 键会被静默丢弃；② 链式调用 `bind_tools`/`with_structured_output` 会被**延迟到真正实例化时**执行，不影响运行时切换；③ 普通模型实例上 config 里的 configurable **不生效**。

## 1.7 bind_tools() —— 绑定工具（告知模型）

- **是什么**：把工具列表传给模型，开启 function calling 能力。模型返回的是**调用请求**（tool_calls），不执行。
- **何时用**：① 直接使用模型而不经过 Agent 时；② 需要自定义工具 schema 时。
- **注意**：在 `create_agent(tools=...)` 里框架会自动调用，不要重复绑定。
- **配合**：拿到 `AIMessage.tool_calls` 后要自己执行并回传 `ToolMessage`——Agent 的 ToolNode 就是干这个的。

## 1.8 with_structured_output() —— 结构化输出（模型方法）

- **是什么**：让模型**直接返回**符合 schema 的对象（Pydantic 实例），而不是 tool_call。
- **签名**：`model.with_structured_output(Schema)`，可选 `method="json_schema" | "function_calling"`。
- **何时用**：纯信息提取/数据解析，不需要多步推理和工具——比建 Agent 更轻量。
- **注意**：依赖模型原生结构化输出能力（DeepSeek 支持），能力弱的模型可能走 function calling 兼容路径。

## 1.9 模型运行方法

| 方法 | 返回 | 何时用 |
|---|---|---|
| `invoke(input)` | 完整 AIMessage | 脚本、单次问答 |
| `stream(input)` | AIMessageChunk 迭代器 | 打字机效果 |
| `batch([...])` | 结果列表 | 批量处理（默认线程池并发） |
| `ainvoke` / `astream` / `abatch` | 异步版 | Web 服务不阻塞事件循环 |
| `batch_as_completed([...])` | (index, result) 元组流 | 大批量、按完成顺序消费（可能乱序） |

## 1.10 with_config() / with_fallbacks()

- `with_config(config)`：给 Runnable 预设运行时配置（tags/metadata/callbacks/thread_id 等）。
- `with_fallbacks([备用模型])`：主模型失败自动降级。生产高可用必备。

---

# 阶段 2：消息层

## 2.1 四大消息类型

| 类 | role | 代表 | 何时用 |
|---|---|---|---|
| `HumanMessage(content)` | user | 用户输入 | 对话起点、每轮提问 |
| `AIMessage(content, tool_calls=...)` | assistant | 模型回复 | 对话历史中的模型回答 |
| `SystemMessage(content)` | system | 角色/规则 | 消息列表最前面，约束行为 |
| `ToolMessage(content, tool_call_id=...)` | tool | 工具执行结果 | 工具返回，必须绑定 tool_call_id |

- **⚠️ 源码验证补充**：ToolMessage 还有 `artifact`（结构化产物）、`status`（success/error 等）字段；AIMessage 完整字段：content/additional_kwargs/response_metadata/type/name/id/tool_calls/invalid_tool_calls/usage_metadata。

- **快捷构造**（Agent 内部自动转换）：
  - `("user", "你好")` / `("human", "你好")`：元组 (role, content)
  - `{"role": "user", "content": "你好"}`：字典
  - 裸字符串：自动当 user 消息
- **注意**：ToolMessage 的 `tool_call_id` 必须与 AIMessage 中对应 `tool_call["id"]` **精确匹配**，否则模型可能忽略结果。

## 2.2 AIMessageChunk —— 流式片段

- **是什么**：`stream()` 逐块返回的增量消息，多个 chunk 可 `+` 拼接成完整消息。
- **何时用**：任何流式场景（打字机、SSE）。
- **注意**：chunk 的 `content` 只是片段；累计拼接后才是全文。推理模型的 `reasoning_content` 在 `additional_kwargs` 里。

## 2.3 ContentBlock —— 结构化消息内容

- **是什么**：消息 content 除了字符串，还可以是内容块列表（多模态）。
- **类型**：`PlainTextContentBlock`（纯文本）、`ImageContentBlock`（base64/URL 图片）、`ToolCall`（工具调用块）。
- **何时用**：单条消息混合文本+图片（多模态输入）；纯文本直接传字符串即可，LangChain 自动处理。

## 2.4 ToolCall —— 工具调用请求

- **签名**：`ToolCall(name="get_weather", args={"city": "杭州"}, id="call_abc123", type="tool_call")`
- **是什么**：模型"想调用工具"的结构化表达。出现在 `AIMessage.tool_calls` 列表里。
- **注意**：`id` 是执行后回填 ToolMessage 的关联键，唯一。

## 2.5 trim_messages() —— 裁剪消息历史

- **签名**：
```python
trim_messages(
    messages,                # 要裁剪的消息列表
    max_tokens=1000,         # 保留上限（token）
    strategy="last",         # "last" 保留最近的 | "first" 保留最早的
    token_counter=model,     # 用什么计数（模型或 tiktoken）
    include_system=True,     # 始终保留 SystemMessage
    start_on="human",        # 裁剪后以 human 消息开头（避免孤立 AI 回复开头）
    allow_partial=False,     # 是否允许截断中间消息
)
```
- **⚠️ 源码验证**：实际参数为 `messages/max_tokens/token_counter/strategy/allow_partial/end_on/start_on/include_system/text_splitter`——漏了 `end_on`（裁剪后以什么结尾）和 `text_splitter`（截断中间消息时的切分器，如按句子）。
- **何时用**：对话变长、逼近上下文窗口时（每轮请求前）。
- **注意**：`strategy="last"` 是"保留系统消息+最近对话"，不是"保留最后 N 条"那么简单。

## 2.6 RemoveMessage —— 删除特定消息

- **签名**：`RemoveMessage(id="msg_3")`
- **是什么**：配合 `add_messages` reducer 使用，从消息列表**删除**指定 id 的消息（不是追加）。
- **何时用**：敏感内容清洗、撤销某条回复、重生成。
- **典型位置**：middleware / after_model 钩子返回值里。

## 2.7 消息通用属性/方法

- `content`：内容（str 或 ContentBlock 列表）
- `type`：角色（human/ai/system/tool）
- `id`：自动生成的唯一 ID（可手动指定）
- `text`：文本内容的快捷提取（非文本返回 ""）
- `pretty_repr()`：格式化打印调试
- `response_metadata`：API 原始元数据（token_usage、finish_reason 等）
- `usage_metadata`：标准化 token 统计（input/output/total + 细节）

---

# 阶段 3：工具层

## 3.1 @tool 装饰器 —— 函数变工具

- **是什么**：把 Python 函数包装成 Tool 对象（有 name/description/args_schema，可 invoke）。
- **签名**：`@tool(name=..., description=..., args_schema=..., return_direct=..., handle_tool_error=...)`，默认取函数名。
- **何时用**：绝大多数工具定义场景。
- **注意**：
  - **docstring 自动成为工具描述**（`__doc__`），Agent 靠它判断何时调用。写清楚：功能 + 参数含义（Args 段）+ 使用场景。
  - 参数类型支持 int/float/bool/Literal/枚举；**关键参数不要设默认值**（缺参会调用失败），默认值让 Agent 少填参数。
- **直接调用**：`hello_tool.invoke({"name": "小明"})` 也可以（绕开 Agent 单独测试）。
- **⚠️ 源码验证补充**：@tool 完整参数 `name_or_callable/runnable/args/description/return_direct/args_schema/infer_schema/response_format/parse_docstring/error_on_invalid_docstring/extras`——支持 `response_format="content_and_artifact"`（返回结构化产物）、`parse_docstring`（自动解析 docstring 成 schema）；**没有 handle_tool_error**（见 3.7）。

## 3.2 工具定义四种方式对比

| 方式 | 代码量 | 适用场景 |
|---|---|---|
| `@tool` | 最少 | 简单~中等复杂度，大多数场景 |
| `@tool(args_schema=Pydantic)` | 中等 | 精细参数校验（API 封装、数据库操作） |
| Pydantic 类作工具 | 较多 | 复杂业务逻辑、内部有状态 |
| 字典格式 | 最少（不推荐） | MCP 工具、服务端工具（描述远程工具） |

## 3.3 args_schema —— 参数校验

- **是什么**：用 Pydantic 模型声明参数结构，工具执行前自动校验类型/必填。
- **何时用**：参数多、类型复杂、要描述字段含义（Field(description=...) 会进工具 schema 给模型看）。

## 3.4 return_direct —— 工具结果即最终答案

- **是什么**：`@tool(return_direct=True)`，工具执行后**跳过模型二次加工**，直接结束循环，工具结果作为最终输出。
- **何时用**：查询类、数据获取类、已格式化好的结果（省 token、降时延）。
- **注意（实测结论，langchain 1.3.14）**：混调（本轮同时调了 return_direct 工具 + 普通工具）时**不会**立即结束——路由是 `all(return_direct)` 才退出，普通工具的结果仍会送回模型加工。文档（菜鸟教程）的"任意一个就退出"说法与当前版本不符。

## 3.5 依赖注入（Injected 系列）—— 框架自动注入参数

- **是什么**：某些工具参数不需要模型提供，由 **LangGraph 运行时**（ToolNode）在执行时自动注入。被标记的参数**不会出现在工具的 schema 里**，模型看不到、也不用填。
- **写法**：`param: Annotated[类型, 注入标记]`

| 标记 | 注入什么 | 作用域 | 何时用 |
|---|---|---|---|
| `InjectedToolCallId` | 当前 tool_call 的 id | 本次调用 | 需要关联调用上下文：审计、日志、构造 ToolMessage |
| `InjectedState` | 整个 AgentState（扁平字典） | 当前对话 | 读取消息历史、中间结果、自定义状态字段 |
| `InjectedStore` | Store 对象（需 `InjectedStore()` 带括号） | 跨会话 | 读写持久化数据（用户偏好、进度） |
| `InjectedToolArg` | 自定义注入（通用基类） | 自定义 | 扩展你自己的注入逻辑 |

- **注意**：三个专门标记都基于 `InjectedToolArg` 实现。不要在 docstring 里把注入参数当"用户要填的参数"描述。

## 3.6 ToolException —— 工具异常

- **是什么**：工具内 `raise ToolException("消息")` 抛出的显式异常。
- **何时用**：参数非法、业务规则不满足（如"用户 ID 必须为正整数"）。
- **默认行为**（`handle_tool_error=False`）：异常向上抛，外层 try/except 可捕获，流程中断。

## 3.7 handle_tool_error —— 工具自己"扛住"错误

- **是什么**：工具出错时不抛异常，把错误信息**转成正常返回值**（ToolMessage）交给模型自行理解修正。
- **⚠️ 源码验证（langchain 1.3.14）**：`@tool(handle_tool_error=...)` **会报 TypeError**——@tool 装饰器签名没有该参数（旧版/文档写法，当前版本不可用）。正确设置方式（均实测可用）：
```python
# 方式 1：装饰后直接赋值
@tool
def my_tool(x: int) -> str: ...
my_tool.handle_tool_error = True

# 方式 2：StructuredTool.from_function
from langchain_core.tools import StructuredTool
my_tool = StructuredTool.from_function(fn, name="my_tool", description="...",
                                        args_schema=None, handle_tool_error=True)
```
- **四种取值**：

| 值 | 行为 | 场景 |
|---|---|---|
| `False`（默认） | 照常抛出 | 不可恢复错误，中断流程 |
| `True` | 捕获 ToolException，内容作为返回值 | Agent 自愈：读错误→修正/重试 |
| `str` | 捕获后返回固定字符串 | 不暴露内部细节 |
| `Callable` | 用函数处理异常生成返回内容 | 按异常类型定制提示 |

- **注意**：
  - 单数是工具级：`@tool(handle_tool_error=...)`（正确写法）。
  - 复数是 Agent/图级：在底层 **ToolNode** 上设置 `handle_tool_errors`，统一所有工具的错误策略。
  - `tool.with_config(handle_tool_errors=True)` 是**无效写法**——with_config 只设 callbacks/tags/metadata，不改变错误行为。

---

# 阶段 4：组装 create_agent()

## 4.1 create_agent() 完整签名

```python
create_agent(
    model,                     # str | BaseChatModel：语言模型（必需）
    tools=None,                # Sequence：工具列表
    *,
    system_prompt=None,        # str | SystemMessage：系统提示
    middleware=(),             # Sequence[AgentMiddleware]：中间件列表
    response_format=None,      # ResponseFormat | type：结构化输出配置
    state_schema=None,         # type[AgentState]：自定义状态结构
    context_schema=None,       # type：运行时上下文结构
    checkpointer=None,         # Checkpointer：对话持久化
    store=None,                # BaseStore：跨会话存储
    interrupt_before=None,     # list[str]：在哪些节点前暂停
    interrupt_after=None,      # list[str]：在哪些节点后暂停
    debug=False,               # bool：输出详细日志
    name=None,                 # str：Agent 名称
    cache=None,                # BaseCache：缓存
    transformers=None,         # ⚠️ v3 流式转换器（StreamTransformer，对应 PII 过滤/自定义投影）
)
# ⚠️ 源码验证：实际参数共 15 个（多一个 transformers），手册初版漏了它
```

## 4.2 各参数详解

- **model**（必需）：三种形式——
  1. 字符串：内部自动 `init_chat_model()` 处理（`"deepseek:deepseek-v4-flash"`）
  2. 已构建实例：精细控制参数（temperature/max_tokens）
  3. 已绑定工具的实例：不常见，通常让 create_agent 自己管理工具绑定
- **tools**：工具列表。框架内部自动 `bind_tools` + 构建 **ToolNode 执行器**。这是完整注册（相比只 bind_tools 多了执行能力）。
- **system_prompt**：str 或 SystemMessage。不传则模型以"通用助手"角色回答。业务应用建议始终设置。
- **middleware**：中间件序列（见阶段 7）。
- **response_format**：Pydantic 模型 / JSON Schema / 策略对象（见阶段 10）。
- **state_schema**：自定义状态扩展（继承 AgentState）。运行时**必须提供自定义字段的初始值**。优先级最高（覆盖 middleware 同名字段）。
- **context_schema**：运行时上下文结构（动态注入用户信息等）。
- **checkpointer**：会话持久化（见阶段 8）。
- **store**：跨会话存储（见阶段 8）。
- **interrupt_before/after**：在指定节点前后暂停（配合 `interrupt()` 人工介入）。
- **debug**：详细日志（开发调试）。
- **name**：图名（LangSmith 显示）。
- **cache**：Runnable 级缓存。

## 4.3 返回值 CompiledStateGraph 的运行方法

| 方法 | 说明 | 场景 |
|---|---|---|
| `invoke(input, config)` | 同步，完整结果 | 脚本、简单接口 |
| `ainvoke(input, config)` | 异步完整结果 | Web 服务 |
| `stream(input, config, stream_mode=...)` | 同步流式 | 实时展示中间步骤 |
| `astream(input, config, stream_mode=...)` | 异步流式 | WebSocket / SSE |
| `get_state(config)` | 读取当前状态 | 查看/恢复对话 |
| `update_state(config, values)` | 手动改状态 | 人工修正、注入 |

- **input 形式**：`{"messages": [...]}` 或直接消息列表；有自定义 state 字段时一并传入。
- **生产建议**：Agent 实例创建为**全局单例**，避免每次请求重新编译图。

---

# 阶段 5：状态层 AgentState

## 5.1 AgentState 三个默认字段

```python
class AgentState(TypedDict):
    messages: Required[Annotated[list[AnyMessage], add_messages]]
    jump_to: NotRequired[Annotated[str | None, EphemeralValue]]
    structured_response: NotRequired[Any]
```

| 字段 | 类型 | 必填 | 含义 |
|---|---|---|---|
| `messages` | list[AnyMessage] | 是 | 消息历史，reducer=add_messages（追加） |
| `jump_to` | str \| None | 否 | 流程跳转（"model"/"tools"/"end"），ephemeral 用后自清 |
| `structured_response` | Any | 否 | 结构化输出结果，**不在 input schema 暴露** |

## 5.2 add_messages reducer —— 追加而非覆盖

- **是什么**：messages 字段的合并规则（reducer）。节点返回 messages 时**追加**到历史，不是替换。
- **智能特性**：
  - 同名覆盖：新消息 id 与已有相同 → 替换
  - RemoveMessage：遇到则删除对应 id 消息
  - 类型安全：自动处理各消息类型
- **何时用**：所有"往对话里加内容"的节点/中间件，返回 `{"messages": [...]}` 即可。

## 5.3 jump_to —— 流程跳转（中间件专属）

- **取值**：
  - `"model"`：回模型节点（通常配合注入工具消息让模型重新处理）
  - `"tools"`：跳过模型直接执行工具
  - `"end"`：结束循环
- **ephemeral**：使用一次后自动清除，无需手动重置。
- **白名单**：必须在钩子装饰器 `@before_model(can_jump_to=["end"])` 声明目标，否则被忽略（安全机制）。

## 5.4 structured_response —— 结构化输出结果

- **是什么**：response_format 配置后，模型产出被解析写入此字段。
- **关键（源码验证）**：它**不是"只记录"**——路由检测到 `"structured_response" in state` 就**退出循环**（正式退出条件之一）。
- **访问**：`result["structured_response"]`，是 Pydantic 实例（可属性访问、IDE 补全）。

## 5.5 state_schema vs middleware state_schema

| 方式 | 使用场景 | 优先级 |
|---|---|---|
| `create_agent(state_schema=...)` | 全局业务状态 | 最高（覆盖同名字段） |
| `AgentMiddleware(state_schema=...)` | 中间件内部字段 | 较低 |

**推荐**：通用业务字段放 create_agent；中间件专属内部字段放中间件自己。职责清晰不污染。

---

# 阶段 6：提示词层

## 6.1 system_prompt 参数

- 两种形式：字符串（简单）或 SystemMessage（可复用）。
- 设计清单：角色定义 / 行为准则 / 工具使用指引 / 边界约束 / 格式要求。
- **注意**：system 是"最先看到"，不是"保证遵守"——关键安全边界要靠代码（工具白名单、校验）兜底。

## 6.2 @dynamic_prompt —— 动态提示词

- **是什么**：装饰器版中间件，**每次模型调用前**执行，根据上下文动态生成 system_prompt。
- **签名**：`@dynamic_prompt def fn(request: ModelRequest) -> str`
- **request 提供**：`request.state`（Agent 状态）、`request.runtime.context`（运行时上下文，如用户信息）、`request.messages` 等。
- **何时用**：个性化提示词（按用户/时间/对话阶段变化）。
- **注意**：① 优先级**高于**静态 system_prompt（覆盖而非合并，要合并需手动拼接）；② request 中不暴露原 system_prompt，保留静态内容需在函数外引用变量；③ 别做重计算，影响响应速度。

---

# 阶段 7：中间件层（六钩子）

## 7.1 六个钩子总览

| 钩子 | 频率 | 位置 | 用途 |
|---|---|---|---|
| `before_agent` | 1 次 | Agent 开始前 | 初始化、权限检查、输入预处理 |
| `before_model` | 每次循环 | 模型调用前 | 消息预处理、动态上下文注入 |
| `wrap_model_call` | 每次循环 | **包裹**模型调用 | 重试、降级、缓存、请求改写 |
| `after_model` | 每次循环 | 模型调用后 | 内容审核、响应过滤、日志 |
| `wrap_tool_call` | 每次工具调用 | **包裹**工具执行 | 工具重试、缓存、参数改写 |
| `after_agent` | 1 次 | Agent 结束后 | 格式化输出、统计、清理 |

**⚠️ 源码验证补充**：① 每个钩子都有异步版本（`abefore_agent`/`abefore_model`/`aafter_model`/`aafter_agent`/`awrap_model_call`/`awrap_tool_call`）；② AgentMiddleware 还有 `name`/`state_schema`/`transformers` 属性（transformers 对应 v3 流式投影）。

**心智模型**：钩子 = 流程的"插眼点"。before/after 是观察+改状态；wrap 是**完全接管**（不调 handler 就跳过执行）。

## 7.2 两种使用方式

- **装饰器**（推荐）：`@before_model` / `@after_model` / `@wrap_model_call` / `@wrap_tool_call` / `@before_agent` / `@after_agent` / `@dynamic_prompt`。
- **类继承**：`class LoggingMiddleware(AgentMiddleware)`，可定义 `name` 属性、state_schema、多个钩子。适合复杂中间件。

## 7.3 中间件返回值

| 返回值 | 效果 |
|---|---|
| `None` | 纯观察（日志、监控） |
| `dict` | 合并进 state（走 reducer；messages 用 add_messages 追加） |
| 含 `jump_to` 的 dict | 合并 + 流程跳转（需 can_jump_to 白名单） |

## 7.4 can_jump_to —— 跳转白名单

- **签名**：`@before_model(can_jump_to=["end"])` 等。
- **取值**：`["end"]` / `["model"]` / `["tools"]` / 组合。
- **机制**：声明了才能跳；未声明的 jump_to 被忽略。防止中间件意外跳到非法节点（安全设计）。

## 7.5 wrap_model_call —— 模型调用拦截

- **核心**：`handler(request)` 回调。**调用才真正执行模型；不调用则跳过模型**（可返回预设回复）。
- **request 内容**：`model`、`messages`、`tools`、`system_message` 等。
- **request.override()**：不可变方法，返回新 request 副本（不修改原对象），用于改 system_message/参数后交给 handler。
- **典型场景**：重试、降级（换模型）、缓存、注入动态信息。
- **洋葱模型**：多个 wrap_model_call 层层包裹，最外层先执行、最后返回。组合互不干扰（外层缓存、内层重试）。

## 7.6 wrap_tool_call —— 工具调用拦截

- **request 内容**：`tool_call`（名称+参数）、`tool`（工具对象）、`state`、`runtime`。
- **返回类型**：`ToolMessage` 或 `Command`。
- **典型场景**：工具重试、参数规范化、结果缓存、日志监控、按结果决定流程（返回空结果直接结束）。
- **返回 Command**：可用 `update` 改状态、向 messages 追加 AI 消息后循环自然结束。

## 7.7 before_agent / after_agent —— Agent 级钩子

- `before_agent(state, runtime)`：输入预处理、权限检查（runtime.context 拿用户信息）。
- `after_agent(state, runtime)`：统计（模型/工具调用次数）、`runtime.stream_writer({...})` 发 custom 流事件、格式化输出。

## 7.8 wrap_model_call vs wrap_tool_call 对比

| 维度 | wrap_model_call | wrap_tool_call |
|---|---|---|
| 拦截目标 | 模型调用 | 工具执行 |
| request | model/messages/tools/system_prompt | tool_call/tool/state/runtime |
| 返回 | ModelResponse 或 AIMessage | ToolMessage 或 Command |
| 场景 | 重试、降级、缓存、prompt 修改 | 工具重试、缓存、参数改写、结果处理 |

---

# 阶段 8：记忆层

## 8.1 checkpointer —— 会话级持久化（短期记忆）

- **是什么**：把图的完整状态（消息历史等）按 `thread_id` 存储，同 id 恢复。
- **签名**：`create_agent(..., checkpointer=InMemorySaver())`；调用时 `config={"configurable": {"thread_id": "t1"}}`。
- **实现**：`InMemorySaver()`（内存，进程重启丢）/ SqliteSaver（SQLite 文件，⚠️ 源码验证：不在 langgraph 主包，需单独 `pip install langgraph-checkpoint-sqlite`）/ PostgresSaver（生产，同样独立包）。
- **何时用**：多轮对话续聊（同一会话跨请求）。
- **注意**：thread_id 相同才恢复；不同 id 互相隔离。`get_state`/`update_state` 可查看/手动修改。

## 8.2 Store —— 跨会话持久化（长期记忆）

- **是什么**：命名空间 + 键的层级存储。`store.put(namespace, key, value)` / `store.get(namespace, key)`。
- **注入**：`create_agent(..., store=store)` 后，工具用 `store: Annotated[BaseStore, InjectedStore()]` 形参读写。
- **何时用**：用户偏好、学习进度、跨会话共享数据。

## 8.3 checkpointer vs Store 对比

| 维度 | checkpointer | Store |
|---|---|---|
| 作用域 | 单会话（thread_id） | 跨会话（namespace+key） |
| 生命周期 | 会话存在期间 | 持久 |
| 数据组织 | 整个 state 快照 | 层级键值 |
| 典型用途 | 续聊恢复 | 用户画像、偏好 |

## 8.4 上下文控制组合拳

`trim_messages`（裁剪）+ 摘要压缩（早期对话总结成几句）+ Store 检索注入（只放相关片段）——共同控制 token 增长曲线。

---

# 阶段 9：运行与流式

## 9.1 运行方式对比

| 方法 | 返回时机 | 场景 |
|---|---|---|
| `invoke()` | 全部完成一次性 | 脚本、API、批处理 |
| `stream()` | 逐步返回 | 聊天界面、展示过程 |
| `ainvoke()` | 异步完成 | Web 服务（不阻塞事件循环） |
| `astream()` | 异步逐步 | WebSocket、SSE |

**异步铁律**：FastAPI 等单线程事件循环里用同步方法会卡住整个服务器；异步方法只是挂起当前请求。

## 9.2 stream_mode 五种

| 模式 | 粒度 | 迭代对象 | 用途 |
|---|---|---|---|
| `messages` | Token 级 | (AIMessageChunk, metadata) | 打字机效果；metadata 有 `langgraph_node` 标明来源节点 |
| `updates` | 节点级 | {node_name: state_update} | 展示思考过程 |
| `values` | 节点级全量 | 完整 state | 状态快照、调试 |
| `custom` | 自定义 | 任意 dict | middleware 用 `runtime.stream_writer()` 发进度 |
| `debug` | 详细 | 调试信息 | 开发期排查 |
| `checkpoints` | ⚠️ 检查点级 | 每个 checkpoint 快照 | 配合 checkpointer 观察状态演进 |
| `tasks` | ⚠️ 任务级 | (task_name, payload) | 并行子任务（Send）的粒度流 |

**⚠️ 源码验证**：StreamMode 完整取值 `['values','updates','checkpoints','tasks','debug','messages','custom']`，手册初版漏了 checkpoints/tasks。

**可组合**：`stream_mode=["updates", "custom"]` 混流，按 mode 分支处理。

## 9.3 config 参数（所有运行方法可传）

| 键 | 作用 | 何时用 |
|---|---|---|
| `run_name` | 本次运行的名称 | LangSmith 定位 |
| `tags` | 标签列表 | 分类过滤 |
| `callbacks` | 回调处理器 | LangSmith/自定义监控 |
| `metadata` | 任意键值（user_id 等） | 业务上下文传递 |
| `max_concurrency` | 并发上限 | batch 时保护 API/资源 |
| `recursion_limit` | 递归深度上限 | 防 Agent 死循环 |
| `configurable` | 可配置参数（模型切换等） | _ConfigurableModel 运行时切换 |
| `configurable.thread_id` | 会话 id | checkpointer 恢复 |
| `run_id` | ⚠️ 内部运行 id（一般不手动设置） | 溯源 |

**⚠️ 源码验证**：RunnableConfig 完整字段 `['tags','metadata','callbacks','run_name','max_concurrency','recursion_limit','configurable','run_id']`，手册初版漏了 run_id。

**优先级**：config 运行时 > 初始化参数（"就近原则"，源码 `{**default, **config}` 证实）。

## 9.4 Agent 退出条件（4 种）

| 条件 | 说明 |
|---|---|
| 无 tool_calls | 模型认为完成，直接回复 |
| return_direct 工具 | 本轮工具**全部** return_direct 时结束（1.3.14 实测） |
| structured_response | state 中出现即退出 |
| jump_to="end" | middleware 主动结束 |

---

# 阶段 10：结构化输出

## 10.1 response_format 参数

- **传入形式**：Pydantic 模型 / JSON Schema 字典 / 策略对象。
- **产出**：`result["structured_response"]`（Pydantic 实例，非字典）。
- **与工具共存**：Agent 可先调工具再结构化输出（真工具优先执行，结构化收尾）。

## 10.2 三种策略

| 策略 | 原理 | 模型要求 | 速度 | 何时用 |
|---|---|---|---|---|
| `ToolStrategy` | Schema 伪装成"假工具"，模型调用它输出 | 支持 function calling 即可 | 较慢 | 兼容性优先 |
| `ProviderStrategy` | 模型原生 response_format | GPT-4o+/Claude 3+ 等 | 快 | 追求性能、模型确定支持 |
| `AutoStrategy` | 自动检测选最优 | 自适应 | 最优 | **默认推荐**（直接传 Pydantic 即用此策略） |

- **ToolStrategy.handle_errors**：结构化输出出错自动重试（`True` / 自定义错误模板字符串，模板可用 `{error}`）。只有 ToolStrategy 支持。
- **降级**：ProviderStrategy 不支持时自动降级 ToolStrategy。
- **检查模型能力**：`model.profile`。

## 10.3 with_structured_output vs response_format

| | with_structured_output | response_format |
|---|---|---|
| 归属 | 模型方法 | create_agent 参数 |
| 需要 Agent 吗 | 否（纯提取） | 是（Agent 状态+退出机制） |
| 场景 | 信息提取、数据解析 | Agent 工作流内结构化收尾 |

---

# 阶段 11：流程控制（jump_to / Command）

## 11.1 jump_to —— 中间件级跳转

- **出现位置**：`AgentState.jump_to` 字段 + 钩子返回值 dict 的键。
- **取值**：`"model"` / `"tools"` / `"end"`。
- **声明**：`can_jump_to` 白名单。
- **特性**：ephemeral 用后自清。

## 11.2 Command —— 节点级指令

```python
Command(
    update={"messages": [...]},   # 改状态（走 reducer）
    goto="tools",                 # 跳转目标：节点名 / [Send(...)] / PARENT
    resume="用户确认了",           # 配合 interrupt() 恢复
    graph=Command.PARENT,         # 发给当前图或父图
)
```

- **何时返回 Command**（不是 ToolMessage）：需要"改状态 + 控制流程"时——
  1. 追加消息并跳转
  2. interrupt() 后 resume 恢复执行
  3. 多 Agent：向父图发指令
- **纯结果返回**：ToolMessage 就够。

## 11.3 流程控制三层级（易混）

| 层级 | 原语 | 白名单 | 状态 |
|---|---|---|---|
| 节点层（图） | `Command(goto=...)` | 无 | ✅ ToolNode 原生支持 |
| 中间件层 | `jump_to` | `can_jump_to` | ✅ |
| wrap 包裹层 | `Command(goto=...)` | — | ❌ wrap_model_call 抛 NotImplementedError |

**原因**：多个 wrap 嵌套时 goto 归属语义冲突 + 绕过图路由统一决策 + 无白名单安全机制（源码注释 "not yet supported"）。

---

# 阶段 12：错误处理

## 12.1 错误处理全景

| 层 | 手段 | 行为 |
|---|---|---|
| 模型调用 | `max_retries`（SDK 重试）| 网络错误/429/5xx 指数退避 |
| 模型调用 | `@wrap_model_call` | 自定义重试/降级/缓存 |
| 工具执行 | `ToolException` + `handle_tool_error` | 工具级 |
| 工具执行 | ToolNode `handle_tool_errors`（复数） | Agent 级统一兜底 |
| 整体 | try/except | 兜底中断 |

## 12.2 异常类型提示

- `ToolException`：工具业务错误。
- SDK 异常：`openai.BadRequestError`（400）、`RateLimitError`（429）、`APITimeoutError` 等——按需分层捕获。
- 配置错误：缺 key 等（ValueError 类）。
- **⚠️ 源码验证补充**：BaseTool 还有 `handle_validation_error` 字段（参数校验失败的处理，与 handle_tool_error 并列）——校验类错误走它，业务异常走 handle_tool_error。

---

# 阶段 13：速查决策表

| 我要... | 用... |
|---|---|
| 多模型切换/配置驱动 | `init_chat_model(configurable_fields=[...])` + 运行时 config |
| 纯提取结构化数据 | `with_structured_output()`（不建 Agent） |
| Agent 内结构化收尾 | `create_agent(response_format=...)` |
| 完整工具注册 | `create_agent(tools=...)`（bind + ToolNode） |
| 工具结果直接当答案 | `@tool(return_direct=True)` |
| 工具读对话上下文 | `InjectedState` 形参 |
| 工具读写持久数据 | `store=` + `InjectedStore()` |
| 工具出错自愈 | `@tool(handle_tool_error=True)` |
| 工具带副作用返回 | `return Command(update=..., goto=...)` |
| 会话续聊 | checkpointer + 相同 thread_id |
| 控制流程/拦截 | middleware 钩子 + jump_to（can_jump_to 声明） |
| 模型重试/降级/缓存 | `@wrap_model_call`（handler） |
| 上下文防膨胀 | `trim_messages` + 摘要 + Store 检索注入 |
| 打字机效果 | `stream_mode="messages"` |
| 展示过程 | `stream_mode="updates"` |
| 前端进度通知 | `stream_mode="custom"` + `stream_writer()` |
| 批量限流 | `batch(..., config={"max_concurrency": 5})` |
| 高可用降级 | `with_fallbacks([...])` |
| 调试追踪 | `debug=True`、config 里 run_name/tags/metadata + LangSmith |

---

**收尾**：整份手册对应一条执行链路——**配好 .env → 工厂建模型 → 备好消息 → 定义工具 → create_agent 组装 → state 流转 → 中间件干预 → 记忆加持 → 运行/流式 → 结构化交付 → 流程控制 → 错误兜底**。每个概念在链路上都有明确位置；写代码时先问自己"我现在处于哪个阶段"，再查对应条目。
