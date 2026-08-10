# LangChain 学习笔记（菜鸟教程 + 实战整理）

在当前目录下也就是D:\Code\LangChain_1.2目录下，使用langchain搭建属于我们自己的agent，下面是一些需求，也就是需要使用要的技术：
我们选用deepseek作为我们的大脑，apikey如下sk-REPLACE_WITH_YOUR_KEY，关于模型调用使用init_chat_model() 
模型选用deepseek-v4-flash，其中model参数如果不指定提供商前缀，LangChain 会尝试从模型名推断，
 .env 用来存储我们的相关配置信息，我们的模型名称还有 apikey 都可以存储在其中，到时候直接加载即可，关于kwargs 参数会直接传递给底层模型类，

控制输出随机性（0~2），值越小输出越稳定

temperature=0.3,

限制输出最大 token 数（控制成本）

max_tokens=200,

请求超时时间（秒）

timeout=30,

失败重试次数

max_retries=2,我们可以使用
相关参数 按实际情况进行设计，
*temperature 和 top_p 通常不同时设置。temperature 控制的是"分布的形状"，top_p 控制的是"候选范围"。对于大多数场景，只调整 temperature 就足够了。*ConfigurableModel——运行时切换模型

ConfigurableModel 是 init_chat_model() 的高级用法，允许在运行时动态指定模型和参数configurable_fields 的取值：

| 值                       | 含义                                               |
| :----------------------- | :------------------------------------------------- |
| None                     | 不可配置，返回普通的 BaseChatModel（固定模型模式） |
| "any"                    | 所有参数可配置（注意安全：api_key 等也能被修改）   |
| ["model", "temperature"] | 只有列表中指定的字段可配置                         |

我们选用deepseek-v4-pro模型 进行一个替代即可，运行时怎么动态执行模型啊，我只在开始任务之前选择过模型，
bind_tools()（绑定工具）和 with_structured_output()（结构化输出）。它们是 Agent 和结构化数据提取的基础。
我们后面会进行工具的实现 并进行配置，而且还会设计模型的结构化输出
*bind_tools() 只是告诉模型"你有一个工具可以用"，模型返回的是工具调用的请求。真正的执行由 Agent 或你自己编写的代码来完成。*用 Pydantic 模型描述工具

对于复杂工具，使用 Pydantic 模型定义参数结构比手写字典更清晰with_structured_output()——让模型返回结构化数据

**with_structured_output()** 是比 tool_calling 更直接的方式。它让模型按你指定的格式（Schema）返回数据，而不是返回 tool_call。with_structured_output() vs bind_tools()

这两个方法看起来相似，但用途不同：

| 对比维度 | with_structured_output()         | bind_tools()                        |
| :------- | :------------------------------- | :---------------------------------- |
| 用途     | 从文本中提取结构化数据           | 让模型知道可用的工具列表            |
| 返回格式 | 直接返回 Pydantic 对象           | 返回 AIMessage，其中包含 tool_calls |
| 适用场景 | 信息提取、数据解析               | Agent 工具调用、需要外部执行的场景  |
| 模型支持 | 需模型支持原生 structured output | 所有支持 function calling 的模型    |

什么是function calling呢？嵌套结构化输出

with_structured_output() 支持复杂的嵌套结构：**lass** Ingredient(BaseModel):
  """食材信息"""
  name: str = Field(description="食材名称")
  amount: str = Field(description="用量，如 '200g'、'2个'")


**class** CookingStep(BaseModel):
  """烹饪步骤"""
  step_number: int = Field(description="步骤编号")
  description: str = Field(description="步骤描述")
  duration_minutes: int = Field(description="此步骤需要的时间（分钟）")

**class** Recipe(BaseModel):
  """菜谱"""
  dish_name: str = Field(description="菜名")
  difficulty: str = Field(description="难度：简单、中等、困难")
  ingredients: list[Ingredient] = Field(description="食材列表")
  steps: list[CookingStep] = Field(description="烹饪步骤")
我们在实现我们的agent的时候可以设计并且实现一下

JSON Schema 模式

除了 Pydantic 模型，也可以直接传入 JSON Schema：
    # 直接传入 JSON Schema
json_schema = {
  "title": "SentimentAnalysis",
  "description": "情感分析结果",
  "type": "object",
  "properties": {
    "sentiment": {
      "type": "string",
      "enum": ["positive", "negative", "neutral"],
      "description": "情感倾向"
    },
    "confidence": {
      "type": "number",
      "description": "置信度，0~1"
    },
    "keywords": {
      "type": "array",
      "items": {"type": "string"},
      "description": "关键情感词"
    }
  },
  "required": ["sentiment", "confidence"]
}

ConfigurableModel 上的 bind_tools 和 with_structured_output

可配置模型也支持这两个方法，用法完全相同：

*在 ConfigurableModel 上链式调用 bind_tools 或 with_structured_output 时，实际操作会被延迟执行——直到模型实例化时才真正绑定，因此不会影响运行时动态切换模型的功能。*

在 LangChain 中，所有的对话都通过消息（Message）对象传递。理解各种消息类型的用途是编写 Agent 的基础。
有四大消息类型：

| 类型          | 角色    | 说明                               | 典型内容                 |
| :------------ | :------ | :--------------------------------- | :----------------------- |
| HumanMessage  | 用户    | 用户发送的消息                     | "今天天气怎么样？"       |
| AIMessage     | AI 助手 | 模型的回复，可能包含 tool_calls    | "今天杭州晴天，25°C"     |
| SystemMessage | 系统    | 系统指令，定义 AI 的角色和行为规则 | "你是一个专业的天气助手" |
| ToolMessage   | 工具    | 工具执行后的返回结果               | "晴，25°C，湿度 60%"     |

### HumanMessage——用户消息

HumanMessage 代表用户发送给 AI 的消息。它是最常见的消息类型，也是对话的起点。HumanMessage 的快捷创建方式

在构建消息列表时，可以使用元组或字典作为快捷方式：

    # 方式 1：标准构造
msg1 = HumanMessage(content="你好")

    # 方式 2：元组快捷方式 (role, content)
msg2 = ("user", "你好")
msg3 = ("human", "你好")

    # 方式 3：字典快捷方式
msg4 = {"role": "user", "content": "你好"}

    # 四种方式等价，都会在 Agent 内部被转换为 HumanMessage
**print**(type(msg1)) # <class 'langchain_core.messages.human.HumanMessage'>

### AIMessage——AI 回复

AIMessage 代表模型的回复。与普通文本不同，AIMessage 可能包含 **tool_calls**（工具调用请求）。

### SystemMessage——系统指令

SystemMessage 用于设定 AI 的行为、角色和约束。它放在消息列表的最前面，指导模型如何回复。
关于systemmessage 这个内容，他是不是领先于其他的所有消息或者说提示词，最先提交给llm的？

### ToolMessage——工具返回结果

ToolMessage 包含工具执行后的返回结果。它必须与对应的 tool_call 关联。

*ToolMessage 的 tool_call_id 必须与 AIMessage 中 tool_call 的 id 精确匹配。如果不匹配，模型可能会忽略这个工具结果，或者产生混乱的行为。*

### AIMessageChunk——流式输出的消息片段

当你使用 stream() 流式输出时，每个到达的片段是 AIMessageChunk，而非完整的 AIMessage：

    # stream() 返回的是 AIMessageChunk 迭代器
**for** chunk **in** model.stream("用一句话介绍菜鸟教程 RUNOOB"):
  # 每个 chunk 是一小段文本
  **print**(chunk.content, end="", flush=True)
**print**() # 换行

### ContentBlock——结构化消息内容

到目前为止，我们使用的消息内容都是纯字符串。但实际上每条消息的内容可以是多个 **ContentBlock**（内容块）组成的列表。

最常用的三种内容块：

| 类型                  | 说明                      | 用途                 |
| :-------------------- | :------------------------ | :------------------- |
| PlainTextContentBlock | 纯文本内容                | 普通文字消息         |
| ImageContentBlock     | 图片内容（base64 或 URL） | 多模态模型的图片输入 |
| ToolCall              | 工具调用请求              | AI 请求调用工具      |

*当你只需要发送纯文本时，直接传字符串即可，LangChain 会自动处理。只有当你需要在单条消息中混合文本和图片时，才需要手动构建 ContentBlock 列表。*

### ToolCall——工具调用消息

AIMessage 中的 tool_calls 字段是一个 ToolCall 列表，每个 ToolCall 代表模型请求调用一个工具：

    # 手动构建一个 ToolCall
tool_call = ToolCall(
  name="get_weather",     # 工具名称
  args={"city": "杭州"},   # 调用参数
  id="call_abc123",     # 唯一标识
  type="tool_call",     # 固定值
)

### trim_messages()——裁剪消息历史

当对话越来越长时，消息列表可能超出模型的上下文窗口。**trim_messages()** 函数帮助你智能地裁剪消息历史。

| strategy="last"  | 保留 system 消息 + 最近的对话 | 长对话中只关心最新上下文 |
| ---------------- | ----------------------------- | ------------------------ |
| strategy="first" | 保留 system 消息 + 最早的对话 | 确保关键上下文不被裁剪   |

    # 裁剪消息以适应模型的上下文窗口（最多 1000 tokens）
    # strategy="last" 保留最后的系统消息和最近的对话
trimmed = trim_messages(
  messages,
  max_tokens=1000,      # 最多保留 1000 tokens
  strategy="last",      # 保留最后的系统消息 + 最近的对话
  token_counter=model,    # 使用模型的 token 计数方式
  include_system=True,    # 始终保留 SystemMessage
  start_on="human",      # 裁剪后以 human 消息开头
)

*start_on="human" 确保裁剪后的消息列表以用户消息开头（而不是 AI 消息），避免让模型收到一条孤立的 AI 回复开头。*

### RemoveMessage——删除特定消息

在某些高级场景中，你可能需要从消息历史中删除特定消息（如敏感内容清洗、重新生成回复等）：

    # 使用 RemoveMessage 删除特定消息（通过 ID）
    # RemoveMessage 配合 add_messages reducer 使用
    # 在更新 Agent 状态时，RemoveMessage 会从列表中移除对应 ID 的消息
removal = RemoveMessage(id="msg_3")

*RemoveMessage 通常配合 AgentState 的 add_messages reducer 使用。在 middleware 或 after_model 钩子中返回 RemoveMessage 可以动态清理消息历史。*

### 消息属性的通用方法

所有消息类型都继承自 BaseMessage，共享一些通用方法：

msg = HumanMessage(content="你好，菜鸟教程")

    # 基本属性
**print**(f"content: {msg.content}")    # 消息内容
**print**(f"type: {msg.type}")       # 消息类型（human/ai/system/tool）
**print**(f"id: {msg.id}")         # 自动生成的唯一 ID

    # text 属性：如果是文本内容，返回文本；否则返回 ""
**print**(f"text: {msg.text}")

    # pretty_repr()：格式化打印，适合调试
**print**(f"美化输出:**\n**{msg.pretty_repr()}")

## LangChain @tool 装饰器——定义工具

工具（Tool）是 Agent 与外部世界交互的桥梁。

通过 **@tool** 装饰器，你可以将任何 Python 函数快速转换为 Agent 可调用的工具。

### @tool 基本语法

@tool 是 LangChain 提供的装饰器，用法极其简单：在函数上加上 @tool 装饰器，函数就变成了一个工具。

    # 最简单的工具：一个普通函数 + @tool 装饰器
@tool
**def** hello_tool(name: str) -> str:
  """向指定的人打招呼。

  Args:
    name: 要打招呼的人的名字
  """
  **return** f"你好，{name}！欢迎来到菜鸟教程 RUNOOB。"

    # 工具也是普通的 Python 函数，可以直接调用
result = hello_tool.invoke({"name": "小明"})
**print**(result)

    # 工具包含自动生成的描述信息
**print**(f"**\n**工具名称: {hello_tool.name}")
**print**(f"工具描述: {hello_tool.description}")

*函数的文档字符串（docstring）会自动成为工具的描述。Agent 依赖这个描述来判断"这个工具能做什么"和"什么情况下应该调用它"。文档字符串写得越清晰，Agent 使用工具就越准确。在描述中说明参数含义、函数功能和使用场景。*

所谓的文档字符串就是 """ 123 """ 吗
@tool 支持多种参数类型，包括 int、float、bool 和枚举值

**def** search_courses(
  keyword: str,
  level: Literal["入门", "进阶", "高级"],
  max_results: int = 5,
  free_only: bool = True,
) -> str:

### 注册工具到 Agent

将定义好的工具传给 create_agent() 的 tools 参数，Agent 就能使用它了
是不是这里也可以使用 bind_tools() 进行工具的注册

一个 Agent 可以注册多个工具，模型会自动判断何时使用哪个工具：
为工具参数设置默认值，可以让工具用起来更灵活

*默认值让 Agent 在调用工具时不用每次都指定所有参数。但注意：如果某个参数没有默认值且 Agent 没有提供，调用会失败。关键参数不要设默认值。*

### 工具的 args_schema——自定义参数校验

对于复杂参数校验需求，可以使用 Pydantic 模型作为 args_schema
@tool(args_schema=CourseSearchInput)

### 工具定义方式对比

| 方式                | 代码量           | 适用场景               | 示例                 |
| :------------------ | :--------------- | :--------------------- | :------------------- |
| @tool 装饰器        | 最少             | 简单到中等复杂度的工具 | 大多数场景           |
| @tool + args_schema | 中等             | 需要精细参数校验的工具 | API 封装、数据库操作 |
| Pydantic 类作为工具 | 较多             | 复杂业务逻辑的工具     | 内部包含状态的工具   |
| 字典格式            | 最少（但不推荐） | 描述远程/内置工具      | MCP 工具、服务端工具 |

### return_direct——直接返回最终结果

默认情况下，工具执行后结果会返回给模型，模型再基于工具结果生成最终回复。但有时工具结果本身就是你想要的最终答案。

设置 **return_direct=True** 后，工具执行完就立即结束 Agent 循环，工具返回内容直接作为最终输出。

@tool(return_direct=True)

| return_direct=False（默认） | 模型收到工具结果 → 模型继续思考 → 生成最终回复 | 需要分析/总结/进一步决策             |
| --------------------------- | ---------------------------------------------- | ------------------------------------ |
| return_direct=True          | 工具执行后立即结束 → 工具结果就是最终输出      | 查询类、数据获取类、已格式化好的结果 |

*当你设置 return_direct=True 时，Agent 会跳过后续的模型思考步骤，直接返回工具结果。这在节省 Token 和降低时延方面非常有价值，但也意味着模型不会对工具结果做任何二次加工。*

*如果一个 Agent 同时挂载了多个工具，其中既有 return_direct=True 的工具，也有普通工具，那么只要模型在这一轮调用中触发了任意一个 return_direct 工具，Agent 循环就会立即结束——即使同一轮还并行调用了其他普通工具，它们的结果也不会再被模型加工总结。设计包含多个工具的 Agent 时要留意这一点，避免"该总结的内容被跳过"。*

也就是说多工具调用的时候 只要有一个设置了直接返回结果，那么所有的其他工具 都默认直接返回结果，不会送入llm 进行二次加工

### InjectedToolCallId——获取工具调用 ID

有时工具需要知道"是谁调用了它"——**InjectedToolCallId** 可以在工具函数中注入当前的 tool_call_id

tool_call_id: Annotated[str, InjectedToolCallId]

tool_call_id: 系统自动注入的工具调用 ID

*带有* **InjectedToolArg** *标记的参数不需要由 Agent（模型）提供， 这些参数会由 LangChain 运行时在执行工具时自动注入。 由于这些参数不会出现在工具的 schema 中，模型无法看到它们， 因此不应该把它们作为用户需要填写的工具参数进行描述。*

例如 **InjectedToolCallId** 是一种特殊的注入参数， LangChain 会在工具执行阶段自动传入当前 ToolCall 的唯一 ID。 它不会出现在模型生成的参数中，但会和当前这一次工具调用自动绑定。

直接调用 **.invoke()** 并手动构造 **ToolCall**， 只是为了演示 LangChain 如何完成参数注入机制。 实际项目中通常不会手动传递这个 ID。

在 Agent 工作流中， **InjectedToolCallId** 更常用于需要关联当前调用上下文的场景， 例如配合 **Command** 对象在工具内部更新 Agent 状态， 向 **messages** 列表追加关联当前调用的 **ToolMessage**， 或者实现工具调用追踪、审计、日志关联等功能。

这也是它被设计为"注入参数"而不是普通参数的原因： 它代表的是 LangChain Runtime 当前正在执行的这一次 ToolCall 上下文， 而不是用户输入的一部分。

### ToolException——工具异常处理

工具执行过程中可能会出错。使用 **ToolException** 抛出明确的工具异常，让 Agent 知道出了问题。

**raise** ToolException(f"用户 ID 必须为正整数，收到了: {user_id}")

    # 异常调用 1：无效 ID
**try**:
  get_user_info.invoke({"user_id": -1})
**except** ToolException **as** e:
  **print**(f"工具异常: {e}")

*这里之所以能在 .invoke() 外层用 try/except 捕获到 ToolException，是因为工具默认的* **handle_tool_error** *为 False——异常不会被工具自己吞掉，而是照常向上抛出。如果希望工具"自己扛住"错误、把错误信息转成字符串返回给模型而不是抛异常，就是下一节要讲的 handle_tool_error。*

### handle_tool_error——让工具自己处理错误

当希望某个工具出错时不中断程序，而是把错误信息转成一段文本、当作正常返回值交给模型自己去理解和修正时，可以在定义工具时设置 **handle_tool_error**（注意是单数，没有 s）。这是 BaseTool 上的一个属性，最简单的设置方式是直接写在 @tool 装饰器里：

    # handle_tool_error=True：出错时不再抛异常，
    # 而是把 ToolException 的内容转成字符串，正常返回给模型
@tool(handle_tool_error=True)

    # 如果想让所有工具的错误都由 Agent 统一处理（而不是逐个工具设置），
    # 可以在 create_agent 内部使用的 ToolNode 层面配置。
    # 在较新版本的 langchain 中，可以这样为整个 Agent 打开错误兜底

| handle_tool_error              | 行为                                                     | 适用场景                               |
| :----------------------------- | :------------------------------------------------------- | :------------------------------------- |
| False（默认）                  | ToolException 照常向上抛出，调用方需自行 try/except      | 不可恢复的错误，需要中断流程           |
| True                           | 捕获 ToolException，将其内容作为工具的正常返回值交给模型 | 希望 Agent 自行阅读错误信息并修正/重试 |
| str                            | 捕获异常后，用指定的固定字符串替换错误信息               | 不想暴露具体报错细节，只给统一提示     |
| Callable[[ToolException], str] | 捕获异常后，用自定义函数处理异常并生成返回内容           | 需要按异常类型/内容定制不同的提示      |

*不要用* **tool.with_config(handle_tool_errors=True)** *这种写法——***with_config()** *是 Runnable 通用的运行时配置方法，只用于设置 callbacks、tags、metadata 等，并不会真正修改工具的错误处理行为。控制单个工具的错误处理，要用* **@tool(handle_tool_error=...)***（单数）；如果想在 Agent/图（Graph）层面统一为所有工具配置错误处理策略，则是在底层的 ToolNode 上设置 handle_tool_errors（复数），而不是作用在单个工具对象上。两者字段名相似但作用层级不同，使用时注意区分。*

*工具设置了* **handle_tool_error=True** *后，ToolException 的内容会以一条 ToolMessage 的形式正常出现在对话历史里（上面输出中的 [tool] 那一行），而不会让程序崩溃；模型看到这条错误信息后，会用自然语言把它转达给用户，并给出可用的替代方案。这正是 return_direct、InjectedToolCallId、ToolException 和 handle_tool_error 组合使用时的典型效果：既保证了工具调用的健壮性，又不牺牲用户体验。*

## LangChain 工具访问 -- InjectedState 与 InjectedStore

有时工具需要访问更多上下文信息，比如当前对话的状态、用户的持久化数据等。

LangChain 通过依赖注入机制，让工具函数能够自动获取这些信息。

------

### InjectedState——在工具中访问 Agent 状态

默认情况下，工具只能通过参数接收模型传来的数据。但有时工具需要知道当前对话的上下文——比如之前的对话历史、用户已确认的信息等。

**InjectedState** 让工具可以直接读取 Agent 的完整状态。

state: Annotated[dict[str, Any], InjectedState]

state: 系统自动注入的当前 Agent 状态

都是由系统自动注入，这是由langchain 来实现的吗
    # InjectedState 会被自动注入，Agent 不需要传这个参数

    # state 不需要传，由框架自动注入

*InjectedState 让你可以访问 AgentState 中的所有字段。如果你扩展了 state_schema（增加了自定义字段），这些字段也可以被 InjectedState 读取到。*

### InjectedStore——在工具中访问持久化存储

Agent 状态（state）是对话级别的，对话结束就没了。而 **Store** 是跨会话的持久化存储，可以用来保存用户偏好、学习进度等长期信息。

**InjectedStore** 让工具可以直接读写 Store。

在 Agent 中结合 Store

将 Store 传给 create_agent()，Agent 中的所有工具都能通过 InjectedStore 访问它：

agent = create_agent(
  model=model,
  tools=[query_course_price],
  store=store, # 将 Store 传入 Agent
)

### InjectedState vs InjectedStore 对比

| 维度     | InjectedState                    | InjectedStore                 |
| :------- | :------------------------------- | :---------------------------- |
| 作用域   | 当前对话（单次 Agent 运行）      | 跨会话（多次 Agent 运行共享） |
| 生命周期 | 对话结束即消失                   | 持久化存储                    |
| 典型用途 | 读取消息历史、当前对话的中间结果 | 用户偏好、学习进度、配置信息  |
| 传入方式 | InjectedState（自动注入）        | InjectedStore()（需要括号）   |
| 数据组织 | 扁平字典                         | 命名空间 + 键的层级结构       |

state 是 会话级别的 存储
store 是 全局级别的 存储

state 通过 依赖注入的方式 进行自动注入，具体表现形式就是函数的 形参列表 进行自动注入
store 通过 InjectedStore() 手动调用 put ()方法，进行存储，

具体的数据组织形式

什么是扁平字典 ？

什么是 命名空间 + 键 的层级结构 

### InjectedToolArg——标记通用注入参数

除了专门的 InjectedState 和 InjectedStore，你还可以用 **InjectedToolArg** 标记任何需要框架注入的参数

injected_param: Annotated[str, InjectedToolArg]

injected_param: 这个参数由框架注入（Agent 不需要提供）

具体表现形式 通过 函数中的 形参 通过 依赖注入 进行 自动注入

*InjectedToolArg 是通用的注入标记，InjectedState、InjectedStore 和 InjectedToolCallId 都是基于它实现的。大多数情况下使用专门的注入标记即可，InjectedToolArg 用于扩展自定义注入逻辑。*

## create_agent() 函数

create_agent() 是 LangChain 最核心的函数，它会创建一个完整的 Agent 图（StateGraph），包含模型调用、工具执行、循环控制等全部逻辑。

```python
agent = create_agent(
    model,                     # str | BaseChatModel：语言模型
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
    debug=False,               # bool：是否输出详细日志
    name=None,                 # str：Agent 名称
    cache=None,                # BaseCache：缓存配置
)
```

### model 参数——模型配置

model 可以接受两种形式：字符串（由 init_chat_model() 处理）或已构建好的 BaseChatModel 实例。

```python
# 方式 1：传字符串（最常用）
# create_agent 内部会调用 init_chat_model() 处理
agent = create_agent(
    model="deepseek:deepseek-v4-flash",
    system_prompt="你是菜鸟教程 RUNOOB 的助手",
)

# 方式 2：传已构建好的模型实例
# 适合需要精细控制模型参数的场景
model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0.3, max_tokens=500)
agent = create_agent(
    model=model,
    system_prompt="你是菜鸟教程 RUNOOB 的助手",
)

# 方式 3：传已绑定工具的模型实例
# less common，通常让 create_agent 自己管理工具绑定
model_with_tools = init_chat_model("deepseek:deepseek-v4-flash").bind_tools([...])
```

三种形式，任选其一 即可

*system_prompt 是可选的，但不传的话模型会以"通用助手"的角色回答。对于有明确业务场景的应用，建议始终设置 system_prompt 来约束模型的行为边界。*

### state_schema 参数——自定义状态

默认的 AgentState 只包含 messages、jump_to 和 structured_response。如果你需要额外的状态字段，可以扩展它

```python
# 扩展 AgentState，添加自定义字段
class LearningAgentState(AgentState):
    """自定义状态，增加学习进度相关字段"""
    user_level: str                       # 用户等级
    completed_topics: list[str]           # 已完成的主题列表
    
agent = create_agent(
    model="deepseek:deepseek-v4-flash",
    tools=[track_progress],
    state_schema=LearningAgentState,  # 使用自定义状态
)

# 运行时需要提供自定义状态的初始值
result = agent.invoke({
    "messages": [HumanMessage(content="我学完了 Python 基础，帮我记录一下")],
    "user_level": "入门",
    "completed_topics": ["HTML 基础"],
})
```

### 返回值——CompiledStateGraph

create_agent() 返回一个 **CompiledStateGraph** 对象，这是 LangGraph 的编译后的图，提供了多种运行方式：

| 方法                                | 说明                   | 适用场景          |
| :---------------------------------- | :--------------------- | :---------------- |
| invoke(input, config)               | 同步运行，等待完整结果 | 脚本、简单接口    |
| ainvoke(input, config)              | 异步运行，等待完整结果 | Web 服务          |
| stream(input, config, stream_mode)  | 同步流式运行           | 实时展示中间步骤  |
| astream(input, config, stream_mode) | 异步流式运行           | WebSocket、SSE    |
| get_state(config)                   | 获取当前状态           | 查看/恢复对话状态 |
| update_state(config, values)        | 更新状态               | 手动修改对话状态  |

    # 使用 stream_mode="updates" 可以看到每一个步骤

agent.stream(
  {"messages": [HumanMessage(content="杭州现在天气怎么样？几点了？")]},
  stream_mode="updates",
)

### stream_mode 详解

stream() 支持多种 stream_mode，每种提供不同粒度的信息：

| 模式     | 返回内容                 | 适用场景                                     |
| :------- | :----------------------- | :------------------------------------------- |
| updates  | 每个节点执行后的状态更新 | 追踪 Agent 执行步骤，显示中间结果            |
| values   | 每个节点执行后的完整状态 | 需要在每一步看到完整消息历史                 |
| messages | 逐 Token 的消息流        | 前端流式展示 AI 打字效果                     |
| custom   | 自定义事件               | Middleware 通过 stream_writer 发送自定义事件 |

### Agent 的退出条件

Agent 什么时候停止？主要有以下几种情况：

| 退出条件            | 说明                                    | 示例                                 |
| :------------------ | :-------------------------------------- | :----------------------------------- |
| 无工具调用          | 模型返回的 AIMessage 中 tool_calls 为空 | 模型认为任务完成，直接回复           |
| return_direct=True  | 工具标记为直接返回，执行后立即结束      | 查询类工具，结果即最终答案           |
| structured_response | 模型产出了结构化输出                    | response_format 指定的结构化输出完成 |
| jump_to="end"       | Middleware 通过状态控制主动结束         | 检测到问题越权，提前终止             |

前两个我清楚 和工具相关的 调用工具返回结果为空时 直接回复 调用工具后信息直接返回，也是立即回复
在 结构化响应输出 中 在哪里进行这样一个配置，还是说llm 自动进行检测
jump_to 好像有很多对象中的属性 都会有这样一个字段名 jump_to  具体有哪些，可以帮我整理一下吗

invoke vs stream 对比

| 方法      | 返回时机             | 适用场景               | 用户体验           |
| :-------- | :------------------- | :--------------------- | :----------------- |
| invoke()  | 全部完成后一次性返回 | 脚本、API 接口、批处理 | 等待后看到完整结果 |
| stream()  | 逐步返回中间状态     | 聊天界面、需要展示过程 | 实时看到进展       |
| ainvoke() | 异步全部完成后返回   | Web 服务、异步框架     | 不阻塞事件循环     |
| astream() | 异步逐步返回         | WebSocket、SSE 推送    | 服务端实时推送     |

关于异步场景，有什么例子可以举例吗
stream 是不是可以搭配这 带有深度思考 比如reasoning_context 字段的 模型 进行搭配使用

### 在 with_config 中传入线程 ID

如果你使用了 checkpointer，需要通过 config 传入 thread_id 来管理对话线程
checkpointer 是有关于 记忆相关的内容吧，好像是会话级别的短期记忆
checkpointer 是不是 会存储一些 thread_id 之类的东西，会话信息 可以持久化到数据库中 然后再恢复

## LangChain AgentState 状态管理

Agent 在执行过程中需要维护状态——消息历史、结构化响应、流程控制等。理解 AgentState 的结构和用法，是自定义 Agent 行为的关键。

------

### AgentState 结构

AgentState 是一个 TypedDict，默认包含三个字段

    # AgentState 的实际定义（简化版）
**class** AgentState(TypedDict):
  # messages：消息历史，使用 add_messages 作为 reducer
  # Required 表示调用时必须提供
  messages: Required[Annotated[list[AnyMessage], add_messages]]

  # jump_to：流程跳转控制，ephemeral（使用后自动清除）
  # NotRequired 表示可选
  jump_to: NotRequired[Annotated[str | None, EphemeralValue]]

  # structured_response：结构化输出结果
  # NotRequired 表示可选，仅在 response_format 设置时出现
  structured_response: NotRequired[Any]

| 字段                | 类型             | 是否必填 | 说明                                                         |
| :------------------ | :--------------- | :------- | :----------------------------------------------------------- |
| messages            | list[AnyMessage] | 是       | 消息历史，使用 add_messages reducer 追加                     |
| jump_to             | str 或 None      | 否       | 流程跳转控制，可选值：tools、model、end。ephemeral 属性，使用后自动清除 |
| structured_response | Any              | 否       | 结构化输出结果，不在 input schema 中暴露                     |

我们create_agent() 中 声明 response_format 是什么  那么 AgentState 中的structured_response 是不是 只用来做一个记录呢？并不会有什么 影响

### messages——消息历史的 Reducer 机制

messages 字段使用了 **add_messages** reducer。这意味着更新 messages 时不是覆盖，而是**追加**

add_messages 的智能特性：

- **同名覆盖**：如果新消息 ID 与已有消息相同，会替换而非追加
- **RemoveMessage 支持**：遇到 RemoveMessage 时，从列表中删除对应消息
- **类型安全**：自动处理 HumanMessage、AIMessage、ToolMessage 等不同类型

result = add_messages(existing, [new_msg])

### jump_to——流程跳转控制

jump_to 是 Middleware 中最常用的字段，用于在 Agent 的各个节点间跳转。

jump_to 是一个 **ephemeral**（瞬态）字段——使用一次后自动清除，不需要手动重置。

    # 声明可跳转目标 "end"
@before_model(can_jump_to=["end"])

    # 检查是否包含不当内容（简化示例）
  **if** "密码" **in** str(last_msg.content):
    # jump_to="end" 直接结束 Agent，不让模型回复
    **return** {
      "jump_to": "end",
      # 使用 AIMessage
      "messages": [AIMessage(
        content="抱歉，出于安全原因，不能回答关于密码的问题。"
      )]
    }
  **return** None

| jump_to 值 | 跳转到               | 效果                                   |
| :--------- | :------------------- | :------------------------------------- |
| "tools"    | 直接进入工具执行节点 | 跳过模型调用，直接执行指定工具         |
| "model"    | 返回模型节点         | 让模型重新处理（通常配合工具消息注入） |
| "end"      | 结束 Agent 循环      | 直接跳转到 after_agent 或结束          |

*jump_to 是 ephemeral 的——每次节点执行后自动清除。这意味着你不需要在跳转后手动将 jump_to 设回 None，Agent 会自动处理。*

### structured_response——获取结构化输出

当使用 response_format 参数时，Agent 会将结构化输出存储在 structured_response 字段中：

response_format=CourseRecommendation,

### 自定义 State 扩展

在实际应用中，你可能需要 Agent 维护额外的状态。通过继承 AgentState 来扩展

### state_schema vs middleware state_schema

既可以通过 create_agent() 的 state_schema 参数扩展状态，也可以通过 Middleware 的 state_schema 扩展。两者的区别：

| 方式                              | 使用场景                   | 优先级                           |
| :-------------------------------- | :------------------------- | :------------------------------- |
| create_agent(state_schema=...)    | 全局状态扩展，所有节点共享 | 最高（覆盖 middleware 同名字段） |
| AgentMiddleware(state_schema=...) | 特定 middleware 的状态扩展 | 较低，可被 create_agent 覆盖     |

*推荐做法：将通用的业务状态字段放在 state_schema 中，将特定 middleware 相关的内部字段放在 middleware 的 state_schema 中。这样职责清晰，不会相互污染。*

## System Prompt 与 Dynamic Prompt

System Prompt（系统提示词）是控制 Agent 行为的核心手段。

### system_prompt 参数

create_agent() 的 system_prompt 参数接受两种形式

    # 方式 1：字符串（最简单）
agent = create_agent(
  model=model,
  system_prompt="你是菜鸟教程 RUNOOB 的学习顾问，回答要简洁专业。",
)

    # 方式 2：SystemMessage 对象（可复用）
system_msg = SystemMessage(
  content="你是菜鸟教程 RUNOOB 的学习顾问，回答要简洁专业。"
)
agent = create_agent(model=model, system_prompt=system_msg)

```
# 一个设计良好的 system_prompt
GOOD_PROMPT = """你是菜鸟教程 RUNOOB 的学习顾问。

## 你的职责
- 帮助用户找到合适的编程课程
- 回答编程学习相关的问题
- 根据用户水平推荐学习路径

## 行为准则
- 回答要简洁，每次不超过 3 句话
- 优先使用 search_course 工具查询课程信息
- 如果用户是零基础，优先推荐入门课程
- 不使用 emoji 表情
- 不知道的就说不知道，不要编造"""

model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
agent = create_agent(
    model=model,
    tools=[search_course],
    system_prompt=GOOD_PROMPT,
)
```

### @dynamic_prompt——动态生成提示词

静态 system_prompt 对所有用户一视同仁。但实际应用中，你可能需要根据用户信息、对话上下文、时间等动态调整提示词。

**@dynamic_prompt** 装饰器让你在每次模型调用前动态生成 system_prompt
    # @dynamic_prompt 装饰器：接收 ModelRequest，返回新的 system_prompt
@dynamic_prompt
**def** personalized_prompt(request: ModelRequest) -> str:
  """根据对话上下文动态生成个性化提示词"""
  messages = request.state.get("messages", [])
  message_count = len(messages)

middleware=[personalized_prompt], # 通过 middleware 注入

### @dynamic_prompt 进阶——结合运行时上下文

@dynamic_prompt 的 request 参数提供了丰富的信息

"""根据用户信息、时间和对话阶段动态生成提示词"""
  # 从 runtime.context 获取用户信息
  context = request.runtime.context

*@dynamic_prompt 在每次模型调用前都会执行，所以提示词可以随对话推进而变化。但注意不要在里面做太重的计算，否则会影响响应速度。*

如果同时设置了 create_agent() 的 system_prompt 和 @dynamic_prompt middleware，middleware 的优先级更高——它会覆盖静态提示词。

*如果你希望 middleware 的提示词和静态提示词合并而不是覆盖，可以在 @dynamic_prompt 中手动拼接。request 中没有直接暴露原有的 system_prompt，所以如果需要保留原有内容，建议将静态提示词作为变量在函数中引用。*

### System Prompt 设计清单

| 要素         | 说明                     | 示例                                  |
| :----------- | :----------------------- | :------------------------------------ |
| 角色定义     | 明确 AI 的身份和职责     | 你是菜鸟教程 RUNOOB 的学习顾问        |
| 行为准则     | 约束回复的风格和边界     | 回答简洁，每次不超过 3 句话           |
| 工具使用指引 | 告诉模型何时使用哪些工具 | 查询课程时优先使用 search_course 工具 |
| 边界约束     | 明确什么不能做           | 不知道的就说不知道，不要编造          |
| 格式要求     | 指定回复的格式（可选）   | 回复使用 Markdown 格式                |

## LangChain 流式输出 Streaming

流式输出让 AI 的回复像打字一样逐字显示，极大地提升了用户体验。LangChain 的 Agent 内置了完善的流式输出支持。

------

### 为什么需要流式输出

如果使用 invoke()，用户需要等待 Agent 完成所有步骤（多次模型调用 + 工具执行）才能看到结果。对于复杂任务，这可能耗时十几秒甚至更长。

流式输出解决了这个问题：每生成一个 Token 就立即返回，用户可以实时看到进展。

| 方式     | 用户体验                  | 适用场景           |
| :------- | :------------------------ | :----------------- |
| invoke() | 等待 → 一次性看到完整结果 | 脚本、API、批处理  |
| stream() | 实时看到每一个 Token      | 聊天界面、实时展示 |

### stream_mode="messages"——逐 Token 流式

这是最细粒度的流式模式，每个 chunk 对应一个 Token

agent.stream(
  {"messages": [HumanMessage(content="用一句话介绍菜鸟教程 RUNOOB")]},
  stream_mode="messages",
)

### metadata

metadata 包含了这个 chunk 的来源信息

metadata.get('langgraph_node')

### stream_mode="updates"——逐步查看 Agent 执行过程

这个模式在构建需要显示"思考过程"的界面时非常有用

### stream_mode="custom"——发送自定义事件

通过 Middleware 的 runtime.stream_writer()，你可以向流中发送自定义事件

可以进行一个混合设计

    # 使用 stream_mode=["updates", "custom"] 同时接收两种事件
**print**("=== 混合流式输出 ===**\n**")
**for** mode, chunk **in** agent.stream(
  {"messages": [HumanMessage(content="查一下 Python 课程")]},
  stream_mode=["updates", "custom"],
):
  **if** mode == "custom":
    **print**(f"[自定义事件] 状态: {chunk['message']}")
  **elif** mode == "updates":
    **for** node_name, update **in** chunk.items():
      **if** "messages" **in** update:
        **for** msg **in** update["messages"]:
          **if** msg.type == "ai" **and** msg.content:
            **print**(f"[回复] {msg.content}")

*stream_mode 可以组合使用，如 stream_mode=["updates", "custom", "messages"]。但过多的模式会增加流中的事件量，建议按需选择。*

### 异步流式输出

在 Web 服务中，使用异步流式可以避免阻塞事件循环

> 生产环境中，建议将 Agent 实例创建为全局单例，避免每次请求都重新创建。Agent 的创建开销很小（主要是编译图），但复用实例更高效。

### stream_mode 速查表

| 模式     | 粒度           | 迭代对象                   | 典型用途           |
| :------- | :------------- | :------------------------- | :----------------- |
| messages | Token 级       | (AIMessageChunk, metadata) | 打字效果、实时聊天 |
| updates  | 节点级         | {node_name: state_update}  | 展示思考过程       |
| values   | 节点级（全量） | 完整 state                 | 状态快照、调试     |
| custom   | 自定义         | 任意 dict                  | 进度通知、状态推送 |
| debug    | 详细           | 调试信息                   | 开发阶段排查问题   |

## LangChain 结构化输出

大多数时候，你需要的不是一段自由文本，而是结构化的数据——比如 JSON 对象。

LangChain 结构化输出(Structured Output) 让 Agent 按照你指定的格式返回结果，方便程序直接使用。

------

### 为什么需要结构化输出

假设你需要从一段用户描述中提取姓名、年龄和职业：

| 方式       | 输出格式                               | 后续处理                     |
| :--------- | :------------------------------------- | :--------------------------- |
| 普通回复   | "张三今年28岁，是一名工程师"           | 需要正则或再次调用模型来解析 |
| 结构化输出 | {name: "张三", age: 28, job: "工程师"} | 直接作为 Python 对象使用     |

结构化输出省去了"从文本中解析数据"这一步，让 AI 的输出可以直接被程序使用。

### 传入 Pydantic 模型

将 Pydantic 模型传给 response_format 参数即可

agent = create_agent(
  model=model,
  response_format=CourseInfo, # 传入 Pydantic 模型
  system_prompt="你是菜鸟教程 RUNOOB 的课程助手，从用户描述中提取课程信息。",
)

*返回的 structured_response 是 Pydantic 模型实例，而不是普通字典。这意味着你可以使用 .course_name 等属性访问，IDE 也能提供自动补全。*

### 与工具共存的 Structured Output

response_format 和 tools 可以同时使用——Agent 在需要时调用工具，最终输出结构化数据：

**class** CourseRecommendation(BaseModel):
  """课程推荐结果"""
  course_name: str = Field(description="推荐课程名称")
  reason: str = Field(description="推荐理由")
  difficulty: str = Field(description="难度：入门/进阶/高级")

agent = create_agent(
  model=model,
  tools=[search_course],
  response_format=CourseRecommendation,
  system_prompt="你是菜鸟教程 RUNOOB 的课程顾问。先查询课程再给出推荐。",
)

rec = result["structured_response"]

    # 查看完整过程
**print**("**\n**=== 执行过程 ===")
**for** msg **in** result["messages"]:
  **if** msg.type == "tool":
    **print**(f"  调用 {msg.name}: {msg.content}")

### 复杂嵌套结构

Pydantic 支持嵌套、列表、枚举等复杂结构

**class** Topic(BaseModel):
  """知识点"""
  name: str = Field(description="知识点名称")
  order: int = Field(description="学习顺序，从 1 开始")
  minutes: int = Field(description="建议学习分钟数")

**class** LearningPlan(BaseModel):
  """学习计划"""
  goal: str = Field(description="学习目标概述")
  level: Literal["入门", "进阶", "高级"] = Field(description="难度级别")
  total_hours: float = Field(description="总时长（小时）")
  topics: list[Topic] = Field(description="知识点列表")

agent = create_agent(
  model=model,
  response_format=LearningPlan,
)

plan = result["structured_response"]

### 从消息中获取结构化输出

如果不需要 Agent 的工具调用能力，只是想从文本中提取结构化信息，可以直接用模型

    # 直接在模型上使用 with_structured_output()
    # 不需要 Agent
structured_model = model.with_structured_output(SentimentResult)

*with_structured_output() 是 model 的方法，不需要 Agent 就可以使用。如果你的场景是"信息提取"而非"多步骤推理"，直接用 with_structured_output() 更简洁高效。*

## LangChain 输出策略

LangChain 提供了三种结构化输出策略，理解它们的区别和工作原理，能帮助你在不同场景下做出最佳选择。

------

### 三种策略概述

| 策略             | 原理                                                         | 模型支持                         | 响应速度               |
| :--------------- | :----------------------------------------------------------- | :------------------------------- | :--------------------- |
| ToolStrategy     | 将 Schema 伪装成工具，模型"调用"这个工具来输出结构化数据     | 所有支持 function calling 的模型 | 较慢（多一次工具调用） |
| ProviderStrategy | 使用模型原生的结构化输出能力（如 OpenAI 的 response_format） | 部分模型（GPT-4o+、Claude 3+等） | 较快（直接输出）       |
| AutoStrategy     | 自动检测模型能力，选择最佳策略                               | 自动适配                         | 自动选择最优           |

### ToolStrategy——工具调用模式

ToolStrategy 是兼容性最好的方式。它将你的 Schema 转换为一个"假工具"，模型通过调用这个工具来输出结构化数据。

    # 显式指定使用 ToolStrategy
agent = create_agent(
  model=model,
  response_format=ToolStrategy(schema=WeatherReport),
  system_prompt="你是天气助手，根据用户描述生成结构化天气报告。",
)

ToolStrategy 多了一个工具调用步骤（调用名为 WeatherReport 的"假工具"），然后才有结构化输出。

### handle_errors——错误重试

ToolStrategy 支持在结构化输出出错时自动重试：

    # handle_errors=True：输出格式错误时，将错误信息反馈给模型重试
strategy_with_retry = ToolStrategy(
  schema=WeatherReport,
  handle_errors=True, # 默认 False
)

    # handle_errors 也可以是一个自定义错误消息模板
strategy_custom_error = ToolStrategy(
  schema=WeatherReport,
  handle_errors="格式有误，请按 {error} 修正后重新输出",
)

### ProviderStrategy——原生结构化输出

ProviderStrategy 使用模型提供商的原生能力（如 OpenAI 的 response_format 参数）。不是所有模型都支持。

    # 显式指定 ProviderStrategy
agent = create_agent(
  model=model,
  response_format=ProviderStrategy(schema=CourseInfo),
  system_prompt="你是菜鸟教程 RUNOOB 的课程助手。",
)

*ProviderStrategy 目前主要被 OpenAI 的 GPT-4o 及以上和 Claude 3 及以上支持。如果模型不支持，LangChain 会自动降级到 ToolStrategy。检查模型是否支持可以用 model.profile 查看。*

### AutoStrategy——自动选择

这是最推荐的方式。传入 Pydantic 模型（而不是策略对象），LangChain 会自动选择最佳策略：

    # 直接传入 Pydantic 模型——LangChain 自动选择策略
model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
agent = create_agent(
  model=model,
  response_format=Analysis, # 直接传模型，自动选择策略
  system_prompt="你是课程评估专家，评估用户描述的课程。",
)

**class** Analysis(BaseModel):
  """分析结果"""
  summary: str = Field(description="一句话总结")
  score: int = Field(description="评分 1~10")
  pros: list[str] = Field(description="优点列表")
  cons: list[str] = Field(description="缺点列表")

### 三种策略选择指南

| 场景                       | 推荐策略                         | 原因                                   |
| :------------------------- | :------------------------------- | :------------------------------------- |
| 不确定模型是否支持原生输出 | AutoStrategy（直接传 Pydantic）  | 自动选择最优策略                       |
| 需要兼容各种模型           | ToolStrategy                     | 所有支持 function calling 的模型都可用 |
| 追求极致性能               | ProviderStrategy                 | 跳过工具调用环节，速度更快             |
| 需要错误重试               | ToolStrategy(handle_errors=True) | 只有 ToolStrategy 支持 handle_errors、 |

*大多数情况下，直接传入 Pydantic 模型（即使用 AutoStrategy）就够了。只有在需要错误重试或明确控制策略行为时，才需要显式指定 ToolStrategy 或 ProviderStrategy。*

## LangChain 中间件（Middleware）

LangChain Middleware（中间件）是 LangChain 最强大的特性。它让你在 Agent 执行的各个环节插入自定义逻辑，实现重试、降级、缓存、内容过滤、日志记录等功能——而不需要修改 Agent 本身的代码。

### 什么是 Middleware

Middleware 是 Agent 执行流程中的**钩子（Hook）**。每个钩子让你在特定的时间点执行自定义代码

```
# Middleware 的直观理解：
# 假设 Agent 的执行流程是这样的：

# 1. 用户输入 → 2. 模型思考 → 3. 可能调用工具 → 4. 模型再思考 → 5. 输出结果

# Middleware 让你可以在这 5 个环节之间插入自定义逻辑：
# 1. 用户输入
#    ↓ [before_agent 钩子：日志记录、权限检查]
# 2. 模型思考
#    ↓ [before_model 钩子：消息预处理]
#    ↓ [wrap_model_call 钩子：重试、降级、缓存]
#    ↓ [after_model 钩子：内容审核]
# 3. 工具执行
#    ↓ [wrap_tool_call 钩子：工具调用重试]
# 4. 回到模型思考（循环直到完成）
#    ↓ [after_agent 钩子：结果格式化、统计分析]
# 5. 输出结果
```

### 六个钩子点

LangChain 的 Middleware 提供了 6 个钩子，按执行时机分为两类：

| 钩子            | 执行频率     | 执行位置     | 主要用途                     |
| :-------------- | :----------- | :----------- | :--------------------------- |
| before_agent    | 一次         | Agent 开始前 | 初始化、权限检查、输入预处理 |
| before_model    | 每次循环     | 模型调用前   | 消息预处理、动态上下文注入   |
| wrap_model_call | 每次循环     | 包裹模型调用 | 重试、降级、缓存、请求改写   |
| after_model     | 每次循环     | 模型调用后   | 内容审核、响应过滤、日志     |
| wrap_tool_call  | 每次工具调用 | 包裹工具执行 | 工具重试、结果缓存、参数改写 |
| after_agent     | 一次         | Agent 结束后 | 格式化输出、统计、清理资源   |

### 两种使用方式

Middleware 可以通过类继承或装饰器两种方式使用：

### 方式 1：装饰器（推荐）

    # 装饰器方式：简单、直观
@before_model

@after_model

### 方式 2：类继承（适合复杂逻辑）

**class** LoggingMiddleware(AgentMiddleware):
  """自定义日志中间件"""

  @property
  **def** name(self) -> str:
    # 自定义中间件名称（默认是类名）
    **return** "logging"

  **def** before_agent(self, state, runtime):
    """Agent 开始前的逻辑"""
    **print**("[Logging] Agent 开始执行")
    **return** None

  **def** before_model(self, state, runtime):
    """模型调用前的逻辑"""
    msg_count = len(state.get("messages", []))
    **print**(f"[Logging] 准备调用模型，当前 {msg_count} 条消息")
    **return** None

  **def** after_model(self, state, runtime):
    """模型调用后的逻辑"""
    **print**("[Logging] 模型调用完成")
    **return** None

  **def** after_agent(self, state, runtime):
    """Agent 结束后的逻辑"""
    **print**("[Logging] Agent 执行结束")
    **return** None

- **before_agent 和 after_agent**：每个问题只执行一次
- **before_model 和 after_model**：每次模型调用都执行（第一个问题调用了两次模型，所以各执行两次）

### Middleware 的返回值

Middleware 的返回值决定了是否要修改 Agent 状态或控制流程：

| 返回值             | 效果                              | 示例                           |
| :----------------- | :-------------------------------- | :----------------------------- |
| None               | 不修改任何状态，继续正常流程      | 纯日志记录                     |
| dict               | 更新 Agent 状态（合并到当前状态） | 返回 {"custom_field": "value"} |
| 含 jump_to 的 dict | 跳转到指定节点                    | 返回 {"jump_to": "end"}        |

> 返回的 dict 会通过 Agent 状态的 reducer 合并。对于 messages 字段，使用 add_messages reducer，所以返回的 messages 会追加而非覆盖。

## LangChain 中间件钩子 -- @before_model 与 @after_model

before_model 和 after_model 是最常用的两个 中间件（Middleware） 钩子。它们在每次模型调用前后执行，适合做内容过滤、消息预处理、响应审核等。

------

### @before_model——模型调用前拦截

@before_model 在每次调用模型之前执行。你可以在这里修改消息、注入上下文条件、或直接跳过模型调用。

### 场景 1：消息预处理——限制对话长度

### 场景 2：内容过滤——屏蔽敏感词

### @after_model——模型调用后处理

@after_model 在模型回复后执行，适合审核模型输出、提取关键信息、追加后续指令等。

### 场景 3：响应内容审核

### 场景 4：自动追加提示信息

### can_jump_to——流程跳转控制

在 before_model 和 after_model 中，你可以通过 can_jump_to 参数和 jump_to 状态来控制 Agent 的流程：

@before_model(can_jump_to=["end"]) # 声明可以跳转到的目标
**def** conditional_exit(state, runtime):
  """在特定条件下直接结束 Agent"""
  messages = state.get("messages", [])
  **if** **not** messages:
    **return** None

  # 如果用户说"再见"，直接结束
  last_content = str(messages[-1].content)
  **if** last_content.strip() **in** ["再见", "拜拜", "bye"]:
    **return** {
      "jump_to": "end", # 直接结束 Agent
      "messages": [{"role": "assistant", "content": "再见！期待下次为您服务。"}]
    }

  **return** None

| can_jump_to 值   | 含义               | 适用场景             |
| :--------------- | :----------------- | :------------------- |
| ["end"]          | 可跳转到结束       | 条件退出、安全拦截   |
| ["model"]        | 可跳转回模型       | 需要让模型重新处理   |
| ["tools"]        | 可跳转到工具节点   | 跳过模型直接执行工具 |
| ["model", "end"] | 可跳转到模型或结束 | 多种条件分支         |

> 如果不在 can_jump_to 中声明目标，jump_to 会被忽略。这是一种安全机制，防止中间件意外跳转到不合法的节点。

## LangChain 模型调用拦截 -- @wrap_model_call

@wrap_model_call 是中间件（Middleware）中最强大的钩子。

@wrap_model_call 不像 before/after 那样只是观察，而是可以**完全控制模型的执行过程**——重试、降级、缓存、甚至跳过模型直接用预设回复。

------

### handler 回调

@wrap_model_call 的核心是一个 **handler 回调函数**。调用 handler(request) 才会真正执行模型；不调用则跳过模型

@wrap_model_call
**def** my_middleware(request, handler):
  # 在模型调用前可以做任何事
  **print**("模型即将被调用...")

  # 调用 handler(request) 才真正执行模型
  response = handler(request)

  # 在模型调用后可以做任何事
  **print**("模型调用完成")

  **return** response

### 场景 1：重试机制

这是最常见的场景——模型调用可能因网络问题失败，自动重试可以提高可靠性：

### 场景 2：模型降级/故障转移

当主模型不可用时，自动切换到备用模型：

*request.override() 是一个不可变方法——它返回一个新的 request 副本，不会修改原始对象。这确保了每次调用都是独立和安全的。*

### 场景 3：缓存模型响应

对于重复的查询，可以缓存模型响应以减少 API 调用成本

### 场景 4：修改 request——动态注入系统消息

    # 在原有 system_message 基础上追加时间信息
  **if** request.system_message:
    new_content = f"{request.system_message.content}**\n****\n**{time_context}"
  **else**:
    new_content = time_context

  # 用 override 创建新的 request
  new_request = request.override(
    system_message=SystemMessage(content=new_content)
  )

  **return** handler(new_request)

### 场景 5：多个 wrap_model_call 的组合

多个 wrap_model_call 中间件会自动按顺序组合——第一个在最外层

*多个 wrap_model_call 就像洋葱一样层层包裹。最外层最先执行、最后返回。这让你可以组合多个独立的功能——比如外层做缓存检查，内层做重试，互不干扰。*

```python
@wrap_model_call
def outer_middleware(request, handler):
    """最外层中间件"""
    print("[外层] 开始")
    result = handler(request)       # 这里会进入 inner_middleware
    print("[外层] 结束")
    return result


@wrap_model_call
def inner_middleware(request, handler):
    """内层中间件"""
    print("  [内层] 开始")
    result = handler(request)       # 这里才真正调用模型
    print("  [内层] 结束")
    return result


# 执行顺序：
# [外层] 开始
#   [内层] 开始
#     → 真正调用模型
#   [内层] 结束
# [外层] 结束
```

## LangChain 工具调用拦截 -- @wrap_tool_call

@wrap_tool_call 让你在工具执行层面实现与 @wrap_model_call 类似的控制能力——重试、缓存、参数改写、结果后处理。

------

### 基本结构

@wrap_tool_call 的结构与 @wrap_model_call 类似，接收 request 和 handler 两个参数

```python
from langchain.agents.middleware import wrap_tool_call

@wrap_tool_call
def my_tool_wrapper(request, handler):
    # request.tool_call: 包含工具名称和参数
    # request.tool: 工具对象本身
    # request.state: Agent 当前状态
    # request.runtime: 运行时上下文

# 调用 handler(request) 才会真正执行工具
result = handler(request)

# result 是 ToolMessage 或 Command
return result
```

### 场景 1：工具调用重试

工具执行可能因外部服务不稳定而失败，自动重试可以提升可靠性：

### 场景 2：修改工具参数

在工具执行之前动态修改参数，可以在不修改工具代码的情况下实现参数转换

"""自动规范化城市名称（全角转半角、去除多余空格等）"""
  tool_call = request.tool_call

### 场景 3：工具结果缓存

对于重复的工具调用（相同工具 + 相同参数），可以缓存结果

"""缓存工具执行结果"""
  # 生成缓存键：工具名 + 参数
  tool_name = request.tool_call.get("name", "unknown")
  tool_args = str(request.tool_call.get("args", {}))

### 场景 4：工具调用日志与监控

记录所有工具调用的详细信息

"""监控工具调用的性能指标"""
  tool_name = request.tool_call.get("name", "unknown")
  tool_args = request.tool_call.get("args", {})

### 场景 5：根据结果决定后续流程

你可以根据工具执行结果决定是否继续 Agent 循环

"""如果工具返回空结果，直接结束 Agent，不浪费模型调用"""
  result = handler(request)

*当 wrap_tool_call 返回 Command 时，可以通过 update 参数修改 Agent 状态。使用 Command 可以直接向消息列表追加 AI 消息，然后 Agent 循环会自然结束。*

我们什么时候 会返回一个 Command 

### @wrap_model_call vs @wrap_tool_call

| 维度         | @wrap_model_call                      | @wrap_tool_call                    |
| :----------- | :------------------------------------ | :--------------------------------- |
| 拦截目标     | 模型调用                              | 工具执行                           |
| request 内容 | model、messages、tools、system_prompt | tool_call、tool、state、runtime    |
| 返回类型     | ModelResponse 或 AIMessage            | ToolMessage 或 Command             |
| 适用场景     | 模型重试、降级、缓存、prompt 修改     | 工具重试、缓存、参数改写、结果处理 |

## LangChain @before_agent 与 @after_agent

before_agent 和 after_agent 是 Agent 级别的钩子，分别在 Agent 执行之前和完成之后各执行一次。适合做初始化、预处理、后处理和统计分析。

------

### before_agent -- Agent 开始前的准备工作

before_agent 在 Agent 正式开始执行前运行，只执行一次。你可以在这里做输入预处理、用户信息验证、资源初始化等。

### 场景 1：输入预处理 -- 自动修正用户输入

"""在 Agent 开始前处理用户输入"""
  messages = state.get("messages", [])

### 场景 2：访问控制 -- 权限检查

"""检查用户是否有权限使用 Agent"""
  # 从 runtime.context 获取用户信息
  context = runtime.context

### after_agent -- Agent 完成后的处理

after_agent 在 Agent 完成所有处理后执行（只执行一次）。你可以在这里格式化最终输出、记录统计信息、清理资源等。

### 场景 3：统计分析 -- 记录对话数据

"""统计对话信息并追加到结果中"""
  messages = state.get("messages", [])

  # 统计数据
  model_calls = 0
  tool_calls = 0
  total_chars = 0

    # 通过 custom stream 发送统计信息
  runtime.stream_writer({
    "type": "stats",
    "model_calls": model_calls,
    "tool_calls": tool_calls,
    "total_messages": len(messages),
    "total_chars": total_chars,
  })

### 场景 4：格式化输出 -- 统一回复风格

"""在结果中追加格式化的总结信息"""
  messages = state.get("messages", [])

### 四个钩子的完整协作示例

### 实例

**from** langchain.agents.middleware **import** (
  before_agent, after_agent, before_model, after_model
)
**from** langchain.agents **import** create_agent
**from** langchain.chat_models **import** init_chat_model
**from** langchain.messages **import** HumanMessage
**from** langchain.tools **import** tool

    # ----- 定义所有钩子 -----

@before_agent
**def** init_session(state, runtime):
  """开始：初始化会话"""
  **print**(">>> 会话开始")
  **return** None


@before_model
**def** pre_model_check(state, runtime):
  """每次模型调用前"""
  msg_count = len(state.get("messages", []))
  **print**(f"  [model前] 消息数: {msg_count}")
  **return** None


@after_model
**def** post_model_check(state, runtime):
  """每次模型调用后"""
  last = state["messages"][-1] **if** state.get("messages") **else** None
  **if** last **and** hasattr(last, 'tool_calls') **and** last.tool_calls:
    **print**(f"  [model后] 需要工具调用")
  **return** None


@after_agent
**def** finish_session(state, runtime):
  """结束：清理资源"""
  total = len(state.get("messages", []))
  **print**(f"<<< 会话结束，共 {total} 条消息")
  **return** None


    # ----- 创建 Agent -----

@tool
**def** get_weather(city: str) -> str:
  """查询天气"""
  **return** f"{city}: 晴"


model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
agent = create_agent(
  model=model,
  tools=[get_weather],
  middleware=[init_session, pre_model_check, post_model_check, finish_session],
  system_prompt="你是助手。",
)

result = agent.invoke({
  "messages": [HumanMessage(content="杭州天气？")]
})
**print**(f"**\n**最终回复: {result['messages'][-1].content}")

运行结果：

```
>>> 会话开始
  [model前] 消息数: 2
  [model后] 需要工具调用
  [model前] 消息数: 3
<<< 会话结束，共 4 条消息

最终回复: 杭州今天晴天，适合出行。
```

------

### Middleware 钩子总结

| 钩子            | 执行次数     | 何时使用                         | 关键能力                  |
| :-------------- | :----------- | :------------------------------- | :------------------------ |
| before_agent    | 1 次         | 权限检查、输入预处理、资源初始化 | 可 jump_to="end" 提前终止 |
| before_model    | 每次循环     | 消息裁剪、内容过滤、上下文注入   | 可 jump_to 控制流程       |
| wrap_model_call | 每次循环     | 重试、降级、缓存、prompt 修改    | 完全控制模型执行          |
| after_model     | 每次循环     | 响应审核、内容追加、日志         | 可替换模型输出            |
| wrap_tool_call  | 每次工具调用 | 工具重试、缓存、参数改写         | 完全控制工具执行          |
| after_agent     | 1 次         | 输出格式化、统计分析、清理       | 最终状态修改              |

## LangChain 对话记忆 -- Checkpointer

默认情况下，每次 agent.invoke() 都是独立的，Agent 不记得之前聊过什么。

Checkpointer（检查点保存器）让 Agent 能够记住对话历史，实现真正的多轮对话。

------

### 使用 Checkpointer 记住对话

添加 Checkpointer 后，同一 thread_id 下的对话会自动关联：

    # 创建一个内存 Checkpointer
checkpointer = InMemorySaver()

model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
agent = create_agent(
  model=model,
  checkpointer=checkpointer, # 传入 Checkpointer
  system_prompt="你是菜鸟教程 RUNOOB 的助手。",
)

    # 使用 thread_id 来标识对话线程
config = {"configurable": {"thread_id": "user-001"}}

    # 第一轮
result1 = agent.invoke(
  {"messages": [HumanMessage(content="我叫小明，我在学 Python")]},
  config=config,
)
**print**(f"第一轮: {result1['messages'][-1].content}")

    # 第二轮——使用相同的 thread_id，Agent 记住了！
result2 = agent.invoke(
  {"messages": [HumanMessage(content="我叫什么名字？我在学什么？")]},
  config=config,
)
**print**(f"第二轮: {result2['messages'][-1].content}")

*thread_id 是关键。同一个 thread_id 下的对话是连续的，不同 thread_id 之间的对话完全隔离。这让你可以用一个 Agent 实例同时服务多个用户。*

### Checkpointer 的工作原理

Checkpointer 在每次 Agent 执行后自动保存状态快照（checkpoint）。下一次使用相同 thread_id 调用时，自动从最近的 checkpoint 恢复状态。

具体工作流程：

1. 调用 agent.invoke()，传入 config（含 thread_id）
2. Agent 检查是否有该 thread_id 的 checkpoint
3. 如果有，加载历史消息，追加新消息后继续
4. 执行完成后，自动保存新的 checkpoint

### Checkpointer 类型

| 类型          | 存储位置      | 持久化               | 安装                          | 适用场景             |
| :------------ | :------------ | :------------------- | :---------------------------- | :------------------- |
| InMemorySaver | 内存          | 否（程序退出后丢失） | 内置                          | 开发调试、单元测试   |
| SqliteSaver   | SQLite 数据库 | 是                   | langgraph-checkpoint-sqlite   | 单机部署、小型应用   |
| PostgresSaver | PostgreSQL    | 是                   | langgraph-checkpoint-postgres | 生产环境、多实例共享 |

如果 Agent 使用 `ainvoke()` 异步调用，需要改用对应的异步版本 `AsyncSqliteSaver` / `AsyncPostgresSaver`，用法与同步版本类似，只是需要配合 `async with` 使用。

### SqliteSaver 示例

`InMemorySaver` 只保存在内存中，程序退出后数据会丢失。如果希望对话能够持久化保存，可以使用 `SqliteSaver`。

首次使用需要安装 SQLite Checkpointer：

```
pip install langgraph-checkpoint-sqlite
```

### 实例

**from** langgraph.checkpoint.sqlite **import** SqliteSaver
**from** langchain.agents **import** create_agent
**from** langchain.chat_models **import** init_chat_model
**from** langchain.messages **import** HumanMessage

model = init_chat_model(
  "deepseek:deepseek-v4-flash",
  temperature=0,
)

    # 必须用 with 语句进入，退出时会自动关闭数据库连接
**with** SqliteSaver.from_conn_string("conversations.db") **as** checkpointer:
  agent = create_agent(
    model=model,
    checkpointer=checkpointer,
    system_prompt="你是菜鸟教程 RUNOOB 的助手。",
  )

  config = {"configurable": {"thread_id": "user-001"}}

  result = agent.invoke(
    {
      "messages": [
        HumanMessage(content="你好")
      ]
    },
    config=config,
  )

  **print**(result["messages"][-1].content)

> SqliteSaver 会将每次对话产生的 checkpoint 保存在 SQLite 数据库中。即使程序退出或重新启动，只要继续使用相同的数据库文件和 `thread_id`，Agent 就能够恢复之前保存的对话状态。注意：`with` 代码块结束后数据库连接会关闭，因此涉及多次调用 Agent 的逻辑都应该放在 `with` 块内部；如果是长期运行的服务（如 Web 应用），建议在应用生命周期内持有该连接，或改用异步版本的 `AsyncSqliteSaver`。

------

### 管理对话线程

### 查看对话状态

### 实例

    # 查看对话状态
state = agent.get_state(config)
**print**(f"下一步: {state.next}") # () 表示空闲
**print**(f"消息数: {len(state.values.get('messages', []))}")

    # 查看对话历史
**for** msg **in** state.values.get("messages", []):
  **print**(f"  [{msg.type}] {str(msg.content)[:60]}")

### 创建新线程

### 实例

    # 不同的 thread_id = 不同的独立对话
config_alice = {"configurable": {"thread_id": "alice"}}
config_bob = {"configurable": {"thread_id": "bob"}}

    # Alice 的对话
agent.invoke(
  {"messages": [HumanMessage(content="我是 Alice")]},
  config=config_alice,
)

    # Bob 的对话——完全独立，不知道 Alice 说了什么
agent.invoke(
  {"messages": [HumanMessage(content="我是 Bob")]},
  config=config_bob,
)

    # 验证隔离性
alice_state = agent.get_state(config_alice)
bob_state = agent.get_state(config_bob)
**print**(f"Alice 对话消息数: {len(alice_state.values['messages'])}")
**print**(f"Bob 对话消息数: {len(bob_state.values['messages'])}")

------

### 更新状态——手动修改对话

有时你需要手动修改对话状态，比如清空对话、插入系统消息等：

### 实例

**from** langchain.messages **import** SystemMessage

    # 更新状态——插入一条系统消息
agent.update_state(
  config,
  {
    "messages": [
      SystemMessage(content="（用户升级到了 VIP 会员）")
    ]
  }
)

    # 之后的对话会包含这条插入的消息

> update_state() 的参数会通过 add_messages reducer 处理（对 messages 字段而言），所以新消息会追加而不是覆盖。如果想清空历史重新开始，最简单的方式是换一个新的 `thread_id`；如果确实需要删除某几条历史消息，可以在 update_state() 中传入对应的 `RemoveMessage`（来自 `langchain.messages`）来精确移除指定的消息。

## LangChain 跨会话存储 —— Store

Checkpointer 解决了"单个对话内记忆"的问题。但如果你需要在不同对话之间共享数据——比如用户偏好、学习进度——就需要用到 **Store**。

------

### Checkpointer vs Store

| 维度     | Checkpointer               | Store                    |
| :------- | :------------------------- | :----------------------- |
| 作用域   | 单个对话线程（thread_id）  | 跨所有对话线程           |
| 数据类型 | Agent 状态快照（自动管理） | 任意键值数据（手动管理） |
| 典型用途 | 多轮对话记忆               | 用户偏好、知识库、配置   |
| 数据组织 | thread_id → checkpoint     | (namespace, key) → value |

------

### Store 的基本操作

Store 使用 **命名空间 + 键** 的层级结构来组织数据：
store = InMemoryStore()

    # 写入数据：put(namespace, key, value)
    # namespace 是元组，key 是字符串，value 是字典
store.put(
  ("users", "user_001"),      # 命名空间
  "preferences",           # 键
  {                 # 值
    "theme": "dark",
    "language": "zh-CN",
    "level": "入门",
  }
)

    # 读取数据：get(namespace, key)
prefs = store.get(("users", "user_001"), "preferences")

    # 搜索数据：search(namespace)
all_user_data = store.search(("users", "user_001"))

    # 删除数据：delete(namespace, key)
store.delete(("users", "user_001"), "preferences")

### 在 Agent 中使用 Store

将 Store 传给 create_agent()，Agent 中的所有工具都能通过 InjectedStore 访问它

@tool
**def** query_course_info(
  course_name: str,
  store: Annotated[BaseStore, InjectedStore()],
) -> str:

agent = create_agent(
  model=model,
  tools=[query_course_info, get_user_membership],
  store=store,
  system_prompt="你是菜鸟教程 RUNOOB 的课程顾问。",
)

### Store 的持久化

InMemoryStore 的数据在程序重启后丢失。生产环境可以使用 PostgresStore 等持久化方案：

### 实例

    # 开发阶段
**from** langgraph.store.memory **import** InMemoryStore
store = InMemoryStore()

    # 生产环境（需要 PostgreSQL）
    # from langgraph.store.postgres import PostgresStore
    # store = PostgresStore.from_conn_string("postgresql://...")

### Store 使用建议

| 场景     | namespace 示例          | key 示例    | 说明                                         |
| :------- | :---------------------- | :---------- | :------------------------------------------- |
| 用户偏好 | ("users", user_id)      | preferences | 主题、语言、通知设置                         |
| 学习进度 | ("users", user_id)      | progress    | 已完成课程、学习时长                         |
| 知识库   | ("kb", collection)      | doc_id      | 文档、FAQ、产品信息                          |
| 会话摘要 | ("sessions", thread_id) | summary     | 长对话的摘要，供 Checkpointer 之外的场景使用 |

> Checkpointer 负责"对话到哪了"，Store 负责"用户是谁、会什么、喜欢什么"。两者配合使用，才能构建出有持续记忆的智能 Agent。

## LangChain 人工介入

在生产环境中，有些操作需要人工确认——比如发送邮件、执行删除、处理支付。

人工介入（HITL，Human-in-the-Loop）让 Agent 在关键时刻暂停，等待人工审批后继续。

------

### interrupt()——在工具中暂停执行

**interrupt()** 函数可以让工具执行到一半时暂停，等待外部输入后再继续：

    # 在工具中使用 interrupt() 暂停
**def** send_email(to: str, subject: str, body: str) -> str:
  """发送邮件（需要人工审批）"""
  # 暂停执行，向外部发送审批请求
  approval = interrupt({
    "action": "send_email",
    "to": to,
    "subject": subject,
    "body": body,
    "message": "请确认是否发送此邮件？"
  })

  # 等待外部传入 approval 后继续
  **if** approval.get("approved"):
    **return** f"邮件已发送给 {to}"
  **else**:
    **return** f"邮件发送已被拒绝：{approval.get('reason', '用户取消')}"

interrupt() 的工作流程：

1. 工具调用 interrupt() → Agent 暂停执行
2. 外部系统获取中断信息，展示给用户
3. 用户做出决定后，通过 Command(resume=...) 恢复执行
4. interrupt() 返回用户传入的值，工具继续执行

interrupt({
    "action": "send_email",
    "to": to,
    "subject": subject,
    "body": body,
    "message": "请确认是否发送此邮件？"
  })



### interrupt_before / interrupt_after 参数

除了在工具中使用 interrupt()，你还可以在 create_agent() 中设置全局中断点

 # 在工具节点之前暂停（每次调用工具前都需要审批）
  interrupt_before=["tools"],

  # 在模型节点之后暂停（每次模型回复后都可以检查）
  # interrupt_after=["model"],



| 参数                       | 暂停时机       | 适用场景                     |
| :------------------------- | :------------- | :--------------------------- |
| interrupt_before=["tools"] | 每次执行工具前 | 所有工具调用都需要审批       |
| interrupt_before=["model"] | 每次模型调用前 | 在模型处理前人工审查消息     |
| interrupt_after=["model"]  | 每次模型回复后 | 审查模型输出后再决定是否继续 |
| interrupt_after=["tools"]  | 每次工具执行后 | 检查工具结果后再决定下一步   |

*HITL 需要 Checkpointer 配合使用。因为 Agent 在 interrupt() 处暂停时，其状态必须被持久化，才能在恢复时正确继续执行。*

## LangChain 多 Agent

当一个任务太复杂，单个 Agent 难以胜任时，你可以创建多个各司其职的 Agent，让它们像团队一样协作。

------

### 为什么需要多 Agent

单个 Agent 的问题：

- system_prompt 太长会导致模型注意力分散
- 工具太多会增加模型选择工具的出错概率
- 不同类型的任务需要不同的专业知识和行为风格

多 Agent 的方案：每个 Agent 专注于一个领域，通过协作完成复杂任务。

------

### 方式 1：子 Agent 作为工具

将 Agent 编译成 CompiledStateGraph，然后作为一个工具注册给父 Agent：

```python
from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)

# 子 Agent 1：天气专家
@tool
def get_weather(city: str) -> str:
    """查询天气"""
    data = {"杭州": "晴，25°C", "北京": "多云，18°C"}
    return data.get(city, f"{city}: 数据暂缺")

weather_agent = create_agent(
    model=model,
    tools=[get_weather],
    name="weather_expert",  # 名字用于标识和日志
    system_prompt="你是天气专家，专门回答天气相关问题。回答要简洁。",
)

# 子 Agent 2：计算专家
@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    result = eval(expression, {"__builtins__": {}}, {})
    return f"{expression} = {result}"

math_agent = create_agent(
    model=model,
    tools=[calculate],
    name="math_expert",
    system_prompt="你是数学专家，专门进行数学计算。回答要简洁。",
)

# 父 Agent：协调者
# 将子 Agent 作为工具注册
@tool
def ask_weather_expert(question: str) -> str:
    """向天气专家咨询天气相关问题。

    Args:
        question: 关于天气的问题
    """
    result = weather_agent.invoke(
        {"messages": [HumanMessage(content=question)]}
    )
    return result["messages"][-1].content


@tool
def ask_math_expert(question: str) -> str:
    """向数学专家咨询数学计算问题。

    Args:
        question: 数学计算问题
    """
    result = math_agent.invoke(
        {"messages": [HumanMessage(content=question)]}
    )
    return result["messages"][-1].content


coordinator = create_agent(
    model=model,
    tools=[ask_weather_expert, ask_math_expert],
    system_prompt="""你是协调助手。根据用户问题选择合适的专家：
- 天气相关问题 → 使用 ask_weather_expert
- 数学计算问题 → 使用 ask_math_expert
- 如果同时涉及多个领域，依次咨询各个专家""",
)

# 测试复合问题
result = coordinator.invoke({
    "messages": [HumanMessage(
        content="杭州今天天气怎么样？如果温度是 25 度，换算成华氏度是多少？"
        "（公式：华氏度 = 摄氏度 × 9/5 + 32）"
    )]
})
print(result["messages"][-1].content)
```

### 方式 2：用 name 参数区分 Agent

当你将子 Agent 作为工具嵌入时，设置 name 参数有助于追踪消息来源

```python
# name 参数的作用：
# 1. 编译后的图中使用该名称
# 2. 作为子图节点嵌入父图时使用该名称
# 3. 所有 AI 消息被标记为该名称

agent = create_agent(
    model=model,
    tools=[...],
    name="customer_service",  # 给 Agent 命名
)
```

### 方式 3：Middleware 实现 Agent 路由

更复杂的多 Agent 场景可以通过 Middleware 实现动态路由：

```python
from langchain.agents.middleware import before_model


# 定义不同专家使用的工具集
general_tools = [tool_a, tool_b]
admin_tools = [tool_c, tool_d]


@before_model
def route_by_user_role(state, runtime):
    """根据用户角色动态切换可用工具"""
    context = runtime.context
    if context is None:
        return None

    user_role = context.get("user_role", "user")

    # 不同角色看到不同的工具
    if user_role == "admin":
        available_tools = general_tools + admin_tools
    else:
        available_tools = general_tools

    # 注意：before_model 不能直接修改 tools，
    # 需要配合 wrap_model_call 或 request.override 来实现
    return None
```

### 多 Agent 架构模式

| 模式       | 结构                             | 适用场景                       |
| :--------- | :------------------------------- | :----------------------------- |
| 协调者模式 | 一个父 Agent → 多个子 Agent 工具 | 任务类型明确可分类             |
| 接力模式   | Agent A 的输出 → Agent B 的输入  | 流水线式处理（生成→审核→润色） |
| 辩论模式   | 多个 Agent 并行输出 → 汇总决策   | 需要多角度分析的问题           |

> 多 Agent 系统增加了复杂度和 Token 消耗。不要为了"多 Agent"而多 Agent——先用单个 Agent + 良好设计的 system_prompt 和 Middleware 解决问题。只有当确实需要领域隔离或独立上下文时，才引入多 Agent。

## LangChain RAG

RAG（Retrieval-Augmented Generation，检索增强生成）让 AI 能够基于你的私有文档回答问题，不需要微调模型，只需将文档向量化存储，Agent 就能检索相关内容来回答。

------

### RAG 是什么

普通的大模型只能回答训练数据中有的内容。如果你的文档是私有的（公司内部文档、个人笔记），模型就"不知道"。RAG 解决了这个问题：

- **离线阶段**：将文档切分成小块 → 用 Embedding 模型转换为向量 → 存入向量数据库
- **在线阶段**：用户提问 → 将问题转为向量 → 在向量数据库中搜索最相似的内容 → 将检索到的内容作为上下文发给模型 → 模型基于检索内容回答

### Embedding 模型初始化

### 实例

**from** dotenv **import** load_dotenv
load_dotenv()

**from** langchain_openai **import** OpenAIEmbeddings

    # OpenAI 的文本嵌入模型
    # 将文本转换为向量（一组浮点数）
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 测试：将一段文本转为向量
text = "菜鸟教程 RUNOOB 是一个编程学习平台"
vector = embeddings.embed_query(text)

**print**(f"文本: {text}")
**print**(f"向量维度: {len(vector)}")  # text-embedding-3-small 是 1536 维
**print**(f"向量前 5 个值: {vector[:5]}")

另外如果没有 OpenAI 的 key，可以使用阿里百炼的 Embedding 服务，**.env** 的配置文件需要加上阿里百炼的 key：

```
DASHSCOPE_API_KEY="sk-xxx"
```

配置后的代码：

embeddings = OpenAIEmbeddings(
  model="text-embedding-v4",
  api_key=os.getenv("DASHSCOPE_API_KEY"),
  base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
  check_embedding_ctx_length=False,
  chunk_size=10,
)

### 创建向量存储

    # 初始化 Embedding 模型
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 创建 Chroma 向量存储（数据保存在本地目录）
vector_store = Chroma(
  collection_name="runoob_docs",
  embedding_function=embeddings,
  persist_directory="./chroma_db", # 持久化目录
)

    # 添加文档（最简单的形式：文本列表）
texts = [
  "菜鸟教程（RUNOOB）是一个免费的编程学习网站，提供 HTML、CSS、JavaScript、Python 等教程。",
  "Python3 基础教程共 30 章，适合零基础入门，包含环境搭建、语法基础、面向对象等内容。",
  "HTML 基础教程共 25 章，覆盖 HTML 标签、表单、多媒体等基础知识。",
]

    # add_texts 自动将文本转为向量并存储
vector_store.add_texts(texts)

### 语义检索

### 实例

    # 语义搜索——不依赖关键词匹配，而是语义相似度
results = vector_store.similarity_search(
  "我想学 Python，有什么教程推荐？",
  k=2, # 返回最相似的 2 个结果
)

> 注意第一个搜索结果比第二个更相关——虽然第一个包含 "Python" 关键词，但它按 **语义相似度** 而非关键词匹配排序。这就是向量检索的优势。

------

### 创建 Retriever 检索器

Retriever 是 Vector Store 的标准化接口：

    # 从 vector_store 创建 retriever
retriever = vector_store.as_retriever(
  search_type="similarity", # 相似度搜索
  search_kwargs={"k": 3},   # 返回前 3 个结果
)

    # 使用 retriever
docs = retriever.invoke("Python 学习路线")

## LangChain 文档加载与切分

之前的文章我们手动输入文本，但在实际项目中，文档可能来自 PDF、网页、Markdown 文件等。

本节介绍如何使用 Document Loader 加载各类文档，以及如何用 Text Splitter 将文档切分成适合检索的小块。

------

### Document Loader——加载文档

LangChain 提供了数十种文档加载器，覆盖常见文件格式：

| Loader                     | 来源          | 安装包                               |
| :------------------------- | :------------ | :----------------------------------- |
| TextLoader                 | .txt 文件     | langchain（内置）                    |
| PyPDFLoader                | PDF 文件      | langchain-community + pypdf          |
| WebBaseLoader              | 网页 URL      | langchain-community + beautifulsoup4 |
| CSVLoader                  | CSV 文件      | langchain-community                  |
| UnstructuredMarkdownLoader | Markdown 文件 | langchain-community + unstructured   |

    # 加载文本文件（内置，无需额外安装）
**from** langchain_community.document_loaders **import** TextLoader

loader = TextLoader("knowledge.txt", encoding="utf-8")
docs = loader.load()

    # 加载网页
    # pip install langchain-community beautifulsoup4
**from** langchain_community.document_loaders **import** WebBaseLoader

loader = WebBaseLoader("https://www.runoob.com/python/python-tutorial.html")
docs = loader.load()

### Text Splitter——文档切分

文档通常太长，需要切分成小块（chunk）才能有效检索。切分策略直接影响 RAG 效果：

### 实例

**from** langchain_text_splitters **import** RecursiveCharacterTextSplitter

    # 创建切分器
text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=500,     # 每块最多 500 个字符
  chunk_overlap=50,    # 块之间重叠 50 个字符
  separators=["**\n****\n**", "**\n**", "。", "！", "？", "；", "，", " ", ""],
  # 优先按段落分割，然后是句子，最后是字符
)

    # 切分文档
chunks = text_splitter.split_text(long_text)

> chunk_overlap 很重要。如果块之间没有重叠，一个完整的句子可能被切成两半，导致检索时遗漏关键信息。50-100 字符的重叠是常见的设置。

------

### 切分参数设置指南

| 场景        | chunk_size | chunk_overlap | 原因                   |
| :---------- | :--------- | :------------ | :--------------------- |
| FAQ 问答    | 200~500    | 20~50         | 问答对较短，小块即可   |
| 技术文档    | 500~1000   | 50~100        | 技术内容需要更多上下文 |
| 长文章/论文 | 1000~2000  | 100~200       | 需要保留段落完整性     |
| 代码库      | 500~1500   | 0~50          | 函数/类作为自然边界    |

------

### 完整流程：加载 → 切分 → 向量化

    # 流程 1：加载
    # loader = TextLoader("runoob_knowledge.txt", encoding="utf-8")
    # docs = loader.load()

    # 为演示直接使用示例文本
docs = [
  "菜鸟教程（RUNOOB）是一个免费的编程学习网站。",
  "网站提供 Python、Java、HTML 等多种编程语言的教程。",
  "Python3 基础教程共 30 章，适合零基础入门学习。",
  "HTML 基础教程共 25 章，包含表单、多媒体等内容。",
  "菜鸟教程的所有基础教程都是免费的。",
]

    # 流程 2：切分
text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=100,
  chunk_overlap=20,
)
chunks = text_splitter.create_documents(docs)

    # 流程 3：向量化存储
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma.from_documents(
  documents=chunks,
  embedding=embeddings,
  persist_directory="./runoob_db",
)

**print**(f"已建立索引：{len(chunks)} 个文档块")

    # 流程 4：检索
results = vector_store.similarity_search("Python 教程有多少章？", k=2)
**for** doc **in** results:
  **print**(f"检索结果: {doc.page_content}")

## LangChain 构建 RAG Agent

前两篇我们准备了向量存储和检索器。

本篇将它们集成到 Agent 中，构建一个完整的 RAG Agent——能够基于私有知识库回答问题的智能助手。

------

### 创建 Retriever 工具

将检索器包装成一个工具，Agent 就能在需要时自动搜索知识库。

如果没有 OpenAI 的 key 可以采用阿里百炼的 Embedding 服务，参考配置 https://www.runoob.com/langchain/langchain-alibailian.html：

```
# OpenAI 的文本嵌入模型
# 将文本转换为向量（一组浮点数）

# 使用阿里云百炼（DashScope）的通义千问 Embedding 服务
embeddings = OpenAIEmbeddings(
    model="text-embedding-v4",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    check_embedding_ctx_length=False,
    chunk_size=10,
)
```

### 实例（OpenAI）

**import** os
**from** dotenv **import** load_dotenv
load_dotenv()

**from** langchain.tools **import** tool
**from** langchain.agents **import** create_agent
**from** langchain.chat_models **import** init_chat_model
**from** langchain.messages **import** HumanMessage
**from** langchain_openai **import** OpenAIEmbeddings
**from** langchain_chroma **import** Chroma
**from** langchain_text_splitters **import** RecursiveCharacterTextSplitter

    # ----- 步骤 1：准备知识库 -----

    # 模拟菜鸟教程 RUNOOB 的知识文档
knowledge_docs = [
  "菜鸟教程（RUNOOB）创立于 2013 年，是一个完全免费的编程学习平台。",
  "平台已上线 300+ 套教程，涵盖前端、后端、数据库、移动开发等领域。",
  "Python3 基础教程是平台最受欢迎的课程，累计学习人次超过 500 万。",
  "Python3 基础教程共 30 章，包含环境搭建、基本语法、函数、类、异常处理等内容。",
  "HTML 基础教程共 25 章，从 HTML 基本结构讲到表单与多媒体元素。",
  "菜鸟教程支持在线运行代码，学习者无需安装任何软件即可编写和运行代码。",
  "平台提供移动端适配，用户可以在手机上随时随地学习编程。",
  "菜鸟教程的会员服务提供视频课程、项目实战、一对一答疑等增值服务。",
]

    # 切分
text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=200, chunk_overlap=30
)
chunks = text_splitter.create_documents(knowledge_docs)

    # 向量化存储 -- 这里可以修改为阿里百炼的，如果没有 OpenAI 的 key
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


vector_store = Chroma.from_documents(
  documents=chunks,
  embedding=embeddings,
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # ----- 步骤 2：创建检索工具 -----

@tool
**def** search_knowledge_base(query: str) -> str:
  """在菜鸟教程 RUNOOB 知识库中搜索相关信息。

  当用户询问关于菜鸟教程的具体信息时（如课程数量、平台历史、功能特性等），
  必须使用此工具查询知识库获取准确信息。

  Args:
    query: 搜索关键词或问题
  """
  docs = retriever.invoke(query)
  **if** **not** docs:
    **return** "知识库中未找到相关信息。"

  results = []
  **for** i, doc **in** enumerate(docs, 1):
    results.append(f"[{i}] {doc.page_content}")

  **return** "**\n****\n**".join(results)


    # ----- 步骤 3：创建 RAG Agent -----

model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
agent = create_agent(
  model=model,
  tools=[search_knowledge_base],
  system_prompt="""你是菜鸟教程 RUNOOB 的智能客服助手。

## 规则

1. 当用户询问关于菜鸟教程的具体信息时，必须使用 search_knowledge_base 工具查询
2. 基于检索到的信息回答，不要编造知识库中没有的内容
3. 如果知识库中没有相关信息，诚实地告诉用户
4. 回答要友好、简洁、准确""",

)

    # ----- 步骤 4：测试 -----

questions = [
  "菜鸟教程是什么时候创立的？",
  "Python3 基础教程有多少章？",
  "菜鸟教程一共有多少套教程？",
]

**for** q **in** questions:
  result = agent.invoke({"messages": [HumanMessage(content=q)]})
  **print**(f"Q: {q}")
  **print**(f"A: {result['messages'][-1].content}")
  **print**("-" * 60)



### RAG Agent 的执行流程

对于上面的第三个问题"菜鸟教程一共有多少套教程？"，Agent 的执行流程是：

1. 用户提问
2. 模型判断需要查询知识库 → 调用 search_knowledge_base("菜鸟教程 教程数量")
3. 检索器从向量数据库中搜索语义最相似的文档块
4. 将检索结果返回给模型
5. 模型基于检索结果生成准确回答

### 添加引用来源

专业的 RAG 系统通常会附带引用来源，让用户知道信息来自哪里

from langchain_core.documents import Document


@tool
def search_with_sources(query: str) -> str:
    """在菜鸟教程知识库中搜索，返回带来源标注的结果。

    Args:
        query: 搜索关键词
    """
    docs = retriever.invoke(query)
    if not docs:
        return "未找到相关信息。"
    
    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "菜鸟教程知识库")
        results.append(f"[来源 {i}: {source}]\n{doc.page_content}")
    
    return "\n\n".join(results)


## 如需在文档中保留来源信息，可在创建时添加元数据
doc_with_meta = Document(
    page_content="Python3 基础教程共 30 章...",
    metadata={"source": "Python3 基础教程-课程介绍", "url": "https://www.runoob.com/python3/"}
)

### 向量存储的持久化

在实际项目中，你不会每次都重建向量索引。Chroma 支持持久化到本地：

### 实例

    # 创建持久化向量存储（首次运行）
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma.from_documents(
  documents=chunks,
  embedding=embeddings,
  persist_directory="./runoob_vector_db", # 持久化目录
)

    # 后续运行直接加载
loaded_store = Chroma(
  persist_directory="./runoob_vector_db",
  embedding_function=embeddings,
)
retriever = loaded_store.as_retriever()

    # 无需重新计算向量！

> 向量存储的持久化可以大幅提升启动速度。在文档量大的情况下（成千上万篇），重新计算所有向量的 Embedding 可能花费数十分钟。持久化后只需加载即可。

## LangChain 智能客服机器人

本篇将前面学到的知识整合起来，构建一个完整的智能客服机器人。它能够查询知识库、处理订单、在必要时转接人工。

------

### 需求分析

| 功能         | 实现方式                               |
| :----------- | :------------------------------------- |
| 知识库问答   | RAG 检索 + 模型回答                    |
| 订单查询     | @tool 工具函数                         |
| 对话记忆     | SqliteSaver Checkpointer               |
| 敏感内容过滤 | @before_model Middleware               |
| 人工转接     | HITL interrupt() / Command(resume=...) |

------

### 环境搭建

正式写代码之前，先把运行环境准备好。跟着下面几步操作，几分钟就能搭好。

### 第一步：确认 Python 版本

建议使用 Python 3.10 及以上版本。打开终端，输入：

```
python --version
```

如果版本低于 3.10，建议先升级 Python，再进行后续步骤。

### 第二步：创建并激活虚拟环境（推荐）

为了不污染系统的 Python 环境，建议为这个项目单独创建一个虚拟环境：

```
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境（Windows）
venv\Scripts\activate

# 激活虚拟环境（macOS / Linux）
source venv/bin/activate
```

激活成功后，终端提示符前面会出现 `(venv)` 字样。

### 第三步：安装依赖包

这个项目一共用到 6 个第三方库，分别负责不同的功能，建议按下面的顺序逐个安装，方便出问题时定位是哪一个库装的不对：

| 安装命令                                  | 作用                                                         |
| :---------------------------------------- | :----------------------------------------------------------- |
| `pip install langchain`                   | LangChain 核心库，提供 create_agent、@tool 等基础能力        |
| `pip install langchain-deepseek`          | 让 init_chat_model 能识别 "deepseek:" 前缀，调用 DeepSeek 模型 |
| `pip install langchain-openai`            | 提供 OpenAIEmbeddings，用于把知识库文本转成向量              |
| `pip install langchain-chroma chromadb`   | 本地向量数据库，用于存储和检索知识库向量                     |
| `pip install langgraph-checkpoint-sqlite` | SqliteSaver，把对话记忆持久化到 SQLite 文件                  |
| `pip install python-dotenv`               | 从 .env 文件读取 API 密钥等环境变量                          |

也可以一条命令全部装好：

```
pip install langchain langchain-deepseek langchain-openai langchain-chroma chromadb langgraph-checkpoint-sqlite python-dotenv
```

### 第四步：配置 API 密钥

在项目根目录新建一个名为 `.env` 的文件（注意文件名以点开头，没有后缀），填入以下内容：

```
# DeepSeek 官网申请：https://platform.deepseek.com
DEEPSEEK_API_KEY=sk-你的deepseek密钥

# OpenAI 官网申请：https://platform.openai.com
# 这里只用来调用 embedding 接口，不涉及 Chat 模型
OPENAI_API_KEY=sk-你的openai密钥
```

如果没有 OpenAI 的 key，我们可以采用阿里百炼的，`.env` 的文件的代码如下：

```
# DeepSeek 官网申请：https://platform.deepseek.com
DEEPSEEK_API_KEY=sk-你的deepseek密钥

# 阿里云百炼控制台申请：https://bailian.console.aliyun.com
# 这里用来调用通义千问的 Embedding 服务，给知识库文本做向量化
DASHSCOPE_API_KEY=sk-你的百炼密钥
```

> .env 文件里保存的是私密密钥，务必不要提交到 Git 仓库或分享给他人。可以在项目里新建 `.gitignore` 文件，加入一行 `.env` 来避免误提交。

### 第五步：验证安装

新建一个 `check_install.py` 文件，运行下面的脚本，检查依赖和密钥是否都配置正确：

### 实例

    # 文件路径：check_install.py
**import** os
**from** dotenv **import** load_dotenv

load_dotenv()

    # 检查依赖包能否正常导入
**import** langchain
**import** langchain_deepseek
**import** langchain_openai
**import** langchain_chroma
**import** chromadb
**import** langgraph

**print**(f"langchain 版本: {langchain.__version__}")

    # 检查密钥是否已配置
**assert** os.getenv("DEEPSEEK_API_KEY"), "未检测到 DEEPSEEK_API_KEY，请检查 .env 文件"
    # assert os.getenv("OPENAI_API_KEY"), "未检测到 OPENAI_API_KEY，请检查 .env 文件"
**assert** os.getenv("DASHSCOPE_API_KEY"), "未检测到 DASHSCOPE_API_KEY，请检查 .env 文件"

**print**("环境配置成功~可以开始写客服机器人了！")

运行：

```
python check_install.py
```

如果正常配置了 key，输出如下：

```
langchain 版本: 1.3.0
环境配置成功~可以开始写客服机器人了！
```

如果报错，通常是某个包没装上或者 .env 里的密钥没填对，根据报错信息回到对应步骤检查即可。

------

### 完整代码

环境搭建完成后（依赖安装、.env 配置见上一节"环境搭建"），就可以编写完整的客服机器人代码了：

### 实例

    # 文件路径：customer_service_bot.py
    # 依赖安装、.env 配置见上一节"环境搭建"
**from** dotenv **import** load_dotenv
load_dotenv()

**import** os
**import** sqlite3
**from** typing **import** Annotated
**from** langchain.tools **import** tool
**from** langchain.agents **import** create_agent
**from** langchain.agents.middleware **import** before_model, after_model
**from** langchain.chat_models **import** init_chat_model
**from** langchain.messages **import** HumanMessage, AIMessage
**from** langchain_openai **import** OpenAIEmbeddings
**from** langchain_chroma **import** Chroma
**from** langchain_text_splitters **import** RecursiveCharacterTextSplitter
**from** langgraph.checkpoint.sqlite **import** SqliteSaver
**from** langgraph.types **import** interrupt, Command


    # ========== 1. 准备知识库 ==========

knowledge_base = [
  "菜鸟教程 RUNOOB 创立于 2013 年，是国内领先的免费编程学习平台。",
  "平台提供 300+ 套教程，涵盖 Python、Java、HTML、CSS、JavaScript 等。",
  "Python3 基础教程共 30 章，累计学习人次超 500 万。课程完全免费。",
  "VIP 会员费用为 ¥99/月，¥799/年，包含视频课程和一对一答疑服务。",
  "退款政策：购买 7 天内且在 3 节课以内可全额退款。",
  "平台支持在线编程环境，无需安装任何软件即可编写运行代码。",
  "客服工作时间：周一至周五 9:00-18:00，周末 10:00-16:00。",
]

    # 使用阿里云百炼（DashScope）的通义千问 Embedding 服务
    # 百炼的 Embedding 接口兼容 OpenAI 接口规范，所以直接用 langchain-openai
    # 的 OpenAIEmbeddings，把 base_url 指向百炼的兼容端点即可，
    # 不需要再装 langchain-community / dashscope（该包已停止维护）。
    # text-embedding-v4 是目前推荐的通用向量模型，默认输出 1024 维向量。
    #
    # 两个关键参数不能少：
    # - check_embedding_ctx_length=False：OpenAIEmbeddings 默认会用 tiktoken
    #  把文本预先编码成 token id 数组再发送（OpenAI 官方接口认这个格式），
    #  但百炼的兼容接口只接受原始字符串，不关掉这个选项会报
    #  "contents is neither str nor list of str" 错误。
    # - chunk_size=10：百炼 Embedding 接口单次请求最多接受 10 条文本，
    #  OpenAIEmbeddings 默认一次打包 1000 条，知识库稍大就会超限报错。
embeddings = OpenAIEmbeddings(
  model="text-embedding-v4",
  api_key=os.getenv("DASHSCOPE_API_KEY"),
  base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
  check_embedding_ctx_length=False,
  chunk_size=10,
)
chunks = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30
                     ).create_documents(knowledge_base)
vector_store = Chroma.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})


    # ========== 2. 定义工具 ==========

@tool
**def** search_kb(query: str) -> str:
  """搜索菜鸟教程知识库，获取关于平台、课程、政策等官方信息。

  Args:
    query: 搜索问题或关键词
  """
  docs = retriever.invoke(query)
  **if** **not** docs:
    **return** "未找到相关信息，建议转接人工客服。"
  **return** "**\n**".join(f"- {doc.page_content}" **for** doc **in** docs)


    # 模拟订单数据库
orders_db = {
  "ORD-2024-001": {"user": "小明", "item": "VIP 年费会员",
           "amount": 799, "status": "已完成", "date": "2024-01-15"},
  "ORD-2024-002": {"user": "小明", "item": "Python 实战课程",
           "amount": 199, "status": "配送中", "date": "2024-03-20"},
}


@tool
**def** query_order(order_id: str) -> str:
  """根据订单号查询订单状态和详情。

  Args:
    order_id: 订单号，如 ORD-2024-001
  """
  order = orders_db.get(order_id.upper())
  **if** **not** order:
    **return** f"未找到订单 {order_id}。请确认订单号是否正确。"
  **return** (f"订单 {order_id}：{order['item']} | "
      f"金额 ¥{order['amount']} | "
      f"状态 {order['status']} | "
      f"日期 {order['date']}")


@tool
**def** transfer_to_human(reason: str) -> str:
  """将用户转接给人工客服。

  Args:
    reason: 转接原因
  """
  approval = interrupt({
    "action": "transfer_to_human",
    "reason": reason,
    "message": f"用户请求转接人工客服，原因：{reason}。是否转接？"
  })
  **if** approval.get("confirmed"):
    **return** (f"已为您转接人工客服，预计等待 {approval.get('wait_time', 3)} 分钟。"
        f"工单号：TK-{approval.get('ticket_id', 'N/A')}")
  **return** "转接已取消，我继续为您服务。"


    # ========== 3. 定义 Middleware ==========

@before_model
**def** content_guard(state, runtime):
  """过滤用户输入中的不当内容"""
  last_msg = state["messages"][-1] **if** state.get("messages") **else** None
  **if** **not** last_msg:
    **return** None
  content = str(getattr(last_msg, 'content', ''))
  blocked = ["黄X", "X博", "违法"]
  **for** word **in** blocked:
    **if** word **in** content:
      **return** {
        "jump_to": "end",
        "messages": [HumanMessage(content="抱歉，我不能处理这个请求。")]
      }
  **return** None


@after_model
**def** auto_signature(state, runtime):
  """自动追加客服签名"""
  msgs = state.get("messages", [])
  **if** **not** msgs:
    **return** None
  last = msgs[-1]
  **if** last.type == "ai" **and** last.content **and** **not** (
    hasattr(last, 'tool_calls') **and** last.tool_calls
  ):
    # 关键：复用 last.id，让 add_messages reducer 原地替换该消息，
    # 而不是把它当成一条新消息追加到历史里（否则历史会越滚越大，
    # 每轮多出一条"无签名版"和一条"带签名版"）
    **return** {"messages": [AIMessage(
      id=last.id,
      content=last.content
      \+ "**\n****\n**---**\n**菜鸟教程 RUNOOB 客服中心 | 工作时间 9:00-18:00"
    )]}
  **return** None


    # ========== 4. 创建 Agent ==========

    # SqliteSaver.from_conn_string() 返回的是上下文管理器，只适合"用完即关"的
    # 一次性脚本。客服机器人需要在多次 chat() 调用之间保持同一个数据库连接，
    # 所以这里自己建立连接后传给 SqliteSaver 构造函数。
    # check_same_thread=False 是因为 Web 框架通常会跨线程调用同一个连接。
conn = sqlite3.connect("customer_service.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)

agent = create_agent(
  model=model,
  tools=[search_kb, query_order, transfer_to_human],
  middleware=[content_guard, auto_signature],
  checkpointer=checkpointer,
  system_prompt="""你是菜鸟教程 RUNOOB 的智能客服"小菜"。

## 你的职责

1. 热情接待每一位用户，用"您"称呼
2. 关于平台信息、课程内容、政策等问题，使用 search_kb 查询
3. 关于订单查询，使用 query_order 工具
4. 遇到无法解决的问题，使用 transfer_to_human 转接人工


## 行为准则
\- 回答简洁，每次 2-3 句话
\- 不知道的就查询知识库，查不到就诚实告知
\- 保持友好亲切的语气""",
)


    # ========== 5. 对话接口 ==========

**def** chat(thread_id: str, message: str) -> str:
  """处理用户消息并返回回复"""
  config = {"configurable": {"thread_id": thread_id}}

  # 运行 Agent
  result = agent.invoke(
    {"messages": [HumanMessage(content=message)]},
    config=config,
  )

  # 检查是否需要转接（HITL）
  state = agent.get_state(config)
  **if** state.tasks **and** state.tasks[0].interrupts:
    interrupt_info = state.tasks[0].interrupts[0].value
    **return** f"[需要审批] {interrupt_info.get('message', '')}"

  **return** result["messages"][-1].content


**def** resume_transfer(thread_id: str, confirmed: bool,
           wait_time: int = 3, ticket_id: str = "0001") -> str:
  """人工客服后台审批后，恢复被 interrupt() 中断的转接流程。

  对应 transfer_to_human 工具里等待的 approval 数据。
  """
  config = {"configurable": {"thread_id": thread_id}}
  result = agent.invoke(
    Command(resume={
      "confirmed": confirmed,
      "wait_time": wait_time,
      "ticket_id": ticket_id,
    }),
    config=config,
  )
  **return** result["messages"][-1].content


    # ========== 6. 测试 ==========

**if** __name__ == "__main__":
  user_id = "user_xiaoming"

  **print**("=== 测试 1：知识库查询 ===")
  **print**(chat(user_id, "Python3 教程有多少章？"))
  **print**()

  **print**("=== 测试 2：订单查询 ===")
  **print**(chat(user_id, "我的订单 ORD-2024-001 状态是什么？"))
  **print**()

  **print**("=== 测试 3：VIP 咨询 ===")
  **print**(chat(user_id, "VIP 会员多少钱？"))
  **print**()

  **print**("=== 测试 4：测试记忆 ===")
  **print**(chat(user_id, "我刚才问过什么问题？"))
  **print**()

  **print**("=== 测试 5：人工转接（HITL） ===")
  **print**(chat(user_id, "我要投诉，请转人工"))    # 触发 interrupt，等待审批
  **print**(resume_transfer(user_id, confirmed=True,
              wait_time=5, ticket_id="8823")) # 后台确认后恢复

  conn.close()

运行结果：

```
=== 测试 1：知识库查询 ===
您好！菜鸟教程的 **Python3 基础教程共 30 章**，完全免费，累计学习人次已超 500 万哦！如需了解其他教程，也可以随时问我～

---
菜鸟教程 RUNOOB 客服中心 | 工作时间 9:00-18:00

=== 测试 2：订单查询 ===
您好！您查询的订单 **ORD-2024-001** 状态为 **已完成**。订单内容是 **VIP 年费会员**，金额 **¥799**，下单日期为 **2024-01-15**。请问还有什么可以帮您的吗？

---
菜鸟教程 RUNOOB 客服中心 | 工作时间 9:00-18:00

=== 测试 3：VIP 咨询 ===
您好！VIP 会员有 **¥99/月** 和 **¥799/年** 两种套餐，包含视频课程和一对一答疑服务哦～请问您需要办理哪种呢？

---
菜鸟教程 RUNOOB 客服中心 | 工作时间 9:00-18:00

---
菜鸟教程 RUNOOB 客服中心 | 工作时间 9:00-18:00

=== 测试 4：测试记忆 ===
您好！您刚才问了以下三个问题：
1. **Python3 教程有多少章？** —— 共 30 章，完全免费
2. **我的订单 ORD-2024-001 状态是什么？** —— 状态为"已完成"
3. **VIP 会员多少钱？** —— ¥99/月 或 ¥799/年

还有什么需要我帮忙的吗？

---
菜鸟教程 RUNOOB 客服中心 | 工作时间 9:00-18:00

---
菜鸟教程 RUNOOB 客服中心 | 工作时间 9:00-18:00

=== 测试 5：人工转接（HITL） ===
[需要审批] 用户请求转接人工客服，原因：用户要求投诉，需要转接人工客服处理。是否转接？
您好！已经为您转接人工客服，预计等待约 5 分钟。您的工单号是 **TK-8823**，请稍候，客服会尽快为您处理～

---
菜鸟教程 RUNOOB 客服中心 | 工作时间 9:00-18:00

---
菜鸟教程 RUNOOB 客服中心 | 工作时间 9:00-18:00
```

------

### 项目总结

这个客服机器人整合了以下 LangChain 特性：

| 特性         | 在项目中的使用                                               |
| :----------- | :----------------------------------------------------------- |
| RAG 检索     | search_kb 工具 + Chroma 向量存储                             |
| 工具调用     | query_order、transfer_to_human                               |
| Checkpointer | SqliteSaver 持久化对话，实现多轮记忆                         |
| Middleware   | before_model 内容过滤 + after_model 签名追加（复用消息 id 原地替换） |
| HITL         | interrupt() 暂停执行 + Command(resume=...) 审批后恢复，实现完整的人工转接闭环 |

## LangChain 个人知识库问答系统

本篇构建一个能加载 Markdown 文件、PDF 文档，并基于这些内容进行问答的个人知识库系统。

------

### 系统设计

- **文档加载**：支持 Markdown、TXT、PDF 多种格式
- **向量检索**：Chroma 持久化存储，支持增量更新
- **引用来源**：回答中附带来源文档和片段位置
- **流式输出**：逐 Token 显示回答

------

### 完整代码

运行前需要在 `.env` 文件中配置 `DEEPSEEK_API_KEY`（Chat 模型）和 `DASHSCOPE_API_KEY`（阿里云百炼，用于知识库文本向量化），具体申请方式和常见问题排查见[《LangChain 智能客服机器人》](https://www.runoob.com/langchain/langchain-project-customer-service.html)一节。

### 实例

    # 文件路径：knowledge_qa.py
    # pip install langchain langchain-deepseek langchain-openai langchain-community langchain-chroma chromadb pypdf
**from** dotenv **import** load_dotenv
load_dotenv()

**import** os
**from** pathlib **import** Path
**from** langchain.tools **import** tool
**from** langchain.agents **import** create_agent
**from** langchain.chat_models **import** init_chat_model
**from** langchain.messages **import** HumanMessage
**from** langchain_openai **import** OpenAIEmbeddings
**from** langchain_chroma **import** Chroma
**from** langchain_text_splitters **import** RecursiveCharacterTextSplitter
**from** langchain_community.document_loaders **import** TextLoader, PyPDFLoader


**class** KnowledgeBase:
  """个人知识库管理器"""

  **def** __init__(self, persist_dir: str = "./my_knowledge_db"):
    self.persist_dir = persist_dir

    # 阿里云百炼的 Embedding 接口兼容 OpenAI 规范，用 langchain-openai
    # 的 OpenAIEmbeddings 调用即可，不需要装 langchain-community 里
    # 已停止维护的 DashScopeEmbeddings。
    # check_embedding_ctx_length=False：关掉 tiktoken 预分词，直接发原始文本
    # （百炼接口不接受 token id 数组）。
    # chunk_size=10：百炼 Embedding 接口单次请求最多接受 10 条文本。
    self.embeddings = OpenAIEmbeddings(
      model="text-embedding-v4",
      api_key=os.getenv("DASHSCOPE_API_KEY"),
      base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
      check_embedding_ctx_length=False,
      chunk_size=10,
    )
    self.text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=500, chunk_overlap=50,
      separators=["**\n****\n**", "**\n**", "。", "！", "？", ". ", "! ", "? ", " "],
    )
    self.vector_store = None
    self._load_or_create()

  **def** _load_or_create(self):
    """加载已有向量库或创建新的（Chroma 用同一套参数即可，
    目录是否已存在只影响提示信息，不影响创建逻辑）"""
    is_existing = os.path.exists(self.persist_dir) **and** os.listdir(self.persist_dir)

    self.vector_store = Chroma(
      embedding_function=self.embeddings,
      persist_directory=self.persist_dir,
    )

    **if** is_existing:
      **print**(f"已加载向量库：{self.vector_store._collection.count()} 个文档块")
    **else**:
      **print**("已创建新的向量库")

  **def** add_file(self, file_path: str) -> int:
    """添加文件到知识库，返回添加的文档块数

    根据文件扩展名自动选择合适的 Loader：
    \- .pdf 用 PyPDFLoader（依赖 pypdf 库解析 PDF，按页返回多个 Document）
    \- 其余（.md、.txt 等纯文本文件）用 TextLoader 按原始文本读取
    """
    suffix = Path(file_path).suffix.lower()
    **if** suffix == ".pdf":
      loader = PyPDFLoader(file_path)
    **else**:
      loader = TextLoader(file_path, encoding="utf-8")

    docs = loader.load()

    # 统一补充文件名来源；PyPDFLoader 加载出的每个 Document 还会自带
    # page 字段（第几页），检索时可以一并用来定位片段位置
    **for** doc **in** docs:
      doc.metadata["source"] = Path(file_path).name

    chunks = self.text_splitter.split_documents(docs)
    self.vector_store.add_documents(chunks)
    **print**(f"已添加 {Path(file_path).name}：{len(chunks)} 个文档块")
    **return** len(chunks)

  **def** add_text(self, text: str, source: str = "手动添加") -> int:
    """直接添加文本到知识库"""
    chunks = self.text_splitter.create_documents(
      [text], metadatas=[{"source": source}]
    )
    self.vector_store.add_documents(chunks)
    **return** len(chunks)

  **def** search(self, query: str, k: int = 3) -> list:
    """搜索知识库"""
    **return** self.vector_store.similarity_search(query, k=k)

  **def** get_retriever(self):
    """获取检索器"""
    **return** self.vector_store.as_retriever(search_kwargs={"k": 3})

> 目前 `TextLoader`、`PyPDFLoader` 这类本地文件加载器还没有独立出来的维护包，仍然只能从 `langchain_community.document_loaders` 导入，运行时可能会看到 langchain-community 整体的 DeprecationWarning——这个警告目前可以忽略，等官方推出独立的文档加载器包后再迁移即可；和前面 embedding 那种"已有现成替代方案却没换"的情况不同。

继续 Agent 部分：

### 实例

    # ========== 创建知识库并添加示例数据 ==========

kb = KnowledgeBase("./my_knowledge_db")

    # 添加一些示例知识
kb.add_text(
  "菜鸟教程 RUNOOB 的 Python3 基础教程包含以下章节："
  "1. Python 简介与环境搭建 2. 基本数据类型 3. 运算符与表达式 "
  "4. 条件判断 if-else 5. 循环 for/while 6. 函数定义与调用 "
  "7. 模块与包 8. 文件操作 9. 异常处理 10. 面向对象编程",
  source="Python3 教程大纲"
)

kb.add_text(
  "要成为一名优秀的 Python 开发者，建议按以下路线学习："
  "第一步，掌握 Python 基础语法（1-2 周）；"
  "第二步，学习数据结构和算法基础（2-3 周）；"
  "第三步，选择一个方向深入学习（Web 开发/数据分析/AI）；"
  "第四步，做 2-3 个实战项目巩固知识。",
  source="Python 学习路线"
)

kb.add_text(
  "菜鸟教程的在线编程环境支持 Python、JavaScript、Java、C++ 等多种语言。"
  "用户无需安装任何软件，打开浏览器即可编写和运行代码。"
  "在线环境还支持代码高亮、自动补全和错误提示功能。",
  source="在线编程环境说明"
)

    # 也可以加载本地文件，PDF 和 Markdown/TXT 都会自动识别：
    # kb.add_file("./docs/产品手册.pdf")
    # kb.add_file("./docs/常见问题.md")


    # ========== 创建 RAG Agent ==========

@tool
**def** search_knowledge(query: str) -> str:
  """在个人知识库中搜索相关信息。搜索时使用完整的问题或关键短语。

  Args:
    query: 搜索问题或关键短语
  """
  docs = kb.search(query, k=3)
  **if** **not** docs:
    **return** "知识库中未找到相关信息。"

  results = []
  **for** i, doc **in** enumerate(docs, 1):
    source = doc.metadata.get("source", "未知来源")
    page = doc.metadata.get("page")
    location = f"{source}" + (f" 第 {page + 1} 页" **if** page **is** **not** None **else** "")
    content = doc.page_content[:200]
    results.append(f"[{i}] 来源：{location}**\n**{content}")

  **return** "**\n****\n**---**\n****\n**".join(results)


model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
agent = create_agent(
  model=model,
  tools=[search_knowledge],
  system_prompt="""你是个人知识库助手。

## 规则

1. 所有问题必须先用 search_knowledge 工具检索知识库
2. 回答时注明信息来源（文档名称，如果是 PDF 还要注明页码）
3. 如果知识库中没有相关内容，如实告知
4. 回答要结构化，使用数字列表或分段""",

)


    # ========== 测试：非流式，同时看检索到的内容 ==========

**def** ask(question: str):
  """提问，同时打印检索到的原始片段和最终回答，方便调试"""
  **print**(f"**\n**{'='*60}")
  **print**(f"Q: {question}")
  **print**(f"{'='*60}")

  result = agent.invoke({
    "messages": [HumanMessage(content=question)]
  })

  # 显示检索到的内容
  **for** msg **in** result["messages"]:
    **if** msg.type == "tool":
      **print**(f"**\n**[检索到的内容]")
      **print**(msg.content[:300])

  **print**(f"**\n**[回答]")
  **print**(result["messages"][-1].content)


    # ========== 测试：流式，对应系统设计里的"流式输出" ==========

**def** ask_stream(question: str):
  """提问并逐 Token 流式打印回答，实现打字机效果"""
  **print**(f"**\n**{'='*60}")
  **print**(f"Q: {question}")
  **print**(f"{'='*60}")
  **print**("**\n**[回答] ", end="", flush=True)

  **for** chunk, metadata **in** agent.stream(
    {"messages": [HumanMessage(content=question)]},
    stream_mode="messages",
  ):
    # metadata["langgraph_node"] == "model" 表示这个 chunk 来自模型生成
    # 最终回答的节点，过滤掉工具调用等其他类型的 chunk
    **if** metadata.get("langgraph_node") == "model" **and** chunk.content:
      **print**(chunk.content, end="", flush=True)
  **print**()


ask("Python3 基础教程包含哪些章节？")
ask("如何规划 Python 学习路线？")
ask_stream("菜鸟教程的在线编程环境支持哪些功能？")

运行结果：

```
============================================================
Q: Python3 基础教程包含哪些章节？
============================================================

[检索到的内容]
[1] 来源：Python3 教程大纲
菜鸟教程 RUNOOB 的 Python3 基础教程包含以下章节：...

[回答]
Python3 基础教程包含以下章节（来源：Python3 教程大纲）：
1. Python 简介与环境搭建
2. 基本数据类型
3. 运算符与表达式
...

============================================================
Q: 如何规划 Python 学习路线？
============================================================

[回答]
根据知识库中的 Python 学习路线建议（来源：Python 学习路线）：
第一步：掌握基础语法（1-2 周）
第二步：学习数据结构和算法（2-3 周）
第三步：选择方向深入学习（Web/数据分析/AI）
第四步：做 2-3 个实战项目巩固

============================================================
Q: 菜鸟教程的在线编程环境支持哪些功能？
============================================================

[回答] 根据知识库中的说明（来源：在线编程环境说明），菜鸟教程的在线编程环境：
1. 支持 Python、JavaScript、Java、C++ 等多种语言
2. 无需安装任何软件，打开浏览器即可编写和运行代码
3. 支持代码高亮、自动补全和错误提示功能
```

> 最后一个问题用的是 `ask_stream()`，实际运行时"[回答]"后面的文字会一个字一个（或几个字符一批）地陆续打印出来，就像打字机效果；上面为了方便展示，直接贴出了打印完成后的最终文本。

## LangChain 多工具个人助手

本篇构建一个集天气查询、日程管理、邮件发送于一体的个人助手 Agent，展示多工具协作和结构化输出的完整用法。

------

### 系统设计

- 三个工具：天气查询、日程管理、邮件发送
- 结构化输出：日程汇总格式化为 Markdown
- 流式输出：实时显示 AI 思考和处理过程
- 对话记忆：记住用户偏好和上下文

------

### 完整代码

运行前安装依赖，并在 `.env` 里配置好 `DEEPSEEK_API_KEY`：

```
pip install langchain langchain-deepseek langgraph-checkpoint-sqlite python-dotenv
```

### 实例

    # 文件路径：personal_assistant.py
**from** dotenv **import** load_dotenv
load_dotenv()

**import** sqlite3
**from** datetime **import** datetime
**from** pydantic **import** BaseModel, Field

**from** langchain.tools **import** tool
**from** langchain.agents **import** create_agent
**from** langchain.agents.middleware **import** dynamic_prompt
**from** langchain.chat_models **import** init_chat_model
**from** langchain.messages **import** HumanMessage
**from** langgraph.checkpoint.sqlite **import** SqliteSaver


    # ========== 1. 模拟数据 ==========

calendar_events = [
  {"id": 1, "title": "Python 学习", "date": "2024-03-25",
   "time": "14:00", "duration": "2小时"},
  {"id": 2, "title": "团队周会", "date": "2024-03-25",
   "time": "10:00", "duration": "1小时"},
  {"id": 3, "title": "代码审查", "date": "2024-03-26",
   "time": "15:00", "duration": "1.5小时"},
]

weather_db = {
  "杭州": {"condition": "晴", "temp": 25, "humidity": 60},
  "北京": {"condition": "多云", "temp": 18, "humidity": 45},
  "上海": {"condition": "小雨", "temp": 22, "humidity": 80},
}


    # ========== 2. 定义工具 ==========

@tool
**def** get_weather(city: str) -> str:
  """查询指定城市的实时天气。

  Args:
    city: 城市名称，如 杭州、北京、上海
  """
  data = weather_db.get(city)
  **if** **not** data:
    **return** f"暂不支持查询 {city} 的天气。支持的城市：{', '.join(weather_db.keys())}"
  **return** (f"{city}天气：{data['condition']}，"
      f"温度 {data['temp']}°C，湿度 {data['humidity']}%")


@tool
**def** query_schedule(date: str = None) -> str:
  """查询指定日期的日程安排。不指定日期则查询今天的日程。

  Args:
    date: 日期，格式 YYYY-MM-DD，如 2024-03-25。不传则查询今天
  """
  **if** date **is** None:
    date = datetime.now().strftime("%Y-%m-%d")

  events = [e **for** e **in** calendar_events **if** e["date"] == date]
  **if** **not** events:
    **return** f"{date} 没有日程安排。"

  events.sort(key=**lambda** e: e["time"])
  # 这里必须直接写 emoji 字符，不能写成 &#x1f4c5; 这种 HTML 实体——
  # 这段是 Python 字符串，会被原样打印出来，不会被浏览器解析成图形
  lines = [f"&#x1f4c5; {date} 日程安排："]
  **for** e **in** events:
    lines.append(f"  - {e['time']} {e['title']}（{e['duration']}）")
  **return** "**\n**".join(lines)


@tool
**def** send_email(to: str, subject: str, body: str) -> str:
  """发送邮件（模拟）。

  Args:
    to: 收件人邮箱
    subject: 邮件主题
    body: 邮件正文
  """
  # 模拟发送
  email_id = f"MSG-{datetime.now().strftime('%Y%m%d%H%M%S')}"
  **return** f"邮件已发送！收件人：{to}，主题：{subject}，邮件ID：{email_id}"


    # ========== 3. 结构化输出模型 ==========

**class** DailySummary(BaseModel):
  """每日摘要"""
  date: str = Field(description="日期")
  weather_summary: str = Field(description="天气概述")
  event_count: int = Field(description="日程数量")
  key_events: list[str] = Field(description="重要日程列表")
  suggestion: str = Field(description="今日建议")


**def** to_markdown(summary: DailySummary) -> str:
  """把结构化的 DailySummary 渲染成 Markdown 文本，
  对应系统设计里"结构化输出：日程汇总格式化为 Markdown"这一项"""
  lines = [
    f"## {summary.date} 今日摘要",
    "",
    f"- **天气**：{summary.weather_summary}",
    f"- **日程数量**：{summary.event_count}",
    "",
    "**重要事项：**",
  ]
  **if** summary.key_events:
    lines += [f"- {event}" **for** event **in** summary.key_events]
  **else**:
    lines.append("- 无")
  lines += ["", f"**今日建议**：{summary.suggestion}"]
  **return** "**\n**".join(lines)


    # ========== 4. 定义 Middleware ==========

@dynamic_prompt
**def** inject_date_context(request) -> str:
  """动态在系统提示词后追加当前日期信息。

  之前用 @before_model 把日期消息插进 messages 列表的写法有两个问题：

1. before_model 返回的消息更新走 add_messages reducer，reducer 只按

    "新消息追加到末尾"处理，不认返回列表里的位置，insert(-1, ...)
    想插到用户消息前面的意图其实并不生效；

2. before_model 在多轮工具调用循环里会反复触发，容易把这条消息插进

    AIMessage(tool_calls=...) 和它对应的 ToolMessage 之间，
    打乱严格的消息顺序要求。
  改用 @dynamic_prompt 直接重写系统提示词字符串，每次模型调用都会
  重新计算一遍，不触碰 messages 列表，没有累积或错位的风险。
  """
  now = datetime.now()
  weekday = ["一", "二", "三", "四", "五", "六", "日"][now.weekday()]
  date_hint = (f"**\n****\n**[系统提示] 当前日期是 {now.strftime('%Y年%m月%d日')}，"
         f"星期{weekday}。如果用户没有指定日期，默认查询今天。")
  **return** request.system_prompt + date_hint


    # ========== 5. 创建 Agent ==========

    # 用 SqliteSaver 持久化对话，实现系统设计里的"对话记忆"。
    # 自己建立连接再传给 SqliteSaver 构造函数，而不是用
    # SqliteSaver.from_conn_string()（那是个只适合 with 语句、
    # 用完即关的上下文管理器，详见《LangChain 智能客服机器人》一篇）。
conn = sqlite3.connect("personal_assistant.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0.3)
agent = create_agent(
  model=model,
  tools=[get_weather, query_schedule, send_email],
  middleware=[inject_date_context],
  response_format=DailySummary,
  checkpointer=checkpointer,
  system_prompt="""你是个人助手"小助"。你可以查天气、管理日程、发送邮件。

## 工作方式

1. 当用户问"今天怎么样"或类似问题时：

  \- 先查询今天的天气（get_weather）
  \- 再查询今天的日程（query_schedule）
  \- 然后生成每日摘要

2. 当用户要求发邮件时，使用 send_email 工具



3. 当用户只问天气或只问日程时，只调用对应的工具


## 风格
\- 语气亲切自然
\- 优先使用工具获取实时数据，不要编造""",
)


    # ========== 6. 交互函数 ==========

**def** chat(message: str, thread_id: str = "xiaoming"):
  """与助手对话：
  \- 用 stream_mode="values" 流式展示 AI 的思考和处理过程（对应"流式输出"）
  \- 传入 thread_id，同一个 thread_id 下的对话会被 SqliteSaver 记住（对应"对话记忆"）
  \- 如果本轮生成了结构化摘要，会渲染成 Markdown 一并打印（对应"结构化输出"）
  """
  config = {"configurable": {"thread_id": thread_id}}
  **print**(f"**\n**{'='*60}")
  **print**(f"你: {message}")
  **print**(f"{'='*60}")

  seen = 0
  final_state = {}
  **for** state **in** agent.stream(
    {"messages": [HumanMessage(content=message)]},
    config=config,
    stream_mode="values",
  ):
    final_state = state
    msgs = state.get("messages", [])
    # 每多出几条新消息，就说明 Agent 往前推进了一步，实时打印出来
    **for** msg **in** msgs[seen:]:
      **if** msg.type == "ai" **and** getattr(msg, "tool_calls", None):
        **for** call **in** msg.tool_calls:
          **print**(f"&#x1f914; 决定调用工具: {call['name']}，参数: {call['args']}")
      **elif** msg.type == "tool":
        tool_name = getattr(msg, "name", "") **or** "工具"
        **print**(f"&#x1f527; {tool_name} 返回: {str(msg.content)[:80]}")
      **elif** msg.type == "ai" **and** msg.content:
        **print**(f"&#x1f916; 助手: {msg.content}")
    seen = len(msgs)

  # 注意：response_format 是在 Agent 级别配置的，每一次调用（包括发邮件、
  # 追问这类和"日程摘要"无关的请求）都会强制尝试生成一份 DailySummary，
  # 这是全局 response_format 的已知局限。真要做成多意图助手，更好的做法是
  # 把"生成每日摘要"做成一个单独的工具，让 Agent 自己判断要不要调用，
  # 而不是给整个 Agent 挂一个一直生效的输出 schema。
  **if** "structured_response" **in** final_state:
    summary = final_state["structured_response"]
    **print**("**\n**--- 结构化摘要（Markdown） ---")
    **print**(to_markdown(summary))

  **return** final_state


    # ========== 7. 测试 ==========

**if** __name__ == "__main__":
  chat("杭州今天天气怎么样？看看我的日程，然后给我一个今日总结")
  chat("帮我发一封邮件给 team@runoob.com，主题是'今日总结'，内容是今天日程已确认")
  # 第三轮不带任何新信息，纯粹考验 Agent 是否记得上一轮对话——
  # 因为传的是同一个 thread_id，checkpointer 会把完整历史带回来
  chat("我刚才让你发的那封邮件，主题是什么来着？")

运行结果：

```
============================================================
你: 杭州今天天气怎么样？看看我的日程，然后给我一个今日总结
============================================================
&#x1f914; 决定调用工具: get_weather，参数: {'city': '杭州'}
&#x1f527; get_weather 返回: 杭州天气：晴，温度 25°C，湿度 60%
&#x1f914; 决定调用工具: query_schedule，参数: {}
&#x1f527; query_schedule 返回: &#x1f4c5; 2024-03-25 日程安排：  - 10:00 团队周会（1小时）  - 14:00...
&#x1f916; 助手: 早上好！今天杭州是大晴天，气温 25°C，湿度 60%，很适合出门活动。
今天您有两项日程：上午 10:00 的团队周会和下午 14:00 的 Python 学习，祝您今天顺利！

--- 结构化摘要（Markdown） ---
## 2024-03-25 今日摘要

- **天气**：杭州晴，25°C，湿度 60%
- **日程数量**：2

**重要事项：**
- 10:00 团队周会
- 14:00 Python 学习

**今日建议**：上午先参加团队周会，下午集中精力学习 Python，注意劳逸结合

============================================================
你: 帮我发一封邮件给 team@runoob.com，主题是'今日总结'，内容是今天日程已确认
============================================================
&#x1f914; 决定调用工具: send_email，参数: {'to': 'team@runoob.com', 'subject': '今日总结', 'body': '今天日程已确认'}
&#x1f527; send_email 返回: 邮件已发送！收件人：team@runoob.com，主题：今日总结，邮件ID：MSG-20240325143000
&#x1f916; 助手: 邮件已经发送成功啦！收件人是 team@runoob.com，主题"今日总结"。

--- 结构化摘要（Markdown） ---
## 2024-03-25 今日摘要
...（这里的摘要和邮件内容其实没什么关系，是 response_format 强制生成的，
     属于前面提到的已知局限）

============================================================
你: 我刚才让你发的那封邮件，主题是什么来着？
============================================================
&#x1f916; 助手: 你刚才让我发的那封邮件主题是"今日总结"，收件人是 team@runoob.com。
```

> 第三轮完全没有触发任何工具调用，助手能答上来纯粹是因为 `thread_id` 相同，SqliteSaver 把前两轮的完整消息历史带回来了——这就是"对话记忆"真正生效的样子。如果把 `thread_id` 换成一个新值，第三轮会变成一次孤立对话，助手不会知道"刚才"发生了什么。

------

### 项目总结

这个个人助手展示了：

| 特性         | 实现方式                                                     |
| :----------- | :----------------------------------------------------------- |
| 多工具协作   | 天气+日程+邮件，Agent 自动选择调用顺序                       |
| 结构化输出   | DailySummary Pydantic 模型 + to_markdown() 渲染成 Markdown 文本 |
| 流式输出     | agent.stream(stream_mode="values")，逐步展示工具调用和思考过程 |
| 对话记忆     | SqliteSaver + thread_id，跨多轮调用记住上下文                |
| 日期注入     | @dynamic_prompt 动态重写系统提示词，避免污染消息历史         |
| 自然语言交互 | 用户用自然语言描述需求，Agent 自主规划                       |



## LangChain LangSmith -- 可观测性

LangSmith 是 LangChain 官方的可观测性平台，帮助你追踪 Agent 执行过程、监控性能、调试问题。

------

### LangSmith 是什么

当 Agent 在后台运行时，你看不到它内部发生了什么——调用了哪些模型、执行了哪些工具、每一步消耗了多少 Token。LangSmith 解决了这个"黑盒"问题。

| 功能     | 说明                            |
| :------- | :------------------------------ |
| 执行追踪 | 记录 Agent 每一步的执行轨迹     |
| 性能监控 | 统计每次调用的耗时和 Token 消耗 |
| 调试回放 | 查看历史执行的详细信息          |
| 评估测试 | 创建测试集评估 Agent 表现       |

------

### 快速开始

### 注册与安装

```
$ pip install langsmith
```

在 [smith.langchain.com](https://smith.langchain.com/) 注册账号，获取 API Key，然后在 .env 中配置：

### 实例

*# .env 文件*
LANGCHAIN_TRACING_V2=**true**
LANGCHAIN_API_KEY=lsv2_pt_your_key_here
LANGCHAIN_PROJECT=my-agent-project

### 自动追踪

### 实例

**from** dotenv **import** load_dotenv
load_dotenv() # LangSmith 配置会自动加载

**from** langchain.agents **import** create_agent
**from** langchain.chat_models **import** init_chat_model
**from** langchain.messages **import** HumanMessage
**from** langchain.tools **import** tool

    # 设置环境变量后，所有 Agent 执行都会自动追踪
    # 无需额外代码！

@tool
**def** search_course(keyword: str) -> str:
  """搜索课程"""
  **return** f"搜索结果：{keyword} 相关课程"

model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
agent = create_agent(
  model=model,
  tools=[search_course],
  system_prompt="你是菜鸟教程 RUNOOB 的助手。",
)

    # 这次执行会被自动记录到 LangSmith
result = agent.invoke({
  "messages": [HumanMessage(content="搜索 Python 课程")]
})

    # 打开 https://smith.langchain.com 查看追踪记录
**print**("已完成，请到 LangSmith 控制台查看追踪详情")

------

### 查看追踪记录

在 LangSmith 控制台中，你可以看到每次 Agent 执行的完整轨迹：

- **执行时间线**：模型调用 → 工具调用 → 模型再调用的完整时间线
- **输入/输出**：每一步的输入消息和模型返回结果
- **Token 用量**：每次模型调用的 Token 消耗和费用估算
- **延迟分析**：每一步的耗时分布
- **错误信息**：如果某步出错，可以看到完整的错误堆栈

------

### 手动创建追踪

### 实例

**from** langsmith **import** traceable

    # 使用 @traceable 装饰器标记需要追踪的函数
@traceable
**def** process_user_query(query: str) -> dict:
  """处理用户查询（此函数会被单独追踪）"""
  # 预处理
  cleaned = query.strip().lower()
  # 调用 Agent
  result = agent.invoke({"messages": [HumanMessage(content=cleaned)]})
  **return** {
    "query": cleaned,
    "answer": result["messages"][-1].content,
  }

    # 在 LangSmith 中会看到 process_user_query 作为一个独立步骤
result = process_user_query("  Python 课程推荐  ")
**print**(result["answer"])

------

### 常用配置

| 环境变量             | 说明                      | 示例                            |
| :------------------- | :------------------------ | :------------------------------ |
| LANGCHAIN_TRACING_V2 | 启用追踪（必须设为 true） | true                            |
| LANGCHAIN_API_KEY    | LangSmith API Key         | lsv2_pt_xxx                     |
| LANGCHAIN_PROJECT    | 项目名称（用于分组追踪）  | my-agent                        |
| LANGCHAIN_ENDPOINT   | API 端点（默认即可）      | https://api.smith.langchain.com |

> 生产环境建议将 LangSmith 的追踪采样率设低一些（避免记录所有请求造成成本过高），只在需要调试时开启完整追踪。

## LangChain 错误处理与调试

Agent 开发中不可避免会遇到各种错误。本篇梳理常见错误类型、调试方法和最佳实践。

------

### 常见错误类型

| 错误类型         | 典型原因               | 解决方案                                |
| :--------------- | :--------------------- | :-------------------------------------- |
| ImportError      | 未安装提供商包         | pip install langchain-deepseek 等       |
| API Key 错误     | .env 未配置或 Key 无效 | 检查环境变量和 Key 有效性               |
| 超时错误         | 网络问题或模型响应慢   | 设置 timeout 参数                       |
| Token 超限       | 消息历史过长           | 使用 trim_messages() 裁剪               |
| 工具调用错误     | 工具内部异常           | 使用 ToolException + handle_tool_errors |
| 模型返回格式错误 | 模型输出不符合预期     | 使用结构化输出 + handle_errors          |

------

### ModelRetryMiddleware——模型调用重试

LangChain 提供了内置的重试中间件：

### 实例

**from** langchain.agents.middleware **import** ModelRetryMiddleware
**from** langchain.agents **import** create_agent
**from** langchain.chat_models **import** init_chat_model

    # 内置的模型重试中间件
    # 自动在模型调用失败时重试
agent = create_agent(
  model=init_chat_model("deepseek:deepseek-v4-flash", timeout=30, max_retries=2),
  middleware=[
    ModelRetryMiddleware(
      max_retries=3,     # 最多重试 3 次
      backoff_factor=2.0,  # 退避因子（2s, 4s, 8s）
    )
  ],
  system_prompt="你是菜鸟教程 RUNOOB 的助手。",
)

### ToolRetryMiddleware——工具调用重试

### 实例

**from** langchain.agents.middleware **import** ToolRetryMiddleware

agent = create_agent(
  model="deepseek:deepseek-v4-flash",
  tools=[my_tool],
  middleware=[
    ToolRetryMiddleware(
      max_retries=3,
      backoff_factor=1.5,
    )
  ],
)

> 内置的 RetryMiddleware 和自定义的 @wrap_model_call / @wrap_tool_call 可以共存。内置中间件放在 middleware 列表前面作为最外层保护。

------

### debug=True——详细日志

### 实例

**from** langchain.agents **import** create_agent
**from** langchain.chat_models **import** init_chat_model
**from** langchain.messages **import** HumanMessage

    # 开启 debug 模式，输出详细执行日志
agent = create_agent(
  model=init_chat_model("deepseek:deepseek-v4-flash"),
  debug=True, # 开启调试日志
  system_prompt="你是菜鸟教程 RUNOOB 的助手。",
)

    # 执行时会打印：
    # - 每个节点的输入状态
    # - 每个节点的输出状态
    # - 边（edge）的跳转决策
result = agent.invoke({
  "messages": [HumanMessage(content="你好")]
})

debug=True 输出的示例：

```
[DEBUG] Starting graph execution
[DEBUG] Executing node: model
[DEBUG] Node 'model' input: {'messages': [HumanMessage(content='你好')]}
[DEBUG] Node 'model' output: {'messages': [AIMessage(content='你好！...')]}
[DEBUG] Edge 'model' -> '__end__': routing to __end__
[DEBUG] Graph execution complete
```

------

### stream_mode="debug"——最详细的调试信息

### 实例

    # 通过 stream_mode="debug" 获取最详细的信息
**for** event **in** agent.stream(
  {"messages": [HumanMessage(content="你好")]},
  stream_mode="debug",
):
  # event 包含：节点名、输入、输出、时间戳、任务信息等
  **print**(f"[{event['type']}] {event.get('name', '')}")
  **if** 'input' **in** event:
    **print**(f"  输入: {event['input']}")
  **if** 'output' **in** event:
    **print**(f"  输出: {event['output']}")

------

### 常见问题排查

### 问题 1：模型一直调用工具不停止

可能原因：工具返回的信息不充分，模型无法判断任务是否完成。解决方法：

- 让工具返回更明确的信息（如"任务已完成"）
- 设置 system_prompt 中的停止条件
- 使用 after_model 检查循环次数，超过阈值后 jump_to="end"

### 问题 2：模型调用了错误的工具或参数

可能原因：工具描述不清晰。解决方法：

- 优化工具函数的文档字符串
- 使用 args_schema 限制参数范围
- 使用更好的模型（如 deepseek-v4-pro 替代 deepseek-v4-flash）

### 问题 3：对话记忆不生效

检查清单：

- 是否传入了 checkpointer 参数？
- 是否每次使用了相同的 thread_id？
- 如果使用 SqliteSaver，数据库文件是否存在且可写？

## LangChain Chat Model API

本文档列出 init_chat_model() 和 BaseChatModel 的完整 API 参考。

------

### init_chat_model() 完整参数

| 参数                | 类型                  | 默认值     | 说明                                                       |
| :------------------ | :-------------------- | :--------- | :--------------------------------------------------------- |
| model               | str 或 None           | 无         | 模型名，格式为 provider:model_name。传 None 创建可配置模型 |
| model_provider      | str 或 None           | None       | 单独指定提供商。当 model 无法自动推断时使用                |
| configurable_fields | "any" 或 list 或 None | None       | 可运行时修改的字段。None=固定模型，"any"=全部可配          |
| config_prefix       | str 或 None           | None       | 多模型场景下的配置键前缀                                   |
| temperature         | float                 | 因模型而异 | 控制随机性，0~2。0=确定，2=最大创造性                      |
| max_tokens          | int                   | 模型上限   | 输出最大 Token 数                                          |
| timeout             | int/float 或 None     | None       | 请求超时秒数                                               |
| max_retries         | int                   | 因模型而异 | 失败重试次数                                               |
| base_url            | str 或 None           | 官方地址   | 自定义 API 端点                                            |
| rate_limiter        | BaseRateLimiter       | 无         | 速率限制器                                                 |
| top_p               | float 或 None         | 因模型而异 | 核采样参数，0~1                                            |
| stop                | list[str]             | 无         | 停止序列                                                   |

------

### BaseChatModel 方法

| 方法                                     | 说明                  | 返回值                          |
| :--------------------------------------- | :-------------------- | :------------------------------ |
| invoke(input, config=None, **kwargs)     | 同步调用模型          | AIMessage                       |
| ainvoke(input, config=None, **kwargs)    | 异步调用模型          | AIMessage                       |
| stream(input, config=None, **kwargs)     | 同步流式调用          | Iterator[AIMessageChunk]        |
| astream(input, config=None, **kwargs)    | 异步流式调用          | AsyncIterator[AIMessageChunk]   |
| batch(inputs, config=None, **kwargs)     | 批量调用              | list[AIMessage]                 |
| bind_tools(tools, **kwargs)              | 绑定工具列表          | Runnable[input, AIMessage]      |
| with_structured_output(schema, **kwargs) | 绑定结构化输出 Schema | Runnable[input, BaseModel/dict] |
| bind(**kwargs)                           | 绑定运行时参数        | Runnable                        |

------

### 支持的模型提供商速查

| provider 名  | 安装包                 | 示例 model 值                        |
| :----------- | :--------------------- | :----------------------------------- |
| openai       | langchain-deepseek     | deepseek:deepseek-v4-flash           |
| anthropic    | langchain-anthropic    | anthropic:claude-sonnet-4-5-20250929 |
| deepseek     | langchain-deepseek     | deepseek:deepseek-chat               |
| google_genai | langchain-google-genai | google_genai:gemini-2.5-flash        |
| ollama       | langchain-ollama       | ollama:llama3.2                      |
| groq         | langchain-groq         | groq:llama-3.3-70b                   |
| xai          | langchain-xai          | xai:grok-3                           |
| mistralai    | langchain-mistralai    | mistralai:mistral-large              |
| openrouter   | langchain-openrouter   | openrouter:openai/gpt-4o             |
| perplexity   | langchain-perplexity   | perplexity:sonar-pro                 |

------

### 常用用法示例

### 实例

**from** langchain.chat_models **import** init_chat_model

    # 固定模型
model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
response = model.invoke("你好")

    # 可配置模型
model = init_chat_model(configurable_fields=("model", "temperature"))
response = model.invoke("你好", config={
  "configurable": {"model": "deepseek:deepseek-v4-flash", "temperature": 0.3}
})

    # 绑定工具
model_with_tools = model.bind_tools([my_tool])
response = model_with_tools.invoke("查询天气")

    # 结构化输出
model_structured = model.with_structured_output(MySchema)
result = model_structured.invoke("提取信息")



## LangChain Agent API

------

### create_agent() 完整参数

| 参数             | 类型                                   | 默认值     | 说明                                           |
| :--------------- | :------------------------------------- | :--------- | :--------------------------------------------- |
| model            | str 或 BaseChatModel                   | 无（必填） | 语言模型                                       |
| tools            | Sequence 或 None                       | None       | 工具列表。支持 @tool 函数、Pydantic 模型、dict |
| system_prompt    | str 或 SystemMessage 或 None           | None       | 系统提示词                                     |
| middleware       | Sequence[AgentMiddleware]              | ()         | 中间件列表                                     |
| response_format  | ResponseFormat 或 type 或 dict 或 None | None       | 结构化输出配置                                 |
| state_schema     | type[AgentState] 或 None               | None       | 自定义状态结构                                 |
| context_schema   | type 或 None                           | None       | 运行时上下文结构                               |
| checkpointer     | Checkpointer 或 None                   | None       | 对话持久化                                     |
| store            | BaseStore 或 None                      | None       | 跨会话存储                                     |
| interrupt_before | list[str] 或 None                      | None       | 在这些节点前暂停                               |
| interrupt_after  | list[str] 或 None                      | None       | 在这些节点后暂停                               |
| debug            | bool                                   | False      | 是否输出详细日志                               |
| name             | str 或 None                            | None       | Agent 名称                                     |
| cache            | BaseCache 或 None                      | None       | 缓存配置                                       |

------

### CompiledStateGraph 方法

| 方法                                               | 说明                   |
| :------------------------------------------------- | :--------------------- |
| invoke(input, config=None)                         | 同步运行，返回最终状态 |
| ainvoke(input, config=None)                        | 异步运行，返回最终状态 |
| stream(input, config=None, stream_mode="updates")  | 同步流式运行           |
| astream(input, config=None, stream_mode="updates") | 异步流式运行           |
| get_state(config)                                  | 获取当前状态快照       |
| update_state(config, values)                       | 手动更新状态           |

------

### AgentState 结构

| 字段                | 类型                                | 是否必填 | 说明                             |
| :------------------ | :---------------------------------- | :------- | :------------------------------- |
| messages            | list[AnyMessage]                    | 是       | 消息历史（add_messages reducer） |
| jump_to             | "tools" 或 "model" 或 "end" 或 None | 否       | 流程跳转（ephemeral）            |
| structured_response | Any                                 | 否       | 结构化输出结果（OmitFromInput）  |

------

### 常用用法示例

### 实例

**from** langchain.agents **import** create_agent

    # 基本用法
agent = create_agent(model="deepseek:deepseek-v4-flash", tools=[tool1, tool2])
result = agent.invoke({"messages": [HumanMessage(content="你好")]})

    # 完整配置
agent = create_agent(
  model="deepseek:deepseek-v4-flash",
  tools=[tool1, tool2],
  system_prompt="你是助手。",
  middleware=[my_middleware],
  response_format=MySchema,
  checkpointer=checkpointer,
  store=store,
  name="my_agent",
)

    # 流式运行
**for** chunk **in** agent.stream(inputs, stream_mode="updates"):
  **print**(chunk)

    # 获取状态
state = agent.get_state({"configurable": {"thread_id": "1"}})



## LangChain Messages API

------

### 所有消息类型

| 类型           | role      | type   | 关键属性                            | 说明                  |
| :------------- | :-------- | :----- | :---------------------------------- | :-------------------- |
| HumanMessage   | user      | human  | content                             | 用户消息              |
| AIMessage      | assistant | ai     | content, tool_calls, usage_metadata | AI 回复               |
| AIMessageChunk | assistant | ai     | content（增量）                     | 流式输出的 Token 片段 |
| SystemMessage  | system    | system | content                             | 系统指令              |
| ToolMessage    | tool      | tool   | content, tool_call_id, name         | 工具执行结果          |
| RemoveMessage  | -         | remove | id                                  | 删除指定消息          |

------

### ContentBlock 类型（多模态内容）

| 类型                  | 说明                  | 使用场景             |
| :-------------------- | :-------------------- | :------------------- |
| PlainTextContentBlock | 纯文本                | 普通文字             |
| ImageContentBlock     | 图片（base64 或 URL） | 多模态模型的图片输入 |
| AudioContentBlock     | 音频                  | 语音输入             |
| VideoContentBlock     | 视频                  | 视频输入             |
| FileContentBlock      | 文件                  | 文档输入             |
| ToolCall              | 工具调用请求          | 模型请求调用工具     |
| ServerToolCall        | 服务端工具调用        | 内置/MCP 工具调用    |

------

### 常用辅助函数

| 函数            | 说明                         | 签名                                                         |
| :-------------- | :--------------------------- | :----------------------------------------------------------- |
| trim_messages() | 裁剪消息历史以适应上下文窗口 | trim_messages(messages, *, max_tokens, strategy, token_counter, include_system, start_on) |

------

### 常用用法示例

### 实例

**from** langchain.messages **import** (
  HumanMessage, AIMessage, SystemMessage, ToolMessage, trim_messages
)

    # 创建消息
human = HumanMessage(content="你好")
system = SystemMessage(content="你是助手")
ai = AIMessage(content="你好！有什么可以帮你的？")
tool = ToolMessage(content="结果", tool_call_id="call_1", name="my_tool")

    # 快捷方式
msg1 = ("user", "你好")      # 元组
msg2 = {"role": "user", "content": "你好"} # 字典

    # 消息属性
**print**(human.type)  # human
**print**(human.content) # 你好
**print**(ai.tool_calls) # [] 或 [ToolCall]

    # 裁剪消息
trimmed = trim_messages(
  messages, max_tokens=1000, strategy="last",
  token_counter=model, include_system=True,
)



## LangChain Tools API

------

### @tool 装饰器

| 参数          | 类型              | 默认值         | 说明                                   |
| :------------ | :---------------- | :------------- | :------------------------------------- |
| args_schema   | BaseModel 或 None | None           | 参数校验模型。不传则从函数签名自动生成 |
| return_direct | bool              | False          | 是否直接返回（跳过模型再思考）         |
| name          | str 或 None       | 函数名         | 工具名称                               |
| description   | str 或 None       | 函数文档字符串 | 工具描述                               |

------

### BaseTool 主要属性与方法

| 属性/方法      | 说明                       |
| :------------- | :------------------------- |
| name           | 工具名称（字符串）         |
| description    | 工具描述（字符串）         |
| args_schema    | 参数 Pydantic 模型         |
| return_direct  | 是否直接返回（bool）       |
| invoke(input)  | 调用工具，input 是参数字典 |
| ainvoke(input) | 异步调用工具               |

------

### 依赖注入标记

| 标记               | 用途            | 用法                                  |
| :----------------- | :-------------- | :------------------------------------ |
| InjectedState      | 注入 Agent 状态 | Annotated[dict, InjectedState]        |
| InjectedStore      | 注入跨会话存储  | Annotated[BaseStore, InjectedStore()] |
| InjectedToolCallId | 注入工具调用 ID | Annotated[str, InjectedToolCallId]    |
| InjectedToolArg    | 通用注入标记    | Annotated[T, InjectedToolArg]         |

------

### 常用用法示例

### 实例

**from** langchain.tools **import** tool, InjectedState, InjectedStore, ToolException
**from** typing **import** Annotated
**from** langgraph.store.base **import** BaseStore

    # 基本工具
@tool
**def** my_tool(param: str) -> str:
  """工具描述"""
  **return** f"结果: {param}"

    # 带参数校验
**from** pydantic **import** BaseModel, Field

**class** MyInput(BaseModel):
  param: str = Field(description="参数说明", min_length=1)

@tool(args_schema=MyInput)
**def** validated_tool(param: str) -> str:
  **return** param

    # 直接返回
@tool(return_direct=True)
**def** query_tool(query: str) -> str:
  **return** f"结果: {query}"

    # 注入状态
@tool
**def** stateful_tool(
  param: str,
  state: Annotated[dict, InjectedState],
) -> str:
  **return** f"消息数: {len(state.get('messages', []))}"

    # 注入 Store
@tool
**def** store_tool(
  key: str,
  store: Annotated[BaseStore, InjectedStore()],
) -> str:
  item = store.get(("ns",), key)
  **return** str(item.value **if** item **else** "无")

    # 异常处理
@tool
**def** safe_tool(param: int) -> str:
  **if** param < 0:
    **raise** ToolException(f"参数必须为正数: {param}")
  **return** f"OK: {param}"

## LangChain Middleware API

------

### AgentMiddleware 基类钩子方法

| 方法             | 签名                                            | 执行频率     | 返回值          |
| :--------------- | :---------------------------------------------- | :----------- | :-------------- |
| before_agent     | (state, runtime) → dict 或 None                 | 1 次         | 状态更新或 None |
| abefore_agent    | (state, runtime) → dict 或 None                 | 1 次         | 状态更新或 None |
| before_model     | (state, runtime) → dict 或 None                 | 每次循环     | 状态更新或 None |
| abefore_model    | (state, runtime) → dict 或 None                 | 每次循环     | 状态更新或 None |
| wrap_model_call  | (request, handler) → ModelResponse 或 AIMessage | 每次循环     | 模型调用结果    |
| awrap_model_call | (request, handler) → ModelResponse 或 AIMessage | 每次循环     | 模型调用结果    |
| after_model      | (state, runtime) → dict 或 None                 | 每次循环     | 状态更新或 None |
| aafter_model     | (state, runtime) → dict 或 None                 | 每次循环     | 状态更新或 None |
| wrap_tool_call   | (request, handler) → ToolMessage 或 Command     | 每次工具调用 | 工具执行结果    |
| awrap_tool_call  | (request, handler) → ToolMessage 或 Command     | 每次工具调用 | 工具执行结果    |
| after_agent      | (state, runtime) → dict 或 None                 | 1 次         | 状态更新或 None |
| aafter_agent     | (state, runtime) → dict 或 None                 | 1 次         | 状态更新或 None |

------

### 装饰器一览

| 装饰器           | 参数                                   | 说明               |
| :--------------- | :------------------------------------- | :----------------- |
| @before_agent    | state_schema, tools, can_jump_to, name | Agent 开始前执行   |
| @after_agent     | state_schema, tools, can_jump_to, name | Agent 结束后执行   |
| @before_model    | state_schema, tools, can_jump_to, name | 模型调用前执行     |
| @after_model     | state_schema, tools, can_jump_to, name | 模型调用后执行     |
| @wrap_model_call | state_schema, tools, name              | 拦截模型执行       |
| @wrap_tool_call  | tools, name                            | 拦截工具执行       |
| @dynamic_prompt  | 无                                     | 动态生成系统提示词 |

------

### ModelRequest 关键属性

| 属性            | 类型                   | 说明                            |
| :-------------- | :--------------------- | :------------------------------ |
| model           | BaseChatModel          | 当前模型                        |
| messages        | list[AnyMessage]       | 消息列表（不含 system message） |
| system_message  | SystemMessage 或 None  | 当前系统消息                    |
| tools           | list                   | 可用工具列表                    |
| tool_choice     | Any                    | 工具选择策略                    |
| response_format | ResponseFormat 或 None | 结构化输出格式                  |
| state           | dict                   | Agent 当前状态                  |
| runtime         | Runtime                | 运行时上下文                    |

ModelRequest 的 override() 方法用于创建带修改的新请求：

### 实例

new_request = request.override(
  model=different_model,
  system_message=SystemMessage(content="新提示"),
  tools=[new_tool],
)

------

### ModelResponse 结构

| 属性                | 类型              | 说明               |
| :------------------ | :---------------- | :----------------- |
| result              | list[BaseMessage] | 模型返回的消息列表 |
| structured_response | Any 或 None       | 结构化输出结果     |

------

### 特殊返回值 ExtendedModelResponse

| 属性           | 说明                                  |
| :------------- | :------------------------------------ |
| model_response | 底层的 ModelResponse                  |
| command        | 可选的 Command 对象，用于额外状态更新 |

## LangChain 配置与错误类

------

### RunnableConfig 配置项

| 字段            | 类型                      | 说明                                            |
| :-------------- | :------------------------ | :---------------------------------------------- |
| configurable    | dict                      | 运行时配置。最常用：thread_id 用于 Checkpointer |
| recursion_limit | int                       | 最大递归深度（默认 9999）                       |
| metadata        | dict                      | 附加元数据                                      |
| tags            | list[str]                 | 标签列表，用于过滤和分组追踪                    |
| callbacks       | list[BaseCallbackHandler] | 回调处理器                                      |

### 实例

config = {
  "configurable": {"thread_id": "user-001"},
  "metadata": {"source": "web"},
  "tags": ["production", "chat"],
}
result = agent.invoke(inputs, config=config)

------

### Checkpointer 实现类

| 类            | 导入路径                      | 持久化 |
| :------------ | :---------------------------- | :----- |
| InMemorySaver | langgraph.checkpoint.memory   | 否     |
| SqliteSaver   | langgraph.checkpoint.sqlite   | 是     |
| PostgresSaver | langgraph.checkpoint.postgres | 是     |

### 实例

    # 内存
**from** langgraph.checkpoint.memory **import** InMemorySaver
checkpointer = InMemorySaver()

    # SQLite
**from** langgraph.checkpoint.sqlite **import** SqliteSaver
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

    # PostgreSQL
    # from langgraph.checkpoint.postgres import PostgresSaver
    # checkpointer = PostgresSaver.from_conn_string("postgresql://...")

------

### Store 实现类

| 类            | 导入路径                 | 持久化 |
| :------------ | :----------------------- | :----- |
| InMemoryStore | langgraph.store.memory   | 否     |
| PostgresStore | langgraph.store.postgres | 是     |

### 实例

**from** langgraph.store.memory **import** InMemoryStore

store = InMemoryStore()
store.put(("namespace",), "key", {"data": "value"})
item = store.get(("namespace",), "key")
items = store.search(("namespace",))
store.delete(("namespace",), "key")

------

### 常见异常类

| 异常                            | 来源                               | 说明                                              |
| :------------------------------ | :--------------------------------- | :------------------------------------------------ |
| ToolException                   | langchain.tools                    | 工具内部异常。Agent 可捕获并重新决策              |
| ImportError                     | Python 内置                        | 缺少依赖包。错误信息会提示安装命令                |
| ValueError                      | Python 内置                        | 参数验证失败或配置错误                            |
| NotImplementedError             | Python 内置                        | Middleware 方法未实现（如只定义了同步但异步调用） |
| StructuredOutputError           | langchain.agents.structured_output | 结构化输出相关错误（格式不符、多个输出等）        |
| StructuredOutputValidationError | langchain.agents.structured_output | 结构化输出校验失败                                |
| MultipleStructuredOutputsError  | langchain.agents.structured_output | 模型返回了多个结构化输出                          |
| TimeoutError                    | Python 内置 / 各 SDK               | 请求超时                                          |

------

### LaunchDarkly 配置检查清单

| 检查项         | 命令/方法                                                    |
| :------------- | :----------------------------------------------------------- |
| Python 版本    | python --version（需要 3.10+）                               |
| langchain 版本 | python -c "import langchain; print(langchain.__version__)"   |
| 依赖安装       | pip list \| grep langchain                                   |
| API Key 配置   | python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('DEEPSEEK_API_KEY', 'NOT SET')[:10])" |
| 模型连通性     | 使用 init_chat_model() 发送简单请求测试                      |

> 本教程的 API 参考基于 LangChain v1.3.0。由于 LangChain 仍在快速发展，建议在使用时查阅最新的官方文档以获取最新 API 信息。
---

## 实战笔记：wrap 中间件 sync/async 双版本（阶段 5 踩坑，2026-08）

### 现象
Web 端（astream/ainvoke）流式卡死，后端报：
`NotImplementedError: awrap_model_call is not available`

### 根因链（三层坑）
1. `@wrap_model_call` 装饰**同步函数** → 异步上下文（astream/ainvoke）不可用
   （before/after 钩子有自动 sync→async 包装，**wrap 没有**）
2. 改 async 后 `return handler(...)` 漏 `await` → `'coroutine' object has no attribute 'result'`
3. 只写 async 版 → 同步上下文（invoke）反向报错

### 结论
**wrap 中间件必须同时实现 wrap_model_call + awrap_model_call**（AgentMiddleware 子类，公共逻辑抽方法）。
`@dynamic_prompt` 底层也是 wrap，同样处理。

### 教训
- 中间件改动必须**双端验证**（CLI 同步 + Web 异步），单端通过不算数
- 报错信息里的 NotImplementedError 提示了三种解法：子类化 / async 函数 / 同步调用

## 实战笔记：记忆层 async 四连坑（阶段 6，2026-08）

1. 同步 SqliteSaver 不支持异步方法（aget_tuple → NotImplementedError）→ 用 AsyncSqliteSaver
2. 直接 aiosqlite.connect() 构造有 loop 生命周期问题（Event loop is closed）→ from_conn_string + async with
3. AsyncSqliteStore 的同步方法 put/get 在主事件循环禁用（InvalidStateError）→ 工具必须 async def + await aput/aget
4. BaseStore 双体系：langchain_core.stores.BaseStore ≠ langgraph.store.base.BaseStore，
   AsyncSqliteStore 继承的是 langgraph 的，类型标注用错 → Pydantic 校验失败

规律：langgraph 持久化组件（checkpointer/store）都是同步/异步双实现，
Web 异步场景必须 Async 类 + async 方法 + async 上下文管理（memory_ctx 模式）。

## 实战笔记：会话管理与标题生成修复（阶段 6.5，2026-08）

### 三个修复
1. **Vite 代理漏配**：新增后端接口（/threads）后 vite.config.js 没加代理 → 前端请求打到 Vite 自身返回 404。教训：加接口必查代理；vite.config.js 改动必须重启 Vite。
2. **asyncio.create_task 无引用被 GC**：后台任务（标题生成）"时灵时不灵"。必须保存 task 引用（模块级集合 + done_callback 清理）。这是 asyncio 官方文档明确警告的经典坑。
3. **checkpoint["ts"]**：LangGraph 写 checkpoint 时自动生成的时间戳（ISO 8601），不是自己传的；是"最近对话时间"排序的可靠键（step 是执行步数，不代表时间）。

### 会话管理能力（用户需求迭代）
- 排序：按 checkpoint.ts 降序（最近聊的排最前）
- 删除：DELETE /threads/{tid}（adelete_thread + Store 删标题；用户进度独立 namespace 不受影响）
- 重命名：POST /threads/{tid}/rename（Store 标题）
- 标题自动总结：后台 LLM 总结前几条消息 → ≤12 字存 Store（namespace ("threads", tid)），仅首次生成
- 历史消息保留思考过程：/threads/{tid}/messages 返回 reasoning_content，前端映射显示
