"""消息层：消息构造、快捷方式、历史裁剪。

知识点覆盖（对应 agent_api_reference.md 阶段 2）：
- 四大消息：SystemMessage / HumanMessage / AIMessage
- 快捷构造：元组、字典（Agent 内部自动转换）
- trim_messages：上下文防膨胀（strategy/start_on/include_system/token_counter）
"""
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.messages.base import BaseMessage

from app.model import get_model

# 全局系统提示（阶段 6 会升级为动态 prompt）
SYSTEM_PROMPT = """你是编程学习助手，回答简洁专业，用中文回答。

## 思考语言规定（重要）
- 你的整个思考过程（reasoning）必须使用中文，与用户输入语言保持一致"""


def demo_quick_construct() -> None:
    """演示消息的四种等价构造方式（都最终变成 HumanMessage）。"""
    msg1 = HumanMessage(content="你好")            # 标准构造
    msg2 = ("user", "你好")                        # 元组 (role, content)
    msg3 = ("human", "你好")                       # 元组（human 别名）
    msg4 = {"role": "user", "content": "你好"}     # 字典
    print("[构造演示] msg1:", type(msg1).__name__)
    print("[构造演示] msg2:", type(msg2).__name__, "（元组是懒转换，传给模型时才变消息）")
    print("[构造演示] msg4:", type(msg4).__name__, "（字典同样懒转换）")
    print("[构造演示] 四种方式语义等价：字符串/元组/字典都会被 LangChain 转成 HumanMessage\n")


def build_conversation(
    history: list[BaseMessage] | None = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[BaseMessage]:
    """构造一条完整消息链：SystemMessage 在最前，历史消息跟随。

    为什么 system 放最前：API 要求 messages 数组第一个通常是 system，
    模型按顺序"最先看到"它（遵守程度靠提示工程，见阶段 6）。
    """
    return [SystemMessage(content=system_prompt)] + (history or [])


def _count_tokens(messages: list[BaseMessage]) -> int:
    """自定义 token 计数（trim_messages 的 token_counter 回调）。

    为什么不用模型自带：deepseek-v4-flash 未实现 get_num_tokens_from_messages()
    （对消息列表计数），调用会抛 NotImplementedError。这里逐个消息 content
    用 get_num_tokens（单文本计数，已实现）求和，近似总量（不含角色开销）。
    """
    model = get_model()
    return sum(model.get_num_tokens(m.content) for m in messages)


def trim_history(
    messages: list[BaseMessage],
    max_tokens: int = 200,
    strategy: str = "last",
) -> list[BaseMessage]:
    """裁剪消息历史以适配上下文窗口。

    参数说明：
    - max_tokens: 保留上限（token 数）
    - strategy="last": 保留 system + 最近的对话
    - token_counter=_count_tokens: 自定义计数回调（绕过 deepseek 未实现的
      get_num_tokens_from_messages）
    - include_system=True: 始终保留 SystemMessage（角色设定不能丢）
    - start_on="human": 裁剪后以 human 消息开头（避免孤立 AI 回复开头）
    """
    return trim_messages(
        messages,
        max_tokens=max_tokens,
        strategy=strategy,
        token_counter=_count_tokens,
        include_system=True,
        start_on="human",
    )
