"""阶段 1-2 验证 CLI：模型调用（invoke/stream）+ 消息层（chat/trim）。

用法：
    python -m app.cli invoke "你好，介绍一下自己"
    python -m app.cli stream "用一句话介绍 FastAPI"
    python -m app.cli chat            # 多轮对话（引用前文验证）
    python -m app.cli trim            # trim_messages 裁剪演示
    python -m app.cli construct       # 消息快捷构造演示
"""
import sys

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.messages import (
    SYSTEM_PROMPT,
    build_conversation,
    demo_quick_construct,
    trim_history,
)
from app.model import get_model


def demo_invoke(prompt: str) -> None:
    print(f"[invoke] 输入: {prompt!r}\n")
    msg = get_model().invoke(prompt)
    print(f"[invoke] 回复: {msg.content}\n")
    print(f"[invoke] 元数据: model={msg.response_metadata.get('model_name')}, "
          f"finish_reason={msg.response_metadata.get('finish_reason')}")
    if msg.usage_metadata:
        u = msg.usage_metadata
        print(f"[invoke] token 用量: 输入={u['input_tokens']}, 输出={u['output_tokens']}, 总计={u['total_tokens']}")


def demo_stream(prompt: str) -> None:
    print(f"[stream] 输入: {prompt!r}\n[stream] 输出: ", end="", flush=True)
    for chunk in get_model().stream(prompt):
        # chunk 是 AIMessageChunk，content 是增量片段
        print(chunk.content, end="", flush=True)
    print("\n[stream] 完成")


def _print_ai(msg) -> None:
    """打印 AIMessage：思考过程（若有）+ 正式回复。

    deepseek-v4-flash 是推理模型：思考内容在 additional_kwargs["reasoning_content"]，
    正式回复在 content。两者都打印，顺便演示 reasoning_tokens 的计费。
    """
    reasoning = msg.additional_kwargs.get("reasoning_content")
    if reasoning:
        print(f"      🤔 思考: {reasoning}")
    print(f"      💬 回复: {msg.content}")


def demo_chat() -> None:
    """多轮对话演示：模型必须"记住"前文才能答对。

    消息链：system + 用户自报家门 + AI 回应 + 追问名字
    如果模型回答"小明"，说明 system+历史 正确传递并被引用。
    """
    print("=" * 50)
    print("[chat] 演示：模型引用前文（消息历史传递）")
    print("=" * 50)

    history = [
        HumanMessage(content="我叫小明，是个编程零基础的新手，想学 Python。"),
    ]
    first = get_model().invoke(build_conversation(history))
    print(f"[chat] 第1轮 user: 我叫小明，是个编程零基础的新手")
    _print_ai(first)
    print()

    # 把第 1 轮 AI 回复加入历史，再问第 2 轮
    history.append(first)
    history.append(HumanMessage(content="根据我刚才说的，你建议我先学什么？另外我叫什么名字？"))

    second = get_model().invoke(build_conversation(history))
    print(f"[chat] 第2轮 user: 根据我刚才说的，你建议我先学什么？另外我叫什么名字？")
    _print_ai(second)

    if "小明" in second.content:
        print("\n[chat] ✅ 模型引用了前文（记得用户叫小明）")
    else:
        print("\n[chat] ⚠️ 模型未提及名字——检查消息传递是否有误")


def demo_trim() -> None:
    """trim_messages 演示：模拟 20 轮对话后裁剪。"""
    print("=" * 50)
    print("[trim] 演示：trim_messages 上下文防膨胀")
    print("=" * 50)

    # 构造 20 轮模拟对话（system + 40 条消息）
    history = [HumanMessage(content="开始学习，请记住编号。")]
    for i in range(20):
        history.append(AIMessage(content=f"这是第 {i + 1} 轮的回复，包含一些课程相关的说明内容，用于模拟长对话中的历史消息。"))
        history.append(HumanMessage(content=f"继续，这是第 {i + 2} 轮的问题，请继续回答。"))

    messages = build_conversation(history)
    token_counter = get_model()
    # 用模型 tokenizer 统计（get_num_tokens 按文本近似计数）
    total = sum(token_counter.get_num_tokens(m.content) for m in messages)
    print(f"[trim] 裁剪前：{len(messages)} 条消息，约 {total} tokens")

    trimmed = trim_history(messages, max_tokens=120)
    after_total = sum(token_counter.get_num_tokens(m.content) for m in trimmed)
    print(f"[trim] 裁剪后：{len(trimmed)} 条消息，约 {after_total} tokens")
    print(f"[trim] 首条: {type(trimmed[0]).__name__}: {str(trimmed[0].content)[:30]}...")
    print(f"[trim] 末条: {type(trimmed[-1]).__name__}: {str(trimmed[-1].content)[:30]}...")
    print(f"[trim] 是否保留 system: {isinstance(trimmed[0], SystemMessage)}")
    print(f"[trim] 是否以 human 开头: {trimmed[0].type == 'human' or isinstance(trimmed[0], SystemMessage) and trimmed[1].type == 'human'}")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "invoke"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "你好，用一句话介绍你自己"
    if cmd == "stream":
        demo_stream(prompt)
    elif cmd == "chat":
        demo_chat()
    elif cmd == "trim":
        demo_trim()
    elif cmd == "construct":
        demo_quick_construct()
    else:
        demo_invoke(prompt)


if __name__ == "__main__":
    main()
