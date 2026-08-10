"""阶段 1 验证 CLI：模型调用（invoke + stream）。

用法：
    python -m app.cli invoke "你好，介绍一下自己"
    python -m app.cli stream "用一句话介绍 FastAPI"
"""
import sys

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


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "invoke"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "你好，用一句话介绍你自己"
    if cmd == "stream":
        demo_stream(prompt)
    else:
        demo_invoke(prompt)


if __name__ == "__main__":
    main()
