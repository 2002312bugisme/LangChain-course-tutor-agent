"""中间件集（阶段 5）。

知识点覆盖（agent_api_reference.md 阶段 7）：
1. 类继承方式：LoggingMiddleware（before_agent / after_model / after_agent）
2. 装饰器方式 + can_jump_to 白名单：goodbye_filter（道别提前结束）
3. @dynamic_prompt：动态提示词（时间/轮数注入，优先级高于静态 system_prompt）
4. @wrap_model_call 洋葱模型：思考语言守卫 + 重试（需求备选方案落地）

注意：开发阶段用 print 输出日志；生产应换 logging 模块。
"""
from datetime import datetime

from langchain.agents.middleware import (
    AgentMiddleware,
    before_model,
    dynamic_prompt,
    wrap_model_call,
)
from langchain_core.messages import AIMessage, SystemMessage

from app.prompts import AGENT_SYSTEM_PROMPT


# ============ 1. 类继承方式：日志中间件 ============
class LoggingMiddleware(AgentMiddleware):
    """观察型中间件：三个钩子全部返回 None（不修改状态）。"""

    name = "logging"

    def before_agent(self, state, runtime):
        """Agent 开始前（整个对话只执行一次）。"""
        n = len(state.get("messages", []))
        print(f"[日志] ── Agent 开始，当前 {n} 条消息 ──")
        return None

    def after_model(self, state, runtime):
        """每次模型调用后执行（可能执行多次）。"""
        last = state["messages"][-1]
        n_tool_calls = len(getattr(last, "tool_calls", []) or [])
        print(f"[日志] 模型调用完成：回复 {len(str(last.content))} 字符，"
              f"{'含 ' + str(n_tool_calls) + ' 个工具调用' if n_tool_calls else '无工具调用'}")
        return None

    def after_agent(self, state, runtime):
        """Agent 结束后（整个对话只执行一次）。"""
        print(f"[日志] ── Agent 结束，共 {len(state.get('messages', []))} 条消息 ──")
        return None


# ============ 2. 装饰器方式 + can_jump_to 白名单 ============
@before_model(can_jump_to=["end"])
def goodbye_filter(state, runtime):
    """检测道别语，提前结束 Agent（不浪费模型调用）。

    can_jump_to=["end"] 是白名单声明：不声明则 jump_to 被忽略（安全机制）。
    """
    messages = state.get("messages", [])
    if not messages:
        return None
    last = str(messages[-1].content).strip()
    if last in ("再见", "拜拜", "bye", "谢谢", "结束"):
        return {
            "jump_to": "end",
            "messages": [AIMessage(content="再见！期待下次为你服务。")],
        }
    return None


# ============ 3. @dynamic_prompt：动态提示词 ============
@dynamic_prompt
def personalized_prompt(request) -> str:
    """每次模型调用前生成提示词（覆盖静态 system_prompt）。

    为什么手动拼接 AGENT_SYSTEM_PROMPT：@dynamic_prompt 是"覆盖"而非"合并"，
    且 request 中拿不到原静态提示词——要保留工具指引必须手动引用（见手册 6.2）。
    """
    n = len(request.state.get("messages", []))
    now = datetime.now().strftime("%H:%M")
    return (
        f"{AGENT_SYSTEM_PROMPT}\n\n"
        f"## 本次对话上下文\n"
        f"- 当前时间：{now}（早晚问好可用）\n"
        f"- 这是第 {n // 2 + 1} 轮对话，记得结合前面的对话内容回答"
    )


# ============ 4. @wrap_model_call：洋葱包裹（守卫 + 重试） ============
@wrap_model_call
def guard_and_retry(request, handler):
    """包裹模型调用：① 注入思考语言强约束 ② 失败重试一次。

    洋葱模型：这里是最外层（也是唯一一层），调用 handler(request) 才真正执行模型。
    需求备选方案落地：阶段 3 用提示词约束思考语言是"概率性"的，
    这里每次调用前在 system_message 上追加强约束指令（更强的软控制）。
    """
    # ① 请求改写：追加思考语言强约束
    base = request.system_message.content if request.system_message else AGENT_SYSTEM_PROMPT
    new_request = request.override(
        system_message=SystemMessage(content=f"{base}\n【强制】你的思考过程必须使用中文。")
    )

    # ② 重试一次（网络抖动/限流自愈）
    try:
        return handler(new_request)
    except Exception as e:
        print(f"[wrap] 首次调用失败（{type(e).__name__}），重试一次…")
        return handler(new_request)


# 挂到 Agent 上的中间件列表（顺序即执行顺序）
MIDDLEWARES = [
    LoggingMiddleware(),
    goodbye_filter,
    personalized_prompt,
    guard_and_retry,
]
