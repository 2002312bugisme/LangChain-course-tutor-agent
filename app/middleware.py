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


# ============ 3. 动态提示词（类继承 + sync/async 双版本） ============
class DynamicPromptMiddleware(AgentMiddleware):
    """@dynamic_prompt 的类实现。

    为什么不用 @dynamic_prompt 装饰器：装饰 async 函数只生成 awrap_model_call，
    同步上下文（invoke/stream）会报 NotImplementedError；装饰 sync 函数则反过来。
    wrap 中间件没有自动包装，必须用类同时提供 sync + async 两个版本。
    """

    name = "dynamic_prompt"

    def _make_prompt(self, request) -> str:
        """每次模型调用前生成提示词（覆盖静态 system_prompt）。

        手动拼接 AGENT_SYSTEM_PROMPT：动态提示词是"覆盖"而非"合并"，
        要保留工具指引必须手动引用（见手册 6.2）。
        """
        n = len(request.state.get("messages", []))
        now = datetime.now().strftime("%H:%M")
        return (
            f"{AGENT_SYSTEM_PROMPT}\n\n"
            f"## 本次对话上下文\n"
            f"- 当前时间：{now}（早晚问好可用）\n"
            f"- 这是第 {n // 2 + 1} 轮对话，记得结合前面的对话内容回答"
        )

    def wrap_model_call(self, request, handler):
        """同步版（CLI invoke/stream 用）。"""
        prompt = self._make_prompt(request)
        return handler(request.override(system_message=SystemMessage(content=prompt)))

    async def awrap_model_call(self, request, handler):
        """异步版（Web ainvoke/astream 用）。"""
        prompt = self._make_prompt(request)
        return await handler(request.override(system_message=SystemMessage(content=prompt)))


# ============ 4. wrap 守卫：思考语言强约束 + 重试（sync/async 双版本） ============
class GuardRetryMiddleware(AgentMiddleware):
    """包裹模型调用：① 注入思考语言强约束 ② 失败重试一次。

    ⚠️ 必须同时实现 wrap_model_call（sync）和 awrap_model_call（async）：
    只写一个版本会在另一种上下文抛 NotImplementedError（阶段 5 双踩坑实证）。
    """

    name = "guard_retry"

    def _inject(self, request):
        """在 system_message 上追加思考语言强约束。"""
        base = request.system_message.content if request.system_message else AGENT_SYSTEM_PROMPT
        return request.override(
            system_message=SystemMessage(content=f"{base}\n【强制】你的思考过程必须使用中文。")
        )

    def wrap_model_call(self, request, handler):
        new_request = self._inject(request)
        try:
            return handler(new_request)
        except Exception as e:
            print(f"[wrap] 首次调用失败（{type(e).__name__}），重试一次…")
            return handler(new_request)

    async def awrap_model_call(self, request, handler):
        new_request = self._inject(request)
        try:
            return await handler(new_request)
        except Exception as e:
            print(f"[wrap] 首次调用失败（{type(e).__name__}），重试一次…")
            return await handler(new_request)


# 挂到 Agent 上的中间件列表（顺序即执行顺序，第一个=最外层）
MIDDLEWARES = [
    LoggingMiddleware(),
    goodbye_filter,
    DynamicPromptMiddleware(),
    GuardRetryMiddleware(),
]
