# 阶段 2：消息层（说明文档）

> 状态：✅ 已完成并验证（真实 API 调用）
> 对应设计文档：DESIGN.md 阶段 2
> 验证时间：2026-08

## 1. 目标

掌握消息层的正确用法：四大消息构造、快捷方式、多轮对话历史传递、`trim_messages` 上下文防膨胀。

## 2. 搭建步骤

1. 新建 `app/messages.py`：消息构造 + trim 封装
2. 扩展 `app/cli.py`：新增 `chat`（多轮对话）、`trim`（裁剪演示）、`construct`（快捷构造演示）三个子命令
3. 修复两个实战坑（见第 5 节）：推理模型 max_tokens 预算、deepseek token 计数未实现
4. 修改 `.env`：`MAX_TOKENS` 200 → 1000

## 3. 目录变更

```
app/
├── messages.py   # 新增：消息层工具（构造/快捷方式/trim）
├── cli.py        # 修改：新增 chat/trim/construct 子命令
└── config.py     # 未变
.env              # 修改：MAX_TOKENS 1000（原因见坑1）
```

## 4. 新增文件详解

### 4.1 `app/messages.py` —— 消息层核心

```python
SYSTEM_PROMPT = "你是编程学习助手，回答简洁专业，用中文回答。"

def demo_quick_construct():  # 四种等价构造演示
    msg1 = HumanMessage(content="你好")        # 标准
    msg2 = ("user", "你好")                    # 元组
    msg3 = {"role": "user", "content": "你好"} # 字典

def build_conversation(history, system_prompt=SYSTEM_PROMPT):
    # system 永远排最前：API 要求 messages[0] 是 system，
    # 模型"最先看到"它（角色设定生效的前提）
    return [SystemMessage(content=system_prompt)] + history

def _count_tokens(messages):  # 自定义计数回调（见坑 2）
    return sum(get_model().get_num_tokens(m.content) for m in messages)

def trim_history(messages, max_tokens=200, strategy="last"):
    return trim_messages(
        messages, max_tokens=max_tokens, strategy=strategy,
        token_counter=_count_tokens,          # 绕过 deepseek 未实现的列表计数
        include_system=True,                  # 角色设定永远保留
        start_on="human",                     # 避免孤立 AI 回复开头
    )
```

### 4.2 `cli.py` 新增命令

- `chat`：两轮对话——用户自报家门 → 追问名字。**验证消息历史传递**（模型记得"小明"才算通过）
- `trim`：模拟 20 轮对话（42 条消息）→ 裁剪 → 对比 token 前后
- `construct`：四种消息构造方式演示
- 新增 `_print_ai()`：同时打印推理模型的思考过程（`additional_kwargs["reasoning_content"]`）和正式回复（`content`）

## 5. 实战踩坑（本阶段最重要的学习素材）

### 坑 1：推理模型的 max_tokens 预算 —— 思考把回复吃掉了

**现象**：第 2 轮对话回复为空字符串；`finish_reason=length`；`usage.output_token_details.reasoning=200`

**原因**：`deepseek-v4-flash` 是**推理模型**——正式回复前先输出思考过程（`reasoning_content`），**思考 token 与回复 token 共享 max_tokens 预算**。`MAX_TOKENS=200` 时模型思考完（200 tokens）就撞到上限，正式回复一个字都没输出。

**修复**：`.env` 中 `MAX_TOKENS` 提到 1000；CLI 里把思考过程也显示出来（`🤔 思考` / `💬 回复`）。

**启示（对应手册 0.2/1.3）**：
- 推理模型调参时，`max_tokens` 必须为"思考 + 回复"留足预算
- `reasoning_tokens` 同样计费（这就是 AIMessage 里 `output_token_details.reasoning` 字段的真实含义）
- 流式场景下思考内容先于回复到达——前端可以做"先灰字显示思考，再显示正式回复"（阶段 4 实现）

### 坑 2：deepseek 未实现消息列表 token 计数

**现象**：`trim_messages` 报 `NotImplementedError: get_num_tokens_from_messages() is not presently implemented for model deepseek-v4-flash`

**原因**：trim_messages 默认对"消息列表"计数（`get_num_tokens_from_messages`），deepseek 集成只实现了单文本计数（`get_num_tokens`）。

**修复**：`token_counter` 参数传自定义回调——逐个消息 content 用 `get_num_tokens` 求和（近似，不含角色开销）。

**启示**：`token_counter` 参数是**可替换策略**（callable），这是 LangChain 常见的扩展点设计——任何模型都能用自定义计数接入 trim。

## 6. 验证记录

```bash
python -m app.cli chat
# ✅ 第2轮: 🤔 思考: 我们根据对话，回答用户问题。用户叫小明...
#        💬 回复: 你叫小明，零基础学 Python，**建议第一步先安装 Python...**
#        ✅ 模型引用了前文（记得用户叫小明）

python -m app.cli trim
# ✅ 裁剪前：42 条消息，约 1297 tokens
# ✅ 裁剪后：4 条消息，约 114 tokens
# ✅ 首条 SystemMessage 保留，以 human 开头，末条为最新消息

python -m app.cli construct
# ✅ 四种构造方式语义等价
```

## 7. 测试用例表

| 编号 | 前置条件 | 操作步骤 | 预期结果 |
|---|---|---|---|
| TC-05 | .env 正常 | `python -m app.cli chat` | 第 2 轮回复提及"小明"；输出含 🤔 思考 和 💬 回复 两段 |
| TC-06 | 同上 | `python -m app.cli trim` | 裁剪前约 1297 tokens → 裁剪后约 114 tokens；首条为 SystemMessage；第二条为 human |
| TC-07 | 同上 | `python -m app.cli construct` | 打印四种构造方式，msg1 为 HumanMessage |
| TC-08 | 同上 | `python -m app.cli invoke "1+1=?"` | 正常回复（回归：阶段 1 功能未破坏） |

## 7.5 人工测试记录（2026-08 用户实测，全部通过 ✅）

- **TC-05** chat：第 1/2 轮均正常输出 🤔 思考 + 💬 回复；第 2 轮回复"你叫小明。根据你的情况，**建议先学 Python 基础语法**…"→ 引用前文 ✅
- **TC-06** trim：裁剪前 42 条约 1297 tokens → 裁剪后 4 条约 114 tokens；首条 SystemMessage、human 开头 ✅
- **TC-07** construct：四种构造方式打印正常 ✅
- **TC-08** invoke 回归：`1+1=?` → `2`，finish_reason=stop，token 输入 87 输出 17 总计 104 ✅

> 用户反馈：流式体验优于 invoke（阶段 4 流式 UI 重点）；提出新需求"交流过程中流式展示模型思考过程"→ 已登记 DESIGN.md

## 8. 知识点映射

| 手册条目 | 落地 |
|---|---|
| 2.1 四大消息 + system 排最前 | build_conversation |
| 2.1 快捷构造 | demo_quick_construct |
| 2.5 trim_messages 全参数 | trim_history（含 end_on 之外的常用参数） |
| 2.7 usage_metadata / reasoning | 坑 1 实战 |
| 1.3 推理类参数 max_tokens | 坑 1（推理模型预算） |
