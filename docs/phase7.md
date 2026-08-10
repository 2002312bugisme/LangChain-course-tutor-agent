# 阶段 7：结构化输出（学习计划卡片化）+ 会话导出

> 上一阶段：阶段 6.5（UI 现代化 + Markdown 渲染）✅
> 本阶段新增：`app/schemas.py`、`POST /plan`、`GET /threads/{tid}/export`、前端计划卡片 + 导出按钮

---

## 1. 本阶段解决的两个问题

### 1.1 用户疑问：聊天输出已经规整，还需要结构化输出吗？

**要区分两个层面**：

| | 普通聊天输出 | 结构化输出 |
|---|---|---|
| 模型返回 | 人类可读文本（Markdown） | 符合 schema 的 JSON |
| 消费方 | 人（聊天界面展示） | 程序（渲染卡片 / 存库 / 后续逻辑） |
| 前端拿数据 | 直接展示文本 | `plan.topics[0].name` 取字段 |
| 类比 | 人写的一份报告 | 一份 Excel 表格 / 数据库记录 |

- 聊天"规整"是 **Markdown 渲染**（展示层）的功劳，模型返回的仍是自由文本；
- 学习计划要渲染成**卡片**（步骤、时长、顺序），前端需要结构化数据——这就是结构化输出的场景；
- 一句话：**给人看用文本，给程序用结构化输出**。

### 1.2 新需求：会话导出为 Markdown

每个会话可一键导出 `.md` 文件，格式规整（标题 / 导出时间 / 会话 ID / 用户与助手分节 / 思考过程用引用块）。

---

## 2. 架构与文件

```
app/
├── schemas.py        # 【新增】LearningPlan / Topic Pydantic 模型（结构化输出 schema）
└── main.py           # 【修改】+ POST /plan（结构化输出）、GET /threads/{tid}/export
frontend/src/App.vue  # 【修改】+ 计划卡片渲染、📥 导出按钮、生成计划示例按钮
```

### 2.1 app/schemas.py —— 结构化输出的"契约"

```python
class Topic(BaseModel):
    name: str        # 知识点名称
    order: int       # 学习顺序（从 1 开始）
    minutes: int     # 建议时长（分钟）

class LearningPlan(BaseModel):
    goal: str                       # 学习目标
    level: Literal["入门", "进阶", "高级"]
    total_hours: float              # 总时长
    topics: list[Topic]             # 有序知识点列表
```

**为什么这么设计**：schema 是模型与前端之间的**契约**——模型按它输出 JSON，前端按它渲染，任何一端不匹配都能在解析时立刻暴露（Pydantic 校验），而不是前端解析文本猜结构。

### 2.2 POST /plan —— 结构化输出接口

```python
model = ChatOpenAI(..., extra_body={"thinking": {"type": "disabled"}})
structured_model = model.with_structured_output(LearningPlan, method="function_calling")
result = await structured_model.ainvoke("请为用户规划编程学习路线：...")
return result.model_dump()
```

**两种结构化方式对比**（源码实验结论）：

| 方式 | 原理 | 本项目中 |
|---|---|---|
| `with_structured_output`（本阶段） | 模型实例包装：输出直接转 Pydantic 对象 | ✅ 用于独立 /plan 接口 |
| `create_agent(response_format=...)` | Agent 内建"结构化响应"节点，写入 `state["structured_response"]` | 阶段 5 手册已记录，本阶段不重复 |

### 2.3 ⚠️ deepseek 推理模型的关键坑（实测记录）

| 尝试 | 结果 |
|---|---|
| thinking 开启 + `with_structured_output`（默认 function_calling） | ❌ 400：`Thinking mode does not support this tool_choice` |
| thinking 开启 + `method="json_schema"` | ❌ 400：同上（json_schema 也走 tool_choice） |
| thinking 关闭 + 默认方式 | ❌ 400：`This response_format type is unavailable now` |
| **thinking 关闭 + `method="function_calling"`** | ✅ **成功** |

**结论**：deepseek-v4-flash 推理模型，结构化输出必须：
1. `extra_body={"thinking": {"type": "disabled"}}` 关闭思考；
2. `method="function_calling"`（不能用 json_schema）。

### 2.4 GET /threads/{tid}/export —— 会话导出

从 checkpointer 读会话全部消息 → 拼 Markdown：

```markdown
# 会话标题

> 导出时间：2026-08-10 05:39 UTC
> 会话 ID：`ui-check-1`

---

## 👤 用户

推荐一门 Python 入门课

---

## 🤖 助手

> **思考过程**
>
> 搜索到了 Python 入门课程 py-101...

为你找到一门合适的 Python 入门课...
```

格式要点：标题取 Store 中的 LLM 总结标题（或手动重命名）；思考过程用引用块 `>` 与正文区分。

### 2.5 前端

- **📥 导出按钮**（顶栏右侧）：fetch `/export` → Blob → 浏览器下载 `${title}.md`（前端生成下载，不经过后端磁盘）；
- **计划卡片**（空状态新按钮"📊 生成学习计划"）：调 `/plan` → 返回 JSON → 渲染 goal/难度徽章/总时长/主题有序列表（序号圆点 + 名称 + 小时数）；
- 卡片是"真数据"渲染（v-for topics），不是文本——这就是结构化输出的价值体现。

---

## 3. 测试用例

| 编号 | 操作 | 预期 |
|---|---|---|
| TC-45 | 点击"📊 生成学习计划"示例按钮 | 出现"正在规划学习路线…"，随后渲染计划卡片（goal/难度/总时长/主题列表） |
| TC-46 | 计划卡片主题顺序 | order 从 1 递增，每项显示名称与小时数 |
| TC-47 | 顶栏"📥 导出对话"按钮 | 下载 `会话标题.md` 文件，内容含标题/时间/会话 ID/用户消息/助手消息/思考引用块 |
| TC-48 | 导出的 md 用 Typora/VSCode 预览 | 格式正常：标题层级、引用块、分隔线正确 |
| TC-49 | 无会话时点导出 | 按钮禁用，无报错 |
| TC-50 | 聊天功能回归 | 正常对话 + 流式 + 思考展示不受影响 |

---

## 4. 验证记录

- `POST /plan`：✅ 返回 `goal: 零基础掌握 Python 编程，并转型数据分析方向`，`topics` 8 项（Python 基础语法 900 分钟 → Pandas 900 分钟...）
- `GET /threads/{tid}/export`：✅ 标题/时间/引用块格式正确
- 前端 HMR 无编译错误；后端重启后 /chat/stream 回归正常

## 5. 提交

- `git commit: feat(phase7): 结构化输出学习计划卡片 + 会话导出 Markdown`
