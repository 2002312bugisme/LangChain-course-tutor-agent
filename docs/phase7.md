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

---

## 6. 对话补充：计划功能后续修复记录（用户实测驱动）

### 6.1 Vite 代理 404（第二次踩坑）

- **现象**：点"生成计划"→ "计划生成失败：HTTP 404"
- **根因**：`/plan` 接口新增后未加进 `vite.config.js` 代理白名单（同 `/threads` 坑）
- **修复**：proxy 加 `'/plan'`；**vite.config.js 改动必须重启 Vite**
- **教训固化**：新增后端接口 → 第一件事检查代理白名单

### 6.2 前端请求超时保护

- 浏览器 fetch 无超时，后端异常时可能永久转圈
- 修复：`AbortController` 45s 超时 + 响应格式校验（`Array.isArray(data.topics)`），挂起时显示明确错误而非永久 loading

### 6.3 学习计划实时生成（非硬编码）

- 用户反馈：示例按钮固定文案=硬编码，不能按输入内容规划
- 修复：输入框旁新增"📊 计划"按钮，规划主题 = 输入框内容；示例按钮改为"填入输入框"引导

### 6.4 计划接入会话系统

- 用户反馈：生成计划后侧边栏不出现会话
- 根因：`/plan` 独立接口不写 checkpointer
- 修复：`/plan` 把 HumanMessage + AIMessage（plan 存 `additional_kwargs["plan"]`）写入当前 thread + 异步标题生成；`/threads/{tid}/messages` 返回 plan 数据；前端历史回看渲染卡片（实时/历史共用模板 + `_planning` 状态标记）

### 6.5 会话标题实时更新

- 用户反馈：标题自动生成要手动刷新才出现
- 根因：标题是后台 `asyncio.create_task` 异步生成，前端只在请求返回时刷新一次
- 修复：前端 `waitTitle()` 短时轮询（1.5s × 最多 8 次 ≈12s），标题出现即停；已命名会话立即返回不无谓轮询

### 6.6 多轮工具循环渲染修复（重大）

- **现象**：思考+回复只显示第一轮；"刷新页面才能看到第二轮及以后的内容"
- **三个根因叠加**：
  1. 后端 `stream_mode="messages"` 把 tools 节点的 **ToolMessage 当 token 发出** → 工具结果混入回复文本。修复：`metadata.get("langgraph_node") != "model"` 过滤
  2. 前端 `attachRender` 的 **`!m._html` 短路** → 第一次 token 渲染后不再重新 parse → 打字机卡死（v-html 永远显示第一个 token 的结果）。修复：去掉短路，每次 token 重新渲染
  3. 多轮内容合并一条消息 → 思考区 160px 截断。修复：**工具调用后的新一轮内容开新 AI 消息**（与历史接口的多条 AIMessage 一致）
- **验证**：SSE 事件流 3 轮思考/回复干净无工具结果混入
