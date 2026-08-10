# 阶段 6：记忆（checkpointer + Store）（说明文档）

> 状态：✅ 已完成并验证（CLI + Web 双端 + 重启持久化）
> 对应设计文档：DESIGN.md 阶段 6
> 验证时间：2026-08

## 1. 目标

两层记忆落地：checkpointer（会话级：thread_id 续聊）+ Store（跨会话：学习进度持久化），SQLite 持久化，Web 端前端支持多会话。

## 2. 搭建步骤

1. `pip install langgraph-checkpoint-sqlite`（含 aiosqlite/sqlite-vec）
2. `app/memory.py`：memory_ctx 常驻上下文（checkpointer + store）
3. `app/tools/memory_tools.py`：save_progress / get_progress（InjectedStore 注入）
4. `app/agent.py`：ensure_agent 异步惰性单例（checkpointer/store 由 ctx 提供）
5. `app/main.py`：FastAPI lifespan 管理记忆连接生命周期；ChatRequest.thread_id
6. `app/cli.py`：agent 命令适配 async 包装
7. `frontend/App.vue`：thread_id 存 localStorage（刷新续聊）+ 新建会话按钮

## 3. 目录/文件变更

```
app/memory.py              # 新增：memory_ctx（AsyncSqliteSaver + AsyncSqliteStore）
app/tools/memory_tools.py  # 新增：进度读写工具（InjectedStore）
app/agent.py               # 重构：ensure_agent（异步惰性单例）
app/main.py                # 修改：lifespan + thread_id 透传
app/cli.py                 # 修改：async 包装
frontend/src/App.vue       # 修改：thread_id + 新建会话
data/agent_memory.db       # 运行时生成（SQLite，已 gitignore *.db）
```

## 4. 核心设计

### 4.1 memory_ctx —— 记忆连接的"常驻上下文"

```python
@asynccontextmanager
async def memory_ctx():
    async with AsyncSqliteSaver.from_conn_string(DB) as cp:
        async with AsyncSqliteStore.from_conn_string(DB) as st:
            yield cp, st
```

**为什么这么设计**：AsyncSqliteSaver 构造需要运行中的事件循环（`__init__` 里 `asyncio.get_running_loop()`），且连接生命周期必须管理。Web 端在 FastAPI lifespan 里进入 ctx（应用运行期间连接保持），CLI 脚本在 asyncio.run 里进入 ctx——一套代码两个场景。

### 4.2 checkpointer 会话续聊（验证结果）

```
第1轮 (thread=s1): 我叫小王，正在学 Vue
第2轮 (thread=s1): 我叫什么名字？→ "你叫小王，正在学 Vue 前端开发" ✅ 记住
第3轮 (thread=other): 我叫什么名字？→ "这是我们第一次对话" ✅ 隔离
重启服务后 (thread=s1): 还记得小王 ✅（SQLite 持久化）
```

### 4.3 Store 跨会话（验证结果）

```
第4轮 (thread=s1): 记录进度 Python 基础语法(入门) → 已记录 ✅
第5轮 (thread=other): 我掌握了哪些主题？→ "Python 基础语法（入门）" ✅ 跨会话
```

**state vs store 数据组织**：state 是扁平字典（会话内），store 是 namespace+key 层级（跨会话），如 `("users", "default", "progress")` + key `"topics"`。

### 4.4 Web 端 thread_id

- 前端：`crypto.randomUUID()` 生成，存 localStorage（刷新续聊）；"新建会话"按钮换新 id + 清空消息
- 后端：`_thread_config(thread_id)` → `config["configurable"]["thread_id"]` 传给 agent 调用
- 无 thread_id 的请求：不启用记忆（config 为空）

## 5. 踩坑记录（本阶段 4 个，全是 async 相关）

1. **同步 SqliteSaver 不支持异步方法**：`aget_tuple` 抛 NotImplementedError → 必须 `AsyncSqliteSaver`（与 wrap 中间件同款：异步环境必须异步实现）
2. **连接构造的 loop 生命周期**：直接 `AsyncSqliteSaver(aiosqlite.connect(...))` 报 "Event loop is closed"（Python 3.13 + aiosqlite 行为）→ 官方 `from_conn_string` + async with
3. **AsyncSqliteStore 同步方法在主事件循环禁用**：工具里 `store.put()` 报 `InvalidStateError: use the asynchronous interface` → 工具改 `async def` + `await store.aput()/aget()`
4. **BaseStore 双体系**：`langchain_core.stores.BaseStore` ≠ `langgraph.store.base.BaseStore`——AsyncSqliteStore 继承的是 **langgraph** 的；工具类型标注用错会 Pydantic 校验失败（"Input should be an instance of BaseStore"）

**规律总结**：langgraph 生态的持久化组件（checkpointer/store）都是"同步/异步双实现"，Web 异步场景必须：Async 类 + async 方法 + async 上下文管理。

## 6. 验证记录

```bash
# CLI 五步验证（asyncio 脚本）：同会话记忆 ✅ 会话隔离 ✅ 记进度 ✅ 跨会话查 ✅
# 重启持久化：新进程查询 → 还记得小王 + Python 基础 ✅
# Web SSE：带 thread_id 两轮 → 第2轮记住"小红" ✅
```

## 7. 测试用例表

| 编号 | 操作 | 预期 |
|---|---|---|
| TC-24 | 浏览器发"我叫小美，正在学 Vue" | 正常流式回复 |
| TC-25 | 浏览器**刷新页面**后发"我叫什么名字？" | 回复"小美"（localStorage thread_id 续聊） |
| TC-26 | 点"新建会话"后发"我叫什么名字？" | 回复"不知道/第一次对话"（新 thread_id 隔离） |
| TC-27 | 发"记录一下：我学完了 Python 基础，入门水平" | 回复"已记录" |
| TC-28 | 新建会话后发"我掌握了哪些主题？" | 回复含"Python 基础（入门）"（Store 跨会话） |
| TC-29 | CLI `python -m app.cli agent "推荐一门 Python 课程"` | 正常（同步回归） |

## 8. 知识点映射

| 手册条目 | 落地 |
|---|---|
| 8.1 checkpointer（thread_id/持久化） | memory_ctx + lifespan |
| 8.2 Store（namespace+key/InjectedStore） | memory_tools.py |
| 8.3 checkpointer vs Store 对比 | 验证结果 4.2/4.3 |
| 3.5 InjectedStore 注入 | 工具形参（异步版） |
| 9.3 config.configurable.thread_id | _thread_config |

## 8.5 迭代：侧边栏会话列表（用户需求，2026-08）

**问题**：刷新浏览器后，checkpointer 记住了对话（模型知道历史），但**前端消息列表是空的**——用户看不到之前聊过什么，体验差。

**方案**：会话侧边栏 + 历史加载
- 后端两个新接口：
  - `GET /threads`：从 checkpointer `alist(None)` 枚举全部会话（每会话取 step 最大者），按 step 降序
  - `GET /threads/{tid}/messages`：`agent.aget_state()` 读该会话消息历史
- 前端：左侧边栏（240px）= 顶部"＋新建会话"按钮 + 会话列表；点击会话 → 切换 thread_id + 加载历史渲染；当前会话高亮；发送后刷新列表
- localStorage 保存 thread_id：刷新页面后自动恢复上次会话及其历史

**顺带修复**：
- CLI `agent` 命令 ImportError（阶段 6 重构后 `get_agent` 残留导入）→ 已清理
- CLI 带 checkpointer 后报 "Checkpointer requires configurable.thread_id" → 固定 `thread_id="cli-session"`（跨次运行可续聊）

**关键 API（alist）**：`checkpointer.alist(None)` 枚举所有 checkpoint（config=None 跳过过滤，否则强制 thread_id 条件）；每个 checkpoint 的 `item.config["configurable"]["thread_id"]` 是会话 id，`item.metadata["step"]` 是步数（取最新）。

## 9. 测试用例表（6.5 追加）

| 编号 | 操作 | 预期 |
|---|---|---|
| TC-30 | 浏览器发"我叫小美，正在学 Vue"，**刷新页面** | 自动恢复上次会话，历史消息可见 |
| TC-31 | 侧边栏点击其他会话 | 切换 thread_id，加载该会话历史 |
| TC-32 | 点"＋新建会话"再提问 | 新会话，回复"第一次对话"；侧边栏出现新会话 |
| TC-33 | CLI `python -m app.cli agent "你好"` | 正常（ImportError 已修） |
