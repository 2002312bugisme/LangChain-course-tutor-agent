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
