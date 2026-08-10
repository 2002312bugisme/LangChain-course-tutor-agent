"""记忆层（阶段 6）：checkpointer（会话）+ Store（跨会话）。

知识点覆盖（agent_api_reference.md 阶段 8）：
- checkpointer: 按 thread_id 存整个 state 快照（会话续聊、SQLite 持久化）
- Store: namespace+key 层级存储（跨会话共享，SQLite 持久化）

⚠️ AsyncSqliteSaver 构造要求运行中的事件循环（__init__ 里
asyncio.get_running_loop()），官方用法是 async with from_conn_string。
所以这里提供 memory_ctx() 常驻上下文：连接在上下文生命周期内保持。
"""
from contextlib import asynccontextmanager

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore

DB_PATH = "data/agent_memory.db"


@asynccontextmanager
async def memory_ctx():
    """常驻式记忆上下文：yield (checkpointer, store)，退出时关闭连接。

    用法（Web - FastAPI lifespan）：
        async with memory_ctx() as (cp, st):
            agent = await ensure_agent(cp, st)
            yield          # 应用运行期间连接保持

    用法（CLI 脚本）：
        async with memory_ctx() as (cp, st):
            agent = await ensure_agent(cp, st)
            ...调用...
    """
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as cp:
        async with AsyncSqliteStore.from_conn_string(DB_PATH) as st:
            yield cp, st


# Store 命名空间常量：("users", user_id) 下存每个用户的学习数据
DEFAULT_USER_NS = ("users", "default")
