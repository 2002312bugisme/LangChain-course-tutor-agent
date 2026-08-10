"""RAG 模块（阶段 8）：BM25 中文知识库检索。

- bm25.py    : BM25 算法自实现（jieba 分词）
- chunker.py : RecursiveCharacterTextSplitter 切片 + 章节元数据
- ingest.py  : 索引构建 + Retriever 抽象（预留向量库升级）
"""
from app.rag.ingest import Retriever, build_index, get_retriever

__all__ = ["Retriever", "build_index", "get_retriever"]
