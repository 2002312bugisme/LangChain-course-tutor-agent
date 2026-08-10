"""RAG 索引构建（阶段 8）。

流程：扫描 data/knowledge/*.md → 逐篇切片 → 构建 BM25 索引。
索引常驻内存（进程生命周期），构建结果缓存为 JSON 便于调试查看。

设计要点：
- lazy 构建 + 模块级单例：CLI 与 Web 共用，首次调用构建（jieba 首次加载词典较慢）
- Retriever 抽象：预留向量库升级路径（后续可换 sqlite-vec / 真实 embedding），
  上层只依赖 Retriever 接口，不依赖 BM25 实现
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from app.rag.bm25 import BM25
from app.rag.chunker import Chunk, split_markdown

KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "knowledge")
CACHE_FILE = os.path.join(KNOWLEDGE_DIR, "index_cache.json")


@dataclass
class Retriever:
    """RAG 检索器（BM25 实现，接口可替换为向量检索）。"""

    documents: list[str]          # 切片文本（BM25 的"文档"）
    metadata: list[dict]          # 每片元数据（source/chapter/idx）
    bm25: BM25

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """检索 top-k 切片，带分数与元数据。"""
        results = []
        for i, score in self.bm25.search(query, top_k):
            item = {"score": round(score, 3), **self.metadata[i]}
            item["text"] = self.documents[i]  # 切片正文
            results.append(item)
        return results

    def format(self, query: str, top_k: int = 3) -> str:
        """检索结果拼成给模型的文本（带出处标注）。"""
        parts = []
        for r in self.retrieve(query, top_k):
            parts.append(
                f"【来源：{r['source']} | 章节：{r['chapter']} | 相关度 {r['score']}】\n{r['text']}"
            )
        return "\n\n".join(parts)


_retriever: Retriever | None = None


def build_index(force: bool = False) -> Retriever:
    """构建（或复用）全局检索器。

    force=True 强制重建（知识库文件变化后手动调用）。
    """
    global _retriever
    if _retriever is not None and not force:
        return _retriever

    chunks: list[Chunk] = []
    for fname in sorted(os.listdir(KNOWLEDGE_DIR)):
        if not fname.endswith(".md"):
            continue
        path = os.path.join(KNOWLEDGE_DIR, fname)
        with open(path, encoding="utf-8") as f:
            chunks.extend(split_markdown(f.read(), source=fname))

    if not chunks:
        raise FileNotFoundError(f"知识库目录 {KNOWLEDGE_DIR} 下没有 .md 文件")

    docs = [c.text for c in chunks]
    meta = [
        {"source": c.source, "chapter": c.chapter, "idx": c.idx, "len": len(c.text)}
        for c in chunks
    ]
    _retriever = Retriever(documents=docs, metadata=meta, bm25=BM25(docs))

    # 缓存（调试/可视化用；检索本身走内存索引）
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {"total_chunks": len(chunks), "chunks": meta},
                f, ensure_ascii=False, indent=1,
            )
    except OSError:
        pass
    return _retriever


def get_retriever() -> Retriever:
    """懒加载单例。"""
    return build_index()
