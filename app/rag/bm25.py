"""BM25 检索算法实现（阶段 8，自实现学习 BM25 原理）。

BM25（Best Matching 25）是经典概率检索模型，评分公式：

    score(q, d) = Σ_{qi ∈ q} IDF(qi) · f(qi, d) · (k1 + 1)
                                   ─────────────────────────
                        f(qi, d) + k1 · (1 - b + b · dl / avgdl)

- f(qi, d)   : 词 qi 在文档 d 中的词频
- dl / avgdl : 文档长度相对平均长度的比值（长度归一化）
- k1 = 1.5   : 词频饱和控制（词频增长对分数的贡献递减）
- b  = 0.75  : 长度归一化强度（0 = 完全不归一化，1 = 完全归一化）
- IDF(qi)    : ln((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
                N = 文档总数，n(qi) = 包含词 qi 的文档数（稀有词权重高）

中文分词用 jieba（lcut_for_search 搜索引擎模式，召回更好）。
"""
from __future__ import annotations

import math

import jieba

K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    """jieba 搜索引擎模式分词（比精确模式召回更多词）。"""
    return [w for w in jieba.lcut_for_search(text or "") if w.strip()]


class BM25:
    """BM25 索引：给定文档列表，支持按查询排序检索。"""

    def __init__(self, documents: list[str], k1: float = K1, b: float = B):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_len = [len(tokenize(d)) for d in documents]  # 每篇词数
        self.avgdl = sum(self.doc_len) / max(len(documents), 1)
        self.df: dict[str, int] = {}   # 词 → 包含它的文档数
        self.tf: list[dict[str, int]] = []  # 每篇文档的词频表
        self._build()

    def _build(self) -> None:
        for i, doc in enumerate(self.documents):
            freq: dict[str, int] = {}
            for w in tokenize(doc):
                freq[w] = freq.get(w, 0) + 1
            self.tf.append(freq)
            for w in freq:
                self.df[w] = self.df.get(w, 0) + 1

    def _idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log((len(self.documents) - n + 0.5) / (n + 0.5) + 1)

    def score(self, query: str, doc_index: int) -> float:
        """查询对单篇文档的 BM25 分数（公式见模块 docstring）。"""
        tf_map = self.tf[doc_index]
        dl = self.doc_len[doc_index]
        total = 0.0
        for w in set(tokenize(query)):
            f = tf_map.get(w, 0)
            if f == 0:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            total += self._idf(w) * f * (self.k1 + 1) / denom
        return total

    def search(self, query: str, top_k: int = 3) -> list[tuple[int, float]]:
        """返回 [(doc_index, score)]，按分数降序。"""
        scored = [(i, self.score(query, i)) for i in range(len(self.documents))]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in scored if s > 0][:top_k]
