"""文档切片（阶段 8）。

用 langchain 的 RecursiveCharacterTextSplitter 把长文档切成有意义的块：
- 优先按标题层级（## / ### / ####）切——保住章节语义
- 再按空行、句号兜底
- chunk_size=800 字符 / chunk_overlap=120（相邻块重叠，防止跨块语义丢失）
- 每块带元数据：source（来源文件名）、chapter（所属章节标题）、idx

切分参数影响检索质量（覆盖知识点：切片参数调优）：
- chunk 太小 → 单块信息不足，模型看不到完整上下文
- chunk 太大 → 命中块混入大量无关内容，稀释答案；检索精度下降
- overlap 太小 → 边界处的关键句可能被截断丢失
"""
from __future__ import annotations

from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    """一个知识块：文本 + 溯源元数据。"""

    text: str
    source: str          # 来源文件名
    chapter: str         # 所属章节标题（取最近的 ## 标题）
    idx: int = 0         # 文件内序号


_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
    separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", "。", "；"],
    keep_separator=True,
)


def split_markdown(text: str, source: str) -> list[Chunk]:
    """把一篇 markdown 切成 Chunk 列表，记录每块所属章节。"""
    pieces = _splitter.split_text(text)
    chunks: list[Chunk] = []
    current_chapter = "未分类"
    cursor = 0  # 用原文位置近似定位章节（splitter 不返回偏移，按块文本是否含标题判断）

    for idx, piece in enumerate(pieces):
        # 近似章节：若块内以标题开头或包含标题行，取第一个 ## 标题
        for line in piece.splitlines():
            if line.startswith("## ") and len(line) > 3:
                current_chapter = line[3:].strip()
                break
        chunks.append(Chunk(text=piece, source=source, chapter=current_chapter, idx=idx))
        cursor += len(piece)
    return chunks
