"""RAG 检索工具（阶段 8）。

search_knowledge：Agent 的知识库检索工具——用户问"xx 是什么/有什么区别/怎么用"等
知识性问题时，先检索项目知识库（LangChain 学习笔记），再基于检索结果回答。
"""
from langchain_core.tools import tool

from app.rag import get_retriever


@tool
def search_knowledge(query: str) -> str:
    """在编程学习知识库（LangChain/Agent 笔记）中检索与 query 相关的内容。

    当用户询问概念、API 用法、区别对比等知识性问题时使用。
    返回匹配的知识片段（含来源文件与章节），应基于这些内容组织回答并标注出处。
    """
    return get_retriever().format(query, top_k=3)
