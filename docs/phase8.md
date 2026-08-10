# 阶段 8：RAG 知识库问答（BM25 + 中文分词）

> 上一阶段：阶段 7（结构化输出 + 会话导出）✅
> 本阶段新增：`app/rag/` 模块（BM25 自实现 + 切片）、`search_knowledge` 工具、知识源 4 篇笔记

---

## 1. 本阶段做什么

让 Agent 能"查自己的笔记"回答知识性问题：用户问"init_chat_model 和 ChatOpenAI 有什么区别"，Agent 先检索知识库（BM25），再基于检索结果回答并**标注出处**——而不是凭空编造。

**RAG 全流程**：文档 → 切片 → 索引 → 检索（BM25）→ 注入上下文 → 生成回答

## 2. 架构与文件

```
app/rag/
├── __init__.py      # 导出 Retriever / build_index / get_retriever
├── bm25.py          # BM25 算法自实现（jieba 分词，lcut_for_search）
├── chunker.py       # RecursiveCharacterTextSplitter 切片 + 章节元数据
└── ingest.py        # 扫描 data/knowledge/*.md → 切片 → BM25 索引（内存单例 + JSON 缓存）

app/tools/knowledge_tools.py   # 【新增】search_knowledge 检索工具
app/agent.py                   # 【修改】tools 注入 search_knowledge
app/prompts.py                 # 【修改】system_prompt 增加"知识性问题先检索再回答，标注出处"
data/knowledge/                # 【新增】知识源（gitignore）：4 篇学习笔记
```

## 3. 核心知识点

### 3.1 切片（Chunking）—— 为什么切、怎么切

长文档不能整篇喂给检索/模型：检索要"块级"精度，模型上下文有长度限制。

```python
RecursiveCharacterTextSplitter(
    chunk_size=800,        # 每块最多 800 字符
    chunk_overlap=120,     # 相邻块重叠 120（防止跨块语义被截断）
    separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", "。", "；"],
    keep_separator=True,   # 保留分隔符（标题行留在块内）
)
```

**separators 顺序是"递归"关键**：先按最大粒度（## 标题）切，切不动再降级到 ### → 空行 → 句号。标题优先保证章节语义完整。

**参数权衡**：
- 太小（~200）→ 单块信息不足，模型看不到完整上下文
- 太大（~2000）→ 命中块混入无关内容，稀释答案
- overlap 太小 → 边界处的关键句被截断丢失

**切片结果**：4 篇笔记 → 303 块，每块带 `source / chapter / idx` 元数据（chapter 取最近的 `## ` 标题）。

### 3.2 BM25 检索 —— 经典概率模型（自实现）

评分公式（bm25.py docstring 有完整推导）：

```
score(q, d) = Σ IDF(qi) · f(qi,d)·(k1+1) / (f(qi,d) + k1·(1-b+b·dl/avgdl))
```

- **词频饱和**：f 越大单次贡献越少（k1=1.5），"编程"出现 10 次 ≠ 10 倍相关
- **长度归一化**：长文档天然词频高，用 dl/avgdl 惩罚（b=0.75）
- **IDF**：稀有词权重高（ln((N-n+0.5)/(n+0.5)+1)），"langchain" 比 "的" 值钱

**为什么中文要 jieba**：BM25 以"词"为单位，中文不分词就整句成"词"，检索失效。`lcut_for_search`（搜索引擎模式）会把"学习路线"切出"学习/路线/学习路线"，召回更好。

### 3.3 检索 → 工具 → Agent 协同

```python
@tool
def search_knowledge(query: str) -> str:
    """在编程学习知识库中检索与 query 相关的内容..."""
    return get_retriever().format(query, top_k=3)
```

- 检索结果格式化：`【来源：文件 | 章节 | 相关度】+ 正文`，模型据此回答
- **Agent 自主决定何时检索**：system_prompt 指示"概念/API/区别对比类问题先检索"，工具调用在聊天界面可见（🔧 调用工具）
- **不编造**：检索不到相关内容时模型应说明"知识库中没有"

### 3.4 Retriever 抽象 —— 预留升级路径

```python
@dataclass
class Retriever:  # 上层只依赖这个接口
    def retrieve(query, top_k) -> list[dict]  # 目前是 BM25，可替换为向量检索
```

后续换 embedding + sqlite-vec（已装 0.1.9）只需实现同一接口——这就是"面向接口编程"在 RAG 上的落点。

## 4. 验证记录

- 切片 303 块，检索 3 组查询全部命中正确章节：
  - "init_chat_model 和 ChatOpenAI 有什么区别" → `agent_api_reference.md | 1.2 具体模型类`（23.05 分）
  - "return_direct 是什么" → `LangChain.md | return_direct`（11.38 分）
  - "interrupt 函数怎么用" → `LangChain.md | interrupt()——在工具中暂停执行`（7.51 分）
- 端到端：Agent 自动 2 次调用 search_knowledge → 回答"入口与产物的关系" + 表格对比 + **标注出处** ✅

## 5. 测试用例

| 编号 | 操作 | 预期 |
|---|---|---|
| TC-51 | 问"init_chat_model 和 ChatOpenAI 有什么区别？" | 自动调 search_knowledge，回答标注【来源：agent_api_reference.md…】 |
| TC-52 | 问"return_direct 是什么？" | 命中知识库，回答含"所有工具 return_direct 才退出"等要点 |
| TC-53 | 问"interrupt 函数怎么用？" | 命中 LangChain.md interrupt 章节 |
| TC-54 | 问"推荐一门 Python 入门课"（回归） | 走 search_courses，不误调知识库 |
| TC-55 | 问"什么是三体问题"（知识库没有） | 如实说明知识库中没有，不编造 |
| TC-56 | 前端聊天验证 | 界面显示 🔧 调用工具 search_knowledge，回答带来源标注 |

## 6. 提交

- `git commit: feat(phase8): RAG 知识库问答（BM25 自实现 + jieba 切片 + search_knowledge 工具）`

---

## 7. 对话补充：RAG 能力盘点（已实现 vs 未实现）

用户提问"RAG 是不是全部实现了"——诚实回答：**实现的是词法检索最小闭环，主流 RAG 进阶能力未做**。

### ✅ 已实现（最小闭环）

| 环节 | 实现 |
|---|---|
| 切分 | RecursiveCharacterTextSplitter（800/120 + 章节元数据） |
| 索引+检索 | BM25（jieba 词法匹配），自实现 |
| 注入 | top-3 切片拼进工具结果 → Agent 生成 |
| 溯源 | 来源/章节/相关度标注 |

### ❌ 未实现（进阶方向，按需扩展）

| 方向 | 说明 |
|---|---|
| 语义检索（向量） | "LLM" 搜不到"大语言模型"——BM25 只认字面；**sqlite-vec 0.1.9 已装未用**，Retriever 抽象已预留 |
| 混合检索 | BM25 + 向量加权融合（RRF） |
| 重排 Rerank | 检索 top-20 → 重排 → top-3 |
| 查询改写/HyDE | 检索前改写问题（"它怎么用"→"interrupt 用法"） |
| 父子分块 | 检索小块、上下文送大块 |
| 索引增量更新 | 文件变更自动重建（现有 force 参数靠手动） |
| 知识库管理界面 | 上传文档/查看切片/删除 |
| 检索评估 | 命中率/MRR 指标 |

**为什么当前够用**：知识库是自己写的学习笔记，术语一致，BM25 实测 3/3 命中正确章节；语义检索的价值在知识库变大、来源杂、提问口语化时。**决策：保持现状，进阶段 9**（用户确认）。
