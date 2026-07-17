# GraphRAG V2 — 垂直领域专家 Agent

基于微软 GraphRAG 论文复现的知识图谱增强检索系统。通过 LLM 自动从文档中抽取实体/关系、构建知识图谱、运行 Leiden 社区发现并生成层级摘要，实现对"全局性问题"（如"这批文档的核心主题是什么？"）的高质量回答——这是传统 top-k 向量检索无法做到的。

---

## 快速启动

### 1. 环境安装

```bash
pip install -e ".[dev]"
```

### 2. 配置 API Key

编辑 `configs/default.yaml`，填入 DeepSeek API Key：

```yaml
llm:
  api_key: sk-your-key-here
  model: deepseek-chat
  base_url: https://api.deepseek.com
```

### 3. 准备文档

将 PDF / TXT / Markdown 文件放入 `data/raw/` 目录。

### 4. 一键构建知识库

```bash
bash scripts/build_pipeline.sh
```

支持参数：

```bash
# 指定文档目录
bash scripts/build_pipeline.sh --source /path/to/docs

# 增量更新：跳过已完成的步骤
bash scripts/build_pipeline.sh --skip-ingest --skip-graph   # 只重跑社区摘要
bash scripts/build_pipeline.sh --skip-ingest                # 只重跑图谱 + 社区

# 查看帮助
bash scripts/build_pipeline.sh --help
```

构建产物：

| 文件 | 说明 |
|------|------|
| `data/processed/chunks.json` | 分块后的文档片段 |
| `data/graphs/knowledge_graph.json` | 知识图谱（实体 + 关系） |
| `data/graphs/communities.json` | 社区层级结构 + LLM 摘要 |

### 5. 启动 Web UI

```bash
python3 app.py
# 访问 http://localhost:7860
```

### 6. CLI 对话

```bash
python3 -m src.main chat
```

### 7. 运行评测

```bash
python3 eval/run_eval.py
# 结果保存至 eval/results/detailed.json
```

---

## 系统架构

```
用户输入
    │
    ▼
Query Router（启发式关键词分类）
    │
    ├─── Local Query ──► BM25 + Graph Local 混合检索
    │                        │
    │                    相关 Chunks → LLM 生成回答
    │
    └─── Global Query ──► Graph Global（Map-Reduce）
                              │
                          社区摘要 Map → 相关点提取
                              │
                          Reduce → LLM 综合回答

索引层（离线构建）：
  文档 → 分块 → LLM 实体/关系抽取 → NetworkX 图谱
       → Leiden 社区发现 → LLM 社区摘要
       → BM25 倒排索引
```

---

## 评测结果（4篇 RAG 论文语料，10题）

| 方法 | LLM-as-Judge (0–10) | Keyword Coverage | 平均延迟 |
|------|---------------------|-----------------|---------|
| BM25 | 3.9 | 20% | ~1s |
| Graph Local | 2.5 | 16% | ~0.9s |
| **Graph Global** | **4.6** | 13% | ~265s |

- 全局问题（G1–G5）：Graph Global 3.8 vs BM25 3.0，**提升 27%**
- 局部问题（L1–L5）：Graph Global 5.4 vs BM25 4.8，**提升 13%**

---

## 面试 QA

> 假设面试官看到简历上的这个项目，由易到难的 20 个高质量问答。

### 基础概念（Q1–Q7）

**Q1: 什么是 RAG？它解决了什么问题？**

A: RAG（Retrieval-Augmented Generation）将信息检索与生成模型结合：先从外部知识库检索相关文档，再将其作为上下文输入 LLM 生成回答。它解决了 LLM 知识截止、幻觉、以及无法访问私有文档的问题。本项目在 RAG 基础上引入知识图谱，进一步解决了传统 RAG 无法回答全局性问题的局限。

**Q2: GraphRAG 相比传统 RAG 的核心创新是什么？**

A: 传统 RAG 用 top-k 向量检索，只能召回局部相似片段，无法回答"整个语料库的核心主题是什么"这类需要全局综合的问题。GraphRAG 的创新在于：从文档中抽取实体/关系构建知识图谱，用 Leiden 算法做社区发现，对每个社区生成 LLM 摘要，查询时用 Map-Reduce 遍历所有社区摘要并综合答案，从而覆盖全局语义。

**Q3: Leiden 算法是什么？为什么选它而不是 Louvain？**

A: Leiden 和 Louvain 都是基于模块度优化的图社区发现算法。Louvain 的缺陷是可能产生内部不连通的社区（"断裂社区"），Leiden 通过引入"精炼"步骤保证每个社区内部连通，质量更高。本项目用 `leidenalg` + `igraph` 库实现，支持多分辨率参数（`resolution`）生成不同粒度的层级社区结构。

**Q4: BM25 的原理是什么？它和向量检索有什么区别？**

A: BM25 是基于词频（TF）和逆文档频率（IDF）的稀疏检索算法，通过关键词精确匹配打分，对专有名词、术语匹配效果好，但无法理解语义相似性（如"汽车"和"车辆"不相关）。向量检索将文本编码为稠密向量，用余弦相似度衡量语义距离，能处理同义词和语义相关性，但对精确关键词匹配不如 BM25。本项目的 BM25 实现对中文做了字符级分词处理。

**Q5: 向量检索的实现原理是什么？FAISS 做了什么优化？**

A: 向量检索将文本用 Sentence Transformer（本项目用 BAAI/bge-m3）编码为高维向量，查询时计算查询向量与所有文档向量的相似度并取 top-k。FAISS 是 Facebook 开发的高效向量索引库，`IndexFlatIP` 做精确内积搜索，配合 L2 归一化后等价于余弦相似度。本项目在 `src/retrieval/dense.py` 中实现，支持 `save()`/`load()` 持久化索引。

**Q6: 知识图谱的节点和边分别存储什么信息？**

A: 节点（实体）存储：`entity_name`、`entity_type`（人物/组织/概念等）、`description`、`source_chunks`（来源 chunk 索引列表）。边（关系）存储：`relation_type`、`description`、`weight`（关系强度）、`source_chunks`。图用 NetworkX 存储在内存中，通过 `node_link_data` 格式序列化为 JSON 持久化。

**Q7: 社区摘要的作用是什么？它是如何生成的？**

A: 社区摘要将知识图谱中一组语义相关的实体/关系压缩为结构化文本，使 Global 检索时不需要遍历原始 chunk，而是遍历更高层次的语义摘要。生成方式：对每个社区，将其包含的实体描述和关系描述拼接后输入 LLM，要求输出 `title`、`summary`、`key_findings`（要点列表）、`importance_score`（0–1 重要性评分）的 JSON。

---

### 系统设计（Q8–Q14）

**Q8: Query Router 是如何判断一个问题是 Local 还是 Global 的？**

A: 本项目用启发式关键词方法：维护一个 `GLOBAL_KEYWORDS` 列表（如"总结"、"概述"、"所有"、"整体"、"比较"等），每匹配一个关键词加 0.3 分，超过阈值（0.3）判定为 Global。关键设计细节：只有在没有匹配到任何 Global 关键词时，才对"是什么"等局部问题施加惩罚，避免"核心观点是什么"被误判为 Local。也支持切换为 LLM-based 分类模式。

**Q9: Graph Global 的 Map-Reduce 流程具体是怎么做的？**

A: Map 阶段：对每个社区摘要，调用 LLM 判断该社区与查询的相关性（0–1 分）并提取相关要点，过滤掉相关性低于阈值的社区。Reduce 阶段：将所有 Map 阶段提取的要点汇总，调用 LLM 综合生成最终回答。本项目在 `src/retrieval/graph_global.py` 实现，Map 阶段是主要延迟来源（321 个社区 × 1 次 LLM 调用 ≈ 265s）。

**Q10: 实体去重是如何实现的？有什么局限？**

A: 在 `src/graph/builder.py` 的 `_deduplicate()` 方法中，用 Python `difflib.SequenceMatcher` 计算实体名称相似度，相似度 ≥ 0.85 的实体合并为同一节点（保留描述最长的那个）。局限：纯字符串相似度无法处理语义等价但字面不同的实体（如"GPT-4"和"OpenAI 的大模型"），更好的方案是用 embedding 相似度或 LLM 二次判断，但成本更高。

**Q11: 混合检索（BM25 + Graph Local）是如何融合结果的？**

A: 在 `src/retrieval/hybrid.py` 中，Local 查询同时调用 BM25 和 Graph Local，两者都返回 `(Chunk, score)` 列表。融合策略：以 `chunk_index` 为 key，对同一 chunk 取两路中的最高分（max fusion），去重后按分数降序排列，取 top-k。这是一种简单的 late fusion，没有做分数归一化，可能存在量纲不一致问题。

**Q12: LLM-as-Judge 评测是如何设计的？有什么局限？**

A: 给 LLM 提供问题、参考答案、系统回答，要求输出 0–10 的评分和一句理由（JSON 格式）。优点：比关键词覆盖率更能反映回答质量，不需要精确字符串匹配。局限：评分受 LLM 自身偏好影响（如偏好更长的回答）、同一问题多次评分可能不一致、用同一个 LLM 既生成回答又评判存在自我偏袒风险。生产环境应用更强的 judge 模型或多模型交叉评判。

**Q13: LLM 响应缓存是如何实现的？为什么这样设计？**

A: 在 `src/extraction/llm_client.py` 中，以 `model + system_prompt + user_prompt` 的 SHA256 哈希前 16 位为文件名，将 LLM 响应存储为 JSON 文件（`data/processed/.llm_cache/`）。下次相同请求直接读缓存，不调用 API。这样设计的原因：实体抽取需要对每个 chunk 调用一次 LLM，136 个 chunk 约需 136 次调用；开发调试时频繁重跑，缓存可节省大量 API 费用和时间。

**Q14: 文档分块策略是如何设计的？为什么用递归分块？**

A: 在 `src/document/chunker.py` 中实现递归分块：按优先级尝试分隔符（`\n\n` → `\n` → `。` → `.` → ` `），先用高优先级分隔符切分，若某段仍超过 `chunk_size`（默认 1000 tokens）则递归用下一级分隔符继续切分，最后合并过小的片段（< 0.5 × chunk_size）。Token 估算：中文字符 /1.5，其他字符 /4。递归分块的优势是尽量保留语义完整性（优先在段落边界切分），而不是机械地按固定长度截断。

---

### 深度追问（Q15–Q20）

**Q15: Graph Global 平均延迟 265 秒，如何优化到生产可用？**

A: 主要瓶颈是 Map 阶段对 321 个社区串行调用 LLM。优化方向：① **并发**：用 `asyncio` + `aiohttp` 并发调用 LLM API，理论上可将延迟降低 10–50 倍；② **社区过滤**：Map 前先用 BM25 或向量相似度对社区摘要做粗筛，只对 top-N 个社区调用 LLM；③ **分层检索**：优先查询高层（粗粒度）社区，相关性高再下钻到子社区；④ **预计算**：离线为每个社区生成 embedding，查询时先向量检索再 LLM 精排。

**Q16: 评测中 Graph Local 得分（2.5）低于 BM25（3.9），原因是什么？**

A: Graph Local 的检索依赖实体匹配：先从查询中识别实体名称（子串匹配 + SequenceMatcher），再做 BFS 子图扩展。问题在于：① 查询中的实体表述与图谱中的实体名称不一致时（如查询用"原始 RAG 论文"，图谱中是"Lewis et al. 2020"），匹配失败；② 对于"Self-RAG 如何决定是否检索"这类问题，关键词"reflection tokens"在图谱中可能是边的属性而非节点，BFS 无法找到；③ BM25 直接在原始文本上做关键词匹配，对论文中的专有术语反而更准确。

**Q17: 这个评测集有哪些局限性？如何改进？**

A: 局限：① 只有 10 个问题，样本量太小，结论统计显著性不足；② 问题由人工设计，可能存在对某种检索方式的隐性偏好；③ 参考答案也是人工写的，LLM-as-Judge 评分受参考答案质量影响；④ 只测了 3 种检索方式，没有 Dense 检索和 Hybrid 的对比基线。改进方向：扩充到 50–100 题，引入多个 judge 模型交叉评分，增加 Dense 和 Hybrid 基线，用 RAGAS 等专业评测框架。

**Q18: 如果用户新增了文档，如何增量更新知识库而不全量重建？**

A: 当前实现不支持真正的增量更新，每次都全量重建。增量更新的设计思路：① **Ingest 层**：记录已处理文件的哈希，只对新增/修改文件重新分块；② **Graph 层**：新 chunk 的实体/关系抽取后，与现有图谱合并（新增节点/边，更新已有节点的 `source_chunks`）；③ **Community 层**：图结构变化后需重跑 Leiden（社区结构是全局的，局部变化会影响整体划分），这是增量更新最难的部分；④ 实践中可接受"定期全量重建"（如每天一次）而非实时增量。

**Q19: 这个系统如果要上生产，还需要解决哪些工程问题？**

A: ① **存储**：NetworkX 内存图在大规模文档（>10 万节点）时会 OOM，需迁移到 Neo4j 或 TigerGraph；② **并发**：当前所有检索都是同步串行，需改为异步 + 连接池；③ **安全**：API Key 目前在配置文件中，生产环境应用 Secret Manager；④ **监控**：需要记录每次查询的检索方式、延迟、LLM 调用次数；⑤ **成本控制**：Graph Global 每次查询调用 300+ 次 LLM，需要限流和预算告警；⑥ **评测持续化**：建立 CI 评测流水线，防止代码变更导致质量回退。

**Q20: 与直接用向量数据库（如 Pinecone、Weaviate）的方案相比，GraphRAG 的优劣势是什么？**

A: 向量数据库方案优势：部署简单、延迟低（毫秒级）、支持实时更新、成熟的生产化工具链。GraphRAG 优势：① 对全局性/综合性问题质量更高（本项目实测全局问题提升 27%）；② 知识图谱显式存储实体关系，支持关系推理和路径解释；③ 社区摘要提供了文档集合的"鸟瞰视角"，适合分析类场景。GraphRAG 劣势：① 索引构建成本高（需大量 LLM 调用）；② Global 检索延迟极高；③ 实体抽取质量依赖 LLM，存在错误传播。实践中两者互补：用向量检索处理大多数局部问题，GraphRAG 处理需要全局综合的问题。
