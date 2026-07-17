# GraphRAG V2 — 垂直领域专家 Agent

基于微软 GraphRAG 论文复现的知识图谱增强检索系统。通过 LLM 自动从文档中抽取实体/关系、构建知识图谱、运行 Leiden 社区发现并生成层级摘要，实现对"全局性问题"（如"这批文档的核心主题是什么？"）的高质量回答——这是传统 top-k 向量检索无法做到的。

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

## 项目结构

```
graphRAG/
├── src/
│   ├── document/      # 文档加载与分块
│   ├── extraction/     # LLM 实体/关系抽取
│   ├── graph/           # 知识图谱构建、社区发现、摘要
│   ├── retrieval/       # Dense / Sparse / Graph Local / Graph Global / Hybrid
│   ├── agent/            # 对话 Agent
│   ├── evaluation/       # 评测指标
│   ├── config.py          # 配置加载
│   └── main.py             # CLI 入口
├── configs/default.yaml   # 系统配置
├── eval/                    # 评测脚本与题库
├── scripts/build_pipeline.sh  # 一键构建脚本
└── app.py                       # Web UI (Gradio)
```

## 快速启动

### 1. 环境安装

```bash
pip install -e ".[dev]"
```

### 2. 配置 API Key

复制 `.env.example` 为 `.env`，填入 DeepSeek API Key：

```bash
cp .env.example .env
# 编辑 .env，设置 DEEPSEEK_API_KEY
export $(cat .env | xargs)
```

`configs/default.yaml` 中的 `api_key` 字段通过 `${DEEPSEEK_API_KEY}` 从环境变量读取，避免密钥硬编码在配置文件中。

### 3. 准备文档

将 PDF / TXT / Markdown 文件放入 `data/raw/` 目录（该目录已被 `.gitignore` 排除，需自行准备）。

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

## 核心特性

- **多格式文档解析**：PDF / Markdown / TXT / HTML，基于 tiktoken 的语义分块
- **LLM 实体/关系抽取**：结合基于 embedding 与 LLM 的实体去重
- **知识图谱构建**：NetworkX 图存储，Leiden 算法做层级社区发现，LLM 生成社区摘要
- **混合检索**：Dense（ChromaDB + OpenAI embeddings）、Sparse（BM25）、GraphRAG（社区遍历）、Hybrid（RRF 融合）
- **有依据的问答**：基于检索上下文生成带引用的回答
- **评测框架**：Recall@K、Precision@K、MRR、NDCG，以及 LLM-as-Judge（忠实度、完整性、相关性）

## 评测结果（4篇 RAG 论文语料，10题）

| 方法 | LLM-as-Judge (0–10) | Keyword Coverage | 平均延迟 |
|------|---------------------|-----------------|---------|
| BM25 | 3.9 | 20% | ~1s |
| Graph Local | 2.5 | 16% | ~0.9s |
| **Graph Global** | **4.6** | 13% | ~265s |

- 全局问题：Graph Global 3.8 vs BM25 3.0，提升 27%
- 局部问题：Graph Global 5.4 vs BM25 4.8，提升 13%

## 已知局限

- 评测集样本量较小（10题），结论统计显著性有限
- Graph Global 平均延迟 265s，串行调用 LLM 是主要瓶颈
- 不支持真正的增量更新，每次构建为全量重建
- 实体去重基于字符串相似度，无法处理语义等价但字面不同的实体

## License

MIT
