# GraphRAG V2 — 垂直领域专家 Agent

[![CI](https://github.com/shijezhang/graphRAG/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/shijezhang/graphRAG/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

基于微软 GraphRAG 论文思路实现的知识图谱增强检索系统。标准向量检索（top-k nearest neighbor）擅长回答"X 是什么"式局部问题，但对"整篇文档的核心主题是什么"这类需要跨文档综合的全局性问题回答质量极差。GraphRAG V2 通过构建显式知识图谱、运行 Leiden 社区发现并用 LLM 生成层级摘要，以 Map-Reduce 方式回答全局查询，同时保留 BM25 + 图遍历的局部检索路径，兼顾两类查询场景。

---

## 系统架构

### 离线索引阶段

```
原始文档 (PDF / TXT / MD)
    │
    ▼
RecursiveChunker
  tiktoken 令牌计数，支持 CJK 字符级 overlap
    │
    ▼
EntityRelationExtractor                    LLM 并发抽取 (ThreadPoolExecutor, concurrency=8)
  → 实体列表 (name, type, description)
  → 关系列表 (source, target, relation, weight)
    │
    ▼
GraphBuilder (NetworkX)
  字符串相似度去重 (SequenceMatcher, threshold=0.85)
  → knowledge_graph.json
    │
    ▼
CommunityDetector (Leiden / leidenalg + igraph)
  最多 3 层层级社区
    │
    ▼
CommunitySummarizer                        LLM 并发生成摘要
  → communities.json (title, summary, key_findings, importance_score)
    │
SparseRetriever                            BM25，CJK 字符级分词
  → BM25 倒排索引 (内存)

DenseRetriever (可选)
  BAAI/bge-m3 (sentence-transformers) + FAISS IndexFlatIP
  → dense.index
```

### 在线检索阶段

```
用户查询
    │
    ▼
QueryRouter (启发式，可配置关键词列表)
    │
    ├── LOCAL ──► BM25 检索 ─┐
    │            Graph Local  ├─► RRF 融合 (k=60) ──► top-k chunks ──► LLM 回答
    │            (并行执行)   ┘
    │
    └── GLOBAL ──► Map: 并发评估各社区相关性 + 提取要点
                   Reduce: LLM 综合所有要点 ──► 最终回答
```

---

## 核心特性

- **全局查询支持**：Leiden 分层社区 + Map-Reduce，解决跨文档综合问题
- **CJK 原生支持**：分词器字符级处理汉字，chunker overlap 不依赖空格切分
- **LLM 并发加速**：抽取和摘要均采用 `ThreadPoolExecutor`，受 `llm.concurrency` 统一控制
- **指数退避重试**：瞬时错误（429 / 5xx / 超时）自动重试，次数与延迟可配置
- **RRF 得分融合**：解决 BM25（量级 0–20+）与 graph/dense（量级 0–1）不可直接比较的问题
- **可配置路由**：`RouterConfig` 将关键词列表和阈值外置到 YAML，支持多领域定制
- **LLMClientProtocol**：`@runtime_checkable` Protocol + 工厂函数，测试时可注入 `MockLLMClient`，无需 monkeypatch
- **响应缓存**：基于 SHA-256 的 LLM 响应磁盘缓存，避免重复 API 调用
- **安全默认值**：API Key 存为 `SecretStr`，服务器默认绑定 `127.0.0.1`，支持可选 Basic Auth
- **可观测性**：每次 `build-graph` / `build-communities` 结束后打印 LLM 调用统计表（总量 / 成功 / 失败 / 重试 / 缓存命中 / 成功率）

---

## 快速开始

### 1. 安装

需要 Python 3.10 或以上。

```bash
# 生产依赖
pip install -r requirements.txt
pip install -e .

# 开发依赖（包含 pytest / ruff / ipykernel）
pip install -r requirements-dev.txt
pip install -e .
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入 DeepSeek API Key：
# DEEPSEEK_API_KEY=sk-xxxxxxxxxxxx
```

`.env` 文件在进程启动时由 `python-dotenv` 自动加载，无需 `export`。API Key 为空时系统在启动阶段立即报错，不会等到首次 LLM 调用才失败。

### 3. 准备文档

将 PDF、TXT 或 Markdown 文件放入 `data/raw/`（该目录已被 `.gitignore` 排除）。

### 4. 构建知识库

```bash
# 一键全流程
bash scripts/build_pipeline.sh

# 指定文档目录
bash scripts/build_pipeline.sh --source /path/to/docs

# 增量重建（跳过已完成步骤）
bash scripts/build_pipeline.sh --skip-ingest          # 跳过分块，从图谱构建开始
bash scripts/build_pipeline.sh --skip-ingest --skip-graph  # 只重跑社区摘要
```

构建产物：

| 文件 | 说明 |
|------|------|
| `data/processed/chunks.json` | 分块后的文档片段 |
| `data/graphs/knowledge_graph.json` | 知识图谱（实体 + 关系） |
| `data/graphs/communities.json` | 社区层级结构与 LLM 摘要 |

### 5. 启动 Web UI

```bash
python app.py
# 访问 http://localhost:7860
```

界面右侧面板实时显示查询类型、检索来源、响应耗时和路由置信度。

### 6. CLI 使用

```bash
# 交互式对话
graphrag chat

# 查看当前配置
graphrag info

# 分步执行索引流程
graphrag ingest data/raw/
graphrag build-graph
graphrag build-communities
```

所有子命令均接受 `--config path/to/config.yaml` 参数以覆盖默认配置路径。

---

## 配置参考

主配置文件：`configs/default.yaml`。

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `llm.model` | `deepseek-chat` | 模型名称 |
| `llm.base_url` | `https://api.deepseek.com` | 兼容 OpenAI 协议的任意端点 |
| `llm.concurrency` | `8` | 批量调用并发数 |
| `llm.max_retries` | `3` | 瞬时错误最大重试次数 |
| `llm.retry_base_delay` | `1.0` | 首次重试延迟（秒），指数增长 |
| `embedding.model` | `BAAI/bge-m3` | sentence-transformers 模型 |
| `embedding.device` | `cpu` | 可改为 `cuda` |
| `chunking.chunk_size` | `500` | 每块最大 token 数 |
| `chunking.chunk_overlap` | `100` | 块间重叠 token 数 |
| `graph.community_algorithm` | `leiden` | 可选 `louvain` |
| `graph.max_community_levels` | `3` | 层级社区最大层数 |
| `retrieval.fusion.strategy` | `rrf` | 融合策略：`rrf` 或 `max_score` |
| `retrieval.fusion.rrf_k` | `60` | RRF 常数，越大越不敏感于排名 |
| `retrieval.router.global_threshold` | `0.5` | 路由到 GLOBAL 的最低得分 |
| `server.host` | `127.0.0.1` | Web UI 绑定地址 |
| `server.port` | `7860` | Web UI 端口 |
| `server.auth_username` | `null` | Basic Auth 用户名（可选） |

路由关键词列表（`retrieval.router.global_keywords`）默认内置中英文通用词和金融领域词，可在 YAML 中完整覆盖以适配其他领域。

---

## 项目结构

```
graphRAG/
├── src/
│   ├── document/
│   │   ├── loader.py          # PDF / TXT / MD 文档加载
│   │   └── chunker.py         # RecursiveChunker，tiktoken 令牌计数，CJK overlap
│   ├── extraction/
│   │   ├── llm_client.py      # LLMClientProtocol, LLMClient, LLMStats, create_llm_client()
│   │   ├── entity_extractor.py
│   │   └── prompts.py
│   ├── graph/
│   │   ├── builder.py         # NetworkX 图谱构建 + 字符串相似度去重
│   │   ├── community.py       # CommunityDetector (Leiden / Louvain)
│   │   └── summarizer.py      # CommunitySummarizer，LLM 并发摘要
│   ├── retrieval/
│   │   ├── sparse.py          # BM25，CJK 字符级分词
│   │   ├── dense.py           # FAISS IndexFlatIP + bge-m3（可选）
│   │   ├── graph_local.py     # 实体匹配 + 邻域遍历
│   │   ├── graph_global.py    # Map-Reduce 全局检索
│   │   ├── hybrid.py          # HybridRetriever，并行本地检索 + 融合
│   │   ├── router.py          # QueryRouter，可配置启发式路由
│   │   └── fusion.py          # RRF / max_score 融合策略
│   ├── agent/
│   │   └── agent.py           # ExpertAgent，流式对话 + 历史管理
│   ├── config.py              # Settings, RouterConfig, FusionConfig, ServerConfig
│   ├── main.py                # CLI 入口 (Typer)
│   └── utils.py               # string_similarity (SequenceMatcher)
├── configs/
│   └── default.yaml
├── tests/
│   ├── helpers.py             # MockLLMClient (LLMClientProtocol 兼容)
│   ├── unit/                  # 11 个测试模块，75 个测试用例
│   └── integration/
├── eval/
│   ├── run_eval.py            # 评测脚本（Keyword Coverage + LLM-as-Judge）
│   └── questions.json         # 10 道标注题目
├── scripts/
│   └── build_pipeline.sh      # 一键索引流程
├── app.py                     # Gradio Web UI
├── pyproject.toml
├── requirements.txt           # 锁定版本生产依赖
└── requirements-dev.txt       # 锁定版本开发依赖
```

---

## 开发

```bash
# 运行测试
pytest tests/ -v

# 代码风格检查
ruff check .

# 格式化
ruff format .
```

CI 在 GitHub Actions 中对 Python 3.10 和 3.11 同时运行 lint 与测试，配置见 `.github/workflows/ci.yml`。

---

## 评测

```bash
python eval/run_eval.py
# 结果保存至 eval/results/detailed.json
```

评测集包含 10 道标注题目（金融领域），每题有参考答案和关键知识点列表，评测指标：

- **Keyword Coverage**：答案覆盖参考关键点的比例（召回率代理指标）
- **LLM-as-Judge**：由 LLM 对答案进行 0–10 打分，维度包括忠实度、完整性和相关性

---

## 已知局限

- **评测集规模小**：仅 10 题，统计显著性有限，结论仅供参考
- **全量重建**：不支持增量更新图谱，每次需从头运行索引流程
- **实体去重**：基于字符串相似度（SequenceMatcher），字面不同但语义等价的实体可能无法合并
- **Dense 检索为可选**：FAISS + bge-m3 在首次运行时需下载模型，默认关闭，需在 `index_from_files(use_dense=True)` 显式启用
- **单机部署**：当前架构不支持分布式图谱构建

---

## License

MIT License — Copyright (c) 2026 zhangshijie
