# GraphRAG 系统级优化升级计划

## 背景

基于深度审计报告，当前项目存在以下主要问题：
- **零测试覆盖**：pyproject.toml 声明 tests/ 但目录不存在，无任何自动化测试
- **关键性能瓶颈**：CommunitySummarizer 和 EntityExtractor 串行调用 LLM（265s+ 延迟）
- **配置系统缺陷**：PathsConfig 定义但未使用，API key 验证缺失，.env 未自动加载
- **多用户并发 bug**：app.py 单例 agent 共享 history，多用户会话会互相污染
- **安全问题**：Gradio 0.0.0.0 无认证绑定，API key 用明文 str 存储
- **代码质量债务**：重复代码（chunk 加载、相似度计算），hybrid 融合不规范（非 RRF），实体去重 O(n²)

## 目标

分 3 个阶段系统性提升项目质量，每个阶段独立可交付：

### P0 - 基础设施与关键修复（立即执行）
修复影响正确性、安全性、可维护性的核心问题

### P1 - 性能优化与测试建设（短期）
解决已知性能瓶颈，建立测试覆盖

### P2 - 架构优化与增强（中期）
重构提升可扩展性，优化算法和融合策略

---

## P0: 基础设施与关键修复

### 1. 配置系统修复

**问题**：
- `.env` 不自动加载，用户必须手动 export
- `PathsConfig` 定义但无人使用，路径硬编码在各处
- API key 空值未校验，启动后才失败
- Gradio 端口硬编码

**方案**：
```python
# 1. 添加 python-dotenv 依赖，在 config.py 顶部自动加载
from dotenv import load_dotenv
load_dotenv()  # 优先级：.env > 环境变量

# 2. 使用 pydantic-settings 的原生 env 支持替换手工解析
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_nested_delimiter='__',  # 支持 DEEPSEEK__API_KEY 或 LLM__API_KEY
        env_file='.env',
        env_file_encoding='utf-8'
    )
    
    @field_validator('llm')
    def validate_api_key(cls, v):
        if not v.api_key or v.api_key == "":
            raise ValueError("LLM API key must be set via DEEPSEEK_API_KEY env var or config file")
        return v

# 3. 统一使用 Settings.paths.*，移除所有硬编码路径
# main.py, agent.py, hybrid.py 等改为接受 settings: Settings 参数
# CLI 命令从 settings.paths.* 读取默认值

# 4. 新增 ServerConfig
class ServerConfig(BaseModel):
    host: str = "127.0.0.1"  # 默认本地，非 0.0.0.0
    port: int = 7860
    auth: tuple[str, str] | None = None  # (username, password) 可选

# app.py 改为：
settings = get_settings()
app.launch(
    server_name=settings.server.host,
    server_port=settings.server.port,
    auth=settings.server.auth
)
```

**文件变更**：
- `pyproject.toml`: 添加 `python-dotenv>=1.0`
- `src/config.py`: 重构 Settings.from_yaml，添加 field_validator，新增 ServerConfig
- `src/main.py`: 所有命令改为 `settings = get_settings(config)` 并使用 `settings.paths.*`
- `src/agent/agent.py`: `initialize()` 默认路径改为从 settings 读取
- `src/retrieval/hybrid.py`: 同上
- `app.py`: 读取 ServerConfig，默认绑定 127.0.0.1
- `README.md`: 更新快速启动，说明 .env 自动加载

**测试验证**：
- 手动测试：删除所有环境变量，只创建 .env，运行 `python app.py` 应正常启动
- 手动测试：.env 中 API key 为空，运行任何命令应立即报错而非等到 LLM 调用

---

### 2. 安全加固

**问题**：
- API key 用 `str` 存储，会出现在 repr/traceback
- Gradio 默认 0.0.0.0 无认证
- 无路径遍历保护（低优先级但需标注）

**方案**：
```python
# 1. config.py 使用 SecretStr
from pydantic import SecretStr

class LLMConfig(BaseModel):
    api_key: SecretStr = SecretStr("")  # 不会出现在 repr
    # 使用时: config.api_key.get_secret_value()

# 2. llm_client.py 适配
self.client = OpenAI(
    api_key=config.api_key.get_secret_value(),
    base_url=config.base_url
)

# 3. ServerConfig 默认值改为本地绑定（已在上一步包含）
# 4. README 添加安全提示
```

**文件变更**：
- `src/config.py`: `api_key: SecretStr`
- `src/extraction/llm_client.py`: `.get_secret_value()`
- `README.md`: 新增 "## 安全注意事项" 章节

---

### 3. 多用户并发 Bug 修复

**问题**：
- `app.py` 的 `agent` 是模块全局单例，多用户共享 history

**方案**：
```python
# app.py 改为 session state 存储
def respond(message, history):
    # Gradio 的 history 参数本身是 per-session 的，但我们的 agent 是全局的
    # 方案：每次调用用传入的 history 覆盖 agent.history
    global agent
    agent.history = history  # 同步 Gradio session history 到 agent
    # ... 原有逻辑
    
# 更好的方案：移除全局 agent，改为工厂函数
def create_agent(settings: Settings) -> ExpertAgent:
    """Create a new agent instance"""
    agent = ExpertAgent(settings.llm, cache_dir=None)
    # 注意：initialize() 很重，每次都调不现实
    # 需区分 agent state（轻，per-session）和 KB state（重，全局共享）
    return agent

# 重构为：KB 全局单例，Agent 实例 per-session
_kb: HybridRetriever | None = None  # 全局，懒加载

def get_kb(settings: Settings) -> HybridRetriever:
    global _kb
    if _kb is None:
        _kb = HybridRetriever(settings)
        _kb.initialize(...)  # 耗时操作仅一次
    return _kb

def respond(message, history):
    kb = get_kb(get_settings())
    # 为本次会话创建轻量 agent，注入共享 KB
    # （需要 agent 架构小改，支持传入已构建的 retriever）
```

**文件变更**：
- `app.py`: 重构为 KB 单例 + Agent per-request 模式
- `src/agent/agent.py`: `initialize()` 拆分为接受外部 retriever 的构造方式
- 或更简单的方案：`agent.history = history` 在每次调用时同步

**最简方案**（先修复正确性）：
```python
def respond(message, history):
    agent.history = history  # 每次请求从 Gradio session 同步
    # ... rest
```

---

### 4. 代码重复消除与工具函数提取

**问题**：
- chunk 加载逻辑重复 3 次
- 相似度计算重复 2 次

**方案**：
```python
# src/document/chunker.py 新增
@staticmethod
def load_from_json(path: str | Path) -> list[Chunk]:
    """Load chunks from JSON file"""
    with open(path) as f:
        data = json.load(f)
    return [Chunk(**item) for item in data]

# src/utils.py 新增（新文件）
from difflib import SequenceMatcher

def string_similarity(a: str, b: str) -> float:
    """Calculate string similarity using SequenceMatcher (0.0-1.0)"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()
```

**文件变更**：
- `src/document/chunker.py`: 添加 `load_from_json` 类方法
- `src/utils.py`: 新文件，提取 `string_similarity`
- `src/main.py`, `src/retrieval/hybrid.py`, `eval/run_eval.py`: 改用 `Chunk.load_from_json()`
- `src/graph/builder.py`, `src/retrieval/graph_local.py`: 改用 `string_similarity()`

---

### 5. 依赖清理与 Lockfile

**问题**：
- langchain 三件套未使用但声明
- 无 lockfile，无法复现构建

**方案**：
```bash
# 1. 移除未使用依赖
# pyproject.toml 删除 langchain, langchain-community, langchain-openai

# 2. 生成 requirements.txt lockfile
pip install uv  # 或用 pip-tools
uv pip compile pyproject.toml -o requirements.txt
uv pip compile pyproject.toml --extra dev -o requirements-dev.txt

# 3. README 更新安装说明
pip install -r requirements.txt
# 或开发环境
pip install -r requirements-dev.txt
```

**文件变更**：
- `pyproject.toml`: 移除 langchain 相关
- `requirements.txt`: 新增（生成）
- `requirements-dev.txt`: 新增（生成）
- `README.md`: 更新安装说明

---

### 6. Ruff 配置完善与 Python 版本声明修正

**问题**：
- `requires-python = ">=3.9"` 但 `target-version = "py311"` 不一致
- Ruff 规则集不全（缺 bugbear 等）

**方案**：
```toml
[project]
requires-python = ">=3.10"  # 实际用了 | union，3.10+ 才原生支持

[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = [
    "E", "F",   # pyflakes, pycodestyle
    "I",        # isort
    "N",        # pep8-naming
    "W",        # warnings
    "B",        # bugbear (常见陷阱)
    "UP",       # pyupgrade
    "C4",       # comprehensions
    "SIM",      # simplify
]
ignore = [
    "B008",     # function call in default arg (typer 常用模式)
]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]  # 允许 __init__ 中未使用的导入
```

**文件变更**：
- `pyproject.toml`: 修正 Python 版本，扩展 ruff 规则
- 运行 `ruff check --fix .` 修复自动可修复的问题
- 手动修复其余问题（如 bare except 改为具体异常）

---

### 7. 基础测试框架搭建

**问题**：
- tests/ 目录不存在
- 无任何测试

**方案**：
```bash
mkdir -p tests/{unit,integration,fixtures}
touch tests/__init__.py
touch tests/conftest.py
```

```python
# tests/conftest.py
import pytest
from pathlib import Path

@pytest.fixture
def fixtures_dir():
    return Path(__file__).parent / "fixtures"

@pytest.fixture
def sample_chunks(fixtures_dir):
    # 提供测试用 chunks
    pass
```

```python
# tests/unit/test_chunker.py (首个测试)
from src.document.chunker import RecursiveChunker

def test_chunker_chinese_overlap():
    """Test that Chinese text overlap doesn't break on whitespace split"""
    chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
    text = "这是一个测试文本。" * 20  # 纯中文无空格
    chunks = chunker.chunk_text(text, "test.txt")
    assert len(chunks) > 1
    # 验证 overlap 逻辑正常工作
    assert chunks[1].content.startswith(chunks[0].content[-20:])
```

**文件变更**：
- `tests/` 目录结构创建
- `tests/conftest.py`: 基础 fixtures
- `tests/unit/test_chunker.py`: 第一个测试（覆盖审计报告中的 CJK bug）
- `tests/unit/test_config.py`: 测试环境变量加载
- `tests/unit/test_llm_client.py`: 测试 `_parse_json_response` 截断恢复

**目标**：
- 能 `pytest tests/` 通过
- 覆盖审计报告"最高价值首批测试"中的前 3 项

---

### 8. CI/CD 基础设施

**方案**：
```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - run: pip install -r requirements-dev.txt
      - run: ruff check .
      - run: ruff format --check .
      - run: pytest tests/
```

**文件变更**：
- `.github/workflows/ci.yml`: 新增

---

## P0 交付标准

- [ ] `.env` 自动加载，API key 启动时校验
- [ ] 所有路径从 `settings.paths.*` 读取
- [ ] Gradio 默认 127.0.0.1，API key 用 SecretStr
- [ ] app.py 多用户 bug 修复（最简方案：同步 history）
- [ ] 代码重复消除（Chunk.load_from_json, string_similarity）
- [ ] 移除 langchain，生成 requirements.txt
- [ ] Ruff 配置完善，Python 版本改 3.10+
- [ ] tests/ 目录存在，≥3 个单元测试通过
- [ ] CI workflow 运行 lint + test

---

## P1: 性能优化与测试建设

### 1. LLM 调用并发化（解决 265s 瓶颈）

**问题**：
- `CommunitySummarizer.summarize_communities` 串行
- `EntityRelationExtractor.extract_from_chunks` 串行

**方案**：
```python
# 1. CommunitySummarizer 添加并发
from concurrent.futures import ThreadPoolExecutor, as_completed

def summarize_communities(self, hierarchy: list[list[Community]]) -> list[list[Community]]:
    for level_communities in hierarchy:
        with ThreadPoolExecutor(max_workers=min(16, len(level_communities))) as executor:
            futures = {
                executor.submit(self._summarize_single, c): c 
                for c in level_communities
            }
            for future in as_completed(futures):
                community = futures[future]
                try:
                    result = future.result()
                    community.title = result["title"]
                    # ...
                except Exception as e:
                    logger.warning(f"Failed to summarize community {community.id}: {e}")
                    # 保持原有降级逻辑
    return hierarchy

# 2. EntityRelationExtractor 同理
def extract_from_chunks(self, chunks: list[Chunk]) -> tuple[list, list]:
    all_entities, all_relations = [], []
    with ThreadPoolExecutor(max_workers=min(16, len(chunks))) as executor:
        futures = {executor.submit(self._extract_single, c.content): i for i, c in enumerate(chunks)}
        for future in as_completed(futures):
            try:
                entities, relations = future.result()
                all_entities.extend(entities)
                all_relations.extend(relations)
            except Exception as e:
                logger.warning(f"Extraction failed for chunk: {e}")
    return all_entities, all_relations
```

**预期效果**：
- 社区摘要从 321s → ~20-40s（16 并发）
- 实体抽取从 136s → ~10-15s

**文件变更**：
- `src/graph/summarizer.py`: 添加 ThreadPoolExecutor
- `src/extraction/entity_extractor.py`: 添加 ThreadPoolExecutor

---

### 2. Chunker CJK Overlap Bug 修复

**问题**：
- `_get_overlap` 用 `text.split()` 对中文无效

**方案**：
```python
def _get_overlap(self, text: str, overlap_size: int) -> str:
    """Get overlap text from end, respecting token boundaries"""
    if len(text) <= overlap_size:
        return text
    
    # 不用 split()，直接按 token 估算反向取
    # 中文 ~1.5 chars/token，英文 ~4 chars/token
    # 保守取 1.5x overlap_size 个字符
    char_count = int(overlap_size * 1.5)
    return text[-char_count:]
```

**文件变更**：
- `src/document/chunker.py`: 修复 `_get_overlap` 逻辑

---

### 3. 错误处理改进

**问题**：
- 所有 LLM 失败都 silent，无阈值告警
- KeyError 在实体解析时会丢弃整个 chunk

**方案**：
```python
# 1. EntityRelationExtractor 添加成功率跟踪
def extract_from_chunks(self, chunks: list[Chunk]) -> tuple[list, list]:
    all_entities, all_relations = [], []
    failed = 0
    for i, chunk in enumerate(chunks):
        try:
            entities, relations = self._extract_single(chunk.content)
            all_entities.extend(entities)
            all_relations.extend(relations)
        except Exception as e:
            failed += 1
            logger.warning(f"Extraction failed for chunk {i}: {e}")
    
    success_rate = (len(chunks) - failed) / len(chunks)
    if success_rate < 0.5:
        raise RuntimeError(f"Extraction failure rate too high: {failed}/{len(chunks)} failed")
    logger.info(f"Extraction complete: {len(chunks) - failed}/{len(chunks)} succeeded")
    return all_entities, all_relations

# 2. _parse_entities/_parse_relations 改为逐项 try-except
def _parse_entities(self, data: dict) -> list:
    entities = []
    for e in data.get("entities", []):
        try:
            entities.append({
                "name": e["name"],
                "type": e.get("type", "UNKNOWN"),
                # ...
            })
        except KeyError as err:
            logger.warning(f"Skipping malformed entity (missing {err}): {e}")
    return entities
```

**文件变更**：
- `src/extraction/entity_extractor.py`: 成功率跟踪 + 逐项解析

---

### 4. 测试覆盖扩展

新增测试：
- `tests/unit/test_router.py`: QueryRouter 关键词路由
- `tests/unit/test_graph_builder.py`: 实体去重逻辑
- `tests/unit/test_sparse.py`: BM25 tokenizer
- `tests/integration/test_pipeline.py`: 端到端 ingest → extract → graph 流程（小样本）

**目标**：
- 核心纯函数覆盖率 >60%
- 至少 1 个集成测试

---

## P1 交付标准

- [ ] CommunitySummarizer 并发化，≥10x 加速
- [ ] EntityRelationExtractor 并发化，≥10x 加速
- [ ] Chunker CJK bug 修复，有测试验证
- [ ] 错误处理改进，失败率阈值告警
- [ ] 新增 ≥5 个单元测试，1 个集成测试
- [ ] 所有测试在 CI 中通过

---

## P2: 架构优化与增强（规划，不在本次实施）

### 1. LLM Client 抽象与提供商解耦
- 定义 `LLMProtocol` Protocol
- 工厂模式支持多提供商
- 测试时注入 MockLLMClient

### 2. Hybrid 检索融合改进
- 实现真正的 RRF（rank-based fusion）
- 或 min-max 归一化后 score fusion
- 可配置融合策略

### 3. 实体去重优化
- Embedding-based semantic dedup
- 或分桶预过滤降低 O(n²) 复杂度

### 4. 路由器可配置化
- 关键词从 config 读取，支持多领域
- 或训练简单分类器

### 5. Async 重构（可选）
- 全面改用 asyncio + httpx/AsyncOpenAI
- 但改动范围极大，ROI 不如直接 ThreadPoolExecutor

---

## 执行顺序

### Sprint 1 (本次)：P0 全部完成
优先级：1 配置 → 2 安全 → 3 并发 bug → 4 代码清理 → 5 依赖 → 6 ruff → 7 测试 → 8 CI

### Sprint 2 (后续)：P1 性能与测试

### Sprint 3 (未来)：P2 架构优化

---

## 风险与注意事项

1. **配置重构影响面大**：需逐文件改，容易漏
   - 缓解：先写测试锁住行为再重构
2. **并发化可能暴露新 bug**：LLM client 线程安全性未验证
   - 缓解：ThreadPoolExecutor 本身安全，OpenAI SDK 官方支持多线程
3. **测试环境需模拟 LLM**：集成测试成本高
   - 缓解：P0 只测纯函数，P1 再引入 LLM mock
