from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env file automatically
load_dotenv()


class LLMConfig(BaseModel):
    provider: str = "deepseek"
    model: str = "deepseek-chat"
    api_key: SecretStr = Field(default_factory=lambda: SecretStr(os.environ.get("DEEPSEEK_API_KEY", "")))
    base_url: str = "https://api.deepseek.com"
    temperature: float = 0.0
    max_tokens: int = 4096
    concurrency: int = 8
    max_retries: int = 3
    retry_base_delay: float = 1.0


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-m3"
    device: str = "cpu"
    batch_size: int = 32


class ChunkingConfig(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list[str] = Field(default_factory=lambda: ["\n\n", "\n", "。", ".", " "])


class GraphConfig(BaseModel):
    community_algorithm: Literal["leiden", "louvain"] = "leiden"
    resolution: float = 1.0
    max_community_levels: int = 3


class DenseRetrievalConfig(BaseModel):
    top_k: int = 10


class SparseRetrievalConfig(BaseModel):
    top_k: int = 10


class GraphLocalConfig(BaseModel):
    max_hops: int = 2
    top_k: int = 10


class GraphGlobalConfig(BaseModel):
    community_level: int = 1
    top_k: int = 5


DEFAULT_GLOBAL_KEYWORDS: list[str] = [
    # General
    "总结",
    "概括",
    "核心",
    "主要",
    "整体",
    "全部",
    "所有",
    "主题",
    "观点",
    "结论",
    "综述",
    "概述",
    "全文",
    "全书",
    "比较",
    "对比",
    "异同",
    "区别",
    "联系",
    "关系",
    "趋势",
    "规律",
    "特点",
    "特征",
    "影响",
    "作用",
    "summarize",
    "overview",
    "main",
    "overall",
    "all",
    "themes",
    "compare",
    # Finance domain
    "投资策略",
    "市场概况",
    "行业分析",
    "宏观",
    "整个市场",
    "各类",
    "不同类型",
    "哪些方面",
    "综合",
    "全面",
]

DEFAULT_LOCAL_PATTERNS: list[str] = ["什么是", "是什么", "what is", "who is"]


class RouterConfig(BaseModel):
    global_keywords: list[str] = Field(default_factory=lambda: list(DEFAULT_GLOBAL_KEYWORDS))
    local_patterns: list[str] = Field(default_factory=lambda: list(DEFAULT_LOCAL_PATTERNS))
    keyword_score: float = 0.3
    global_threshold: float = 0.5
    local_pattern_penalty: float = 0.2


class FusionConfig(BaseModel):
    strategy: Literal["rrf", "max_score"] = "rrf"
    rrf_k: int = 60  # RRF constant: score = 1/(k + rank)


class RetrievalConfig(BaseModel):
    dense: DenseRetrievalConfig = Field(default_factory=DenseRetrievalConfig)
    sparse: SparseRetrievalConfig = Field(default_factory=SparseRetrievalConfig)
    graph_local: GraphLocalConfig = Field(default_factory=GraphLocalConfig)
    graph_global: GraphGlobalConfig = Field(default_factory=GraphGlobalConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)


class PathsConfig(BaseModel):
    raw_data: Path = Path("data/raw")
    processed_data: Path = Path("data/processed")
    graph_output: Path = Path("data/graphs")
    eval_data: Path = Path("eval")

    @property
    def chunks_file(self) -> Path:
        return self.processed_data / "chunks.json"

    @property
    def graph_file(self) -> Path:
        return self.graph_output / "knowledge_graph.json"

    @property
    def communities_file(self) -> Path:
        return self.graph_output / "communities.json"

    @property
    def llm_cache_dir(self) -> Path:
        return self.processed_data / ".llm_cache"


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 7860
    auth_username: str | None = None
    auth_password: SecretStr | None = None

    @property
    def auth(self) -> tuple[str, str] | None:
        if self.auth_username and self.auth_password:
            return (self.auth_username, self.auth_password.get_secret_value())
        return None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    @field_validator("llm")
    @classmethod
    def validate_api_key(cls, v: LLMConfig) -> LLMConfig:
        if not v.api_key.get_secret_value():
            raise ValueError(
                "LLM API key is not set. Set DEEPSEEK_API_KEY in your .env file "
                "or environment, or configure `llm.api_key` directly in the config file."
            )
        return v

    @classmethod
    def from_yaml(cls, path: str | Path = "configs/default.yaml", validate: bool = True) -> Settings:
        path = Path(path)
        raw: dict = {}
        if path.exists():
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
            # Resolve ${ENV_VAR} interpolation for any string value
            raw = _resolve_env_vars(raw)
        if not validate:
            # model_construct skips the top-level validate_api_key field_validator.
            # Nested models use normal __init__ so type coercion (e.g. str→Path) works.
            nested = {
                "llm": LLMConfig(**(raw.get("llm") or {})),
                "embedding": EmbeddingConfig(**(raw.get("embedding") or {})),
                "chunking": ChunkingConfig(**(raw.get("chunking") or {})),
                "graph": GraphConfig(**(raw.get("graph") or {})),
                "retrieval": RetrievalConfig(**(raw.get("retrieval") or {})),
                "paths": PathsConfig(**(raw.get("paths") or {})),
                "server": ServerConfig(**(raw.get("server") or {})),
            }
            return cls.model_construct(**nested)
        return cls(**raw)


def _resolve_env_vars(value):
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(v) for v in value]
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_var = value[2:-1]
        return os.environ.get(env_var, "")
    return value


def get_settings(config_path: str | Path | None = None, validate: bool = True) -> Settings:
    return Settings.from_yaml(config_path or "configs/default.yaml", validate=validate)
