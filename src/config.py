from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class LLMConfig(BaseModel):
    provider: str = "deepseek"
    model: str = "deepseek-chat"
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    temperature: float = 0.0
    max_tokens: int = 4096


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


class RetrievalConfig(BaseModel):
    dense: DenseRetrievalConfig = Field(default_factory=DenseRetrievalConfig)
    sparse: SparseRetrievalConfig = Field(default_factory=SparseRetrievalConfig)
    graph_local: GraphLocalConfig = Field(default_factory=GraphLocalConfig)
    graph_global: GraphGlobalConfig = Field(default_factory=GraphGlobalConfig)


class PathsConfig(BaseModel):
    raw_data: Path = Path("data/raw")
    processed_data: Path = Path("data/processed")
    graph_output: Path = Path("data/graphs")
    eval_data: Path = Path("eval")


class Settings(BaseSettings):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    @classmethod
    def from_yaml(cls, path: str | Path = "configs/default.yaml") -> Settings:
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path) as f:
            raw = yaml.safe_load(f)
        # Resolve env vars in api_key
        if "llm" in raw and "api_key" in raw["llm"]:
            key = raw["llm"]["api_key"]
            if key.startswith("${") and key.endswith("}"):
                env_var = key[2:-1]
                raw["llm"]["api_key"] = os.environ.get(env_var, "")
        return cls(**raw)


def get_settings(config_path: str | Path | None = None) -> Settings:
    if config_path:
        return Settings.from_yaml(config_path)
    return Settings.from_yaml()
