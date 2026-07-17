"""Shared pytest fixtures."""
from __future__ import annotations

import pytest

from src.config import ChunkingConfig, Settings
from src.document.chunker import Chunk, RecursiveChunker


@pytest.fixture
def chunking_config():
    return ChunkingConfig(chunk_size=100, chunk_overlap=20)


@pytest.fixture
def chunker(chunking_config):
    return RecursiveChunker(chunking_config)


@pytest.fixture
def sample_chunks():
    return [
        Chunk(content="Apple is a technology company.", metadata={}, chunk_index=0),
        Chunk(content="Google develops search engines.", metadata={}, chunk_index=1),
        Chunk(content="Microsoft makes Windows operating system.", metadata={}, chunk_index=2),
    ]


@pytest.fixture
def settings_no_validate():
    return Settings.from_yaml("configs/default.yaml", validate=False)
