"""Unit tests for document chunker, including CJK overlap bug."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.config import ChunkingConfig
from src.document.chunker import Chunk, RecursiveChunker, _estimate_tokens


class TestEstimateTokens:
    def test_english_text(self):
        # "hello world" = 11 chars, ~4 chars/token → ~2
        assert _estimate_tokens("hello world") > 0

    def test_chinese_text(self):
        # Chinese chars counted at 1.5 chars/token
        text = "这是一段中文文本"  # 8 chars
        assert _estimate_tokens(text) > 0

    def test_empty_text(self):
        assert _estimate_tokens("") == 0


class TestChunkLoadFromJson:
    def test_roundtrip(self, tmp_path):
        chunks = [
            Chunk(content="hello world", metadata={"source": "test.txt"}, chunk_index=0),
            Chunk(content="foo bar", metadata={}, chunk_index=1),
        ]
        data = [{"content": c.content, "metadata": c.metadata, "chunk_index": c.chunk_index} for c in chunks]
        p = tmp_path / "chunks.json"
        p.write_text(json.dumps(data), encoding="utf-8")

        loaded = Chunk.load_from_json(p)
        assert len(loaded) == 2
        assert loaded[0].content == "hello world"
        assert loaded[0].chunk_index == 0
        assert loaded[1].content == "foo bar"

    def test_missing_chunk_index_defaults_to_enumerate(self, tmp_path):
        data = [{"content": "a"}, {"content": "b"}]
        p = tmp_path / "chunks.json"
        p.write_text(json.dumps(data), encoding="utf-8")

        loaded = Chunk.load_from_json(p)
        assert loaded[0].chunk_index == 0
        assert loaded[1].chunk_index == 1


class TestRecursiveChunker:
    def test_short_doc_produces_single_chunk(self):
        config = ChunkingConfig(chunk_size=500, chunk_overlap=0)
        chunker = RecursiveChunker(config)
        from src.document.loader import Document

        doc = Document(content="Short text.", metadata={"source": "test"})
        chunks = chunker.chunk_documents([doc])
        assert len(chunks) == 1
        assert chunks[0].content == "Short text."

    def test_empty_doc_produces_no_chunks(self):
        config = ChunkingConfig(chunk_size=100, chunk_overlap=0)
        chunker = RecursiveChunker(config)
        from src.document.loader import Document

        doc = Document(content="   ", metadata={})
        chunks = chunker.chunk_documents([doc])
        assert chunks == []

    def test_long_english_doc_is_split(self):
        config = ChunkingConfig(chunk_size=20, chunk_overlap=0)
        chunker = RecursiveChunker(config)
        from src.document.loader import Document

        # Build text longer than 20 estimated tokens
        long_text = "word " * 200
        doc = Document(content=long_text, metadata={})
        chunks = chunker.chunk_documents([doc])
        assert len(chunks) > 1

    def test_cjk_overlap_bug(self):
        """CJK text with no spaces: _get_overlap uses text.split() which fails.

        This test documents the known bug: for CJK content, overlap produces
        empty or wrong results because split() on whitespace finds no words.
        Once the P1 fix lands, this test should be updated to assert correct behavior.
        """
        config = ChunkingConfig(chunk_size=20, chunk_overlap=10)
        chunker = RecursiveChunker(config)

        # Pure CJK text — no spaces, so text.split() returns a single item
        cjk_text = "这是一段没有空格的中文文本" * 30
        overlap = chunker._get_overlap(cjk_text)
        # Document current (buggy) behavior: overlap is empty because split() finds no tokens
        assert overlap == ""
