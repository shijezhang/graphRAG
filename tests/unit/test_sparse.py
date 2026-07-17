"""Unit tests for SparseRetriever tokenization and search."""
from __future__ import annotations

import pytest

from src.config import SparseRetrievalConfig
from src.document.chunker import Chunk
from src.retrieval.sparse import SparseRetriever, _tokenize


class TestTokenize:
    def test_english_lowercased(self):
        tokens = _tokenize("Hello World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_chinese_chars_individual(self):
        tokens = _tokenize("中文")
        assert "中" in tokens
        assert "文" in tokens

    def test_mixed_text(self):
        tokens = _tokenize("Apple 苹果")
        assert "apple" in tokens
        assert "苹" in tokens
        assert "果" in tokens

    def test_punctuation_stripped(self):
        tokens = _tokenize("hello, world!")
        assert "hello" in tokens
        assert "," not in tokens


class TestSparseRetriever:
    def test_raises_when_not_indexed(self):
        retriever = SparseRetriever(SparseRetrievalConfig(top_k=3))
        with pytest.raises(RuntimeError, match="Index not built"):
            retriever.search("test")

    def test_returns_relevant_results(self, sample_chunks):
        retriever = SparseRetriever(SparseRetrievalConfig(top_k=3))
        retriever.index(sample_chunks)

        results = retriever.search("technology company Apple")
        assert len(results) >= 1
        top_chunk, score = results[0]
        assert score > 0
        assert "Apple" in top_chunk.content

    def test_top_k_respected(self, sample_chunks):
        retriever = SparseRetriever(SparseRetrievalConfig(top_k=1))
        retriever.index(sample_chunks)
        results = retriever.search("company")
        assert len(results) <= 1

    def test_no_results_for_unrelated_query(self, sample_chunks):
        retriever = SparseRetriever(SparseRetrievalConfig(top_k=5))
        retriever.index(sample_chunks)
        # Query with no overlap at all — score is 0, so nothing returned
        results = retriever.search("xyzzy quux blargh")
        assert results == []
