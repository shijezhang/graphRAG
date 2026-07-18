"""Unit tests for score fusion strategies."""

from __future__ import annotations

import pytest

from src.config import FusionConfig
from src.document.chunker import Chunk
from src.retrieval.fusion import fuse


def _chunk(idx: int) -> Chunk:
    return Chunk(content=f"chunk-{idx}", metadata={}, chunk_index=idx)


@pytest.fixture
def chunks():
    return {i: _chunk(i) for i in range(5)}


class TestRRFFusion:
    def test_rrf_rewards_agreement_across_sources(self, chunks):
        # Chunk 0 ranks mid in both sources; chunk 1 tops bm25 only; chunk 2 tops graph only.
        # With plain max-score, bm25's large scores would dominate. RRF should reward
        # the chunk that appears in BOTH lists.
        hits = {
            "bm25": [(chunks[1], 20.0), (chunks[0], 15.0), (chunks[3], 10.0)],
            "graph_local": [(chunks[2], 0.9), (chunks[0], 0.8), (chunks[4], 0.5)],
        }
        config = FusionConfig(strategy="rrf", rrf_k=60)
        result = fuse(hits, config)

        ranked_idx = [chunk.chunk_index for chunk, _, _ in result]
        # Chunk 0 appears in both → highest fused rank
        assert ranked_idx[0] == 0

    def test_rrf_score_is_sum_of_reciprocal_ranks(self, chunks):
        hits = {
            "a": [(chunks[0], 5.0)],  # rank 0 → 1/(60+1)
            "b": [(chunks[0], 0.1)],  # rank 0 → 1/(60+1)
        }
        config = FusionConfig(strategy="rrf", rrf_k=60)
        result = fuse(hits, config)
        assert len(result) == 1
        expected = 1.0 / 61 + 1.0 / 61
        assert result[0][1] == pytest.approx(expected)

    def test_rrf_source_is_highest_scoring(self, chunks):
        hits = {
            "bm25": [(chunks[0], 20.0)],
            "graph_local": [(chunks[0], 0.9)],
        }
        config = FusionConfig(strategy="rrf", rrf_k=60)
        result = fuse(hits, config)
        # bm25 had the higher raw score, so it is recorded as the source
        assert result[0][2] == "bm25"

    def test_rrf_empty_input(self):
        config = FusionConfig(strategy="rrf", rrf_k=60)
        assert fuse({}, config) == []

    def test_rrf_rank_matters_not_score_magnitude(self, chunks):
        # Single source: order must follow rank, and scores decrease monotonically.
        hits = {"bm25": [(chunks[2], 100.0), (chunks[1], 5.0), (chunks[0], 4.9)]}
        config = FusionConfig(strategy="rrf", rrf_k=60)
        result = fuse(hits, config)
        ranked_idx = [chunk.chunk_index for chunk, _, _ in result]
        assert ranked_idx == [2, 1, 0]
        scores = [s for _, s, _ in result]
        assert scores == sorted(scores, reverse=True)


class TestMaxScoreFusion:
    def test_max_score_keeps_highest(self, chunks):
        hits = {
            "bm25": [(chunks[0], 20.0)],
            "graph_local": [(chunks[0], 0.9)],
        }
        config = FusionConfig(strategy="max_score")
        result = fuse(hits, config)
        assert len(result) == 1
        assert result[0][1] == 20.0
        assert result[0][2] == "bm25"

    def test_max_score_empty_input(self):
        config = FusionConfig(strategy="max_score")
        assert fuse({}, config) == []
