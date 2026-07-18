"""Score fusion strategies for combining ranked results from multiple retrievers."""

from __future__ import annotations

from src.config import FusionConfig
from src.document.chunker import Chunk

# One retriever's hits: list of (chunk, score), already sorted best-first by that retriever.
SourceHits = list[tuple[Chunk, float]]


def fuse(hits_by_source: dict[str, SourceHits], config: FusionConfig) -> list[tuple[Chunk, float, str]]:
    """Fuse per-source ranked hits into a single ranked list of (chunk, fused_score, source).

    Each chunk keeps the name of the source that ranked it highest.
    """
    if config.strategy == "rrf":
        return _fuse_rrf(hits_by_source, config.rrf_k)
    return _fuse_max_score(hits_by_source)


def _fuse_rrf(hits_by_source: dict[str, SourceHits], k: int) -> list[tuple[Chunk, float, str]]:
    fused: dict[int, float] = {}
    best_source: dict[int, tuple[str, float]] = {}  # chunk_idx -> (source, source_score) for the top-ranked source
    chunks_by_idx: dict[int, Chunk] = {}

    for source, hits in hits_by_source.items():
        for rank, (chunk, score) in enumerate(hits):
            idx = chunk.chunk_index
            chunks_by_idx[idx] = chunk
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
            if idx not in best_source or score > best_source[idx][1]:
                best_source[idx] = (source, score)

    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return [(chunks_by_idx[idx], fused_score, best_source[idx][0]) for idx, fused_score in ranked]


def _fuse_max_score(hits_by_source: dict[str, SourceHits]) -> list[tuple[Chunk, float, str]]:
    all_chunks: dict[int, tuple[Chunk, float, str]] = {}
    for source, hits in hits_by_source.items():
        for chunk, score in hits:
            idx = chunk.chunk_index
            if idx not in all_chunks or score > all_chunks[idx][1]:
                all_chunks[idx] = (chunk, score, source)
    return sorted(all_chunks.values(), key=lambda x: x[1], reverse=True)
