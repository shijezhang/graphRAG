from __future__ import annotations

import logging
import re

from rank_bm25 import BM25Okapi

from src.config import SparseRetrievalConfig
from src.document.chunker import Chunk

logger = logging.getLogger(__name__)


class SparseRetriever:
    def __init__(self, config: SparseRetrievalConfig):
        self.config = config
        self._bm25: BM25Okapi | None = None
        self._chunks: list[Chunk] = []

    def index(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        tokenized = [_tokenize(c.content) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index built: {len(chunks)} documents")

    def search(self, query: str, top_k: int | None = None) -> list[tuple[Chunk, float]]:
        if self._bm25 is None:
            raise RuntimeError("Index not built. Call index() first.")

        top_k = top_k or self.config.top_k
        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        top_indices = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self._chunks[idx], float(scores[idx])))
        return results


def _tokenize(text: str) -> list[str]:
    # Simple tokenization: split on non-word chars, keep Chinese chars as individual tokens
    tokens = []
    for segment in re.split(r'\s+', text):
        i = 0
        while i < len(segment):
            char = segment[i]
            if '一' <= char <= '鿿':
                tokens.append(char)
                i += 1
            else:
                word = ""
                while i < len(segment) and not ('一' <= segment[i] <= '鿿'):
                    word += segment[i]
                    i += 1
                word = re.sub(r'[^\w]', '', word).lower()
                if word:
                    tokens.append(word)
    return tokens
