from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src.config import DenseRetrievalConfig, EmbeddingConfig
from src.document.chunker import Chunk

logger = logging.getLogger(__name__)


class DenseRetriever:
    def __init__(self, embedding_config: EmbeddingConfig, retrieval_config: DenseRetrievalConfig):
        self.embedding_config = embedding_config
        self.retrieval_config = retrieval_config
        self._model = None
        self._index = None
        self._chunks: list[Chunk] = []

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.embedding_config.model, device=self.embedding_config.device)
        return self._model

    def index(self, chunks: list[Chunk]) -> None:
        import faiss

        self._chunks = chunks
        texts = [c.content for c in chunks]

        logger.info(f"Encoding {len(texts)} chunks...")
        embeddings = self.model.encode(texts, batch_size=self.embedding_config.batch_size, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        self._index.add(embeddings)
        logger.info(f"FAISS index built: {self._index.ntotal} vectors, dim={dim}")

    def search(self, query: str, top_k: int | None = None) -> list[tuple[Chunk, float]]:
        import faiss

        if self._index is None:
            raise RuntimeError("Index not built. Call index() first.")

        top_k = top_k or self.retrieval_config.top_k
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding, dtype=np.float32)
        faiss.normalize_L2(query_embedding)

        scores, indices = self._index.search(query_embedding, min(top_k, len(self._chunks)))

        results = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx < 0:
                continue
            results.append((self._chunks[idx], float(score)))
        return results

    def save(self, path: str | Path) -> None:
        import faiss

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self._index:
            faiss.write_index(self._index, str(path / "dense.index"))
        chunks_data = [
            {"content": c.content, "metadata": c.metadata, "chunk_index": c.chunk_index} for c in self._chunks
        ]
        (path / "dense_chunks.json").write_text(json.dumps(chunks_data, ensure_ascii=False), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        import faiss

        path = Path(path)
        self._index = faiss.read_index(str(path / "dense.index"))
        chunks_data = json.loads((path / "dense_chunks.json").read_text(encoding="utf-8"))
        self._chunks = [
            Chunk(content=c["content"], metadata=c.get("metadata", {}), chunk_index=c.get("chunk_index", i))
            for i, c in enumerate(chunks_data)
        ]
