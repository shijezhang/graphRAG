from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from src.config import Settings
from src.document.chunker import Chunk
from src.graph.builder import GraphBuilder
from src.graph.community import Community
from src.retrieval.dense import DenseRetriever
from src.retrieval.graph_global import GraphGlobalRetriever
from src.retrieval.graph_local import GraphLocalRetriever
from src.retrieval.router import QueryRouter, QueryType, RoutingResult
from src.retrieval.sparse import SparseRetriever

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    query: str
    query_type: QueryType
    routing_confidence: float
    chunks: list[tuple[Chunk, float, str]]  # (chunk, score, source)
    global_answer: str | None = None


class HybridRetriever:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.router = QueryRouter(settings.llm, use_llm=False)
        self.sparse = SparseRetriever(settings.retrieval.sparse)
        self.graph_local = GraphLocalRetriever(settings.retrieval.graph_local)
        self.graph_global = GraphGlobalRetriever(settings.retrieval.graph_global, settings.llm)
        self._dense: DenseRetriever | None = None
        self._indexed = False

    @property
    def dense(self) -> DenseRetriever:
        if self._dense is None:
            self._dense = DenseRetriever(self.settings.embedding, self.settings.retrieval.dense)
        return self._dense

    def index_from_files(
        self,
        chunks_path: str | Path = "data/processed/chunks.json",
        graph_path: str | Path = "data/graphs/knowledge_graph.json",
        communities_path: str | Path = "data/graphs/communities.json",
        use_dense: bool = False,
    ) -> None:
        chunks = self._load_chunks(chunks_path)

        self.sparse.index(chunks)
        logger.info("BM25 index ready")

        if Path(graph_path).exists():
            builder = GraphBuilder.load(graph_path)
            self.graph_local.index(builder.graph, chunks)
            logger.info("Graph local index ready")

        if Path(communities_path).exists():
            self.graph_global.load_communities(communities_path)
            logger.info("Graph global index ready")

        if use_dense:
            self.dense.index(chunks)
            logger.info("Dense index ready")

        self._indexed = True

    def search(self, query: str, use_dense: bool = False) -> RetrievalResult:
        if not self._indexed:
            raise RuntimeError("Not indexed. Call index_from_files() first.")

        routing = self.router.route(query)
        logger.info(f"Query routed as {routing.query_type.value} (conf={routing.confidence:.2f})")

        if routing.query_type == QueryType.GLOBAL:
            return self._search_global(query, routing)
        return self._search_local(query, routing, use_dense)

    def _search_global(self, query: str, routing: RoutingResult) -> RetrievalResult:
        global_answer = self.graph_global.search(query)
        return RetrievalResult(
            query=query,
            query_type=QueryType.GLOBAL,
            routing_confidence=routing.confidence,
            chunks=[],
            global_answer=global_answer,
        )

    def _search_local(self, query: str, routing: RoutingResult, use_dense: bool = False) -> RetrievalResult:
        all_chunks: dict[int, tuple[Chunk, float, str]] = {}

        def _run_sparse():
            return [("bm25", chunk, score) for chunk, score in self.sparse.search(query)]

        def _run_graph_local():
            try:
                return [("graph_local", chunk, score) for chunk, score in self.graph_local.search(query)]
            except RuntimeError:
                return []

        with ThreadPoolExecutor(max_workers=2) as executor:
            sparse_future = executor.submit(_run_sparse)
            graph_future = executor.submit(_run_graph_local)
            sparse_hits = sparse_future.result()
            graph_hits = graph_future.result()

        for source, chunk, score in sparse_hits + graph_hits:
            idx = chunk.chunk_index
            if idx not in all_chunks or score > all_chunks[idx][1]:
                all_chunks[idx] = (chunk, score, source)

        # Dense (optional, sequential — only used when explicitly enabled)
        if use_dense and self._dense:
            dense_results = self.dense.search(query)
            for chunk, score in dense_results:
                idx = chunk.chunk_index
                if idx not in all_chunks or score > all_chunks[idx][1]:
                    all_chunks[idx] = (chunk, score, "dense")

        sorted_chunks = sorted(all_chunks.values(), key=lambda x: x[1], reverse=True)
        top_k = self.settings.retrieval.dense.top_k
        return RetrievalResult(
            query=query,
            query_type=QueryType.LOCAL,
            routing_confidence=routing.confidence,
            chunks=sorted_chunks[:top_k],
        )

    def _load_chunks(self, path: str | Path) -> list[Chunk]:
        return Chunk.load_from_json(path)
