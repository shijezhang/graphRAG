from __future__ import annotations

import logging
from difflib import SequenceMatcher

import networkx as nx

from src.config import GraphLocalConfig
from src.document.chunker import Chunk

logger = logging.getLogger(__name__)


class GraphLocalRetriever:
    def __init__(self, config: GraphLocalConfig):
        self.config = config
        self._graph: nx.Graph | None = None
        self._chunks: list[Chunk] = []

    def index(self, graph: nx.Graph, chunks: list[Chunk]) -> None:
        self._graph = graph
        self._chunks = chunks

    def search(self, query: str, top_k: int | None = None) -> list[tuple[Chunk, float]]:
        if self._graph is None:
            raise RuntimeError("Index not built. Call index() first.")

        top_k = top_k or self.config.top_k

        matched_nodes = self._match_entities(query)
        if not matched_nodes:
            return []

        relevant_chunks = self._traverse_and_collect(matched_nodes)

        scored = sorted(relevant_chunks.items(), key=lambda x: x[1], reverse=True)
        results = []
        for chunk_idx, score in scored[:top_k]:
            if chunk_idx < len(self._chunks):
                results.append((self._chunks[chunk_idx], score))
        return results

    def _match_entities(self, query: str) -> list[str]:
        query_lower = query.lower()
        matches = []
        for node, data in self._graph.nodes(data=True):
            name = data.get("name", node).lower()
            if name in query_lower or query_lower in name:
                matches.append((node, 1.0))
            else:
                sim = SequenceMatcher(None, name, query_lower).ratio()
                if sim > 0.5:
                    matches.append((node, sim))

        matches.sort(key=lambda x: x[1], reverse=True)
        return [m[0] for m in matches[:5]]

    def _traverse_and_collect(self, start_nodes: list[str]) -> dict[int, float]:
        chunk_scores: dict[int, float] = {}

        for node in start_nodes:
            subgraph_nodes = self._get_neighborhood(node, self.config.max_hops)
            for sg_node, distance in subgraph_nodes.items():
                node_data = self._graph.nodes[sg_node]
                source_chunks = node_data.get("source_chunks", [])
                if isinstance(source_chunks, str):
                    import json
                    source_chunks = json.loads(source_chunks)
                decay = 1.0 / (1.0 + distance)
                for chunk_idx in source_chunks:
                    chunk_idx = int(chunk_idx)
                    chunk_scores[chunk_idx] = max(chunk_scores.get(chunk_idx, 0), decay)

            for neighbor in self._graph.neighbors(node):
                edge_data = self._graph.edges[node, neighbor]
                source_chunks = edge_data.get("source_chunks", [])
                if isinstance(source_chunks, str):
                    import json
                    source_chunks = json.loads(source_chunks)
                for chunk_idx in source_chunks:
                    chunk_idx = int(chunk_idx)
                    chunk_scores[chunk_idx] = max(chunk_scores.get(chunk_idx, 0), 0.8)

        return chunk_scores

    def _get_neighborhood(self, node: str, max_hops: int) -> dict[str, int]:
        visited = {node: 0}
        frontier = [node]
        for hop in range(1, max_hops + 1):
            next_frontier = []
            for n in frontier:
                for neighbor in self._graph.neighbors(n):
                    if neighbor not in visited:
                        visited[neighbor] = hop
                        next_frontier.append(neighbor)
            frontier = next_frontier
        return visited
