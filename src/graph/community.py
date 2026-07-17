from __future__ import annotations

import logging
from dataclasses import dataclass, field

import igraph as ig
import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class Community:
    id: int
    level: int
    node_keys: list[str] = field(default_factory=list)
    summary: str = ""
    title: str = ""
    key_findings: list[str] = field(default_factory=list)
    importance_score: float = 0.0


class CommunityDetector:
    def __init__(self, algorithm: str = "leiden", max_levels: int = 3):
        self.algorithm = algorithm
        self.max_levels = max_levels

    def detect(self, graph: nx.Graph, resolutions: list[float] | None = None) -> list[list[Community]]:
        if graph.number_of_nodes() == 0:
            return []

        if resolutions is None:
            resolutions = self._default_resolutions()

        ig_graph, node_mapping = self._nx_to_igraph(graph)
        reverse_mapping = {v: k for k, v in node_mapping.items()}

        hierarchy: list[list[Community]] = []

        for level, resolution in enumerate(resolutions[: self.max_levels]):
            if self.algorithm == "leiden":
                partition = ig_graph.community_leiden(
                    objective_function="modularity",
                    resolution=resolution,
                )
            else:
                partition = ig_graph.community_multilevel(resolution=resolution)

            communities = self._partition_to_communities(partition, reverse_mapping, level)
            if communities:
                hierarchy.append(communities)
                logger.info(
                    f"Level {level} (resolution={resolution:.2f}): "
                    f"{len(communities)} communities"
                )

        return hierarchy

    def _default_resolutions(self) -> list[float]:
        return [0.5, 1.0, 2.0, 4.0][: self.max_levels]

    def _nx_to_igraph(self, graph: nx.Graph) -> tuple[ig.Graph, dict[str, int]]:
        node_list = list(graph.nodes())
        node_mapping = {node: i for i, node in enumerate(node_list)}

        ig_graph = ig.Graph(n=len(node_list), directed=False)

        edges = []
        weights = []
        for u, v, data in graph.edges(data=True):
            edges.append((node_mapping[u], node_mapping[v]))
            weights.append(data.get("weight", 1.0))

        ig_graph.add_edges(edges)
        ig_graph.es["weight"] = weights

        return ig_graph, node_mapping

    def _partition_to_communities(
        self,
        partition: ig.VertexClustering,
        reverse_mapping: dict[int, str],
        level: int,
    ) -> list[Community]:
        communities = []
        for comm_id, members in enumerate(partition):
            if not members:
                continue
            node_keys = [reverse_mapping[m] for m in members]
            communities.append(
                Community(id=comm_id, level=level, node_keys=node_keys)
            )
        return communities
