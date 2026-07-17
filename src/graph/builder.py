from __future__ import annotations

import json
import logging
from difflib import SequenceMatcher
from pathlib import Path

import networkx as nx

from src.extraction.entity_extractor import Entity, ExtractionResult, Relation

logger = logging.getLogger(__name__)


class GraphBuilder:
    def __init__(self):
        self.graph = nx.Graph()
        self._entity_registry: dict[str, Entity] = {}

    def build_from_extraction(self, result: ExtractionResult, deduplicate: bool = True) -> nx.Graph:
        if deduplicate:
            result = self._deduplicate(result)

        for entity in result.entities:
            self._add_entity(entity)

        for relation in result.relations:
            self._add_relation(relation)

        logger.info(
            f"Graph built: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )
        return self.graph

    def _add_entity(self, entity: Entity) -> None:
        key = entity.key
        if self.graph.has_node(key):
            node_data = self.graph.nodes[key]
            node_data["source_chunks"] = list(
                set(node_data.get("source_chunks", []) + entity.source_chunks)
            )
            if entity.description and len(entity.description) > len(node_data.get("description", "")):
                node_data["description"] = entity.description
        else:
            self.graph.add_node(
                key,
                name=entity.name,
                type=entity.type,
                description=entity.description,
                source_chunks=entity.source_chunks,
            )

    def _add_relation(self, relation: Relation) -> None:
        source_key = relation.source.lower().strip()
        target_key = relation.target.lower().strip()

        if not self.graph.has_node(source_key):
            self.graph.add_node(source_key, name=relation.source, type="OTHER", description="", source_chunks=[])
        if not self.graph.has_node(target_key):
            self.graph.add_node(target_key, name=relation.target, type="OTHER", description="", source_chunks=[])

        if self.graph.has_edge(source_key, target_key):
            edge_data = self.graph.edges[source_key, target_key]
            edge_data["weight"] = max(edge_data.get("weight", 0), relation.weight)
            edge_data["source_chunks"] = list(
                set(edge_data.get("source_chunks", []) + relation.source_chunks)
            )
            descriptions = edge_data.get("descriptions", [])
            if relation.description and relation.description not in descriptions:
                descriptions.append(relation.description)
                edge_data["descriptions"] = descriptions
        else:
            self.graph.add_edge(
                source_key,
                target_key,
                relation=relation.relation,
                description=relation.description,
                descriptions=[relation.description] if relation.description else [],
                weight=relation.weight,
                source_chunks=relation.source_chunks,
            )

    def _deduplicate(self, result: ExtractionResult) -> ExtractionResult:
        name_map: dict[str, str] = {}
        unique_entities: dict[str, Entity] = {}

        for entity in result.entities:
            key = entity.key
            matched_key = self._find_match(key, unique_entities)
            if matched_key:
                name_map[key] = matched_key
                existing = unique_entities[matched_key]
                existing.source_chunks = list(set(existing.source_chunks + entity.source_chunks))
                if len(entity.description) > len(existing.description):
                    existing.description = entity.description
            else:
                unique_entities[key] = entity
                name_map[key] = key

        merged_relations: list[Relation] = []
        for rel in result.relations:
            source_key = rel.source.lower().strip()
            target_key = rel.target.lower().strip()
            rel.source = name_map.get(source_key, source_key)
            rel.target = name_map.get(target_key, target_key)
            if rel.source != rel.target:
                merged_relations.append(rel)

        return ExtractionResult(
            entities=list(unique_entities.values()),
            relations=merged_relations,
        )

    def _find_match(self, key: str, registry: dict[str, Entity], threshold: float = 0.85) -> str | None:
        for existing_key in registry:
            if _similarity(key, existing_key) >= threshold:
                return existing_key
        return None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix == ".graphml":
            g = self.graph.copy()
            for _, data in g.nodes(data=True):
                for k, v in data.items():
                    if isinstance(v, list):
                        data[k] = json.dumps(v, ensure_ascii=False)
            for _, _, data in g.edges(data=True):
                for k, v in data.items():
                    if isinstance(v, list):
                        data[k] = json.dumps(v, ensure_ascii=False)
            nx.write_graphml(g, str(path))
        else:
            data = nx.node_link_data(self.graph)
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Graph saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "GraphBuilder":
        path = Path(path)
        builder = cls()
        if path.suffix == ".graphml":
            builder.graph = nx.read_graphml(str(path))
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            builder.graph = nx.node_link_graph(data)
        return builder


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()
