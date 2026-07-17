"""Unit tests for GraphBuilder deduplication and string_similarity."""

from __future__ import annotations

from src.extraction.entity_extractor import Entity, ExtractionResult, Relation
from src.graph.builder import GraphBuilder
from src.utils import string_similarity


class TestStringSimilarity:
    def test_identical_strings(self):
        assert string_similarity("hello", "hello") == 1.0

    def test_empty_strings(self):
        assert string_similarity("", "") == 1.0

    def test_completely_different(self):
        assert string_similarity("abc", "xyz") < 0.5

    def test_case_insensitive(self):
        assert string_similarity("Apple", "apple") == 1.0

    def test_partial_overlap(self):
        sim = string_similarity("Apple Inc", "Apple")
        assert 0.5 < sim < 1.0


class TestGraphBuilderDedup:
    def _make_entity(self, name: str, source_chunks: list[int] | None = None) -> Entity:
        return Entity(
            name=name,
            type="ORG",
            description=f"{name} description",
            source_chunks=source_chunks or [0],
        )

    def test_near_duplicate_entities_merged(self):
        entities = [
            self._make_entity("Apple Inc", [0]),
            self._make_entity("Apple Inc.", [1]),  # very similar, should merge
        ]
        result = ExtractionResult(entities=entities, relations=[])
        builder = GraphBuilder()
        graph = builder.build_from_extraction(result, deduplicate=True)
        # Should have only 1 node since the names are near-identical
        assert graph.number_of_nodes() == 1

    def test_distinct_entities_kept_separate(self):
        entities = [
            self._make_entity("Apple"),
            self._make_entity("Google"),
            self._make_entity("Microsoft"),
        ]
        result = ExtractionResult(entities=entities, relations=[])
        builder = GraphBuilder()
        graph = builder.build_from_extraction(result, deduplicate=True)
        assert graph.number_of_nodes() == 3

    def test_self_loop_relations_dropped(self):
        e = self._make_entity("Apple")
        rel = Relation(source="Apple", target="Apple", relation="owns", description="", weight=1.0, source_chunks=[0])
        result = ExtractionResult(entities=[e], relations=[rel])
        builder = GraphBuilder()
        graph = builder.build_from_extraction(result)
        assert graph.number_of_edges() == 0

    def test_dedup_false_keeps_all_entities(self):
        entities = [
            self._make_entity("Apple Inc", [0]),
            self._make_entity("Apple Inc.", [1]),
        ]
        result = ExtractionResult(entities=entities, relations=[])
        builder = GraphBuilder()
        graph = builder.build_from_extraction(result, deduplicate=False)
        assert graph.number_of_nodes() == 2
