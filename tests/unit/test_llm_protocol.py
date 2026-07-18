"""Tests for LLM client injection via LLMClientProtocol."""

from __future__ import annotations

import networkx as nx
import pytest

from src.config import LLMConfig
from src.document.chunker import Chunk
from src.extraction.entity_extractor import EntityRelationExtractor
from src.extraction.llm_client import LLMClientProtocol
from src.graph.community import Community
from src.graph.summarizer import CommunitySummarizer
from tests.helpers import MockLLMClient


def _chunk(idx: int, text: str = "content") -> Chunk:
    return Chunk(content=text, metadata={}, chunk_index=idx)


@pytest.fixture
def dummy_config() -> LLMConfig:
    return LLMConfig(api_key="sk-test")


class TestMockSatisfiesProtocol:
    def test_isinstance_check(self, dummy_config):
        mock = MockLLMClient(config=dummy_config)
        # runtime_checkable lets us verify structural compatibility
        assert isinstance(mock, LLMClientProtocol)


class TestEntityExtractorInjection:
    def test_uses_injected_client(self, dummy_config):
        json_response = {
            "entities": [{"name": "Alice", "type": "PERSON", "description": "test"}],
            "relations": [],
        }
        mock = MockLLMClient(json_responses=[json_response], config=dummy_config)
        extractor = EntityRelationExtractor(llm_config=dummy_config, llm=mock)
        result = extractor.extract_from_chunks([_chunk(0)])

        assert len(result.entities) == 1
        assert result.entities[0].name == "Alice"
        assert len(mock.calls) == 1

    def test_empty_chunks_skips_llm(self, dummy_config):
        mock = MockLLMClient(config=dummy_config)
        extractor = EntityRelationExtractor(llm_config=dummy_config, llm=mock)
        result = extractor.extract_from_chunks([])
        assert result.entities == []
        assert mock.calls == []


class TestCommunitySummarizerInjection:
    def _make_community(self, comm_id: int, node_key: str) -> Community:
        return Community(id=comm_id, level=0, node_keys=[node_key])

    def test_uses_injected_client(self, dummy_config):
        json_response = {
            "title": "Topic",
            "summary": "Summary text",
            "key_findings": ["finding"],
            "importance_score": 0.8,
        }
        mock = MockLLMClient(json_responses=[json_response], config=dummy_config)
        summarizer = CommunitySummarizer(llm_config=dummy_config, llm=mock)

        graph = nx.Graph()
        graph.add_node("nodeA", name="NodeA", type="CONCEPT", description="desc", source_chunks=[])
        community = self._make_community(0, "nodeA")

        results = summarizer.summarize_communities([community], graph)
        assert results[0].title == "Topic"
        assert len(mock.calls) == 1
