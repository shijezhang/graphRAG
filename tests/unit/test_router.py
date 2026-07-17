"""Unit tests for query router heuristic."""

from __future__ import annotations

import pytest

from src.retrieval.router import QueryRouter, QueryType


@pytest.fixture
def router():
    return QueryRouter(use_llm=False)


class TestQueryRouterHeuristic:
    def test_global_keywords_route_global(self, router):
        result = router.route("总结这篇文章的主要内容")
        assert result.query_type == QueryType.GLOBAL

    def test_what_is_question_routes_local(self, router):
        result = router.route("苹果是什么？")
        assert result.query_type == QueryType.LOCAL

    def test_specific_entity_question_routes_local(self, router):
        result = router.route("第三章提到了哪个公司？")
        assert result.query_type == QueryType.LOCAL

    def test_overview_question_routes_global(self, router):
        result = router.route("整体来看，核心观点是什么")
        assert result.query_type == QueryType.GLOBAL

    def test_english_summarize_routes_global(self, router):
        result = router.route("summarize the main themes")
        assert result.query_type == QueryType.GLOBAL

    def test_confidence_is_bounded(self, router):
        result = router.route("anything")
        assert 0.0 <= result.confidence <= 1.0
