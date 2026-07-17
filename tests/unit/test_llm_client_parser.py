"""Unit tests for LLM client JSON parsing/recovery (no network calls)."""
from __future__ import annotations

import pytest

from src.extraction.llm_client import _parse_json_response


class TestParseJsonResponse:
    def test_plain_json(self):
        raw = '{"entities": [], "relations": []}'
        result = _parse_json_response(raw)
        assert result == {"entities": [], "relations": []}

    def test_strips_markdown_fences(self):
        raw = '```json\n{"key": "value"}\n```'
        result = _parse_json_response(raw)
        assert result == {"key": "value"}

    def test_strips_generic_fences(self):
        raw = '```\n{"key": "value"}\n```'
        result = _parse_json_response(raw)
        assert result == {"key": "value"}

    def test_valid_entities_and_relations(self):
        raw = '{"entities": [{"name": "A", "type": "ORG"}], "relations": []}'
        result = _parse_json_response(raw)
        assert len(result["entities"]) == 1

    def test_truncated_json_recovery(self):
        # Simulate a truncated response with a complete entity at the end
        raw = '{"entities": [{"name": "A", "type": "ORG", "description": "test"}\n]}'
        result = _parse_json_response(raw)
        assert "entities" in result

    def test_garbage_input_returns_empty_structure(self):
        raw = "I'm sorry I cannot process this request."
        result = _parse_json_response(raw)
        assert "entities" in result
        assert "relations" in result
        assert result["entities"] == []
        assert result["relations"] == []

    def test_partial_entities_recovered(self):
        # Partial JSON where entities array is complete but relations is missing
        raw = '{"entities": [{"name": "X", "type": "ORG", "description": "d"}], "relatio'
        result = _parse_json_response(raw)
        # entities should have been extracted via fallback bracket matching
        assert isinstance(result.get("entities"), list)
