"""Shared test doubles."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

from src.config import LLMConfig
from src.extraction.llm_client import LLMStats


class MockLLMClient:
    """In-memory stand-in for LLMClient, satisfying LLMClientProtocol.

    Responses are consumed from a fixed queue (or produced by a callable) so
    tests can inject deterministic LLM output without any network access or
    OpenAI SDK monkeypatching.
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        json_responses: list[dict[str, Any]] | None = None,
        config: LLMConfig | None = None,
    ):
        self.config = config or LLMConfig(api_key="sk-test")
        self.stats = LLMStats()
        self._responses = list(responses or [])
        self._json_responses = list(json_responses or [])
        self.calls: list[tuple[str, str]] = []

    def chat(self, system: str, user: str, temperature: float | None = None) -> str:
        self.calls.append((system, user))
        self.stats.record_success(retried=False)
        if not self._responses:
            return ""
        return self._responses.pop(0)

    def chat_json(self, system: str, user: str) -> dict[str, Any]:
        self.calls.append((system, user))
        self.stats.record_success(retried=False)
        if not self._json_responses:
            return {}
        return self._json_responses.pop(0)

    def chat_stream(self, system: str, user: str) -> Generator[str, None, None]:
        yield self.chat(system, user)

    def batch_chat(self, requests: list[tuple[str, str]], temperature: float | None = None) -> list[str | None]:
        return [self.chat(system, user, temperature) for system, user in requests]

    def batch_chat_json(self, requests: list[tuple[str, str]]) -> list[dict[str, Any] | None]:
        return [self.chat_json(system, user) for system, user in requests]
