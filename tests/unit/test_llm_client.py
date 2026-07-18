"""Unit tests for LLMClient retry, stats, and batch concurrency.

Uses a fake OpenAI-style client to avoid real network calls.
"""

from __future__ import annotations

import httpx
import pytest
from openai import APITimeoutError, RateLimitError

from src.config import LLMConfig
from src.extraction.llm_client import LLMClient, LLMStats


def _make_client(monkeypatch, tmp_path, responder, **config_overrides):
    """Build an LLMClient whose underlying OpenAI call is replaced by `responder`.

    `responder(system, user)` is called for each chat request and should either
    return a string (the assistant message content) or raise an exception.
    """
    cfg = LLMConfig(
        api_key="sk-test",
        max_retries=config_overrides.pop("max_retries", 2),
        retry_base_delay=config_overrides.pop("retry_base_delay", 0.0),
        concurrency=config_overrides.pop("concurrency", 4),
        **config_overrides,
    )
    client = LLMClient(cfg, cache_dir=tmp_path / "cache")

    class _Msg:
        def __init__(self, content):
            self.content = content

    class _Choice:
        def __init__(self, content):
            self.message = _Msg(content)

    class _Resp:
        def __init__(self, content):
            self.choices = [_Choice(content)]

    def _create(*, model, messages, temperature, max_tokens, **kwargs):
        system = messages[0]["content"]
        user = messages[1]["content"]
        content = responder(system, user)
        return _Resp(content)

    monkeypatch.setattr(client.client.chat.completions, "create", _create)
    # sleep is called between retries; make it instant
    monkeypatch.setattr("src.extraction.llm_client.time.sleep", lambda _s: None)
    return client


def _timeout_error() -> APITimeoutError:
    return APITimeoutError(request=httpx.Request("POST", "https://api.deepseek.com/chat"))


def _rate_limit_error() -> RateLimitError:
    resp = httpx.Response(429, request=httpx.Request("POST", "https://api.deepseek.com/chat"))
    return RateLimitError("rate limited", response=resp, body=None)


class TestLLMStats:
    def test_success_rate_empty_is_one(self):
        assert LLMStats().success_rate == 1.0

    def test_success_rate_computed(self):
        stats = LLMStats()
        stats.record_success(retried=False)
        stats.record_success(retried=True)
        stats.record_failure()
        assert stats.total_calls == 3
        assert stats.successful_calls == 2
        assert stats.failed_calls == 1
        assert stats.retried_calls == 1
        assert stats.success_rate == pytest.approx(2 / 3)

    def test_as_dict_keys(self):
        d = LLMStats().as_dict()
        assert set(d) == {
            "total_calls",
            "successful_calls",
            "failed_calls",
            "retried_calls",
            "cache_hits",
            "success_rate",
        }


class TestChatRetry:
    def test_succeeds_first_try(self, monkeypatch, tmp_path):
        client = _make_client(monkeypatch, tmp_path, lambda s, u: "ok")
        assert client.chat("sys", "user") == "ok"
        assert client.stats.successful_calls == 1
        assert client.stats.retried_calls == 0

    def test_retries_then_succeeds(self, monkeypatch, tmp_path):
        calls = {"n": 0}

        def responder(system, user):
            calls["n"] += 1
            if calls["n"] < 2:
                raise _timeout_error()
            return "recovered"

        client = _make_client(monkeypatch, tmp_path, responder, max_retries=3)
        assert client.chat("sys", "user") == "recovered"
        assert calls["n"] == 2
        assert client.stats.successful_calls == 1
        assert client.stats.retried_calls == 1

    def test_exhausts_retries_and_raises(self, monkeypatch, tmp_path):
        def responder(system, user):
            raise _rate_limit_error()

        client = _make_client(monkeypatch, tmp_path, responder, max_retries=2)
        with pytest.raises(RateLimitError):
            client.chat("sys", "user")
        # 1 initial + 2 retries = 3 attempts, but only one failure recorded
        assert client.stats.failed_calls == 1
        assert client.stats.successful_calls == 0

    def test_non_retryable_error_raises_immediately(self, monkeypatch, tmp_path):
        calls = {"n": 0}

        def responder(system, user):
            calls["n"] += 1
            raise ValueError("bad request")

        client = _make_client(monkeypatch, tmp_path, responder, max_retries=3)
        with pytest.raises(ValueError):
            client.chat("sys", "user")
        assert calls["n"] == 1  # no retries for non-retryable errors
        assert client.stats.failed_calls == 1


class TestCache:
    def test_cache_hit_skips_call(self, monkeypatch, tmp_path):
        calls = {"n": 0}

        def responder(system, user):
            calls["n"] += 1
            return "value"

        client = _make_client(monkeypatch, tmp_path, responder)
        first = client.chat("sys", "user")
        second = client.chat("sys", "user")
        assert first == second == "value"
        assert calls["n"] == 1  # second served from cache
        assert client.stats.cache_hits == 1


class TestBatchChat:
    def test_batch_chat_preserves_order(self, monkeypatch, tmp_path):
        def responder(system, user):
            return user.upper()

        client = _make_client(monkeypatch, tmp_path, responder)
        requests = [("sys", "a"), ("sys", "b"), ("sys", "c")]
        assert client.batch_chat(requests) == ["A", "B", "C"]

    def test_batch_chat_empty(self, monkeypatch, tmp_path):
        client = _make_client(monkeypatch, tmp_path, lambda s, u: "x")
        assert client.batch_chat([]) == []

    def test_batch_chat_failure_yields_none(self, monkeypatch, tmp_path):
        def responder(system, user):
            if user == "bad":
                raise _timeout_error()
            return user

        client = _make_client(monkeypatch, tmp_path, responder, max_retries=0)
        result = client.batch_chat([("sys", "good"), ("sys", "bad"), ("sys", "fine")])
        assert result == ["good", None, "fine"]

    def test_batch_chat_json_parses(self, monkeypatch, tmp_path):
        def responder(system, user):
            return '{"entities": [], "relations": []}'

        client = _make_client(monkeypatch, tmp_path, responder)
        result = client.batch_chat_json([("sys", "a"), ("sys", "b")])
        assert result == [{"entities": [], "relations": []}, {"entities": [], "relations": []}]
