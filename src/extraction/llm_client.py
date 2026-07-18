from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import time
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Protocol, runtime_checkable

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from src.config import LLMConfig

logger = logging.getLogger(__name__)

# Transient errors worth retrying with backoff. APIStatusError covers 5xx/429
# responses that aren't already RateLimitError/APITimeoutError instances.
RETRYABLE_ERRORS = (APIConnectionError, APITimeoutError, RateLimitError, APIStatusError)


@dataclass
class LLMStats:
    """Thread-safe counters for tracking LLM call success/failure rates."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    retried_calls: int = 0
    cache_hits: int = 0
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    def record_success(self, retried: bool) -> None:
        with self._lock:
            self.total_calls += 1
            self.successful_calls += 1
            if retried:
                self.retried_calls += 1

    def record_failure(self) -> None:
        with self._lock:
            self.total_calls += 1
            self.failed_calls += 1

    def record_cache_hit(self) -> None:
        with self._lock:
            self.cache_hits += 1

    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 1.0
        return self.successful_calls / self.total_calls

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "retried_calls": self.retried_calls,
            "cache_hits": self.cache_hits,
            "success_rate": round(self.success_rate, 4),
        }


@runtime_checkable
class LLMClientProtocol(Protocol):
    """Structural protocol for LLM clients. Implementations only need the public API."""

    config: LLMConfig
    stats: LLMStats

    def chat(self, system: str, user: str, temperature: float | None = None) -> str: ...
    def chat_json(self, system: str, user: str) -> dict[str, Any]: ...
    def chat_stream(self, system: str, user: str) -> Generator[str, None, None]: ...
    def batch_chat(self, requests: list[tuple[str, str]], temperature: float | None = None) -> list[str | None]: ...
    def batch_chat_json(self, requests: list[tuple[str, str]]) -> list[dict[str, Any] | None]: ...


class LLMClient:
    def __init__(self, config: LLMConfig, cache_dir: str | Path | None = "data/processed/.llm_cache"):
        self.config = config
        self.client = OpenAI(api_key=config.api_key.get_secret_value(), base_url=config.base_url)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = LLMStats()

    def chat(self, system: str, user: str, temperature: float | None = None) -> str:
        cache_key = self._cache_key(system, user)
        cached = self._read_cache(cache_key)
        if cached is not None:
            self.stats.record_cache_hit()
            return cached

        result = self._chat_with_retry(system, user, temperature)
        self._write_cache(cache_key, result)
        return result

    def _chat_with_retry(self, system: str, user: str, temperature: float | None) -> str:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature if temperature is not None else self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                self.stats.record_success(retried=attempt > 0)
                return response.choices[0].message.content or ""
            except RETRYABLE_ERRORS as e:
                last_error = e
                if attempt < self.config.max_retries:
                    delay = self.config.retry_base_delay * (2**attempt)
                    logger.warning(
                        f"LLM call failed (attempt {attempt + 1}/{self.config.max_retries + 1}): {e}. "
                        f"Retrying in {delay:.1f}s"
                    )
                    time.sleep(delay)
            except Exception:
                self.stats.record_failure()
                raise

        self.stats.record_failure()
        raise last_error  # type: ignore[misc]

    def chat_stream(self, system: str, user: str) -> Generator[str, None, None]:
        """Stream response tokens. Does not use cache (streaming responses are not cached)."""
        stream = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def chat_json(self, system: str, user: str) -> dict[str, Any]:
        raw = self.chat(system, user)
        return _parse_json_response(raw)

    def batch_chat_json(self, requests: list[tuple[str, str]]) -> list[dict[str, Any] | None]:
        """Concurrent variant of chat_json. Failed requests yield None at their position."""
        if not requests:
            return []

        max_workers = min(self.config.concurrency, len(requests))
        results: list[dict[str, Any] | None] = [None] * len(requests)

        def _call(idx: int, system: str, user: str) -> tuple[int, dict[str, Any] | None]:
            try:
                return idx, self.chat_json(system, user)
            except Exception as e:
                logger.warning(f"batch_chat_json request {idx} failed: {e}")
                return idx, None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_call, i, system, user) for i, (system, user) in enumerate(requests)]
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        return results

    def batch_chat(self, requests: list[tuple[str, str]], temperature: float | None = None) -> list[str | None]:
        """Run chat() calls concurrently (bounded by config.concurrency).

        Returns results in input order. A failed request (after retries) yields
        None at its position rather than aborting the whole batch.
        """
        if not requests:
            return []

        max_workers = min(self.config.concurrency, len(requests))
        results: list[str | None] = [None] * len(requests)

        def _call(idx: int, system: str, user: str) -> tuple[int, str | None]:
            try:
                return idx, self.chat(system, user, temperature)
            except Exception as e:
                logger.warning(f"batch_chat request {idx} failed: {e}")
                return idx, None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_call, i, system, user) for i, (system, user) in enumerate(requests)]
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        return results

    def _cache_key(self, system: str, user: str) -> str:
        content = f"{self.config.model}|{system}|{user}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _read_cache(self, key: str) -> str | None:
        if not self.cache_dir:
            return None
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return data.get("response")
        return None

    def _write_cache(self, key: str, response: str) -> None:
        if not self.cache_dir:
            return
        cache_file = self.cache_dir / f"{key}.json"
        cache_file.write_text(
            json.dumps({"response": response}, ensure_ascii=False),
            encoding="utf-8",
        )


def create_llm_client(
    config: LLMConfig, cache_dir: str | Path | None = "data/processed/.llm_cache"
) -> LLMClientProtocol:
    """Build an LLM client for the configured provider.

    All currently supported providers (deepseek, openai, and other OpenAI-compatible
    APIs) share the same chat-completions wire format, so this always returns an
    `LLMClient` pointed at `config.base_url`. Centralizing construction here means
    callers depend on `LLMClientProtocol` rather than the concrete class, so tests
    can inject a fake implementing the same protocol.
    """
    return LLMClient(config, cache_dir=cache_dir)


def _parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Attempt to recover truncated JSON by finding the last complete array element
        for end_char in ("}\n]}", "}\n  ]\n}", "  }\n]}", "  }\n  ]\n}"):
            idx = text.rfind(end_char)
            if idx != -1:
                try:
                    return json.loads(text[: idx + len(end_char)])
                except json.JSONDecodeError:
                    continue
        # Last resort: extract whatever entities/relations parsed so far
        result: dict[str, Any] = {"entities": [], "relations": []}
        for key in ("entities", "relations"):
            start = text.find(f'"{key}"')
            if start == -1:
                continue
            bracket = text.find("[", start)
            if bracket == -1:
                continue
            depth, i = 0, bracket
            while i < len(text):
                if text[i] == "[":
                    depth += 1
                elif text[i] == "]":
                    depth -= 1
                    if depth == 0:
                        with contextlib.suppress(json.JSONDecodeError):
                            result[key] = json.loads(text[bracket : i + 1])
                        break
                i += 1
        return result
