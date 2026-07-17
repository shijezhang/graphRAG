from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Generator

from openai import OpenAI

from src.config import LLMConfig


class LLMClient:
    def __init__(self, config: LLMConfig, cache_dir: str | Path | None = "data/processed/.llm_cache"):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def chat(self, system: str, user: str, temperature: float | None = None) -> str:
        cache_key = self._cache_key(system, user)
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        result = response.choices[0].message.content or ""
        self._write_cache(cache_key, result)
        return result

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

    def batch_chat(self, requests: list[tuple[str, str]], temperature: float | None = None) -> list[str]:
        results = []
        for system, user in requests:
            results.append(self.chat(system, user, temperature))
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
                        try:
                            result[key] = json.loads(text[bracket : i + 1])
                        except json.JSONDecodeError:
                            pass
                        break
                i += 1
        return result
