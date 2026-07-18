"""Document chunking with recursive splitting and overlap."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.config import ChunkingConfig
from src.document.loader import Document


@dataclass
class Chunk:
    content: str
    metadata: dict = field(default_factory=dict)
    chunk_index: int = 0

    @property
    def source(self) -> str:
        return self.metadata.get("source", "")

    @property
    def token_count(self) -> int:
        return self.metadata.get("token_count", 0)

    @staticmethod
    def load_from_json(path: str | Path) -> list[Chunk]:
        """Load chunks from JSON file.

        Args:
            path: Path to chunks JSON file

        Returns:
            List of Chunk objects
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return [
            Chunk(content=c["content"], metadata=c.get("metadata", {}), chunk_index=c.get("chunk_index", i))
            for i, c in enumerate(data)
        ]


class RecursiveChunker:
    def __init__(self, config: ChunkingConfig | None = None):
        config = config or ChunkingConfig()
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        self.separators = config.separators

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        chunks = []
        for doc in documents:
            doc_chunks = self._split_document(doc)
            chunks.extend(doc_chunks)
        return chunks

    def _split_document(self, doc: Document) -> list[Chunk]:
        text = doc.content
        if not text.strip():
            return []

        splits = self._recursive_split(text, self.separators)
        merged = self._merge_splits(splits)

        chunks = []
        for i, content in enumerate(merged):
            chunk = Chunk(
                content=content,
                metadata={**doc.metadata, "chunk_index": i, "token_count": _estimate_tokens(content)},
                chunk_index=i,
            )
            chunks.append(chunk)
        return chunks

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return [text] if text else []

        separator = separators[0]
        remaining_separators = separators[1:]

        splits = text.split(separator) if separator else list(text)
        final_splits = []
        current = ""

        for split in splits:
            piece = split if not separator else (split + separator if split != splits[-1] else split)
            if _estimate_tokens(current + piece) <= self.chunk_size:
                current += piece
            else:
                if current:
                    if _estimate_tokens(current) > self.chunk_size and remaining_separators:
                        final_splits.extend(self._recursive_split(current, remaining_separators))
                    else:
                        final_splits.append(current)
                current = piece

        if current:
            if _estimate_tokens(current) > self.chunk_size and remaining_separators:
                final_splits.extend(self._recursive_split(current, remaining_separators))
            else:
                final_splits.append(current)

        return final_splits

    def _merge_splits(self, splits: list[str]) -> list[str]:
        if not splits:
            return []

        merged = []
        current = splits[0]

        for i in range(1, len(splits)):
            candidate = current + splits[i]
            if _estimate_tokens(candidate) <= self.chunk_size:
                current = candidate
            else:
                merged.append(current)
                overlap_text = self._get_overlap(current)
                current = overlap_text + splits[i]

        if current:
            merged.append(current)

        return merged

    def _get_overlap(self, text: str) -> str:
        if self.chunk_overlap <= 0:
            return ""

        # For CJK text (no spaces), text.split() returns one token and overlap
        # would be empty. Fall back to character-by-character accumulation.
        words = text.split()
        if len(words) <= 1 and len(text) > 0:
            # Build overlap from the tail of the text, one char at a time
            overlap_chars = []
            token_count = 0
            for ch in reversed(text):
                ch_tokens = _estimate_tokens(ch)
                # _estimate_tokens("x") rounds down fractions, so force minimum 1
                ch_tokens = ch_tokens or 1
                if token_count + ch_tokens > self.chunk_overlap:
                    break
                token_count += ch_tokens
                overlap_chars.insert(0, ch)
            return "".join(overlap_chars)

        overlap_words = []
        token_count = 0
        for word in reversed(words):
            token_count += _estimate_tokens(word)
            if token_count > self.chunk_overlap:
                break
            overlap_words.insert(0, word)
        return " ".join(overlap_words) + " " if overlap_words else ""


def _estimate_tokens(text: str) -> int:
    # ~1.5 chars per token for Chinese, ~4 chars per token for English
    # Use a blended estimate
    chinese_chars = sum(1 for c in text if "一" <= c <= "鿿")
    other_chars = len(text) - chinese_chars
    return int(chinese_chars / 1.5 + other_chars / 4)
