from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.config import LLMConfig
from src.document.chunker import Chunk
from src.extraction.llm_client import LLMClient
from src.extraction.prompts import (
    ENTITY_EXTRACTION_SYSTEM,
    ENTITY_EXTRACTION_USER,
)

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    name: str
    type: str
    description: str
    source_chunks: list[int] = field(default_factory=list)

    @property
    def key(self) -> str:
        return self.name.lower().strip()


@dataclass
class Relation:
    source: str
    target: str
    relation: str
    description: str
    weight: float = 1.0
    source_chunks: list[int] = field(default_factory=list)


@dataclass
class ExtractionResult:
    entities: list[Entity]
    relations: list[Relation]


class EntityRelationExtractor:
    def __init__(self, llm_config: LLMConfig):
        self.llm = LLMClient(llm_config)

    def extract_from_chunks(self, chunks: list[Chunk]) -> ExtractionResult:
        all_entities: list[Entity] = []
        all_relations: list[Relation] = []

        for i, chunk in enumerate(chunks):
            logger.info(f"Extracting from chunk {i + 1}/{len(chunks)}")
            try:
                result = self._extract_single(chunk, i)
                all_entities.extend(result.entities)
                all_relations.extend(result.relations)
            except Exception as e:
                logger.warning(f"Failed to extract from chunk {i}: {e}")
                continue

        return ExtractionResult(entities=all_entities, relations=all_relations)

    def _extract_single(self, chunk: Chunk, chunk_index: int) -> ExtractionResult:
        user_prompt = ENTITY_EXTRACTION_USER.format(text=chunk.content)
        data = self.llm.chat_json(ENTITY_EXTRACTION_SYSTEM, user_prompt)

        entities = []
        for e in data.get("entities", []):
            entities.append(
                Entity(
                    name=e["name"],
                    type=e.get("type", "OTHER"),
                    description=e.get("description", ""),
                    source_chunks=[chunk_index],
                )
            )

        relations = []
        for r in data.get("relations", []):
            relations.append(
                Relation(
                    source=r["source"],
                    target=r["target"],
                    relation=r.get("relation", "related_to"),
                    description=r.get("description", ""),
                    weight=float(r.get("weight", 1.0)),
                    source_chunks=[chunk_index],
                )
            )

        return ExtractionResult(entities=entities, relations=relations)
