from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.config import LLMConfig
from src.document.chunker import Chunk
from src.extraction.llm_client import LLMClientProtocol, create_llm_client
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
    def __init__(self, llm_config: LLMConfig, llm: LLMClientProtocol | None = None):
        self.llm = llm or create_llm_client(llm_config)

    def extract_from_chunks(self, chunks: list[Chunk]) -> ExtractionResult:
        all_entities: list[Entity] = []
        all_relations: list[Relation] = []

        requests = [(ENTITY_EXTRACTION_SYSTEM, ENTITY_EXTRACTION_USER.format(text=chunk.content)) for chunk in chunks]
        logger.info(
            f"Extracting entities/relations from {len(chunks)} chunk(s), concurrency={self.llm.config.concurrency}"
        )
        responses = self.llm.batch_chat_json(requests)

        failed = 0
        for i, data in enumerate(responses):
            if data is None:
                failed += 1
                logger.warning(f"Failed to extract from chunk {i} after retries")
                continue
            try:
                result = self._parse_result(data, i)
            except (KeyError, TypeError, ValueError) as e:
                failed += 1
                logger.warning(f"Malformed extraction result for chunk {i}: {e}")
                continue
            all_entities.extend(result.entities)
            all_relations.extend(result.relations)

        succeeded = len(chunks) - failed
        rate = succeeded / len(chunks) if chunks else 1.0
        logger.info(f"Extraction complete: {succeeded}/{len(chunks)} chunks succeeded ({rate:.1%})")

        return ExtractionResult(entities=all_entities, relations=all_relations)

    def _parse_result(self, data: dict, chunk_index: int) -> ExtractionResult:
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
