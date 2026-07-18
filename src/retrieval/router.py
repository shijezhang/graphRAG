from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from src.config import LLMConfig, RouterConfig
from src.extraction.llm_client import create_llm_client

logger = logging.getLogger(__name__)


class QueryType(Enum):
    LOCAL = "local"
    GLOBAL = "global"


ROUTER_SYSTEM = """你是一个查询分类器。判断用户问题属于"局部问题"还是"全局问题"。

- 局部问题（LOCAL）：针对具体细节、特定实体、特定事实的问题。例如："X是什么？""A和B的关系是什么？""第三章提到了什么？"
- 全局问题（GLOBAL）：需要综合全文/全局信息才能回答的问题。
  例如："核心观点是什么？""总结主要内容""有哪些主题？""整体结论是什么？"

只输出一个词：LOCAL 或 GLOBAL"""

ROUTER_USER = """问题：{query}

请判断：LOCAL 还是 GLOBAL？"""


@dataclass
class RoutingResult:
    query_type: QueryType
    confidence: float


class QueryRouter:
    def __init__(
        self,
        llm_config: LLMConfig | None = None,
        use_llm: bool = False,
        router_config: RouterConfig | None = None,
    ):
        self.use_llm = use_llm
        self.llm = create_llm_client(llm_config) if llm_config and use_llm else None
        self.config = router_config or RouterConfig()

    def route(self, query: str) -> RoutingResult:
        if self.use_llm and self.llm:
            return self._route_llm(query)
        return self._route_heuristic(query)

    def _route_heuristic(self, query: str) -> RoutingResult:
        query_lower = query.lower()
        score = 0.0
        matched_global = 0
        for keyword in self.config.global_keywords:
            if keyword in query_lower:
                score += self.config.keyword_score
                matched_global += 1

        # Only penalize local-pattern questions when no global keywords matched
        if (
            matched_global == 0
            and ("?" in query or "？" in query)
            and any(p in query_lower for p in self.config.local_patterns)
        ):
            score -= self.config.local_pattern_penalty

        score = max(0.0, min(1.0, score))

        if score >= self.config.global_threshold:
            return RoutingResult(query_type=QueryType.GLOBAL, confidence=score)
        return RoutingResult(query_type=QueryType.LOCAL, confidence=1.0 - score)

    def _route_llm(self, query: str) -> RoutingResult:
        user_prompt = ROUTER_USER.format(query=query)
        response = self.llm.chat(ROUTER_SYSTEM, user_prompt).strip().upper()

        if "GLOBAL" in response:
            return RoutingResult(query_type=QueryType.GLOBAL, confidence=0.9)
        return RoutingResult(query_type=QueryType.LOCAL, confidence=0.9)
