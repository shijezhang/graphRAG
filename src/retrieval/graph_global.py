from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.config import GraphGlobalConfig, LLMConfig
from src.extraction.llm_client import LLMClient
from src.graph.community import Community

logger = logging.getLogger(__name__)

MAP_SYSTEM = """你是一个知识分析助手。给定一个社区摘要和用户问题，判断该社区是否包含与问题相关的信息，并提取相关要点。

输出格式（严格 JSON）：
```json
{
  "relevance": 0.8,
  "points": ["要点1", "要点2"]
}
```

- relevance: 0.0-1.0，该社区与问题的相关程度
- points: 从该社区中提取的与问题相关的要点（如果不相关则为空列表）"""

MAP_USER = """社区标题：{title}
社区摘要：{summary}
关键发现：{findings}

用户问题：{query}

请评估相关性并提取要点。"""

REDUCE_SYSTEM = """你是一个知识综合专家。给定从多个社区中提取的相关要点，综合生成一个完整、连贯的回答。

要求：
1. 综合所有要点，不要遗漏重要信息
2. 按逻辑组织回答，而非简单罗列
3. 如果信息不足以回答问题，明确说明"""

REDUCE_USER = """用户问题：{query}

从各社区提取的相关要点：
{points_text}

请综合以上信息，生成完整的回答。"""


class GraphGlobalRetriever:
    def __init__(self, global_config: GraphGlobalConfig, llm_config: LLMConfig):
        self.config = global_config
        self.llm = LLMClient(llm_config)
        self._communities: list[list[Community]] = []
        self._bm25_index: BM25Okapi | None = None
        self._bm25_communities: list[Community] = []

    def index(self, communities: list[list[Community]]) -> None:
        self._communities = communities
        self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        if not self._communities:
            return
        level = min(self.config.community_level, len(self._communities) - 1)
        self._bm25_communities = self._communities[level]
        corpus = [
            (c.title + " " + c.summary + " " + " ".join(c.key_findings)).lower().split()
            for c in self._bm25_communities
        ]
        self._bm25_index = BM25Okapi(corpus)

    def _prefilter_communities(self, query: str, communities: list[Community], top_n: int = 60) -> list[Community]:
        if self._bm25_index is None or len(communities) <= top_n:
            return communities
        tokens = query.lower().split()
        scores = self._bm25_index.get_scores(tokens)
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_indices = {i for i, _ in indexed[:top_n]}
        return [c for i, c in enumerate(communities) if i in top_indices]

    def search(self, query: str, community_level: int | None = None) -> str:
        level = community_level if community_level is not None else self.config.community_level
        if level >= len(self._communities):
            level = 0

        target_communities = self._communities[level]
        if not target_communities:
            return "没有可用的社区摘要来回答此问题。"

        # BM25 pre-filter to reduce LLM calls
        filtered = self._prefilter_communities(query, target_communities)
        logger.info(f"Map phase: {len(filtered)}/{len(target_communities)} communities after BM25 pre-filter")

        # Map phase: score each community's relevance (concurrent)
        mapped = self._map_phase(query, filtered)

        # Filter by relevance and take top-k
        mapped = [(c, data) for c, data in mapped if data["relevance"] > 0.2]
        mapped.sort(key=lambda x: x[1]["relevance"], reverse=True)
        mapped = mapped[: self.config.top_k]

        if not mapped:
            return "在知识图谱社区中未找到与问题相关的信息。"

        # Reduce phase: synthesize answer
        return self._reduce_phase(query, mapped)

    def _map_phase(self, query: str, communities: list[Community]) -> list[tuple[Community, dict]]:
        max_workers = min(32, len(communities))
        results: list[tuple[Community, dict]] = [None] * len(communities)  # type: ignore[list-item]

        def _call(idx: int, community: Community) -> tuple[int, Community, dict]:
            user_prompt = MAP_USER.format(
                title=community.title,
                summary=community.summary,
                findings="; ".join(community.key_findings),
                query=query,
            )
            try:
                data = self.llm.chat_json(MAP_SYSTEM, user_prompt)
            except Exception as e:
                logger.warning(f"Map failed for community {community.id}: {e}")
                data = {"relevance": 0.0, "points": []}
            return idx, community, data

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_call, i, c): i for i, c in enumerate(communities)}
            for future in as_completed(futures):
                idx, community, data = future.result()
                results[idx] = (community, data)

        return results

    def _reduce_phase(self, query: str, mapped: list[tuple[Community, dict]]) -> str:
        points_lines = []
        for community, data in mapped:
            for point in data.get("points", []):
                points_lines.append(f"- [{community.title}] {point}")

        points_text = "\n".join(points_lines)
        user_prompt = REDUCE_USER.format(query=query, points_text=points_text)
        return self.llm.chat(REDUCE_SYSTEM, user_prompt)

    def load_communities(self, path: str | Path) -> None:
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        self._communities = []
        for level_data in raw:
            communities = []
            for c in level_data:
                communities.append(
                    Community(
                        id=c["id"],
                        level=c["level"],
                        node_keys=c.get("node_keys", []),
                        title=c.get("title", ""),
                        summary=c.get("summary", ""),
                        key_findings=c.get("key_findings", []),
                        importance_score=c.get("importance_score", 0.0),
                    )
                )
            self._communities.append(communities)
        self._build_bm25_index()
