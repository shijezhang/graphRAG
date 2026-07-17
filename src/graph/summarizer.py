from __future__ import annotations

import logging

import networkx as nx

from src.config import LLMConfig
from src.extraction.llm_client import LLMClient
from src.graph.community import Community

logger = logging.getLogger(__name__)

COMMUNITY_SUMMARY_SYSTEM = """你是一个知识图谱分析专家。给定一个社区中的实体和关系信息，生成该社区的结构化摘要。

输出格式（严格 JSON）：
```json
{
  "title": "社区主题的简短标题（10字以内）",
  "summary": "该社区核心内容的概括（2-3句话）",
  "key_findings": ["关键发现1", "关键发现2", "关键发现3"],
  "importance_score": 0.8
}
```

要求：
1. title 要精炼，概括社区的核心主题
2. summary 要涵盖社区中最重要的实体和关系
3. key_findings 列出 2-5 个关键发现
4. importance_score 范围 0.0-1.0，表示该社区在整体知识中的重要程度"""

COMMUNITY_SUMMARY_USER = """请为以下社区生成结构化摘要：

## 社区中的实体
{entities_text}

## 社区中的关系
{relations_text}

请输出 JSON 格式的摘要。"""


class CommunitySummarizer:
    def __init__(self, llm_config: LLMConfig):
        self.llm = LLMClient(llm_config)

    def summarize_communities(
        self, communities: list[Community], graph: nx.Graph
    ) -> list[Community]:
        for i, community in enumerate(communities):
            logger.info(f"Summarizing community {i+1}/{len(communities)} ({len(community.node_keys)} nodes)")
            try:
                self._summarize_single(community, graph)
            except Exception as e:
                logger.warning(f"Failed to summarize community {community.id}: {e}")
                community.title = f"Community {community.id}"
                community.summary = f"包含 {len(community.node_keys)} 个实体的社区"
        return communities

    def _summarize_single(self, community: Community, graph: nx.Graph) -> None:
        entities_text = self._format_entities(community, graph)
        relations_text = self._format_relations(community, graph)

        user_prompt = COMMUNITY_SUMMARY_USER.format(
            entities_text=entities_text, relations_text=relations_text
        )
        data = self.llm.chat_json(COMMUNITY_SUMMARY_SYSTEM, user_prompt)

        community.title = data.get("title", f"Community {community.id}")
        community.summary = data.get("summary", "")
        community.key_findings = data.get("key_findings", [])
        community.importance_score = float(data.get("importance_score", 0.5))

    def _format_entities(self, community: Community, graph: nx.Graph) -> str:
        lines = []
        for key in community.node_keys:
            if graph.has_node(key):
                data = graph.nodes[key]
                name = data.get("name", key)
                etype = data.get("type", "OTHER")
                desc = data.get("description", "")
                lines.append(f"- [{etype}] {name}: {desc}")
        return "\n".join(lines) if lines else "（无实体信息）"

    def _format_relations(self, community: Community, graph: nx.Graph) -> str:
        lines = []
        node_set = set(community.node_keys)
        for u, v, data in graph.edges(data=True):
            if u in node_set and v in node_set:
                relation = data.get("relation", "related_to")
                desc = data.get("description", "")
                u_name = graph.nodes[u].get("name", u)
                v_name = graph.nodes[v].get("name", v)
                lines.append(f"- {u_name} --[{relation}]--> {v_name}: {desc}")
        return "\n".join(lines) if lines else "（无关系信息）"
