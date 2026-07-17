from __future__ import annotations

import logging
import time

import gradio as gr

from src.agent.agent import ExpertAgent
from src.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

agent: ExpertAgent | None = None
_kb_stats: dict = {}


def initialize_agent():
    global agent, _kb_stats
    settings = get_settings()
    agent = ExpertAgent(settings)
    agent.initialize()

    # Collect knowledge base stats for display
    import json

    try:
        chunks = json.loads(settings.paths.chunks_file.read_text())
        _kb_stats["chunks"] = len(chunks)
    except Exception:
        _kb_stats["chunks"] = "?"
    try:
        graph = json.loads(settings.paths.graph_file.read_text())
        _kb_stats["nodes"] = len(graph.get("nodes", []))
        _kb_stats["edges"] = len(graph.get("links", []))
    except Exception:
        _kb_stats["nodes"] = "?"
        _kb_stats["edges"] = "?"
    try:
        communities = json.loads(settings.paths.communities_file.read_text())
        total = sum(len(level) for level in communities)
        _kb_stats["communities"] = total
    except Exception:
        _kb_stats["communities"] = "?"

    logger.info("Agent ready")


def reset_fn():
    if agent:
        agent.reset()
    return [], "", "—", "—", "—", "—"


EXAMPLE_QUESTIONS = [
    "巴菲特的价值投资核心理念是什么？",
    "什么是市盈率（PE），如何用它判断股票估值？",
    "可转债的投资逻辑是什么？",
    "A股市场有哪些主要的投资策略？请综合比较各策略的优劣。",
    "宏观经济指标（GDP、CPI、PMI）如何影响股票市场？",
    "中国证券监管体系的主要构成和近年重要政策有哪些？",
]


def create_app() -> gr.Blocks:
    kb_nodes = _kb_stats.get("nodes", "?")
    kb_edges = _kb_stats.get("edges", "?")
    kb_chunks = _kb_stats.get("chunks", "?")
    kb_communities = _kb_stats.get("communities", "?")

    with gr.Blocks(
        title="金融知识图谱 Agent",
        theme=gr.themes.Soft(),
        css="""
        .status-box { background: #f8f9fa; border-radius: 8px; padding: 12px; font-size: 13px; }
        .meta-label { color: #6c757d; font-size: 12px; margin-bottom: 2px; }
        .meta-value { font-weight: 600; font-size: 14px; }
        """,
    ) as app:
        # Header
        gr.Markdown(
            f"""# 📈 金融知识图谱专家 Agent
基于 GraphRAG 的金融领域智能问答系统 · 知识图谱: **{kb_nodes}** 实体 / **{kb_edges}** 关系 / \
**{kb_communities}** 社区 / **{kb_chunks}** 文本块"""
        )

        with gr.Row():
            # Left: chat panel (70%)
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(
                    height=520,
                    label="对话",
                    bubble_full_width=False,
                )
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="输入金融问题，按 Enter 发送...",
                        label="",
                        scale=9,
                        show_label=False,
                        container=False,
                    )
                    submit_btn = gr.Button("发送", variant="primary", scale=1, min_width=60)

                gr.Examples(
                    examples=EXAMPLE_QUESTIONS,
                    inputs=msg,
                    label="示例问题（点击填入）",
                )

            # Right: info panel (30%)
            with gr.Column(scale=3):
                with gr.Accordion("📊 检索信息", open=True):
                    query_type_box = gr.Textbox(label="查询类型", value="—", interactive=False)
                    retrieval_src_box = gr.Textbox(label="检索来源", value="—", interactive=False)
                    latency_box = gr.Textbox(label="响应耗时", value="—", interactive=False)
                    confidence_box = gr.Textbox(label="路由置信度", value="—", interactive=False)

                with gr.Accordion("💡 使用说明", open=False):
                    gr.Markdown("""
**局部查询**（Local）适合：
- 特定概念解释（"什么是久期？"）
- 具体数据查询（"茅台的ROE是多少？"）
- 人物/机构信息（"巴菲特的投资风格"）

**全局查询**（Global）适合：
- 综合比较（"各类基金的优劣势"）
- 主题归纳（"监管政策的整体趋势"）
- 跨文档分析（"不同投资策略的共同点"）

> 系统自动判断查询类型，无需手动选择。
                    """)

                clear_btn = gr.Button("🗑️ 清空对话", variant="secondary")

        # State for metadata
        meta_state = gr.State({})

        def respond(message: str, chat_history: list, meta: dict):
            if not message.strip() or agent is None:
                yield "", chat_history, meta, "—", "—", "—", "—"
                return

            t0 = time.time()
            chat_history = chat_history + [[message, ""]]
            yield "", chat_history, meta, "处理中...", "检索中...", "—", "—"

            # Stream response — use a fresh per-request history snapshot to avoid
            # the global agent.history being corrupted across concurrent users.
            per_request_history = list(agent.history)
            agent.history = per_request_history

            full_response = ""
            query_type = "—"
            retrieval_src = "—"
            confidence = "—"

            for token in agent.chat_stream(message):
                full_response += token
                chat_history[-1][1] = full_response
                yield "", chat_history, meta, "处理中...", "检索中...", "—", "—"

            # Extract metadata from source note in response
            latency = f"{(time.time() - t0) * 1000:.0f} ms"
            if "📎 检索方式:" in full_response:
                note_line = [ln for ln in full_response.split("\n") if "📎 检索方式:" in ln]
                if note_line:
                    parts = note_line[0].replace("📎 检索方式:", "").split("|")
                    retrieval_src = parts[0].strip() if parts else "—"
                    if len(parts) > 1:
                        query_type = parts[1].replace("类型:", "").strip()

            # Get routing confidence from agent history
            if agent.history and agent.history[-1].metadata:
                conf = agent.history[-1].metadata.get("routing_confidence", None)
                if conf is not None:
                    confidence = f"{conf:.2f}"

            yield "", chat_history, meta, query_type, retrieval_src, latency, confidence

        msg.submit(
            respond,
            inputs=[msg, chatbot, meta_state],
            outputs=[msg, chatbot, meta_state, query_type_box, retrieval_src_box, latency_box, confidence_box],
        )
        submit_btn.click(
            respond,
            inputs=[msg, chatbot, meta_state],
            outputs=[msg, chatbot, meta_state, query_type_box, retrieval_src_box, latency_box, confidence_box],
        )
        clear_btn.click(
            reset_fn,
            outputs=[chatbot, msg, query_type_box, retrieval_src_box, latency_box, confidence_box],
        )

    return app


if __name__ == "__main__":
    initialize_agent()
    settings = get_settings()
    app = create_app()
    app.queue()
    app.launch(
        server_name=settings.server.host,
        server_port=settings.server.port,
        auth=settings.server.auth,
    )
