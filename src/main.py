from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import typer
from rich.console import Console
from rich.table import Table

from src.config import get_settings
from src.document.chunker import RecursiveChunker
from src.document.loader import load_documents

app = typer.Typer(name="graphrag", help="GraphRAG V2 - 垂直领域专家 Agent")
console = Console()


@app.command()
def ingest(
    source: str = typer.Argument(..., help="文档路径（文件或目录）"),
    config: str = typer.Option("configs/default.yaml", help="配置文件路径"),
    output: str = typer.Option("data/processed/chunks.json", help="输出路径"),
):
    """加载文档并分块，输出 chunks 到 JSON 文件"""
    settings = get_settings(config)

    console.print(f"[bold]Loading documents from:[/bold] {source}")
    documents = load_documents(source)
    console.print(f"  Loaded {len(documents)} document(s)")

    chunker = RecursiveChunker(settings.chunking)
    chunks = chunker.chunk_documents(documents)
    console.print(f"  Generated {len(chunks)} chunk(s)")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunk_data = [{"content": c.content, "metadata": c.metadata, "chunk_index": c.chunk_index} for c in chunks]
    output_path.write_text(json.dumps(chunk_data, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"[green]Saved to {output_path}[/green]")

    table = Table(title="Chunk Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Total Documents", str(len(documents)))
    table.add_row("Total Chunks", str(len(chunks)))
    if chunks:
        avg_tokens = sum(c.token_count for c in chunks) / len(chunks)
        table.add_row("Avg Tokens/Chunk", f"{avg_tokens:.0f}")
    console.print(table)


@app.command()
def build_graph(
    chunks_file: str = typer.Argument("", help="chunks JSON 文件路径（默认读取配置）"),
    config: str = typer.Option("configs/default.yaml", help="配置文件路径"),
    output: str = typer.Option("", help="图谱输出路径（默认读取配置）"),
):
    """从 chunks 中抽取实体/关系并构建知识图谱"""
    import logging

    from src.document.chunker import Chunk
    from src.extraction.entity_extractor import EntityRelationExtractor
    from src.graph.builder import GraphBuilder

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    settings = get_settings(config)

    chunks_path = chunks_file or str(settings.paths.chunks_file)
    output_path = Path(output or str(settings.paths.graph_file))

    console.print(f"[bold]Loading chunks from:[/bold] {chunks_path}")
    chunks = Chunk.load_from_json(chunks_path)
    console.print(f"  Loaded {len(chunks)} chunk(s)")

    console.print("[bold]Extracting entities and relations...[/bold]")
    extractor = EntityRelationExtractor(settings.llm)
    result = extractor.extract_from_chunks(chunks)
    console.print(f"  Extracted {len(result.entities)} entities, {len(result.relations)} relations")

    console.print("[bold]Building knowledge graph...[/bold]")
    builder = GraphBuilder()
    graph = builder.build_from_extraction(result)

    builder.save(output_path)
    console.print(f"[green]Graph saved to {output_path}[/green]")

    table = Table(title="Knowledge Graph Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Nodes (Entities)", str(graph.number_of_nodes()))
    table.add_row("Edges (Relations)", str(graph.number_of_edges()))
    table.add_row("Connected Components", str(nx.number_connected_components(graph)))
    console.print(table)


@app.command()
def build_communities(
    graph_file: str = typer.Argument("", help="知识图谱 JSON 文件（默认读取配置）"),
    config: str = typer.Option("configs/default.yaml", help="配置文件路径"),
    output: str = typer.Option("", help="社区输出路径（默认读取配置）"),
    summarize: bool = typer.Option(True, help="是否生成社区摘要"),
):
    """在知识图谱上运行社区发现并生成层级摘要"""
    import logging

    from src.graph.builder import GraphBuilder
    from src.graph.community import CommunityDetector
    from src.graph.summarizer import CommunitySummarizer

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    settings = get_settings(config)

    graph_path = graph_file or str(settings.paths.graph_file)
    output_path = Path(output or str(settings.paths.communities_file))

    console.print(f"[bold]Loading graph from:[/bold] {graph_path}")
    builder = GraphBuilder.load(graph_path)
    graph = builder.graph
    console.print(f"  {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    console.print(f"[bold]Running {settings.graph.community_algorithm} community detection...[/bold]")
    detector = CommunityDetector(
        algorithm=settings.graph.community_algorithm,
        max_levels=settings.graph.max_community_levels,
    )
    hierarchy = detector.detect(graph)

    if summarize and hierarchy:
        console.print("[bold]Generating community summaries...[/bold]")
        summarizer = CommunitySummarizer(settings.llm)
        for level_communities in hierarchy:
            summarizer.summarize_communities(level_communities, graph)

    _save_communities(hierarchy, output_path)
    console.print(f"[green]Communities saved to {output_path}[/green]")

    table = Table(title="Community Hierarchy")
    table.add_column("Level", style="cyan")
    table.add_column("Communities", style="magenta")
    table.add_column("Avg Size", style="green")
    for level_communities in hierarchy:
        level = level_communities[0].level if level_communities else "?"
        avg_size = sum(len(c.node_keys) for c in level_communities) / len(level_communities)
        table.add_row(str(level), str(len(level_communities)), f"{avg_size:.1f}")
    console.print(table)


def _save_communities(hierarchy: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for level_communities in hierarchy:
        level_data = []
        for c in level_communities:
            level_data.append(
                {
                    "id": c.id,
                    "level": c.level,
                    "node_keys": c.node_keys,
                    "title": c.title,
                    "summary": c.summary,
                    "key_findings": c.key_findings,
                    "importance_score": c.importance_score,
                }
            )
        data.append(level_data)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


@app.command()
def chat(config: str = typer.Option("configs/default.yaml", help="配置文件路径")):
    """启动交互式对话 Agent"""
    import logging

    from src.agent.agent import ExpertAgent

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    settings = get_settings(config)

    console.print("[bold]Initializing Expert Agent...[/bold]")
    agent = ExpertAgent(settings)
    agent.initialize()
    console.print("[green]Agent ready! Type 'quit' to exit, 'reset' to clear history.[/green]\n")

    while True:
        try:
            query = console.input("[bold cyan]You:[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("quit", "exit", "q"):
            break
        if query.strip().lower() == "reset":
            agent.reset()
            console.print("[dim]History cleared.[/dim]\n")
            continue
        if not query.strip():
            continue

        answer = agent.chat(query)
        console.print(f"\n[bold green]Agent:[/bold green] {answer}\n")


@app.command()
def info(config: str = typer.Option("configs/default.yaml", help="配置文件路径")):
    """显示当前配置信息"""
    settings = get_settings(config)
    console.print("[bold]Current Configuration[/bold]")
    console.print(f"  LLM: {settings.llm.provider} / {settings.llm.model}")
    console.print(f"  Embedding: {settings.embedding.model}")
    console.print(f"  Chunk Size: {settings.chunking.chunk_size} (overlap: {settings.chunking.chunk_overlap})")
    console.print(f"  Graph Algorithm: {settings.graph.community_algorithm}")


if __name__ == "__main__":
    app()
