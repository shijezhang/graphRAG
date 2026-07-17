"""
M6 Evaluation: Compare BM25 vs Graph Local vs Graph Global retrieval.
Metrics: Keyword Coverage (recall proxy) + LLM-as-Judge (0-10 score).
"""
from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.document.chunker import Chunk
from src.extraction.llm_client import LLMClient
from src.graph.builder import GraphBuilder
from src.retrieval.graph_global import GraphGlobalRetriever
from src.retrieval.graph_local import GraphLocalRetriever
from src.retrieval.sparse import SparseRetriever

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

JUDGE_SYSTEM = """You are an expert evaluator for RAG systems. Given a question, a reference answer, and a system's answer, score the system answer from 0 to 10.

Scoring rubric:
- 10: Complete, accurate, covers all key points
- 7-9: Mostly correct, covers most key points
- 4-6: Partially correct, covers some key points
- 1-3: Mostly incorrect or very incomplete
- 0: Completely wrong or no answer

Output JSON only: {"score": <int>, "reason": "<one sentence>"}"""

JUDGE_USER = """Question: {question}

Reference Answer: {reference}

System Answer: {answer}

Score the system answer (0-10):"""


@dataclass
class EvalResult:
    question_id: str
    question_type: str
    question: str
    method: str
    answer: str
    keyword_coverage: float
    llm_score: float = 0.0
    llm_reason: str = ""
    latency_ms: float = 0.0


def keyword_coverage(answer: str, key_points: list[str]) -> float:
    answer_lower = answer.lower()
    hits = sum(1 for kp in key_points if kp.lower() in answer_lower)
    return hits / len(key_points) if key_points else 0.0


def llm_judge(llm: LLMClient, question: str, reference: str, answer: str) -> tuple[float, str]:
    prompt = JUDGE_USER.format(question=question, reference=reference, answer=answer)
    try:
        data = llm.chat_json(JUDGE_SYSTEM, prompt)
        return float(data.get("score", 0)), data.get("reason", "")
    except Exception as e:
        return 0.0, f"Judge failed: {e}"


def run_evaluation():
    settings = get_settings()
    llm = LLMClient(settings.llm, cache_dir="data/processed/.llm_cache")

    # Load data
    chunks = Chunk.load_from_json("data/processed/chunks.json")
    questions = json.loads(Path("eval/questions.json").read_text())

    # Init retrievers
    bm25 = SparseRetriever(settings.retrieval.sparse)
    bm25.index(chunks)

    builder = GraphBuilder.load("data/graphs/knowledge_graph.json")
    graph_local = GraphLocalRetriever(settings.retrieval.graph_local)
    graph_local.index(builder.graph, chunks)

    graph_global = GraphGlobalRetriever(settings.retrieval.graph_global, settings.llm)
    graph_global.load_communities("data/graphs/communities.json")

    results: list[EvalResult] = []

    for q in questions:
        qid = q["id"]
        qtype = q["type"]
        question = q["question"]
        reference = q["reference_answer"]
        key_points = q["key_points"]

        print(f"\n[{qid}] ({qtype}) {question[:70]}...")

        # --- BM25 ---
        t0 = time.time()
        bm25_chunks = bm25.search(question, top_k=5)
        context = "\n\n".join(c.content[:400] for c, _ in bm25_chunks)
        bm25_answer = llm.chat(
            "Answer the question based only on the provided context. Be concise.",
            f"Context:\n{context}\n\nQuestion: {question}"
        )
        bm25_latency = (time.time() - t0) * 1000
        bm25_kc = keyword_coverage(bm25_answer, key_points)
        bm25_score, bm25_reason = llm_judge(llm, question, reference, bm25_answer)
        results.append(EvalResult(qid, qtype, question, "BM25", bm25_answer, bm25_kc, bm25_score, bm25_reason, bm25_latency))
        print(f"  BM25:         KC={bm25_kc:.2f}  Score={bm25_score:.0f}  ({bm25_latency:.0f}ms)")

        # --- Graph Local ---
        t0 = time.time()
        gl_chunks = graph_local.search(question, top_k=5)
        if gl_chunks:
            context = "\n\n".join(c.content[:400] for c, _ in gl_chunks)
            gl_answer = llm.chat(
                "Answer the question based only on the provided context. Be concise.",
                f"Context:\n{context}\n\nQuestion: {question}"
            )
        else:
            gl_answer = "No relevant entities found in knowledge graph."
        gl_latency = (time.time() - t0) * 1000
        gl_kc = keyword_coverage(gl_answer, key_points)
        gl_score, gl_reason = llm_judge(llm, question, reference, gl_answer)
        results.append(EvalResult(qid, qtype, question, "GraphLocal", gl_answer, gl_kc, gl_score, gl_reason, gl_latency))
        print(f"  Graph Local:  KC={gl_kc:.2f}  Score={gl_score:.0f}  ({gl_latency:.0f}ms)")

        # --- Graph Global ---
        t0 = time.time()
        gg_answer = graph_global.search(question)
        gg_latency = (time.time() - t0) * 1000
        gg_kc = keyword_coverage(gg_answer, key_points)
        gg_score, gg_reason = llm_judge(llm, question, reference, gg_answer)
        results.append(EvalResult(qid, qtype, question, "GraphGlobal", gg_answer, gg_kc, gg_score, gg_reason, gg_latency))
        print(f"  Graph Global: KC={gg_kc:.2f}  Score={gg_score:.0f}  ({gg_latency:.0f}ms)")

    # Save detailed results
    Path("eval/results").mkdir(exist_ok=True)
    results_data = [vars(r) for r in results]
    Path("eval/results/detailed.json").write_text(
        json.dumps(results_data, ensure_ascii=False, indent=2)
    )

    # Print summary table
    print_summary(results, questions)


def print_summary(results: list[EvalResult], questions: list[dict]):
    methods = ["BM25", "GraphLocal", "GraphGlobal"]
    qtypes = ["local", "global"]

    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)

    # Overall
    print(f"\n{'Method':<14} {'Avg KC':>8} {'Avg Score':>10} {'Avg Latency':>12}")
    print("-" * 50)
    for method in methods:
        m_results = [r for r in results if r.method == method]
        avg_kc = sum(r.keyword_coverage for r in m_results) / len(m_results)
        avg_score = sum(r.llm_score for r in m_results) / len(m_results)
        avg_lat = sum(r.latency_ms for r in m_results) / len(m_results)
        print(f"{method:<14} {avg_kc:>8.2f} {avg_score:>10.1f} {avg_lat:>11.0f}ms")

    # By query type
    for qtype in qtypes:
        qtype_ids = {q["id"] for q in questions if q["type"] == qtype}
        print(f"\n--- {qtype.upper()} QUERIES ({len(qtype_ids)} questions) ---")
        print(f"{'Method':<14} {'Avg KC':>8} {'Avg Score':>10}")
        print("-" * 35)
        for method in methods:
            m_results = [r for r in results if r.method == method and r.question_id in qtype_ids]
            if not m_results:
                continue
            avg_kc = sum(r.keyword_coverage for r in m_results) / len(m_results)
            avg_score = sum(r.llm_score for r in m_results) / len(m_results)
            print(f"{method:<14} {avg_kc:>8.2f} {avg_score:>10.1f}")

    # Per-question breakdown
    print("\n--- PER-QUESTION SCORES (LLM Judge) ---")
    print(f"{'QID':<5} {'Type':<7} {'BM25':>6} {'GraphL':>8} {'GraphG':>8}")
    print("-" * 40)
    for q in questions:
        qid = q["id"]
        scores = {r.method: r.llm_score for r in results if r.question_id == qid}
        print(f"{qid:<5} {q['type']:<7} {scores.get('BM25', 0):>6.0f} {scores.get('GraphLocal', 0):>8.0f} {scores.get('GraphGlobal', 0):>8.0f}")

    print("\n" + "="*70)
    print("KC = Keyword Coverage (recall proxy), Score = LLM-as-Judge (0-10)")
    print("="*70)


if __name__ == "__main__":
    run_evaluation()
