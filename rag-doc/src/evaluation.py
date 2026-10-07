import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import time
import re
from typing import List, Dict, Any, Optional
from collections import Counter
from src.rag_pipeline import RAGPipeline


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _load_qa_pairs(path: str = "data/evaluation_qa.json") -> List[Dict[str, Any]]:
    """Load ground-truth Q&A pairs from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize(text: str) -> str:
    """Lower-case, collapse whitespace, strip punctuation for fuzzy comparison."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _token_overlap(reference: str, candidate: str) -> float:
    """
    Compute unigram token overlap (F1) between reference and candidate.
    Used as a lightweight proxy for answer relevance.
    """
    ref_tokens = _normalize(reference).split()
    cand_tokens = _normalize(candidate).split()
    if not ref_tokens or not cand_tokens:
        return 0.0
    common = Counter(ref_tokens) & Counter(cand_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(cand_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _check_retrieval_recall(retrieved_chunks: List[Dict], expected_doc: str) -> bool:
    """
    Check whether at least one retrieved chunk comes from the expected
    source document.
    """
    for chunk in retrieved_chunks:
        doc_name = chunk.get("doc_name", "") or chunk.get("metadata", {}).get("doc_name", "")
        if expected_doc.lower() in doc_name.lower():
            return True
    return False


def _check_citation_accuracy(answer_text: str, expected_doc: str) -> bool:
    """
    Check whether the generated answer cites the expected source document.
    Looks for patterns like [01_LandSegmenter, Page N] or similar.
    """
    doc_stem = expected_doc.replace(".pdf", "")
    return doc_stem.lower() in answer_text.lower()


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_pipeline(
    qa_path: str = "data/evaluation_qa.json",
    top_k: int = 5,
    max_queries: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Comprehensive RAG Pipeline Evaluation:

    Metrics computed per query:
        1. Retrieval Recall     – Did the retriever surface a chunk from the correct source doc?
        2. Answer Relevance     – Token-overlap F1 between generated answer and ground-truth answer.
        3. Citation Accuracy    – Does the answer cite the expected source document?
        4. Citation Presence    – Does the answer contain any in-text citation at all?
        5. Retrieval Latency    – Time (seconds) to retrieve top-k chunks.
        6. End-to-End Latency   – Time (seconds) for the full query (retrieval + LLM generation).

    Aggregated summary:
        - Mean / median of each metric across all queries.
        - Breakdown by question category (factual, method, results, comparison, dataset).
    """
    # --- Load Q&A pairs -------------------------------------------------------
    qa_pairs = _load_qa_pairs(qa_path)
    if max_queries:
        qa_pairs = qa_pairs[:max_queries]

    if verbose:
        print(f"\n[Evaluation] Loaded {len(qa_pairs)} Q&A pairs from {qa_path}")

    # --- Initialise pipeline --------------------------------------------------
    rag = RAGPipeline()
    if not rag.loaded:
        rag.ingest_documents()

    # --- Run evaluation loop --------------------------------------------------
    query_results: List[Dict[str, Any]] = []
    total_retrieval_time = 0.0
    total_query_time = 0.0
    retrieval_hits = 0
    citation_accuracy_hits = 0
    citation_presence_hits = 0
    answer_relevance_scores: List[float] = []
    category_metrics: Dict[str, Dict[str, list]] = {}

    if verbose:
        print("[Evaluation] Starting RAG Pipeline Evaluation ...\n")

    for idx, qa in enumerate(qa_pairs, 1):
        question = qa["question"]
        ground_truth = qa["ground_truth_answer"]
        expected_doc = qa["source_document"]
        category = qa.get("category", "unknown")

        # -- Retrieval (timed separately for latency metric) --
        t0 = time.time()
        retrieved_chunks = rag.retrieve(question, top_k=top_k)
        retrieval_time = time.time() - t0
        total_retrieval_time += retrieval_time

        # -- Generation only (reuse already-retrieved chunks via llm_engine) --
        t1 = time.time()
        result = rag.llm_engine.generate_answer(question, retrieved_chunks)
        generation_time = time.time() - t1
        full_query_time = retrieval_time + generation_time
        total_query_time += full_query_time

        generated_answer = result.get("answer", "")

        # -- Metrics --
        recall_hit = _check_retrieval_recall(retrieved_chunks, expected_doc)
        relevance_f1 = _token_overlap(ground_truth, generated_answer)
        citation_acc = _check_citation_accuracy(generated_answer, expected_doc)
        citation_present = bool(re.search(r'\[.*?Page\s+\d+\]', generated_answer))

        if recall_hit:
            retrieval_hits += 1
        if citation_acc:
            citation_accuracy_hits += 1
        if citation_present:
            citation_presence_hits += 1
        answer_relevance_scores.append(relevance_f1)

        # -- Per-category tracking --
        if category not in category_metrics:
            category_metrics[category] = {
                "retrieval_recall": [],
                "answer_relevance": [],
                "citation_accuracy": [],
            }
        category_metrics[category]["retrieval_recall"].append(1 if recall_hit else 0)
        category_metrics[category]["answer_relevance"].append(relevance_f1)
        category_metrics[category]["citation_accuracy"].append(1 if citation_acc else 0)

        entry = {
            "id": qa["id"],
            "question": question,
            "expected_doc": expected_doc,
            "category": category,
            "retrieval_recall": recall_hit,
            "answer_relevance_f1": round(relevance_f1, 4),
            "citation_accuracy": citation_acc,
            "citation_present": citation_present,
            "top_k_retrieved": len(retrieved_chunks),
            "retrieval_time_sec": round(retrieval_time, 4),
            "total_query_time_sec": round(full_query_time, 4),
            "provider": result.get("provider", "unknown"),
        }
        query_results.append(entry)

        if verbose:
            status = "PASS" if (recall_hit and citation_present) else "MISS"
            print(
                f"  [{idx:>2}/{len(qa_pairs)}] {status} | "
                f"Recall={'Y' if recall_hit else 'N'} | "
                f"Relevance={relevance_f1:.2f} | "
                f"CitAcc={'Y' if citation_acc else 'N'} | "
                f"{retrieval_time:.3f}s | "
                f"Q: {question[:70]}..."
            )

    # --- Aggregate metrics ----------------------------------------------------
    n = len(qa_pairs)
    sorted_relevance = sorted(answer_relevance_scores)
    median_relevance = sorted_relevance[n // 2] if n % 2 == 1 else (
        sorted_relevance[n // 2 - 1] + sorted_relevance[n // 2]
    ) / 2

    # Per-category summary
    category_summary = {}
    for cat, metrics in category_metrics.items():
        cat_n = len(metrics["retrieval_recall"])
        category_summary[cat] = {
            "count": cat_n,
            "retrieval_recall": round(sum(metrics["retrieval_recall"]) / cat_n, 4) if cat_n else 0,
            "avg_answer_relevance": round(sum(metrics["answer_relevance"]) / cat_n, 4) if cat_n else 0,
            "citation_accuracy": round(sum(metrics["citation_accuracy"]) / cat_n, 4) if cat_n else 0,
        }

    summary = {
        "total_queries": n,
        "top_k": top_k,
        "metrics": {
            "retrieval_recall": round(retrieval_hits / n, 4),
            "mean_answer_relevance_f1": round(sum(answer_relevance_scores) / n, 4),
            "median_answer_relevance_f1": round(median_relevance, 4),
            "citation_accuracy": round(citation_accuracy_hits / n, 4),
            "citation_presence_rate": round(citation_presence_hits / n, 4),
        },
        "latency": {
            "avg_retrieval_time_sec": round(total_retrieval_time / n, 4),
            "avg_total_query_time_sec": round(total_query_time / n, 4),
            "total_evaluation_time_sec": round(total_retrieval_time + total_query_time, 2),
        },
        "category_breakdown": category_summary,
        "query_details": query_results,
    }

    if verbose:
        print("\n" + "=" * 70)
        print("  RAG PIPELINE EVALUATION SUMMARY")
        print("=" * 70)
        print(f"  Total Queries Evaluated   : {n}")
        print(f"  Top-K                     : {top_k}")
        print(f"  Retrieval Recall          : {summary['metrics']['retrieval_recall']:.2%}")
        print(f"  Mean Answer Relevance (F1): {summary['metrics']['mean_answer_relevance_f1']:.4f}")
        print(f"  Median Answer Relevance   : {summary['metrics']['median_answer_relevance_f1']:.4f}")
        print(f"  Citation Accuracy         : {summary['metrics']['citation_accuracy']:.2%}")
        print(f"  Citation Presence Rate    : {summary['metrics']['citation_presence_rate']:.2%}")
        print(f"  Avg Retrieval Latency     : {summary['latency']['avg_retrieval_time_sec']:.4f}s")
        print(f"  Avg End-to-End Latency    : {summary['latency']['avg_total_query_time_sec']:.4f}s")
        print("-" * 70)
        print("  Category Breakdown:")
        for cat, vals in category_summary.items():
            print(
                f"    {cat:<12} (n={vals['count']:>2}) | "
                f"Recall={vals['retrieval_recall']:.2%} | "
                f"Relevance={vals['avg_answer_relevance']:.4f} | "
                f"CitAcc={vals['citation_accuracy']:.2%}"
            )
        print("=" * 70)

    return summary


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG Pipeline Evaluation Suite")
    parser.add_argument("--qa-path", default="data/evaluation_qa.json", help="Path to Q&A JSON file")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--max-queries", type=int, default=None, help="Limit number of queries (for quick tests)")
    parser.add_argument("--output", default=None, help="Save full JSON report to this path")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-query output")
    args = parser.parse_args()

    report = evaluate_pipeline(
        qa_path=args.qa_path,
        top_k=args.top_k,
        max_queries=args.max_queries,
        verbose=not args.quiet,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n[Evaluation] Full report saved to {args.output}")
    else:
        print("\n[Evaluation Full Report]")
        print(json.dumps(report, indent=2))
