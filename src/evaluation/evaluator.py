"""Retrieval and generation evaluation metrics.

Implements:
- Recall@K
- Precision@K
- MRR (Mean Reciprocal Rank)
- NDCG@K (Normalised Discounted Cumulative Gain)
- ROUGE-L (for generation quality)

Also provides a :class:`RAGBenchmark` helper to create synthetic Q&A pairs
from filing chunks for self-contained evaluation runs.
"""

from __future__ import annotations

import logging
import math
import random
import re
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level metric helpers
# ---------------------------------------------------------------------------


def _recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = sum(1 for rid in retrieved_ids[:k] if rid in relevant_ids)
    return hits / len(relevant_ids)


def _precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for rid in retrieved_ids[:k] if rid in relevant_ids)
    return hits / k


def _mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k], start=1):
        if rid in relevant_ids:
            dcg += 1.0 / math.log2(i + 1)

    # Ideal DCG: all relevant docs at the top
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _chunk_id_from_result(result: dict[str, Any]) -> str:
    """Extract a stable identifier from a retriever result dict."""
    chunk = result.get("chunk", {})
    chunk_id = chunk.get("chunk_id")
    if chunk_id:
        return str(chunk_id)
    # Fallback: company + period
    company = chunk.get("company", "")
    period = chunk.get("period", "")
    return f"{company}_{period}"


# ---------------------------------------------------------------------------
# RetrievalEvaluator
# ---------------------------------------------------------------------------


class RetrievalEvaluator:
    """Compute retrieval quality metrics against ground-truth relevance labels."""

    def evaluate(
        self,
        queries: list[str],
        relevant_docs_map: dict[str, set[str]],
        retriever: Any,
        k_values: list[int] | None = None,
    ) -> dict[str, Any]:
        """Evaluate a retriever on a set of queries.

        Parameters
        ----------
        queries:
            List of question strings.
        relevant_docs_map:
            Mapping ``query → set[chunk_id]`` of ground-truth relevant chunks.
        retriever:
            A retriever instance with a ``retrieve(query)`` method.
        k_values:
            List of cut-off values for @K metrics.

        Returns
        -------
        dict
            Nested metrics dict: ``{metric_name: {k: value}}``.
        """
        k_values = k_values or [1, 3, 5, 10]
        max_k = max(k_values)

        recall: dict[int, list[float]] = {k: [] for k in k_values}
        precision: dict[int, list[float]] = {k: [] for k in k_values}
        mrr_scores: list[float] = []
        ndcg: dict[int, list[float]] = {k: [] for k in k_values}

        for query in queries:
            relevant_ids = relevant_docs_map.get(query, set())
            results = retriever.retrieve(query, top_k=max_k)
            retrieved_ids = [_chunk_id_from_result(r) for r in results]

            mrr_scores.append(_mrr(retrieved_ids, relevant_ids))

            for k in k_values:
                recall[k].append(_recall_at_k(retrieved_ids, relevant_ids, k))
                precision[k].append(_precision_at_k(retrieved_ids, relevant_ids, k))
                ndcg[k].append(_ndcg_at_k(retrieved_ids, relevant_ids, k))

        def _avg(lst: list[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        metrics: dict[str, Any] = {
            "recall": {k: _avg(recall[k]) for k in k_values},
            "precision": {k: _avg(precision[k]) for k in k_values},
            "mrr": _avg(mrr_scores),
            "ndcg": {k: _avg(ndcg[k]) for k in k_values},
            "num_queries": len(queries),
        }
        return metrics

    # ------------------------------------------------------------------
    # Multi-retriever comparison
    # ------------------------------------------------------------------

    def compare_retrievers(
        self,
        queries: list[str],
        relevant_docs_map: dict[str, set[str]],
        dense_retriever: Any,
        hybrid_retriever: Any,
        k_values: list[int] | None = None,
    ) -> dict[str, Any]:
        """Evaluate both retrievers and return a side-by-side comparison.

        Returns
        -------
        dict
            ``{"dense": metrics_dict, "hybrid": metrics_dict}``
        """
        k_values = k_values or [1, 3, 5, 10]
        logger.info("Evaluating dense retriever…")
        dense_metrics = self.evaluate(queries, relevant_docs_map, dense_retriever, k_values)
        logger.info("Evaluating hybrid retriever…")
        hybrid_metrics = self.evaluate(queries, relevant_docs_map, hybrid_retriever, k_values)
        return {"dense": dense_metrics, "hybrid": hybrid_metrics}

    # ------------------------------------------------------------------
    # Pretty printing
    # ------------------------------------------------------------------

    @staticmethod
    def print_comparison_table(comparison: dict[str, Any]) -> None:
        """Print a formatted comparison table to stdout."""
        dense = comparison.get("dense", {})
        hybrid = comparison.get("hybrid", {})

        k_values = sorted(dense.get("recall", {}).keys())

        # Header
        col_w = 14
        header = f"{'Metric':<22}" + "".join(f"{'Dense@'+str(k):>{col_w}}" for k in k_values)
        header += "".join(f"{'Hybrid@'+str(k):>{col_w}}" for k in k_values)
        sep = "-" * len(header)

        print(sep)
        print("  Dense vs Hybrid Retrieval — Evaluation Results")
        print(sep)
        print(header)
        print(sep)

        for metric_name in ("recall", "precision", "ndcg"):
            dense_vals = dense.get(metric_name, {})
            hybrid_vals = hybrid.get(metric_name, {})
            row = f"{metric_name.upper():<22}"
            row += "".join(f"{dense_vals.get(k, 0):.4f}".rjust(col_w) for k in k_values)
            row += "".join(f"{hybrid_vals.get(k, 0):.4f}".rjust(col_w) for k in k_values)
            print(row)

        # MRR (scalar)
        row = f"{'MRR':<22}"
        d_mrr = dense.get("mrr", 0)
        h_mrr = hybrid.get("mrr", 0)
        row += f"{d_mrr:.4f}".rjust(col_w * len(k_values))
        row += f"{h_mrr:.4f}".rjust(col_w * len(k_values))
        print(row)

        print(sep)

    # ------------------------------------------------------------------
    # Generation quality
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate_generation(
        generated_answers: list[str],
        reference_answers: list[str],
    ) -> dict[str, float]:
        """Compute ROUGE-L scores between generated and reference answers.

        Parameters
        ----------
        generated_answers:
            List of model-generated answer strings.
        reference_answers:
            List of reference (ground-truth) answer strings.

        Returns
        -------
        dict
            ``{"rouge_l_precision": float, "rouge_l_recall": float, "rouge_l_f1": float}``
        """
        try:
            from rouge_score import rouge_scorer  # type: ignore[import-untyped]
        except ImportError:
            logger.error("rouge-score not installed.  Run: pip install rouge-score")
            return {"rouge_l_precision": 0.0, "rouge_l_recall": 0.0, "rouge_l_f1": 0.0}

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        precisions, recalls, f1s = [], [], []
        for gen, ref in zip(generated_answers, reference_answers):
            scores = scorer.score(ref, gen)
            precisions.append(scores["rougeL"].precision)
            recalls.append(scores["rougeL"].recall)
            f1s.append(scores["rougeL"].fmeasure)

        def _avg(lst: list[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        return {
            "rouge_l_precision": _avg(precisions),
            "rouge_l_recall": _avg(recalls),
            "rouge_l_f1": _avg(f1s),
        }


# ---------------------------------------------------------------------------
# RAGBenchmark
# ---------------------------------------------------------------------------


class RAGBenchmark:
    """Create synthetic Q&A pairs and run full benchmarks."""

    # Patterns that indicate a chunk contains a numeric metric worth asking about
    _METRIC_PATTERNS = [
        r"\$[\d,]+(?:\.\d+)?\s*(?:million|billion|thousand)?",
        r"\d+(?:\.\d+)?\s*(?:percent|%)",
        r"(?:revenue|net income|earnings|profit|loss|assets|liabilities|cash)\s+(?:of|was|were|totaled?)\s+\$?[\d,]+",
        r"(?:increased|decreased|grew|declined)\s+(?:by\s+)?\d+(?:\.\d+)?\s*(?:percent|%)",
    ]
    _COMPILED = [re.compile(p, re.IGNORECASE) for p in _METRIC_PATTERNS]

    @classmethod
    def _has_metric(cls, text: str) -> bool:
        return any(pat.search(text) for pat in cls._COMPILED)

    @staticmethod
    def _generate_question(chunk: dict[str, Any]) -> str:
        """Heuristically generate a factual question from a chunk."""
        text = chunk.get("text", "")
        company = chunk.get("company", "the company")
        period = chunk.get("period", "the reporting period")

        # Try to find a dollar amount and generate a revenue/earnings question
        dollar_match = re.search(
            r"(revenue|net income|earnings|sales|profit|loss|assets|liabilities|cash)[^.]{0,60}"
            r"\$[\d,]+(?:\.\d+)?\s*(?:million|billion|thousand)?",
            text,
            re.IGNORECASE,
        )
        if dollar_match:
            snippet = dollar_match.group(0)[:80]
            return f"What was {company}'s {snippet.split()[0].lower()} for the period ending {period}?"

        # Fallback: generic question about the chunk topic
        first_sentence = text.split(".")[0][:120] if "." in text else text[:120]
        return f"What does {company}'s {period} 10-K filing state about: {first_sentence}?"

    def create_qa_pairs_from_filings(
        self,
        chunks: list[dict[str, Any]],
        n: int = 20,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        """Sample *n* chunks and generate synthetic Q&A pairs.

        Parameters
        ----------
        chunks:
            All preprocessed filing chunks.
        n:
            Number of Q&A pairs to generate.
        seed:
            Random seed for reproducibility.

        Returns
        -------
        list[dict]
            Each dict has keys: ``question``, ``answer``, ``relevant_chunk_ids``.
        """
        random.seed(seed)

        # Prefer chunks that contain numeric metrics
        metric_chunks = [c for c in chunks if self._has_metric(c.get("text", ""))]
        pool = metric_chunks if len(metric_chunks) >= n else chunks
        sampled = random.sample(pool, min(n, len(pool)))

        qa_pairs: list[dict[str, Any]] = []
        for chunk in sampled:
            question = self._generate_question(chunk)
            answer = chunk.get("text", "")[:300]
            chunk_id = chunk.get("chunk_id", "")
            qa_pairs.append(
                {
                    "question": question,
                    "answer": answer,
                    "relevant_chunk_ids": {chunk_id} if chunk_id else set(),
                }
            )

        return qa_pairs

    def run_full_benchmark(
        self,
        dense_pipeline: Any,
        hybrid_pipeline: Any,
        qa_pairs: list[dict[str, Any]],
        k_values: list[int] | None = None,
    ) -> dict[str, Any]:
        """Run a full evaluation comparing dense vs hybrid RAG pipelines.

        Parameters
        ----------
        dense_pipeline:
            A :class:`~src.pipeline.rag_pipeline.RAGPipeline` backed by the
            dense retriever.
        hybrid_pipeline:
            A :class:`~src.pipeline.rag_pipeline.RAGPipeline` backed by the
            hybrid retriever.
        qa_pairs:
            Synthetic Q&A pairs (from :meth:`create_qa_pairs_from_filings`).
        k_values:
            Cut-off values for @K metrics.

        Returns
        -------
        dict
            Full comparison report including retrieval and generation metrics.
        """
        k_values = k_values or [1, 3, 5, 10]

        queries = [qa["question"] for qa in qa_pairs]
        relevant_docs_map = {
            qa["question"]: qa["relevant_chunk_ids"] for qa in qa_pairs
        }
        reference_answers = [qa["answer"] for qa in qa_pairs]

        evaluator = RetrievalEvaluator()

        # Retrieval comparison
        retrieval_comparison = evaluator.compare_retrievers(
            queries,
            relevant_docs_map,
            dense_pipeline.retriever,
            hybrid_pipeline.retriever,
            k_values=k_values,
        )

        # Generation quality
        logger.info("Generating answers with dense pipeline…")
        dense_answers = [dense_pipeline.query(q)["answer"] for q in queries]
        logger.info("Generating answers with hybrid pipeline…")
        hybrid_answers = [hybrid_pipeline.query(q)["answer"] for q in queries]

        dense_rouge = evaluator.evaluate_generation(dense_answers, reference_answers)
        hybrid_rouge = evaluator.evaluate_generation(hybrid_answers, reference_answers)

        report: dict[str, Any] = {
            "retrieval": retrieval_comparison,
            "generation": {
                "dense": dense_rouge,
                "hybrid": hybrid_rouge,
            },
            "num_queries": len(queries),
        }

        # Print summary
        evaluator.print_comparison_table(retrieval_comparison)
        print("\n— Generation Quality (ROUGE-L F1) —")
        print(f"  Dense  : {dense_rouge['rouge_l_f1']:.4f}")
        print(f"  Hybrid : {hybrid_rouge['rouge_l_f1']:.4f}")

        return report
