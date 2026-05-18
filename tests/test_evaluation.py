"""Tests for RetrievalEvaluator and RAGBenchmark."""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.evaluation.evaluator import (
    RAGBenchmark,
    RetrievalEvaluator,
    _mrr,
    _ndcg_at_k,
    _precision_at_k,
    _recall_at_k,
)


# ---------------------------------------------------------------------------
# Helper: build a fake retriever from a fixed result list
# ---------------------------------------------------------------------------


def _make_retriever(results_per_query: dict[str, list[str]]) -> Any:
    """Return a mock retriever whose retrieve() method returns pre-set chunk ids."""

    def _retrieve(query: str, top_k: int | None = None) -> list[dict]:
        ids = results_per_query.get(query, [])
        if top_k is not None:
            ids = ids[:top_k]
        return [
            {
                "chunk": {
                    "chunk_id": cid,
                    "text": f"Text for {cid}",
                    "company": "AAPL",
                    "period": "2023",
                },
                "score": 1.0 / (i + 1),
                "rank": i + 1,
            }
            for i, cid in enumerate(ids)
        ]

    mock = MagicMock()
    mock.retrieve.side_effect = _retrieve
    return mock


# ---------------------------------------------------------------------------
# Unit tests for metric functions
# ---------------------------------------------------------------------------


def test_recall_at_k_perfect() -> None:
    """Perfect retrieval → recall = 1.0."""
    retrieved = ["a", "b", "c"]
    relevant = {"a", "b", "c"}
    assert _recall_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)


def test_recall_at_k_partial() -> None:
    retrieved = ["a", "x", "b"]
    relevant = {"a", "b", "c"}
    # 2 hits out of 3 relevant, within top-3
    assert _recall_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)


def test_recall_at_k_none() -> None:
    retrieved = ["x", "y", "z"]
    relevant = {"a", "b", "c"}
    assert _recall_at_k(retrieved, relevant, k=3) == pytest.approx(0.0)


def test_recall_at_k_cutoff() -> None:
    """Only top-k results are considered."""
    retrieved = ["x", "y", "a"]  # "a" is at rank 3
    relevant = {"a"}
    assert _recall_at_k(retrieved, relevant, k=2) == pytest.approx(0.0)
    assert _recall_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)


def test_precision_at_k() -> None:
    retrieved = ["a", "x", "b", "y"]
    relevant = {"a", "b"}
    assert _precision_at_k(retrieved, relevant, k=4) == pytest.approx(2 / 4)
    assert _precision_at_k(retrieved, relevant, k=2) == pytest.approx(1 / 2)


def test_precision_at_k_empty_relevant() -> None:
    assert _precision_at_k(["a", "b"], set(), k=2) == pytest.approx(0.0)


def test_mrr_first_hit() -> None:
    assert _mrr(["a", "b", "c"], {"a"}) == pytest.approx(1.0)


def test_mrr_second_hit() -> None:
    assert _mrr(["x", "a", "b"], {"a"}) == pytest.approx(0.5)


def test_mrr_no_hit() -> None:
    assert _mrr(["x", "y", "z"], {"a"}) == pytest.approx(0.0)


def test_mrr_multiple_relevant() -> None:
    """MRR uses the rank of the *first* relevant document found."""
    assert _mrr(["x", "a", "b"], {"a", "b"}) == pytest.approx(0.5)


def test_ndcg_perfect() -> None:
    """Perfect NDCG = 1.0 when all top-k results are relevant."""
    relevant = {"a", "b", "c"}
    retrieved = ["a", "b", "c", "x", "y"]
    assert _ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)


def test_ndcg_none() -> None:
    retrieved = ["x", "y", "z"]
    relevant = {"a", "b"}
    assert _ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(0.0)


def test_ndcg_partial() -> None:
    """NDCG with one hit at rank 2 should be between 0 and 1."""
    retrieved = ["x", "a", "y"]
    relevant = {"a"}
    score = _ndcg_at_k(retrieved, relevant, k=3)
    assert 0.0 < score < 1.0
    # DCG = 1/log2(3) = 0.630..., IDCG = 1/log2(2) = 1.0
    expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
    assert score == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# RetrievalEvaluator integration tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def evaluator() -> RetrievalEvaluator:
    return RetrievalEvaluator()


def _perfect_retriever(queries: list[str], relevant_map: dict[str, set[str]]) -> Any:
    """Returns a retriever that always puts relevant docs at rank 1."""
    results_map: dict[str, list[str]] = {}
    for q in queries:
        rel = list(relevant_map.get(q, set()))
        results_map[q] = rel + [f"irrelevant_{i}" for i in range(10)]
    return _make_retriever(results_map)


QUERIES = ["q1", "q2", "q3"]
RELEVANT_MAP: dict[str, set[str]] = {
    "q1": {"chunk_a"},
    "q2": {"chunk_b"},
    "q3": {"chunk_c"},
}


def test_evaluate_perfect_retrieval(evaluator: RetrievalEvaluator) -> None:
    retriever = _perfect_retriever(QUERIES, RELEVANT_MAP)
    metrics = evaluator.evaluate(QUERIES, RELEVANT_MAP, retriever, k_values=[1, 3])

    assert metrics["recall"][1] == pytest.approx(1.0)
    assert metrics["precision"][1] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(1.0)
    assert metrics["ndcg"][1] == pytest.approx(1.0)


def test_evaluate_zero_retrieval(evaluator: RetrievalEvaluator) -> None:
    """A retriever that returns nothing relevant → all metrics = 0."""
    results_map = {q: [f"wrong_{i}" for i in range(5)] for q in QUERIES}
    retriever = _make_retriever(results_map)
    metrics = evaluator.evaluate(QUERIES, RELEVANT_MAP, retriever, k_values=[1, 5])
    assert metrics["recall"][1] == pytest.approx(0.0)
    assert metrics["mrr"] == pytest.approx(0.0)
    assert metrics["ndcg"][5] == pytest.approx(0.0)


def test_evaluate_num_queries(evaluator: RetrievalEvaluator) -> None:
    retriever = _perfect_retriever(QUERIES, RELEVANT_MAP)
    metrics = evaluator.evaluate(QUERIES, RELEVANT_MAP, retriever, k_values=[1])
    assert metrics["num_queries"] == len(QUERIES)


def test_compare_retrievers(evaluator: RetrievalEvaluator) -> None:
    """compare_retrievers returns a dict with 'dense' and 'hybrid' keys."""
    dense = _perfect_retriever(QUERIES, RELEVANT_MAP)
    hybrid = _perfect_retriever(QUERIES, RELEVANT_MAP)
    comparison = evaluator.compare_retrievers(
        QUERIES, RELEVANT_MAP, dense, hybrid, k_values=[1, 3]
    )
    assert "dense" in comparison
    assert "hybrid" in comparison

    for key in ("dense", "hybrid"):
        m = comparison[key]
        assert "recall" in m
        assert "precision" in m
        assert "mrr" in m
        assert "ndcg" in m


def test_compare_retrievers_keys_complete(evaluator: RetrievalEvaluator) -> None:
    """Each metrics dict contains all k values."""
    k_values = [1, 3, 5]
    dense = _perfect_retriever(QUERIES, RELEVANT_MAP)
    hybrid = _perfect_retriever(QUERIES, RELEVANT_MAP)
    comparison = evaluator.compare_retrievers(
        QUERIES, RELEVANT_MAP, dense, hybrid, k_values=k_values
    )
    for key in ("dense", "hybrid"):
        for metric in ("recall", "precision", "ndcg"):
            for k in k_values:
                assert k in comparison[key][metric], f"Missing {metric}@{k} for {key}"


# ---------------------------------------------------------------------------
# RAGBenchmark tests
# ---------------------------------------------------------------------------


SAMPLE_CHUNKS = [
    {
        "chunk_id": f"chunk_{i}",
        "text": (
            f"Apple reported revenue of {i * 10 + 100} billion dollars in Q{(i % 4) + 1} 2023. "
            "Net income grew by 15 percent. Capital expenditures totaled $5 billion."
        ),
        "company": "AAPL",
        "period": "2023-12-31",
        "filing_url": "https://example.com",
    }
    for i in range(30)
]


def test_create_qa_pairs_count() -> None:
    benchmark = RAGBenchmark()
    qa_pairs = benchmark.create_qa_pairs_from_filings(SAMPLE_CHUNKS, n=10)
    assert len(qa_pairs) == 10


def test_create_qa_pairs_structure() -> None:
    benchmark = RAGBenchmark()
    qa_pairs = benchmark.create_qa_pairs_from_filings(SAMPLE_CHUNKS, n=5)
    required_keys = {"question", "answer", "relevant_chunk_ids"}
    for qa in qa_pairs:
        assert required_keys.issubset(qa.keys())
        assert isinstance(qa["question"], str)
        assert len(qa["question"]) > 0
        assert isinstance(qa["relevant_chunk_ids"], set)


def test_create_qa_pairs_reproducible() -> None:
    """Same seed → same QA pairs."""
    benchmark = RAGBenchmark()
    qa1 = benchmark.create_qa_pairs_from_filings(SAMPLE_CHUNKS, n=5, seed=99)
    qa2 = benchmark.create_qa_pairs_from_filings(SAMPLE_CHUNKS, n=5, seed=99)
    questions1 = [q["question"] for q in qa1]
    questions2 = [q["question"] for q in qa2]
    assert questions1 == questions2


def test_create_qa_pairs_fewer_than_n() -> None:
    """When n > len(chunks), return len(chunks) pairs."""
    benchmark = RAGBenchmark()
    small_corpus = SAMPLE_CHUNKS[:3]
    qa_pairs = benchmark.create_qa_pairs_from_filings(small_corpus, n=100)
    assert len(qa_pairs) == len(small_corpus)


def test_evaluate_generation_rouge_l(evaluator: RetrievalEvaluator) -> None:
    """Identical answers → ROUGE-L = 1.0."""
    answers = ["The revenue was 100 billion dollars.", "Net income grew by 15 percent."]
    scores = evaluator.evaluate_generation(answers, answers)
    assert scores["rouge_l_f1"] == pytest.approx(1.0, rel=1e-3)


def test_evaluate_generation_no_overlap(evaluator: RetrievalEvaluator) -> None:
    generated = ["alpha beta gamma"]
    reference = ["delta epsilon zeta"]
    scores = evaluator.evaluate_generation(generated, reference)
    assert scores["rouge_l_f1"] == pytest.approx(0.0, abs=1e-3)
