"""Tests for DenseRetriever, SparseRetriever, and HybridRetriever.

Uses a small synthetic corpus so no internet access or GPU is required.
The sentence-transformer model is replaced with a deterministic TF-IDF
mock so the tests run fully offline.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.sparse import SparseRetriever


# ---------------------------------------------------------------------------
# Offline mock: replace SentenceTransformer with a TF-IDF encoder
# ---------------------------------------------------------------------------


class _TfidfEncoder:
    """Deterministic encoder that uses TF-IDF vectors instead of a real model."""

    def __init__(self) -> None:
        self._vectorizer: TfidfVectorizer | None = None
        self._fitted_texts: list[str] = []

    def _fit_if_needed(self, texts: list[str]) -> None:
        if self._vectorizer is None:
            self._vectorizer = TfidfVectorizer(max_features=64)
            self._vectorizer.fit(texts)
            self._fitted_texts = list(texts)

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        self._fit_if_needed(texts if len(texts) > 1 else self._fitted_texts or texts)
        if self._vectorizer is None:
            raise RuntimeError("Vectorizer not fitted — call _fit_if_needed first.")
        matrix = self._vectorizer.transform(texts).toarray().astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            matrix = matrix / norms
        return matrix


def _make_mock_sentence_transformer(corpus_texts: list[str]) -> MagicMock:
    """Return a MagicMock that behaves like SentenceTransformer for offline tests."""
    encoder = _TfidfEncoder()
    # Pre-fit on the full corpus so single-query encoding works too
    encoder._fit_if_needed(corpus_texts)

    mock_model = MagicMock()
    mock_model.encode.side_effect = encoder.encode
    return mock_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CORPUS_TEXTS = [
    "Apple reported revenue of 100 billion dollars in fiscal year 2023.",
    "Microsoft cloud services grew by 25 percent year over year.",
    "Google advertising revenue declined due to macroeconomic headwinds.",
    "Amazon web services profit margin expanded to 30 percent.",
    "Meta increased capital expenditures for artificial intelligence infrastructure.",
    "Tesla vehicle deliveries reached a record high in Q4 2023.",
    "Netflix subscriber growth exceeded analyst expectations by 15 percent.",
    "NVIDIA semiconductor revenue surged on AI chip demand.",
    "Apple iPhone sales contributed 52 percent of total revenue.",
    "Microsoft operating income grew by 20 percent in Q2 fiscal 2024.",
    "Alphabet total revenue was 86 billion dollars in Q3 2023.",
    "Amazon retail segment showed a modest recovery in operating income.",
    "Meta Reality Labs division reported an operating loss of 4 billion.",
    "Tesla energy generation and storage segment doubled its revenue.",
    "Netflix raised subscription prices in major markets including the US.",
    "NVIDIA data center revenue tripled compared to the previous year.",
    "Apple services segment set a new all-time revenue record.",
    "Microsoft gaming revenue benefited from the Activision acquisition.",
    "Google cloud division turned profitable for the first time.",
    "Amazon AWS revenue grew to 24 billion in the third quarter.",
]

CORPUS = [
    {
        "text": text,
        "chunk_id": f"chunk_{i}",
        "company": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"][i % 5],
        "period": "2023-12-31",
        "filing_url": f"https://example.com/filing_{i}",
    }
    for i, text in enumerate(CORPUS_TEXTS)
]


@pytest.fixture(scope="module")
def dense_retriever() -> DenseRetriever:
    """Build a dense retriever over the synthetic corpus using an offline mock."""
    mock_model = _make_mock_sentence_transformer(CORPUS_TEXTS)
    with patch("src.retrieval.dense.SentenceTransformer", return_value=mock_model):
        retriever = DenseRetriever(
            model_name="BAAI/bge-small-en-v1.5",
            top_k=5,
            batch_size=8,
        )
        retriever.build_index(CORPUS)
    return retriever


@pytest.fixture(scope="module")
def sparse_retriever() -> SparseRetriever:
    """Build a sparse BM25 retriever over the synthetic corpus."""
    retriever = SparseRetriever(top_k=5, b=0.75, k1=1.5)
    retriever.build_index(CORPUS)
    return retriever


@pytest.fixture(scope="module")
def hybrid_retriever(dense_retriever: DenseRetriever, sparse_retriever: SparseRetriever) -> HybridRetriever:
    return HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        top_k=5,
        rrf_k=60,
    )


# ---------------------------------------------------------------------------
# DenseRetriever tests
# ---------------------------------------------------------------------------


def test_dense_retriever_build_and_retrieve(dense_retriever: DenseRetriever) -> None:
    """Dense retriever returns correctly structured results."""
    results = dense_retriever.retrieve("What was Apple's revenue?", top_k=5)
    assert len(results) > 0
    assert len(results) <= 5

    for result in results:
        assert "chunk" in result
        assert "score" in result
        assert "rank" in result
        assert isinstance(result["score"], float)
        assert isinstance(result["rank"], int)
        assert isinstance(result["chunk"], dict)
        assert "text" in result["chunk"]


def test_dense_retriever_top_k_respected(dense_retriever: DenseRetriever) -> None:
    """Dense retriever returns at most top_k results."""
    for k in (1, 3, 5):
        results = dense_retriever.retrieve("cloud revenue", top_k=k)
        assert len(results) <= k


def test_dense_retriever_scores_descending(dense_retriever: DenseRetriever) -> None:
    """Dense retriever results are ordered by descending score."""
    results = dense_retriever.retrieve("Apple revenue fiscal year", top_k=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_dense_retriever_save_load(dense_retriever: DenseRetriever) -> None:
    """Save and reload index give identical results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dense_retriever.save_index(tmpdir)

        mock_model = _make_mock_sentence_transformer(CORPUS_TEXTS)
        with patch("src.retrieval.dense.SentenceTransformer", return_value=mock_model):
            loaded = DenseRetriever(
                model_name=dense_retriever.model_name,
                top_k=5,
                batch_size=8,
            )
        loaded.load_index(tmpdir)

        query = "semiconductor revenue NVIDIA"
        original_results = dense_retriever.retrieve(query, top_k=5)
        loaded_results = loaded.retrieve(query, top_k=5)

        orig_ids = [r["chunk"]["chunk_id"] for r in original_results]
        loaded_ids = [r["chunk"]["chunk_id"] for r in loaded_results]
        assert orig_ids == loaded_ids


def test_dense_retriever_raises_without_index() -> None:
    """retrieve() raises RuntimeError when index is not built."""
    mock_model = _make_mock_sentence_transformer(CORPUS_TEXTS)
    with patch("src.retrieval.dense.SentenceTransformer", return_value=mock_model):
        retriever = DenseRetriever(model_name="BAAI/bge-small-en-v1.5")
    with pytest.raises(RuntimeError, match="Index not built"):
        retriever.retrieve("test query")


# ---------------------------------------------------------------------------
# SparseRetriever tests
# ---------------------------------------------------------------------------


def test_sparse_retriever_build_and_retrieve(sparse_retriever: SparseRetriever) -> None:
    """Sparse retriever returns correctly structured results."""
    results = sparse_retriever.retrieve("Apple revenue billion", top_k=5)
    assert len(results) > 0
    assert len(results) <= 5

    for result in results:
        assert "chunk" in result
        assert "score" in result
        assert "rank" in result
        assert isinstance(result["score"], float)


def test_sparse_retriever_top_k_respected(sparse_retriever: SparseRetriever) -> None:
    for k in (1, 3, 5):
        results = sparse_retriever.retrieve("cloud revenue growth", top_k=k)
        assert len(results) <= k


def test_sparse_retriever_scores_descending(sparse_retriever: SparseRetriever) -> None:
    results = sparse_retriever.retrieve("revenue profit quarterly earnings", top_k=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_sparse_retriever_save_load(sparse_retriever: SparseRetriever) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        sparse_retriever.save_index(tmpdir)

        loaded = SparseRetriever(top_k=5)
        loaded.load_index(tmpdir)

        query = "Netflix subscriber growth"
        original_results = sparse_retriever.retrieve(query, top_k=3)
        loaded_results = loaded.retrieve(query, top_k=3)

        orig_ids = [r["chunk"]["chunk_id"] for r in original_results]
        loaded_ids = [r["chunk"]["chunk_id"] for r in loaded_results]
        assert orig_ids == loaded_ids


def test_sparse_retriever_raises_without_index() -> None:
    retriever = SparseRetriever()
    with pytest.raises(RuntimeError, match="Index not built"):
        retriever.retrieve("test query")


# ---------------------------------------------------------------------------
# HybridRetriever tests
# ---------------------------------------------------------------------------


def test_hybrid_retriever_rrf_fusion(hybrid_retriever: HybridRetriever) -> None:
    """Hybrid retriever returns fused results with correct structure."""
    results = hybrid_retriever.retrieve("Apple revenue 2023", top_k=5)
    assert len(results) > 0
    assert len(results) <= 5

    for result in results:
        assert "chunk" in result
        assert "score" in result
        assert "rank" in result
        assert result["score"] > 0


def test_hybrid_retriever_top_k_respected(hybrid_retriever: HybridRetriever) -> None:
    for k in (1, 3, 5):
        results = hybrid_retriever.retrieve("AI chip demand semiconductor", top_k=k)
        assert len(results) <= k


def test_hybrid_retriever_scores_descending(hybrid_retriever: HybridRetriever) -> None:
    results = hybrid_retriever.retrieve("revenue profit growth", top_k=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_retriever_no_duplicates(hybrid_retriever: HybridRetriever) -> None:
    """Each chunk_id appears at most once in hybrid results."""
    results = hybrid_retriever.retrieve("revenue billion", top_k=10)
    chunk_ids = [r["chunk"]["chunk_id"] for r in results]
    assert len(chunk_ids) == len(set(chunk_ids))


# ---------------------------------------------------------------------------
# Rank correctness
# ---------------------------------------------------------------------------


def test_retriever_ranks_are_correct(dense_retriever: DenseRetriever) -> None:
    """Ranks are 1-indexed and sequential."""
    results = dense_retriever.retrieve("quarterly earnings per share", top_k=5)
    ranks = [r["rank"] for r in results]
    assert ranks == list(range(1, len(ranks) + 1))


def test_sparse_retriever_ranks_are_correct(sparse_retriever: SparseRetriever) -> None:
    results = sparse_retriever.retrieve("subscriber growth percent", top_k=5)
    ranks = [r["rank"] for r in results]
    assert ranks == list(range(1, len(ranks) + 1))


def test_hybrid_retriever_ranks_are_correct(hybrid_retriever: HybridRetriever) -> None:
    results = hybrid_retriever.retrieve("cloud services operating income", top_k=5)
    ranks = [r["rank"] for r in results]
    assert ranks == list(range(1, len(ranks) + 1))
