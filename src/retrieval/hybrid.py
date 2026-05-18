"""Hybrid retrieval via Reciprocal Rank Fusion (RRF).

RRF combines ranked lists from multiple retrievers without requiring
score normalisation.  The formula is::

    RRF_score(d) = Σ  1 / (rrf_k + rank_i(d))

where *rank_i(d)* is the 1-indexed rank of document *d* in retriever *i*,
and *rrf_k* (default 60) is a smoothing constant.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combine dense and sparse retrievers using Reciprocal Rank Fusion."""

    def __init__(
        self,
        dense_retriever: Any,
        sparse_retriever: Any,
        top_k: int = 10,
        rrf_k: int = 60,
    ) -> None:
        """
        Parameters
        ----------
        dense_retriever:
            A :class:`~src.retrieval.dense.DenseRetriever` instance with a
            built index.
        sparse_retriever:
            A :class:`~src.retrieval.sparse.SparseRetriever` instance with a
            built index.
        top_k:
            Number of results to return after fusion.
        rrf_k:
            RRF smoothing constant (higher → less weight on top-ranked docs).
        """
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.top_k = top_k
        self.rrf_k = rrf_k

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self, query: str, top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """Retrieve and fuse results from dense and sparse retrievers.

        Parameters
        ----------
        query:
            Natural-language question.
        top_k:
            Override the default top-k.

        Returns
        -------
        list[dict]
            Ranked list of ``{chunk, score, rank}`` dicts sorted by descending
            RRF score.  Rank is 1-indexed.
        """
        k = top_k if top_k is not None else self.top_k

        # Fetch candidates from both retrievers (use larger k for fusion pool)
        fetch_k = max(k * 2, 20)
        dense_results = self.dense_retriever.retrieve(query, top_k=fetch_k)
        sparse_results = self.sparse_retriever.retrieve(query, top_k=fetch_k)

        # Map chunk_id → chunk dict
        chunk_map: dict[str, dict[str, Any]] = {}
        # Map chunk_id → accumulated RRF score
        rrf_scores: dict[str, float] = defaultdict(float)

        for retriever_results in [dense_results, sparse_results]:
            for result in retriever_results:
                chunk = result["chunk"]
                chunk_id = self._chunk_key(chunk)
                chunk_map[chunk_id] = chunk
                rrf_scores[chunk_id] += 1.0 / (self.rrf_k + result["rank"])

        # Sort by descending RRF score, take top-k
        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:k]

        results: list[dict[str, Any]] = []
        for rank, chunk_id in enumerate(sorted_ids, start=1):
            results.append(
                {
                    "chunk": chunk_map[chunk_id],
                    "score": rrf_scores[chunk_id],
                    "rank": rank,
                }
            )

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_key(chunk: dict[str, Any]) -> str:
        """Return a stable unique key for a chunk dict."""
        chunk_id = chunk.get("chunk_id")
        if chunk_id:
            return str(chunk_id)
        # Fallback: use first 120 characters of text
        return chunk.get("text", "")[:120]
