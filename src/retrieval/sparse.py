"""Sparse retrieval using BM25 (rank-bm25 library)."""

from __future__ import annotations

import logging
import pickle
import re
import string
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    return text.split()


class SparseRetriever:
    """BM25-based sparse retriever backed by ``rank-bm25``."""

    def __init__(
        self,
        top_k: int = 10,
        b: float = 0.75,
        k1: float = 1.5,
    ) -> None:
        self.top_k = top_k
        self.b = b
        self.k1 = k1

        self.bm25: BM25Okapi | None = None
        self.chunks: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build_index(self, chunks: list[dict[str, Any]]) -> None:
        """Tokenize all chunk texts and build a BM25Okapi index.

        Parameters
        ----------
        chunks:
            List of chunk dicts — each must have a ``text`` key.
        """
        if not chunks:
            raise ValueError("Cannot build index from empty chunk list.")

        self.chunks = chunks
        tokenized = [_tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        logger.info("BM25 index built: %d documents", len(chunks))

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self, query: str, top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """Retrieve the top-k chunks by BM25 score for *query*.

        Parameters
        ----------
        query:
            Natural-language question.
        top_k:
            Override the default top-k.

        Returns
        -------
        list[dict]
            Ranked list of ``{chunk, score, rank}`` dicts (rank is 1-indexed).
        """
        if self.bm25 is None or not self.chunks:
            raise RuntimeError("Index not built. Call build_index() first.")

        k = top_k if top_k is not None else self.top_k
        k = min(k, len(self.chunks))

        query_tokens = _tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # argsort descending, take top-k
        top_indices = scores.argsort()[::-1][:k]

        results: list[dict[str, Any]] = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append(
                {
                    "chunk": self.chunks[int(idx)],
                    "score": float(scores[int(idx)]),
                    "rank": rank,
                }
            )
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_index(self, path: str | Path) -> None:
        """Pickle the BM25 object and chunk metadata to *path* (directory).

        Creates ``sparse_index.pkl`` inside *path*.
        """
        if self.bm25 is None:
            raise RuntimeError("No index to save. Call build_index() first.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        payload = {"bm25": self.bm25, "chunks": self.chunks, "top_k": self.top_k}
        with open(path / "sparse_index.pkl", "wb") as fh:
            pickle.dump(payload, fh)

        logger.info("Sparse index saved to %s", path)

    def load_index(self, path: str | Path) -> None:
        """Load a previously saved BM25 index from *path*.

        Parameters
        ----------
        path:
            Directory containing ``sparse_index.pkl``.
        """
        path = Path(path)
        pkl_file = path / "sparse_index.pkl"

        if not pkl_file.exists():
            raise FileNotFoundError(f"Sparse index not found: {pkl_file}")

        with open(pkl_file, "rb") as fh:
            payload = pickle.load(fh)

        self.bm25 = payload["bm25"]
        self.chunks = payload["chunks"]
        self.top_k = payload.get("top_k", self.top_k)

        logger.info("Sparse index loaded from %s (%d documents)", path, len(self.chunks))
