"""Dense retrieval using sentence-transformers and FAISS."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Encode text chunks with a sentence-transformer and index with FAISS.

    Uses inner-product search on L2-normalised embeddings, which is
    equivalent to cosine similarity.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        top_k: int = 10,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.top_k = top_k
        self.batch_size = batch_size

        logger.info("Loading sentence-transformer model: %s", model_name)
        self.model = SentenceTransformer(model_name)

        self.index: faiss.Index | None = None
        self.chunks: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode *texts* in batches and return L2-normalised embeddings."""
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine via inner-product
        )
        return embeddings.astype(np.float32)

    def build_index(self, chunks: list[dict[str, Any]]) -> None:
        """Encode all chunks and build a FAISS IndexFlatIP index.

        Parameters
        ----------
        chunks:
            List of chunk dicts — each must have a ``text`` key.
        """
        if not chunks:
            raise ValueError("Cannot build index from empty chunk list.")

        self.chunks = chunks
        texts = [c["text"] for c in chunks]

        logger.info("Encoding %d chunks with %s…", len(texts), self.model_name)
        embeddings = self._encode(texts)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        logger.info("FAISS index built: %d vectors, dim=%d", self.index.ntotal, dim)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self, query: str, top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """Retrieve the top-k most similar chunks for *query*.

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
        if self.index is None or not self.chunks:
            raise RuntimeError("Index not built. Call build_index() first.")

        k = top_k if top_k is not None else self.top_k
        k = min(k, len(self.chunks))

        query_embedding = self._encode([query])
        scores, indices = self.index.search(query_embedding, k)

        results: list[dict[str, Any]] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx == -1:
                continue
            results.append(
                {
                    "chunk": self.chunks[idx],
                    "score": float(score),
                    "rank": rank,
                }
            )
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_index(self, path: str | Path) -> None:
        """Persist the FAISS index and chunk metadata to *path* (directory).

        Creates two files:
        - ``dense.faiss`` — the FAISS binary index
        - ``dense_chunks.json`` — chunk metadata
        """
        if self.index is None:
            raise RuntimeError("No index to save. Call build_index() first.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "dense.faiss"))
        with open(path / "dense_chunks.json", "w", encoding="utf-8") as fh:
            json.dump(self.chunks, fh, ensure_ascii=False)

        logger.info("Dense index saved to %s", path)

    def load_index(self, path: str | Path, chunks: list[dict[str, Any]] | None = None) -> None:
        """Load a previously saved FAISS index from *path*.

        Parameters
        ----------
        path:
            Directory containing ``dense.faiss`` and optionally
            ``dense_chunks.json``.
        chunks:
            If provided, these chunks are used instead of loading from disk.
        """
        path = Path(path)
        index_file = path / "dense.faiss"
        chunks_file = path / "dense_chunks.json"

        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_file}")

        self.index = faiss.read_index(str(index_file))

        if chunks is not None:
            self.chunks = chunks
        elif chunks_file.exists():
            with open(chunks_file, "r", encoding="utf-8") as fh:
                self.chunks = json.load(fh)
        else:
            raise FileNotFoundError(
                f"Chunk metadata not found: {chunks_file}. Pass chunks= explicitly."
            )

        logger.info(
            "Dense index loaded from %s (%d vectors, %d chunks)",
            path,
            self.index.ntotal,
            len(self.chunks),
        )
