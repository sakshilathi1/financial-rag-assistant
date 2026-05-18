"""Semantic chunker: splits text at topic boundaries detected via embedding similarity."""

import re
from typing import Any, Optional

import numpy as np

from src.chunking.base import BaseChunker, Chunk
from src.utils.logging import get_logger

log = get_logger(__name__)

# Sentence splitter: break on sentence-ending punctuation followed by whitespace
# and an uppercase letter or digit.  Handles most 10-K prose without requiring
# an NLP library.
_SENT_SPLIT = re.compile(r'(?<=[.!?])["\']?\s+(?=[A-Z\d"])')


def _split_sentences(text: str, min_length: int = 15) -> list[str]:
    """Split *text* into sentences using lightweight punctuation heuristics.

    Args:
        text: Input text (may span multiple paragraphs).
        min_length: Sentences shorter than this are merged with the previous one
                    to avoid embedding very short fragments.

    Returns:
        List of sentence strings with leading/trailing whitespace stripped.
    """
    # Also split on double-newlines (paragraph breaks in 10-K text).
    parts = re.split(r'(?:\n\s*){2,}|' + _SENT_SPLIT.pattern, text)
    sentences: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if sentences and len(part) < min_length:
            sentences[-1] = sentences[-1] + " " + part
        else:
            sentences.append(part)
    return sentences


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors.

    Args:
        a: First embedding vector.
        b: Second embedding vector.

    Returns:
        Scalar similarity in [-1, 1].
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class SemanticChunker(BaseChunker):
    """Splits text at semantic topic boundaries detected via embedding cosine similarity.

    Each sentence is embedded independently.  Consecutive sentence pairs whose
    cosine similarity falls below *similarity_threshold* are treated as a topic
    boundary; sentences on either side are accumulated into separate chunks.

    Dependency injection: pass a pre-built *embedder* (any object with an
    ``encode(texts: list[str]) -> np.ndarray`` method) to avoid loading a real
    model — useful for unit tests and ablation experiments.

    Args:
        embedder: Optional embedding model.  If ``None``, a
                  ``sentence_transformers.SentenceTransformer`` is loaded lazily
                  on the first call to :meth:`chunk`.
        similarity_threshold: Cosine-similarity value below which a sentence
                               boundary is inserted (default 0.7).
        model_name: HuggingFace model ID used when *embedder* is ``None``
                    (default ``"BAAI/bge-small-en-v1.5"``).
        min_chunk_size: Chunks with fewer characters (after stripping) are
                        dropped (default 100).
    """

    def __init__(
        self,
        embedder: Optional[Any] = None,
        similarity_threshold: float = 0.7,
        model_name: str = "BAAI/bge-small-en-v1.5",
        min_chunk_size: int = 100,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.model_name = model_name
        self.min_chunk_size = min_chunk_size
        self._embedder = embedder  # None triggers lazy init on first use

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def embedder(self) -> Any:
        """Return the embedding model, initialising it lazily if needed."""
        if self._embedder is None:
            log.info(
                "SemanticChunker: loading SentenceTransformer model '{}'",
                self.model_name,
            )
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            self._embedder = SentenceTransformer(self.model_name)
            log.info("SemanticChunker: model loaded.")
        return self._embedder

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _locate_sentences(self, text: str, sentences: list[str]) -> list[int]:
        """Return the start character offset in *text* for each sentence.

        Uses a monotonically advancing search so repeated sentences are mapped
        to distinct positions.

        Args:
            text: Original full text.
            sentences: Ordered sentence list from :func:`_split_sentences`.

        Returns:
            List of start offsets (same length as *sentences*).
        """
        offsets: list[int] = []
        pos = 0
        for sent in sentences:
            idx = text.find(sent, pos)
            if idx == -1:
                idx = pos  # fallback – shouldn't happen in practice
            offsets.append(idx)
            pos = idx + max(1, len(sent))
        return offsets

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[dict] = None,
    ) -> list[Chunk]:
        """Split *text* into semantically coherent chunks.

        Args:
            text: Plain text to split.
            doc_id: Identifier propagated to every chunk.
            metadata: Optional metadata dict.  ``"section"`` is extracted to
                      ``Chunk.section``; remaining keys go into
                      ``Chunk.metadata``.

        Returns:
            Ordered list of :class:`Chunk` objects.
        """
        meta = metadata or {}
        section = meta.get("section")
        chunk_meta = {k: v for k, v in meta.items() if k != "section"}

        if not text or not text.strip():
            return []

        sentences = _split_sentences(text)
        if not sentences:
            return []

        # ── Embed all sentences at once ───────────────────────────────────────
        embeddings: np.ndarray = self.embedder.encode(sentences)

        # ── Identify split boundaries ─────────────────────────────────────────
        # Index i is a split point if similarity(emb[i-1], emb[i]) < threshold.
        boundaries: list[int] = [0]
        for i in range(1, len(embeddings)):
            sim = _cosine_similarity(embeddings[i - 1], embeddings[i])
            if sim < self.similarity_threshold:
                boundaries.append(i)
        boundaries.append(len(sentences))

        # ── Map sentences back to character offsets ───────────────────────────
        sent_starts = self._locate_sentences(text, sentences)

        # ── Build chunks ──────────────────────────────────────────────────────
        chunks: list[Chunk] = []
        for chunk_idx in range(len(boundaries) - 1):
            s_start = boundaries[chunk_idx]
            s_end = boundaries[chunk_idx + 1]
            chunk_sentences = sentences[s_start:s_end]
            chunk_text = " ".join(chunk_sentences).strip()

            if len(chunk_text) < self.min_chunk_size:
                continue

            char_start = sent_starts[s_start]
            # char_end = start of the sentence AFTER the last one in this chunk,
            # or end of text if this chunk runs to the last sentence.
            if s_end < len(sent_starts):
                char_end = sent_starts[s_end]
            else:
                char_end = len(text)

            chunks.append(
                Chunk(
                    id=f"{doc_id}_{chunk_idx:04d}",
                    text=chunk_text,
                    doc_id=doc_id,
                    section=section,
                    char_start=char_start,
                    char_end=char_end,
                    metadata=chunk_meta.copy(),
                )
            )

        log.debug(
            "SemanticChunker produced {} chunks for doc '{}' "
            "(threshold={}, sentences={})",
            len(chunks),
            doc_id,
            self.similarity_threshold,
            len(sentences),
        )
        return chunks
