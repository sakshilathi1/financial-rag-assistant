"""Cross-encoder reranker for re-scoring retrieved chunks."""

from typing import Any, Optional

from src.chunking.base import Chunk
from src.utils.logging import get_logger

log = get_logger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Rerank retrieved chunks using a cross-encoder relevance model.

    The cross-encoder scores each ``(query, chunk_text)`` pair jointly,
    providing more accurate relevance estimates than bi-encoder cosine
    similarity at the cost of higher latency.

    The underlying model is loaded lazily on the first call to
    :meth:`rerank`, so constructing a ``CrossEncoderReranker`` is cheap.

    Args:
        model: A pre-loaded cross-encoder model with a ``predict`` method
               (e.g. ``sentence_transformers.CrossEncoder``).  When ``None``,
               ``cross-encoder/ms-marco-MiniLM-L-6-v2`` is loaded lazily.
    """

    def __init__(self, model: Optional[Any] = None) -> None:
        self._model = model

    @property
    def model(self) -> Any:
        """Return the cross-encoder model, loading it lazily if needed."""
        if self._model is None:
            log.info("CrossEncoderReranker: loading model '{}'", _DEFAULT_MODEL)
            from sentence_transformers import CrossEncoder  # noqa: PLC0415

            self._model = CrossEncoder(_DEFAULT_MODEL)
            log.info("CrossEncoderReranker: model loaded.")
        return self._model

    def rerank(
        self,
        query: str,
        chunks_with_scores: list[tuple[Chunk, float]],
        top_k: int,
    ) -> list[tuple[Chunk, float]]:
        """Re-score *chunks_with_scores* and return the top-*k* by cross-encoder score.

        Args:
            query: The original query string.
            chunks_with_scores: Retrieved ``(Chunk, score)`` pairs.
            top_k: Number of results to return after reranking.

        Returns:
            List of ``(Chunk, cross_encoder_score)`` pairs ordered by
            descending relevance (up to *top_k* items).
        """
        if not chunks_with_scores:
            return []

        pairs = [(query, chunk.text) for chunk, _ in chunks_with_scores]
        ce_scores: list[float] = self.model.predict(pairs).tolist()

        reranked = sorted(
            zip([chunk for chunk, _ in chunks_with_scores], ce_scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        log.debug(
            "CrossEncoderReranker: reranked {} → {} results",
            len(chunks_with_scores),
            len(reranked),
        )
        return reranked
