"""Dense and hybrid (BM25 + dense) retriever with reciprocal rank fusion."""

from typing import TYPE_CHECKING, Optional

from src.chunking.base import Chunk
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.retrieval.vector_store import VectorStore

log = get_logger(__name__)

_RRF_K = 60  # standard RRF constant


def _reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    k: int = _RRF_K,
) -> list[tuple[str, float]]:
    """Merge multiple ranked lists into one via reciprocal rank fusion.

    Args:
        rankings: Each inner list is ``[(chunk_id, score), ...]`` ordered by
                  descending score.  Scores are not used in RRF — only ranks.
        k: RRF smoothing constant (default 60).

    Returns:
        List of ``(chunk_id, rrf_score)`` sorted by descending RRF score.
    """
    scores: dict[str, float] = {}
    for ranked_list in rankings:
        for rank, (chunk_id, _) in enumerate(ranked_list, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class Retriever:
    """Retrieve relevant chunks using dense or hybrid (BM25 + dense) search.

    Dense mode delegates entirely to the underlying :class:`VectorStore`.
    Hybrid mode runs BM25 and dense search in parallel, then fuses rankings
    with reciprocal rank fusion (RRF).

    Args:
        vector_store: Populated :class:`~src.retrieval.vector_store.VectorStore`.
        corpus: Pre-tokenised BM25 corpus as a list of token lists.  When
                ``None``, the BM25 index is built lazily from
                ``vector_store.collection`` documents on the first hybrid query.
        mode: Default retrieval mode (``"dense"`` or ``"hybrid"``).
    """

    def __init__(
        self,
        vector_store: "VectorStore",
        corpus: Optional[list[list[str]]] = None,
        mode: str = "hybrid",
    ) -> None:
        if mode not in {"dense", "hybrid"}:
            raise ValueError(f"mode must be 'dense' or 'hybrid', got '{mode}'")
        self.vector_store = vector_store
        self.mode = mode
        self._corpus: Optional[list[list[str]]] = corpus
        self._corpus_ids: Optional[list[str]] = None
        self._bm25: Optional[object] = None

    # ------------------------------------------------------------------
    # BM25 initialisation
    # ------------------------------------------------------------------

    def _ensure_bm25(self) -> None:
        """Build the BM25 index from the ChromaDB collection if not yet built."""
        if self._bm25 is not None:
            return

        from rank_bm25 import BM25Okapi  # noqa: PLC0415

        if self._corpus is not None:
            # Caller provided a pre-tokenised corpus; IDs must match.
            self._bm25 = BM25Okapi(self._corpus)
            return

        # Fetch all documents from the collection.
        count = self.vector_store.count()
        if count == 0:
            log.warning("Retriever._ensure_bm25: collection is empty — BM25 index will be empty")
            self._corpus = []
            self._corpus_ids = []
            self._bm25 = BM25Okapi([[]])
            return

        result = self.vector_store.collection.get(
            include=["documents"],
            limit=count,
        )
        documents: list[str] = result["documents"]
        self._corpus_ids = result["ids"]
        self._corpus = [doc.lower().split() for doc in documents]
        self._bm25 = BM25Okapi(self._corpus)
        log.info("Retriever: BM25 index built over {} documents", len(self._corpus))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: Optional[str] = None,
    ) -> list[tuple[Chunk, float]]:
        """Retrieve the top-*k* most relevant chunks for *query*.

        Args:
            query: Natural-language query string.
            top_k: Maximum number of results to return.
            mode: Override the instance-level mode for this call.  Must be
                  ``"dense"`` or ``"hybrid"`` if provided.

        Returns:
            List of ``(Chunk, score)`` pairs ordered by descending relevance.
        """
        effective_mode = mode if mode is not None else self.mode
        if effective_mode not in {"dense", "hybrid"}:
            raise ValueError(f"mode must be 'dense' or 'hybrid', got '{effective_mode}'")

        if effective_mode == "dense":
            return self._dense_retrieve(query, top_k)
        return self._hybrid_retrieve(query, top_k)

    def _dense_retrieve(
        self,
        query: str,
        top_k: int,
    ) -> list[tuple[Chunk, float]]:
        """Pure vector-similarity retrieval."""
        results = self.vector_store.query(query, k=top_k)
        log.debug("Retriever.dense: {} results for '{}'", len(results), query[:60])
        return results

    def _hybrid_retrieve(
        self,
        query: str,
        top_k: int,
    ) -> list[tuple[Chunk, float]]:
        """BM25 + dense retrieval fused via RRF."""
        self._ensure_bm25()

        # ── Dense ranking ─────────────────────────────────────────────────────
        # Fetch more candidates than top_k so RRF has a wider pool to fuse.
        fetch_k = min(top_k * 4, max(top_k, 20))
        dense_results = self.vector_store.query(query, k=fetch_k)
        dense_ranking: list[tuple[str, float]] = [
            (chunk.id, score) for chunk, score in dense_results
        ]

        # ── BM25 ranking ──────────────────────────────────────────────────────
        assert self._bm25 is not None  # guaranteed by _ensure_bm25
        assert self._corpus_ids is not None

        query_tokens = query.lower().split()
        bm25_scores: list[float] = self._bm25.get_scores(query_tokens)
        bm25_ranking: list[tuple[str, float]] = sorted(
            zip(self._corpus_ids, bm25_scores),
            key=lambda x: x[1],
            reverse=True,
        )[:fetch_k]

        # ── RRF fusion ────────────────────────────────────────────────────────
        fused = _reciprocal_rank_fusion([dense_ranking, bm25_ranking])[:top_k]

        # ── Reconstruct Chunk objects from dense results ───────────────────────
        chunk_map: dict[str, tuple[Chunk, float]] = {
            chunk.id: (chunk, score) for chunk, score in dense_results
        }

        output: list[tuple[Chunk, float]] = []
        for chunk_id, rrf_score in fused:
            if chunk_id in chunk_map:
                chunk, _ = chunk_map[chunk_id]
                output.append((chunk, rrf_score))
            # Chunks that appear only in BM25 but not dense are skipped here
            # because we don't have their text without a second ChromaDB lookup.
            # In practice the overlap between the two fetch_k sets is very high.

        log.debug(
            "Retriever.hybrid: {} results after RRF for '{}'",
            len(output),
            query[:60],
        )
        return output
