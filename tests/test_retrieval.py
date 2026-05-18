"""Unit tests for the retrieval layer.

All tests use MagicMock objects — no real models or ChromaDB on disk.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.chunking.base import Chunk
from src.retrieval.retriever import Retriever, _reciprocal_rank_fusion


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_chunk(chunk_id: str, text: str = "placeholder text for testing") -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        doc_id="test_doc",
        section=None,
        char_start=0,
        char_end=len(text),
    )


def _make_vector_store(
    chunks: list[Chunk],
    scores: list[float] | None = None,
) -> MagicMock:
    """Return a mock VectorStore whose .query() returns (chunk, score) pairs."""
    if scores is None:
        scores = [1.0 - i * 0.1 for i in range(len(chunks))]

    pairs = list(zip(chunks, scores))

    vs = MagicMock()
    vs.query.side_effect = lambda query, k=5: pairs[:k]
    vs.count.return_value = len(chunks)

    # .collection.get() used by _ensure_bm25
    vs.collection.get.return_value = {
        "ids": [c.id for c in chunks],
        "documents": [c.text for c in chunks],
    }
    return vs


# ──────────────────────────────────────────────────────────────────────────────
# RRF unit tests (pure function, no mocks needed)
# ──────────────────────────────────────────────────────────────────────────────

class TestRRF:

    def test_single_list_preserves_order(self) -> None:
        ranking = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        result = _reciprocal_rank_fusion([ranking])
        ids = [r[0] for r in result]
        assert ids == ["a", "b", "c"]

    def test_two_identical_lists_preserve_order(self) -> None:
        ranking = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        result = _reciprocal_rank_fusion([ranking, ranking])
        ids = [r[0] for r in result]
        assert ids == ["a", "b", "c"]

    def test_rrf_score_formula(self) -> None:
        """Verify 1/(k+rank) arithmetic for k=60."""
        k = 60
        ranking = [("x", 1.0), ("y", 0.5)]
        result = _reciprocal_rank_fusion([ranking], k=k)
        scores = {r[0]: r[1] for r in result}
        assert abs(scores["x"] - 1 / (k + 1)) < 1e-9
        assert abs(scores["y"] - 1 / (k + 2)) < 1e-9

    def test_higher_ranked_item_wins_fusion(self) -> None:
        """Item ranked 1st in both lists must outscore item ranked 2nd in both."""
        list_a = [("alpha", 1.0), ("beta", 0.5)]
        list_b = [("alpha", 0.8), ("beta", 0.3)]
        result = _reciprocal_rank_fusion([list_a, list_b])
        scores = {r[0]: r[1] for r in result}
        assert scores["alpha"] > scores["beta"]

    def test_item_present_in_only_one_list(self) -> None:
        """An item in only one ranking still appears in the output."""
        list_a = [("a", 1.0), ("b", 0.5)]
        list_b = [("c", 1.0)]
        result = _reciprocal_rank_fusion([list_a, list_b])
        ids = {r[0] for r in result}
        assert {"a", "b", "c"} == ids

    def test_item_in_both_lists_beats_item_in_one(self) -> None:
        """An item that appears in both lists (lower ranks) should outscore
        an item that appears in only one list (high rank) if both are low ranks."""
        # "shared" appears at rank 2 in both lists → RRF = 1/62 + 1/62 = 2/62 ≈ 0.032
        # "exclusive" appears at rank 1 in one list only → RRF = 1/61 ≈ 0.016
        list_a = [("exclusive", 1.0), ("shared", 0.5)]
        list_b = [("exclusive2", 1.0), ("shared", 0.4)]
        result = _reciprocal_rank_fusion([list_a, list_b])
        scores = {r[0]: r[1] for r in result}
        assert scores["shared"] > scores["exclusive"]
        assert scores["shared"] > scores["exclusive2"]

    def test_empty_rankings(self) -> None:
        assert _reciprocal_rank_fusion([]) == []

    def test_empty_inner_list(self) -> None:
        result = _reciprocal_rank_fusion([[]])
        assert result == []


# ──────────────────────────────────────────────────────────────────────────────
# Retriever — mode selection
# ──────────────────────────────────────────────────────────────────────────────

class TestRetrieverModeSelection:

    def test_invalid_mode_at_init_raises(self) -> None:
        vs = MagicMock()
        with pytest.raises(ValueError, match="mode"):
            Retriever(vector_store=vs, mode="bm25_only")

    def test_invalid_mode_at_retrieve_raises(self) -> None:
        chunks = [_make_chunk("c1"), _make_chunk("c2")]
        vs = _make_vector_store(chunks)
        retriever = Retriever(vector_store=vs, mode="dense")
        with pytest.raises(ValueError, match="mode"):
            retriever.retrieve("query", mode="bad_mode")

    def test_dense_mode_calls_vector_store_query(self) -> None:
        chunks = [_make_chunk("c1"), _make_chunk("c2")]
        vs = _make_vector_store(chunks)
        retriever = Retriever(vector_store=vs, mode="dense")
        results = retriever.retrieve("revenue growth", top_k=2, mode="dense")
        vs.query.assert_called_once()
        assert len(results) == 2

    def test_retrieve_respects_per_call_mode_override(self) -> None:
        chunks = [_make_chunk("c1")]
        vs = _make_vector_store(chunks)
        # Default mode is hybrid, but override to dense for this call.
        retriever = Retriever(vector_store=vs, mode="hybrid")
        results = retriever.retrieve("profit margin", top_k=1, mode="dense")
        vs.query.assert_called_once()
        assert len(results) == 1


# ──────────────────────────────────────────────────────────────────────────────
# Retriever — dense mode
# ──────────────────────────────────────────────────────────────────────────────

class TestDenseRetriever:

    def test_returns_correct_number_of_results(self) -> None:
        chunks = [_make_chunk(f"c{i}") for i in range(5)]
        vs = _make_vector_store(chunks)
        retriever = Retriever(vector_store=vs, mode="dense")
        results = retriever.retrieve("operating expenses", top_k=3, mode="dense")
        assert len(results) == 3

    def test_returns_chunk_score_tuples(self) -> None:
        chunks = [_make_chunk("c1"), _make_chunk("c2")]
        vs = _make_vector_store(chunks, scores=[0.95, 0.80])
        retriever = Retriever(vector_store=vs, mode="dense")
        results = retriever.retrieve("risk factors", top_k=2, mode="dense")
        assert all(isinstance(r[0], Chunk) for r in results)
        assert all(isinstance(r[1], float) for r in results)

    def test_empty_store_returns_empty_list(self) -> None:
        vs = MagicMock()
        vs.query.return_value = []
        vs.count.return_value = 0
        retriever = Retriever(vector_store=vs, mode="dense")
        results = retriever.retrieve("cash flow", top_k=5)
        assert results == []


# ──────────────────────────────────────────────────────────────────────────────
# Retriever — hybrid mode
# ──────────────────────────────────────────────────────────────────────────────

class TestHybridRetriever:

    def _make_hybrid_retriever(self, chunks: list[Chunk]) -> tuple[Retriever, MagicMock]:
        vs = _make_vector_store(chunks)
        # Provide pre-tokenised corpus so we skip the real collection.get() path.
        corpus = [c.text.lower().split() for c in chunks]
        retriever = Retriever(vector_store=vs, corpus=corpus, mode="hybrid")
        # Corpus IDs must be set for BM25 ranking to work.
        retriever._corpus_ids = [c.id for c in chunks]
        return retriever, vs

    def test_hybrid_returns_chunk_score_tuples(self) -> None:
        chunks = [_make_chunk(f"c{i}", text=f"revenue growth quarter {i}") for i in range(4)]
        retriever, _ = self._make_hybrid_retriever(chunks)
        results = retriever.retrieve("revenue", top_k=3)
        assert all(isinstance(r[0], Chunk) for r in results)
        assert all(isinstance(r[1], float) for r in results)

    def test_hybrid_top_k_respected(self) -> None:
        chunks = [_make_chunk(f"c{i}", text=f"sentence about topic {i}") for i in range(6)]
        retriever, _ = self._make_hybrid_retriever(chunks)
        results = retriever.retrieve("topic", top_k=3)
        assert len(results) <= 3

    def test_hybrid_scores_are_positive(self) -> None:
        chunks = [_make_chunk(f"c{i}", text=f"Apple revenue fiscal year {i}") for i in range(4)]
        retriever, _ = self._make_hybrid_retriever(chunks)
        results = retriever.retrieve("Apple revenue", top_k=4)
        assert all(score > 0 for _, score in results)

    def test_hybrid_builds_bm25_lazily(self) -> None:
        """BM25 index should not exist until first hybrid retrieve call."""
        chunks = [_make_chunk("c1", "financial results quarterly")]
        vs = _make_vector_store(chunks)
        corpus = [c.text.lower().split() for c in chunks]
        retriever = Retriever(vector_store=vs, corpus=corpus, mode="hybrid")
        retriever._corpus_ids = [c.id for c in chunks]
        assert retriever._bm25 is None
        retriever.retrieve("financial results", top_k=1)
        assert retriever._bm25 is not None

    def test_hybrid_uses_rrf_fusion(self) -> None:
        """Scores must come from RRF (small positive floats ~1/60), not raw cosine."""
        chunks = [_make_chunk(f"c{i}", f"term{i} data analysis") for i in range(3)]
        retriever, _ = self._make_hybrid_retriever(chunks)
        results = retriever.retrieve("term0 analysis", top_k=3)
        # RRF scores are bounded by 1/(k+1) ≤ score ≤ 2*(1/61) ≈ 0.033
        for _, score in results:
            assert 0 < score < 1.0, f"Expected RRF score in (0, 1), got {score}"


# ──────────────────────────────────────────────────────────────────────────────
# format_context
# ──────────────────────────────────────────────────────────────────────────────

class TestFormatContext:

    def test_chunk_id_appears_in_output(self) -> None:
        from src.generation.prompts import format_context
        chunk = _make_chunk("AAPL_0001", "Revenue grew significantly.")
        result = format_context([chunk])
        assert "[AAPL_0001]" in result

    def test_section_included_when_present(self) -> None:
        from src.generation.prompts import format_context
        chunk = Chunk(
            id="MSFT_0002",
            text="Cloud revenue increased.",
            doc_id="msft",
            section="mda",
            char_start=0,
            char_end=22,
        )
        result = format_context([chunk])
        assert "(mda)" in result

    def test_multiple_chunks_separated(self) -> None:
        from src.generation.prompts import format_context
        chunks = [_make_chunk(f"c{i}") for i in range(3)]
        result = format_context(chunks)
        assert result.count("[c") == 3

    def test_empty_list_returns_empty_string(self) -> None:
        from src.generation.prompts import format_context
        assert format_context([]) == ""


# ──────────────────────────────────────────────────────────────────────────────
# build_citation_prompt
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildCitationPrompt:

    def test_question_in_prompt(self) -> None:
        from src.generation.prompts import build_citation_prompt
        chunk = _make_chunk("c1", "Apple revenue grew.")
        prompt = build_citation_prompt("What drove revenue growth?", [chunk])
        assert "What drove revenue growth?" in prompt

    def test_chunk_id_in_prompt(self) -> None:
        from src.generation.prompts import build_citation_prompt
        chunk = _make_chunk("NVDA_0005", "GPU sales surged.")
        prompt = build_citation_prompt("GPU performance?", [chunk])
        assert "[NVDA_0005]" in prompt

    def test_json_instruction_present(self) -> None:
        from src.generation.prompts import build_citation_prompt
        chunk = _make_chunk("c1")
        prompt = build_citation_prompt("question", [chunk])
        assert '"answer"' in prompt
        assert '"citations"' in prompt
        assert '"confidence"' in prompt
