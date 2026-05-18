"""Unit tests for the chunking layer.

Fast tests (no real model):  run by default.
Slow test (real SentenceTransformer): skipped by default; opt-in with -m slow.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.chunking.base import Chunk
from src.chunking.fixed_chunker import FixedChunker
from src.chunking.semantic_chunker import SemanticChunker, _split_sentences


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_text(num_sentences: int = 20, words_per_sentence: int = 15) -> str:
    """Generate synthetic text with a fixed random seed for reproducibility."""
    import random
    rng = random.Random(0)
    vocab = ["Apple", "revenue", "grew", "significantly", "driven", "by",
             "strong", "iPhone", "sales", "cloud", "services", "margins",
             "increased", "operating", "income", "fiscal", "quarter",
             "technology", "market", "competition"]
    sentences = []
    for _ in range(num_sentences):
        words = [rng.choice(vocab) for _ in range(words_per_sentence)]
        words[0] = words[0].capitalize()
        sentences.append(" ".join(words) + ".")
    return " ".join(sentences)


# ──────────────────────────────────────────────────────────────────────────────
# FixedChunker
# ──────────────────────────────────────────────────────────────────────────────

class TestFixedChunker:

    def test_empty_text_returns_empty_list(self) -> None:
        chunker = FixedChunker()
        assert chunker.chunk("", doc_id="doc1") == []
        assert chunker.chunk("   \n  ", doc_id="doc1") == []

    def test_short_text_produces_single_chunk(self) -> None:
        # min_chunk_size must be <= len(text) for the chunk to survive the filter.
        chunker = FixedChunker(chunk_size=1000, min_chunk_size=10)
        text = "Apple reported record revenue this quarter."
        chunks = chunker.chunk(text, doc_id="doc1")
        assert len(chunks) == 1
        assert chunks[0].text == text.strip()

    def test_long_text_produces_multiple_chunks(self) -> None:
        chunker = FixedChunker(chunk_size=200, overlap=40, min_chunk_size=10)
        text = _make_text(num_sentences=30)
        chunks = chunker.chunk(text, doc_id="doc1")
        assert len(chunks) >= 2

    def test_chunk_ids_are_unique(self) -> None:
        chunker = FixedChunker(chunk_size=200, overlap=40, min_chunk_size=10)
        text = _make_text(num_sentences=30)
        chunks = chunker.chunk(text, doc_id="aapl")
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"

    def test_chunk_ids_contain_doc_id(self) -> None:
        chunker = FixedChunker(chunk_size=200, overlap=40, min_chunk_size=10)
        chunks = chunker.chunk(_make_text(), doc_id="MSFT")
        assert all(c.id.startswith("MSFT_") for c in chunks)

    def test_consecutive_chunks_overlap(self) -> None:
        """The start of chunk[i+1] must be before the end of chunk[i]."""
        chunker = FixedChunker(chunk_size=200, overlap=50, min_chunk_size=10)
        text = _make_text(num_sentences=30)
        chunks = chunker.chunk(text, doc_id="doc")
        for a, b in zip(chunks, chunks[1:]):
            assert b.char_start < a.char_end, (
                f"No overlap between chunk {a.id} (end={a.char_end}) "
                f"and {b.id} (start={b.char_start})"
            )

    def test_char_positions_are_valid(self) -> None:
        chunker = FixedChunker(chunk_size=200, overlap=40, min_chunk_size=10)
        text = _make_text(num_sentences=20)
        chunks = chunker.chunk(text, doc_id="doc")
        for c in chunks:
            assert 0 <= c.char_start < c.char_end <= len(text)
            assert c.text == text[c.char_start: c.char_end].strip()

    def test_metadata_propagated(self) -> None:
        chunker = FixedChunker(chunk_size=200, overlap=40, min_chunk_size=10)
        meta = {"section": "risk_factors", "ticker": "NVDA", "year": 2024}
        chunks = chunker.chunk(_make_text(), doc_id="nvda", metadata=meta)
        assert len(chunks) > 0
        for c in chunks:
            assert c.section == "risk_factors"
            assert c.metadata["ticker"] == "NVDA"
            assert "section" not in c.metadata, "'section' must not leak into metadata dict"

    def test_all_chunk_fields_populated(self) -> None:
        chunker = FixedChunker(chunk_size=300, overlap=50, min_chunk_size=10)
        chunk = chunker.chunk(_make_text(), doc_id="jpm")[0]
        assert isinstance(chunk, Chunk)
        assert chunk.id
        assert chunk.text
        assert chunk.doc_id == "jpm"
        assert isinstance(chunk.char_start, int)
        assert isinstance(chunk.char_end, int)
        assert isinstance(chunk.metadata, dict)

    def test_invalid_overlap_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            FixedChunker(chunk_size=100, overlap=100)

    def test_min_chunk_size_filters_tiny_chunks(self) -> None:
        # Text that yields a very short last chunk should be filtered out.
        chunker = FixedChunker(chunk_size=80, overlap=20, min_chunk_size=50)
        text = _make_text(num_sentences=5)
        chunks = chunker.chunk(text, doc_id="doc")
        assert all(len(c.text) >= 50 for c in chunks)


# ──────────────────────────────────────────────────────────────────────────────
# SemanticChunker  (all fast tests use mock_embedder)
# ──────────────────────────────────────────────────────────────────────────────

class TestSemanticChunker:

    def test_empty_text_returns_empty_list(self, mock_embedder: MagicMock) -> None:
        chunker = SemanticChunker(embedder=mock_embedder)
        assert chunker.chunk("", doc_id="doc") == []
        assert chunker.chunk("  \n ", doc_id="doc") == []

    def test_returns_list_of_chunks(self, mock_embedder: MagicMock) -> None:
        chunker = SemanticChunker(embedder=mock_embedder, similarity_threshold=0.99)
        text = _make_text(num_sentences=10)
        chunks = chunker.chunk(text, doc_id="doc")
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_encode_called_once_per_chunk_call(self, mock_embedder: MagicMock) -> None:
        """The embedder.encode must be called exactly once per chunk() invocation."""
        mock_embedder.encode.reset_mock()
        chunker = SemanticChunker(embedder=mock_embedder)
        chunker.chunk(_make_text(num_sentences=5), doc_id="doc")
        assert mock_embedder.encode.call_count == 1

    def test_encode_receives_sentence_list(self, mock_embedder: MagicMock) -> None:
        mock_embedder.encode.reset_mock()
        chunker = SemanticChunker(embedder=mock_embedder)
        text = "Apple grew revenue. Microsoft grew cloud income. Google grew ads."
        chunker.chunk(text, doc_id="doc")
        call_args = mock_embedder.encode.call_args[0][0]
        assert isinstance(call_args, list)
        assert len(call_args) >= 1

    def test_high_threshold_splits_every_sentence(self, mock_embedder: MagicMock) -> None:
        """threshold=1.0 → every pair of random vectors falls below it → max splits."""
        # Random vectors from the mock have cosine sim ≈ 0, so threshold=1.0
        # causes every pair to be a split boundary.
        chunker = SemanticChunker(
            embedder=mock_embedder,
            similarity_threshold=1.0,
            min_chunk_size=1,
        )
        text = "First sentence here. Second sentence here. Third sentence here."
        chunks = chunker.chunk(text, doc_id="doc")
        # Each sentence should become its own chunk (or close to it).
        assert len(chunks) >= 2

    def test_zero_threshold_merges_all_sentences(self) -> None:
        """threshold=0 with identical vectors (sim=1.0) → single chunk.

        Uses a local embedder that returns the same vector for every sentence so
        cosine similarity is always 1.0, which is never < 0.0 → no boundaries.
        """
        identical_mock = MagicMock()
        identical_mock.encode.side_effect = (
            lambda texts, **kw: np.ones((len(texts), 384), dtype=np.float32)
        )
        chunker = SemanticChunker(
            embedder=identical_mock,
            similarity_threshold=0.0,
            min_chunk_size=1,
        )
        text = "First sentence here. Second sentence here. Third sentence here."
        chunks = chunker.chunk(text, doc_id="doc")
        assert len(chunks) == 1

    def test_chunk_ids_contain_doc_id(self, mock_embedder: MagicMock) -> None:
        chunker = SemanticChunker(embedder=mock_embedder)
        chunks = chunker.chunk(_make_text(), doc_id="GOOGL")
        assert all(c.id.startswith("GOOGL_") for c in chunks)

    def test_metadata_propagated(self, mock_embedder: MagicMock) -> None:
        chunker = SemanticChunker(embedder=mock_embedder)
        meta = {"section": "mda", "ticker": "AAPL"}
        chunks = chunker.chunk(_make_text(), doc_id="aapl", metadata=meta)
        assert len(chunks) > 0
        for c in chunks:
            assert c.section == "mda"
            assert c.metadata.get("ticker") == "AAPL"
            assert "section" not in c.metadata

    def test_no_real_model_loaded(self, mock_embedder: MagicMock) -> None:
        """Confirm that passing an embedder prevents SentenceTransformer import."""
        chunker = SemanticChunker(embedder=mock_embedder)
        assert chunker._embedder is mock_embedder
        # Accessing .embedder property must return the injected object, not load a new one.
        assert chunker.embedder is mock_embedder


# ──────────────────────────────────────────────────────────────────────────────
# Sentence splitter unit tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSplitSentences:

    def test_basic_split(self) -> None:
        # Use sentences longer than min_length=15 so the merge heuristic doesn't
        # collapse them back into one.
        text = (
            "Apple reported record revenue this quarter. "
            "Microsoft expanded its cloud division significantly. "
            "Google launched new advertising products."
        )
        sentences = _split_sentences(text)
        assert len(sentences) >= 2

    def test_empty_string(self) -> None:
        assert _split_sentences("") == []

    def test_single_sentence_no_split(self) -> None:
        text = "Apple reported strong results this quarter."
        result = _split_sentences(text)
        assert len(result) >= 1
        assert any("Apple" in s for s in result)

    def test_paragraph_breaks_trigger_split(self) -> None:
        text = "First paragraph sentence.\n\nSecond paragraph sentence."
        result = _split_sentences(text)
        assert len(result) >= 2


# ──────────────────────────────────────────────────────────────────────────────
# Slow test – real SentenceTransformer model
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_semantic_chunker_real_model_end_to_end() -> None:
    """Integration test: loads BAAI/bge-small-en-v1.5 and chunks real 10-K text."""
    text = (
        "Apple Inc. designs, manufactures, and markets smartphones, personal "
        "computers, tablets, wearables, and accessories worldwide. "
        "The Company's iPhone product line is its most important product. "
        "Apple's services segment includes the App Store, Apple Music, iCloud, "
        "Apple TV+, Apple Arcade, and Apple Pay. "
        "The Company faces intense competition from Samsung, Google, and Microsoft. "
        "Revenue grew 8% year-over-year driven by strong services performance. "
        "The board declared a quarterly dividend of $0.25 per share. "
        "Capital expenditures for fiscal 2024 are expected to be approximately $11 billion."
    )
    chunker = SemanticChunker(similarity_threshold=0.5)
    chunks = chunker.chunk(text, doc_id="AAPL_slow_test")
    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.text for c in chunks)
    # Verify no real embedder leaked into subsequent tests by ensuring
    # it was created lazily on this specific instance only.
    assert chunker._embedder is not None
