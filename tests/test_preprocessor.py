"""Tests for TextPreprocessor."""

from __future__ import annotations

import pytest

from src.data.preprocessor import TextPreprocessor


@pytest.fixture()
def preprocessor() -> TextPreprocessor:
    return TextPreprocessor(chunk_size=20, chunk_overlap=5, min_chunk_length=10)


@pytest.fixture()
def sample_filing() -> dict:
    words = " ".join([f"word{i}" for i in range(200)])
    return {
        "text": words,
        "company": "AAPL",
        "cik": "0000320193",
        "period": "2023-09-30",
        "filing_url": "https://example.com/filing",
    }


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------


def test_chunk_text_basic(preprocessor: TextPreprocessor) -> None:
    """Basic chunking produces a non-empty list of strings."""
    text = " ".join([f"word{i}" for i in range(100)])
    chunks = preprocessor.chunk_text(text)
    assert len(chunks) > 0
    assert all(isinstance(c, str) for c in chunks)
    assert all(len(c) > 0 for c in chunks)


def test_chunk_text_overlap(preprocessor: TextPreprocessor) -> None:
    """Overlapping chunks share some words between consecutive chunks."""
    words = [f"word{i}" for i in range(60)]
    text = " ".join(words)
    chunks = preprocessor.chunk_text(text, chunk_size=20, overlap=5)
    assert len(chunks) >= 2
    # The last 5 words of chunk[0] should appear at the start of chunk[1]
    last_words_first = chunks[0].split()[-5:]
    first_words_second = chunks[1].split()[:5]
    assert last_words_first == first_words_second


def test_chunk_text_short_text(preprocessor: TextPreprocessor) -> None:
    """Text shorter than min_chunk_length produces no chunks."""
    text = " ".join(["word"] * 5)  # 5 words, min_chunk_length = 10
    chunks = preprocessor.chunk_text(text)
    assert chunks == []


def test_chunk_text_empty(preprocessor: TextPreprocessor) -> None:
    """Empty text returns empty list."""
    assert preprocessor.chunk_text("") == []


def test_chunk_text_whitespace_only(preprocessor: TextPreprocessor) -> None:
    """Whitespace-only text returns empty list."""
    assert preprocessor.chunk_text("   \n\t  ") == []


def test_chunk_length_filtering(preprocessor: TextPreprocessor) -> None:
    """Chunks with fewer words than min_chunk_length are filtered out."""
    # Create text just long enough for one full chunk but leave a small tail
    words = [f"w{i}" for i in range(25)]  # chunk_size=20, overlap=5 → step=15
    text = " ".join(words)
    # With step=15: chunk 0 = words[0:20], chunk 1 = words[15:35] but only 10 words remain
    chunks = preprocessor.chunk_text(text, chunk_size=20, overlap=5)
    # All returned chunks must have at least min_chunk_length (10) words
    for chunk in chunks:
        assert len(chunk.split()) >= preprocessor.min_chunk_length


# ---------------------------------------------------------------------------
# process_filing
# ---------------------------------------------------------------------------


def test_process_filing(preprocessor: TextPreprocessor, sample_filing: dict) -> None:
    """process_filing returns chunk dicts with the correct structure."""
    result = preprocessor.process_filing(sample_filing)
    assert isinstance(result, list)
    assert len(result) > 0

    required_keys = {"text", "chunk_id", "company", "cik", "period", "filing_url"}
    for chunk in result:
        assert required_keys.issubset(chunk.keys()), f"Missing keys: {required_keys - chunk.keys()}"
        assert chunk["company"] == "AAPL"
        assert chunk["period"] == "2023-09-30"
        assert "AAPL" in chunk["chunk_id"]


def test_process_filing_empty_text(preprocessor: TextPreprocessor) -> None:
    """process_filing with empty text returns an empty list."""
    filing = {"text": "", "company": "TEST", "cik": "123", "period": "2023", "filing_url": ""}
    assert preprocessor.process_filing(filing) == []


def test_process_filings(preprocessor: TextPreprocessor, sample_filing: dict) -> None:
    """process_filings aggregates chunks from multiple filings."""
    filing2 = {**sample_filing, "company": "MSFT", "period": "2022-12-31"}
    result = preprocessor.process_filings([sample_filing, filing2])
    assert len(result) > 0
    companies = {c["company"] for c in result}
    assert "AAPL" in companies
    assert "MSFT" in companies


def test_chunk_ids_are_unique(preprocessor: TextPreprocessor, sample_filing: dict) -> None:
    """Each chunk from a single filing has a unique chunk_id."""
    chunks = preprocessor.process_filing(sample_filing)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
