"""Text preprocessing and chunking for SEC filings."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


class TextPreprocessor:
    """Clean and chunk raw filing text into overlapping word-based windows."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_length: int = 100,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length

    # ------------------------------------------------------------------
    # Text cleaning
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize unicode, remove control characters, collapse whitespace."""
        # Normalize unicode (e.g. smart quotes → ASCII equivalents)
        text = unicodedata.normalize("NFKD", text)
        # Remove non-printable / control characters (keep newlines and tabs)
        text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", " ", text)
        # Remove page numbers: lines that are only digits (possibly with dashes)
        text = re.sub(r"(?m)^\s*-?\s*\d+\s*-?\s*$", "", text)
        # Remove repeated dashes / underscores (common table separators in 10-K)
        text = re.sub(r"[-_]{4,}", " ", text)
        # Collapse runs of whitespace (preserve single newlines for readability)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def chunk_text(
        self,
        text: str,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> list[str]:
        """Split *text* into overlapping word-based chunks.

        Parameters
        ----------
        text:
            Input text (will be cleaned first).
        chunk_size:
            Number of words per chunk (defaults to ``self.chunk_size``).
        overlap:
            Number of words shared between consecutive chunks (defaults to
            ``self.chunk_overlap``).

        Returns
        -------
        list[str]
            Non-empty chunks that satisfy the minimum word count.
        """
        chunk_size = chunk_size if chunk_size is not None else self.chunk_size
        overlap = overlap if overlap is not None else self.chunk_overlap

        text = self._clean_text(text)
        words = text.split()

        if not words:
            return []

        step = max(1, chunk_size - overlap)
        chunks: list[str] = []

        for start in range(0, len(words), step):
            end = start + chunk_size
            chunk_words = words[start:end]
            if len(chunk_words) >= self.min_chunk_length:
                chunks.append(" ".join(chunk_words))
            # Stop when we've consumed all words
            if end >= len(words):
                break

        return chunks

    # ------------------------------------------------------------------
    # Filing-level processing
    # ------------------------------------------------------------------

    def process_filing(self, filing: dict[str, Any]) -> list[dict[str, Any]]:
        """Chunk a single filing dict and attach metadata to each chunk.

        Parameters
        ----------
        filing:
            Dict with at least ``text`` key. May also have ``company``,
            ``cik``, ``period``, ``filing_url``.

        Returns
        -------
        list[dict]
            Each dict has keys: ``text``, ``chunk_id``, ``company``,
            ``cik``, ``period``, ``filing_url``.
        """
        raw_text: str = filing.get("text", "")
        company: str = filing.get("company", "unknown")
        cik: str = str(filing.get("cik", ""))
        period: str = filing.get("period", "")
        filing_url: str = filing.get("filing_url", "")

        chunks = self.chunk_text(raw_text)
        result: list[dict[str, Any]] = []
        for idx, chunk_text in enumerate(chunks):
            chunk_id = f"{company}_{period}_{idx}"
            result.append(
                {
                    "text": chunk_text,
                    "chunk_id": chunk_id,
                    "company": company,
                    "cik": cik,
                    "period": period,
                    "filing_url": filing_url,
                }
            )
        return result

    def process_filings(self, filings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Process a list of filings and return all chunks.

        Parameters
        ----------
        filings:
            List of filing dicts (as returned by :class:`SECLoader`).

        Returns
        -------
        list[dict]
            All chunks from all filings, each with full metadata.
        """
        all_chunks: list[dict[str, Any]] = []
        for filing in filings:
            all_chunks.extend(self.process_filing(filing))
        return all_chunks
