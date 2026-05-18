"""Fixed-size character chunker with sentence-boundary awareness."""

import re
from typing import Optional

from src.chunking.base import BaseChunker, Chunk
from src.utils.logging import get_logger

log = get_logger(__name__)

# Sentence-ending punctuation followed by whitespace – used to find split points.
_SENT_END = re.compile(r"[.!?][\"']?\s")


class FixedChunker(BaseChunker):
    """Splits text into fixed-size chunks with configurable overlap.

    Chunk boundaries are nudged backwards to the nearest sentence end when one
    exists within the last ``overlap`` characters of the window, preventing
    mid-sentence cuts.

    Args:
        chunk_size: Target character length of each chunk (default 1000).
        overlap: Number of characters shared between consecutive chunks
                 (default 200).  Must be strictly less than *chunk_size*.
        min_chunk_size: Chunks shorter than this (after stripping) are dropped
                        (default 100).
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
        min_chunk_size: int = 100,
    ) -> None:
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _adjust_boundary(self, text: str, pos: int) -> int:
        """Return a position at or before *pos* aligned to a sentence end.

        Searches backwards within the last ``overlap`` characters for the
        rightmost sentence-ending marker.  Falls back to *pos* unchanged when
        no boundary is found.

        Args:
            text: Full text being chunked.
            pos: Initial candidate end position.

        Returns:
            Adjusted position (≤ *pos*).
        """
        search_start = max(0, pos - self.overlap)
        window = text[search_start:pos]
        best = -1
        for m in _SENT_END.finditer(window):
            # +1 keeps the punctuation char; the trailing space is NOT included
            # so the next chunk starts cleanly at the word after the period.
            best = m.start() + 1
        if best >= 0:
            return search_start + best
        return pos

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[dict] = None,
    ) -> list[Chunk]:
        """Split *text* into fixed-size chunks.

        Args:
            text: Plain text to split.
            doc_id: Identifier propagated to every chunk.
            metadata: Optional metadata dict.  A ``"section"`` key is extracted
                      and placed on ``Chunk.section``; remaining keys are copied
                      into ``Chunk.metadata``.

        Returns:
            List of :class:`Chunk` objects in document order.
        """
        meta = metadata or {}
        section = meta.get("section")
        chunk_meta = {k: v for k, v in meta.items() if k != "section"}

        if not text or not text.strip():
            return []

        stride = self.chunk_size - self.overlap
        chunks: list[Chunk] = []
        chunk_idx = 0
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Nudge the boundary to a sentence end when we are not at EOF.
            if end < len(text):
                end = self._adjust_boundary(text, end)
                # Guard: adjustment must advance past start to avoid infinite loop.
                if end <= start:
                    end = min(start + self.chunk_size, len(text))

            chunk_text = text[start:end].strip()
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(
                    Chunk(
                        id=f"{doc_id}_{chunk_idx:04d}",
                        text=chunk_text,
                        doc_id=doc_id,
                        section=section,
                        char_start=start,
                        char_end=end,
                        metadata=chunk_meta.copy(),
                    )
                )
                chunk_idx += 1

            # Advance by stride; ensure we always make forward progress.
            next_start = start + stride
            if next_start <= start:
                next_start = end
            start = next_start

        log.debug(
            "FixedChunker produced {} chunks for doc '{}' (chunk_size={}, overlap={})",
            len(chunks),
            doc_id,
            self.chunk_size,
            self.overlap,
        )
        return chunks
