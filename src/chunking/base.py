"""Base types for the chunking layer: Chunk dataclass and BaseChunker ABC."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chunk:
    """A single text chunk produced by a chunker.

    Attributes:
        id: Globally unique identifier, typically ``{doc_id}_{index:04d}``.
        text: The chunk's plain text content.
        doc_id: Identifier of the source document (e.g. ticker symbol).
        section: Optional 10-K section name (``business``, ``risk_factors``, …).
        char_start: Start character offset in the original document text.
        char_end: End character offset (exclusive) in the original document text.
        metadata: Arbitrary key/value pairs (filing date, source path, …).
    """

    id: str
    text: str
    doc_id: str
    section: Optional[str]
    char_start: int
    char_end: int
    metadata: dict = field(default_factory=dict)


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    @abstractmethod
    def chunk(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[dict] = None,
    ) -> list[Chunk]:
        """Split *text* into a list of :class:`Chunk` objects.

        Args:
            text: Full plain-text document (or section) to chunk.
            doc_id: Identifier propagated to every produced chunk.
            metadata: Optional dict of key/value pairs propagated to every chunk.
                      A ``"section"`` key is extracted and stored on the chunk's
                      dedicated ``section`` field; remaining keys go into
                      ``chunk.metadata``.

        Returns:
            Ordered list of non-overlapping (or overlapping-by-design) chunks.
            Returns an empty list when *text* is blank.
        """
