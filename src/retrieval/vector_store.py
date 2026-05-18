"""ChromaDB persistent vector store wrapper."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from src.chunking.base import Chunk
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.embeddings.embedder import Embedder

log = get_logger(__name__)

# Metadata keys stored on every chunk in ChromaDB (extracted back on query).
_RESERVED_META_KEYS = frozenset({"doc_id", "section", "char_start", "char_end"})


def _sanitise_metadata(meta: dict) -> dict[str, str | int | float | bool]:
    """Coerce metadata values to types accepted by ChromaDB.

    ChromaDB only stores ``str``, ``int``, ``float``, and ``bool``.  All other
    types are converted to their string representation.

    Args:
        meta: Arbitrary metadata dict.

    Returns:
        New dict with only ChromaDB-compatible value types.
    """
    out: dict[str, str | int | float | bool] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


class VectorStore:
    """Persistent ChromaDB collection for :class:`~src.chunking.base.Chunk` objects.

    All chunk metadata is stored alongside the text and embedding, allowing
    full :class:`Chunk` reconstruction from query results without a separate
    metadata store.

    The underlying ChromaDB client and collection are created lazily on first
    access, so constructing a ``VectorStore`` never touches the filesystem.

    Args:
        embedder: :class:`~src.embeddings.embedder.Embedder` instance used to
                  convert text → embedding vectors for both ingestion and queries.
        persist_dir: Path to the directory where ChromaDB persists its data
                     (default ``"data/processed/chroma"``).
        collection_name: Name of the ChromaDB collection (default
                         ``"financial_10k"``).
    """

    def __init__(
        self,
        embedder: "Embedder",
        persist_dir: str | Path = "data/processed/chroma",
        collection_name: str = "financial_10k",
    ) -> None:
        self.embedder = embedder
        self.persist_dir = str(persist_dir)
        self.collection_name = collection_name
        self._client: Optional[Any] = None
        self._collection: Optional[Any] = None

    # ------------------------------------------------------------------
    # Lazy properties
    # ------------------------------------------------------------------

    @property
    def client(self) -> Any:
        """Return the ChromaDB PersistentClient, creating it on first access."""
        if self._client is None:
            import chromadb  # noqa: PLC0415

            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            log.info("VectorStore: ChromaDB client initialised at '{}'", self.persist_dir)
        return self._client

    @property
    def collection(self) -> Any:
        """Return the ChromaDB collection, creating it on first access."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            log.info(
                "VectorStore: collection '{}' ready ({} docs)",
                self.collection_name,
                self._collection.count(),
            )
        return self._collection

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, chunks: list[Chunk]) -> None:
        """Embed and store a batch of chunks.

        Uses ``upsert`` so re-indexing the same chunks is safe (idempotent).

        Args:
            chunks: Chunks to embed and store.  Skipped if the list is empty.
        """
        if not chunks:
            return

        texts = [c.text for c in chunks]
        log.info("VectorStore.add: embedding {} chunks …", len(chunks))
        embeddings = self.embedder.embed(texts)

        metadatas = []
        for c in chunks:
            meta: dict = {
                "doc_id": c.doc_id,
                "section": c.section or "",
                "char_start": c.char_start,
                "char_end": c.char_end,
            }
            meta.update(_sanitise_metadata(c.metadata))
            metadatas.append(meta)

        self.collection.upsert(
            ids=[c.id for c in chunks],
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )
        log.info(
            "VectorStore.add: stored {} chunks → collection '{}' now has {} docs",
            len(chunks),
            self.collection_name,
            self.collection.count(),
        )

    def query(
        self,
        query_text: str,
        k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        """Retrieve the *k* most relevant chunks for *query_text*.

        Args:
            query_text: Natural-language query string.
            k: Maximum number of results to return.

        Returns:
            List of ``(Chunk, score)`` pairs ordered by descending relevance.
            ``score`` is cosine similarity in [-1, 1] (1 = identical).
        """
        count = self.collection.count()
        if count == 0:
            log.warning("VectorStore.query: collection is empty")
            return []

        n = min(k, count)
        query_emb = self.embedder.embed_query(query_text)

        results = self.collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=n,
            include=["documents", "distances", "metadatas"],
        )

        output: list[tuple[Chunk, float]] = []
        for chunk_id, text, dist, meta in zip(
            results["ids"][0],
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            chunk = Chunk(
                id=chunk_id,
                text=text,
                doc_id=meta.get("doc_id", ""),
                section=meta.get("section") or None,
                char_start=int(meta.get("char_start", 0)),
                char_end=int(meta.get("char_end", 0)),
                metadata={
                    k: v for k, v in meta.items() if k not in _RESERVED_META_KEYS
                },
            )
            # ChromaDB cosine distance ∈ [0, 2]; similarity = 1 − distance.
            score = 1.0 - float(dist)
            output.append((chunk, score))

        return output

    def count(self) -> int:
        """Return the total number of stored chunks."""
        return self.collection.count()

    def delete_collection(self) -> None:
        """Drop and recreate the collection (useful for re-indexing)."""
        self.client.delete_collection(self.collection_name)
        self._collection = None
        log.warning("VectorStore: collection '{}' deleted", self.collection_name)
