"""Sentence-embedding wrapper with batched encoding and L2 normalisation."""

from typing import Any, Optional

import numpy as np

from src.utils.logging import get_logger

log = get_logger(__name__)


class Embedder:
    """Thin wrapper around a ``SentenceTransformer`` model.

    The underlying model is loaded lazily on the first call to :meth:`embed`
    or :meth:`embed_query`, so constructing an ``Embedder`` is always cheap.

    Args:
        model_name: HuggingFace model identifier (default
                    ``"BAAI/bge-small-en-v1.5"``).
        batch_size: Number of texts encoded in a single forward pass (default 64).
        normalize: If ``True``, L2-normalise every embedding so that cosine
                   similarity equals the inner product (default ``True``).
        device: PyTorch device string (``"cpu"``, ``"cuda"``, …).  ``None``
                lets sentence-transformers pick automatically.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 64,
        normalize: bool = True,
        device: Optional[str] = "cpu",
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device
        self._model: Optional[Any] = None  # lazy init

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model(self) -> Any:
        """Return the SentenceTransformer model, loading it on first access."""
        if self._model is None:
            log.info("Embedder: loading model '{}'", self.model_name)
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            kwargs: dict = {}
            if self.device is not None:
                kwargs["device"] = self.device
            self._model = SentenceTransformer(self.model_name, **kwargs)
            log.info("Embedder: model loaded.")
        return self._model

    @property
    def dimension(self) -> int:
        """Embedding dimension of the loaded model."""
        return self.model.get_sentence_embedding_dimension()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, texts: list[str]) -> np.ndarray:
        """Encode a list of strings into a matrix of embeddings.

        Texts are processed in batches of :attr:`batch_size`.  When
        :attr:`normalize` is ``True`` each row is L2-normalised so that
        ``embeddings @ embeddings.T`` gives cosine similarities.

        Args:
            texts: Non-empty list of strings to encode.

        Returns:
            Float32 ``ndarray`` of shape ``(len(texts), dimension)``.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        batches: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            emb = self.model.encode(
                batch,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            batches.append(emb)

        embeddings = np.vstack(batches).astype(np.float32)

        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            embeddings = embeddings / norms

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a single query string into a 1-D embedding vector.

        Args:
            query: Query string to encode.

        Returns:
            Float32 ``ndarray`` of shape ``(dimension,)``.
        """
        return self.embed([query])[0]
