"""Shared pytest fixtures for the financial-rag-assistant test suite."""

from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture(scope="session")
def mock_embedder() -> MagicMock:
    """Session-scoped mock embedding model.

    Returns a :class:`~unittest.mock.MagicMock` whose ``encode`` method returns
    deterministic :class:`numpy.ndarray` arrays of shape ``(N, 384)`` — the
    same dimensionality as ``BAAI/bge-small-en-v1.5``.

    Using ``np.random.seed(42)`` inside the side-effect guarantees identical
    outputs on every call regardless of input, making tests fully reproducible
    without loading any real ML model.
    """
    mock = MagicMock()

    def _encode(texts: list[str], **kwargs) -> np.ndarray:
        np.random.seed(42)
        return np.random.randn(len(texts), 384).astype(np.float32)

    mock.encode.side_effect = _encode
    return mock
