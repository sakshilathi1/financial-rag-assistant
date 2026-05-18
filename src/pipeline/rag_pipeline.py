"""End-to-end RAG pipeline."""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Wrap a retriever and a generator into a single query interface.

    Accepts any retriever that exposes a ``retrieve(query, top_k)`` method
    (dense, sparse, or hybrid) and any generator that exposes a
    ``generate(query, retrieved_chunks, top_k_context)`` method.
    """

    def __init__(
        self,
        retriever: Any,
        generator: Any,
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        retriever:
            A retriever instance with a ``retrieve`` method.
        generator:
            An :class:`~src.generation.generator.AnswerGenerator` instance.
        config:
            Optional config dict (``generation`` sub-key is used if present).
        """
        self.retriever = retriever
        self.generator = generator
        self.config = config or {}
        self._top_k_context: int = (
            self.config.get("generation", {}).get("top_k_context", 5)
        )

    # ------------------------------------------------------------------
    # Single query
    # ------------------------------------------------------------------

    def query(self, question: str) -> dict[str, Any]:
        """Run a single RAG query end-to-end.

        Parameters
        ----------
        question:
            User question string.

        Returns
        -------
        dict
            Keys: ``question``, ``answer``, ``retrieved_chunks``,
            ``retrieval_time``, ``generation_time``.
        """
        # Retrieval
        t0 = time.perf_counter()
        retrieved_chunks = self.retriever.retrieve(question)
        retrieval_time = time.perf_counter() - t0

        # Generation
        t1 = time.perf_counter()
        answer = self.generator.generate(
            question, retrieved_chunks, top_k_context=self._top_k_context
        )
        generation_time = time.perf_counter() - t1

        logger.debug(
            "Query finished: retrieval=%.3fs, generation=%.3fs", retrieval_time, generation_time
        )

        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
        }

    # ------------------------------------------------------------------
    # Batch query
    # ------------------------------------------------------------------

    def batch_query(self, questions: list[str]) -> list[dict[str, Any]]:
        """Run multiple queries sequentially.

        Parameters
        ----------
        questions:
            List of question strings.

        Returns
        -------
        list[dict]
            List of result dicts (same structure as :meth:`query`).
        """
        results: list[dict[str, Any]] = []
        for i, question in enumerate(questions, start=1):
            logger.info("Processing query %d/%d: %s", i, len(questions), question[:80])
            results.append(self.query(question))
        return results
