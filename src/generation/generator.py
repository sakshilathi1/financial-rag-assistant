"""LLM answer generation with OpenAI API and template fallback."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a financial analyst assistant.  Answer the user's question "
    "based solely on the provided context from SEC 10-K filings.  "
    "Be concise, accurate, and cite the source company and filing period "
    "when possible.  If the context does not contain enough information, "
    "say so explicitly."
)


class AnswerGenerator:
    """Generate answers from retrieved chunks using an LLM.

    Tries the OpenAI API first (if ``OPENAI_API_KEY`` is set in the
    environment).  Falls back to a simple extractive template when the
    key is absent or the API call fails.
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._openai_client: Any = None
        self._init_openai()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_openai(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            logger.info("OPENAI_API_KEY not set — will use template-based fallback.")
            return
        try:
            from openai import OpenAI  # type: ignore[import-untyped]
            self._openai_client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialised (model=%s).", self.model)
        except ImportError:
            logger.warning("openai package not installed — using template fallback.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        query: str,
        retrieved_chunks: list[dict[str, Any]],
        top_k_context: int = 5,
    ) -> str:
        """Generate an answer for *query* given *retrieved_chunks*.

        Parameters
        ----------
        query:
            User question.
        retrieved_chunks:
            Ranked list of ``{chunk, score, rank}`` dicts from a retriever.
        top_k_context:
            Maximum number of chunks to include in the prompt context.

        Returns
        -------
        str
            Generated answer string.
        """
        context_chunks = retrieved_chunks[:top_k_context]
        context_text = self._build_context(context_chunks)

        if self._openai_client is not None:
            try:
                return self._call_openai(query, context_text)
            except Exception as exc:
                logger.warning("OpenAI call failed (%s) — using template fallback.", exc)

        return self._template_answer(query, context_chunks)

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context(chunks: list[dict[str, Any]]) -> str:
        """Format retrieved chunks into a numbered context block."""
        parts: list[str] = []
        for i, result in enumerate(chunks, start=1):
            chunk = result["chunk"]
            company = chunk.get("company", "Unknown")
            period = chunk.get("period", "Unknown")
            chunk_id = chunk.get("chunk_id", f"chunk_{i}")
            text = chunk.get("text", "")
            parts.append(
                f"[{i}] Source: {company} ({period}) | ID: {chunk_id}\n{text}"
            )
        return "\n\n".join(parts)

    def _call_openai(self, query: str, context: str) -> str:
        """Call the OpenAI chat completions API."""
        user_message = (
            f"Context from SEC 10-K filings:\n\n{context}\n\n"
            f"Question: {query}"
        )
        response = self._openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def _template_answer(query: str, chunks: list[dict[str, Any]]) -> str:
        """Extractive template answer when OpenAI is unavailable."""
        if not chunks:
            return "No relevant information found in the provided filings."

        best = chunks[0]["chunk"]
        company = best.get("company", "Unknown")
        period = best.get("period", "Unknown")
        text_snippet = best.get("text", "")[:500]

        sources: list[str] = []
        for result in chunks:
            c = result["chunk"]
            src = f"{c.get('company', '?')} ({c.get('period', '?')})"
            if src not in sources:
                sources.append(src)

        source_line = "; ".join(sources)
        return (
            f"Based on the retrieved SEC 10-K filings ({source_line}), "
            f"the most relevant passage states:\n\n\"{text_snippet}\"\n\n"
            f"(This is an extractive summary.  For a more detailed answer, "
            f"configure an OpenAI API key.)"
        )
