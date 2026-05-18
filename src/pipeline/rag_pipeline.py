"""End-to-end RAG pipeline: retrieve → (rerank) → generate → parse citations."""

import json
import time
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field

from src.chunking.base import Chunk
from src.generation.prompts import build_citation_prompt, format_context
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.generation.llm_client import OllamaClient
    from src.retrieval.reranker import CrossEncoderReranker
    from src.retrieval.retriever import Retriever

log = get_logger(__name__)


class Citation(BaseModel):
    """A single source passage cited in a RAG answer."""

    chunk_id: str
    doc_id: str
    snippet: str = Field(max_length=300)
    score: float


class RAGResponse(BaseModel):
    """Complete response from a single RAG pipeline query."""

    answer: str
    citations: list[Citation]
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    confidence: float = Field(ge=0.0, le=1.0)
    mode: str
    num_chunks_retrieved: int
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    context_used: str = ""


class RAGPipeline:
    """Orchestrates retrieval, optional reranking, and LLM generation.

    Args:
        retriever: Populated :class:`~src.retrieval.retriever.Retriever`.
        llm_client: Async :class:`~src.generation.llm_client.OllamaClient`.
        reranker: Optional :class:`~src.retrieval.reranker.CrossEncoderReranker`.
                  Only used when ``use_reranker=True`` is passed to
                  :meth:`query`.
    """

    def __init__(
        self,
        retriever: "Retriever",
        llm_client: "OllamaClient",
        reranker: Optional["CrossEncoderReranker"] = None,
    ) -> None:
        self.retriever = retriever
        self.llm_client = llm_client
        self.reranker = reranker

    async def query(
        self,
        question: str,
        top_k: int = 5,
        retrieval_mode: str = "hybrid",
        use_reranker: bool = False,
    ) -> RAGResponse:
        """Run the full RAG pipeline for *question*.

        Args:
            question: Natural-language question to answer.
            top_k: Number of chunks to retrieve (and keep after reranking).
            retrieval_mode: ``"dense"`` or ``"hybrid"``.
            use_reranker: Whether to apply cross-encoder reranking.

        Returns:
            Populated :class:`RAGResponse` with answer, citations, and timing.
        """
        pipeline_start = time.perf_counter()

        # ── Retrieval ─────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        chunks_with_scores = self.retriever.retrieve(
            question, top_k=top_k, mode=retrieval_mode
        )
        retrieval_ms = (time.perf_counter() - t0) * 1000.0

        log.info(
            "RAGPipeline: retrieved {} chunks in {:.1f}ms (mode={})",
            len(chunks_with_scores),
            retrieval_ms,
            retrieval_mode,
        )

        # ── Optional reranking ────────────────────────────────────────────────
        if use_reranker and self.reranker is not None:
            chunks_with_scores = self.reranker.rerank(
                question, chunks_with_scores, top_k=top_k
            )
            log.debug("RAGPipeline: reranked to {} chunks", len(chunks_with_scores))

        chunks: list[Chunk] = [c for c, _ in chunks_with_scores]
        chunk_score_map: dict[str, float] = {c.id: s for c, s in chunks_with_scores}

        # ── Prompt construction ───────────────────────────────────────────────
        context_str = format_context(chunks)
        prompt = build_citation_prompt(question, chunks)

        # ── Generation ───────────────────────────────────────────────────────
        t1 = time.perf_counter()
        raw_output = await self.llm_client.generate(prompt)
        generation_ms = (time.perf_counter() - t1) * 1000.0

        log.info(
            "RAGPipeline: generation complete in {:.1f}ms ({} chars)",
            generation_ms,
            len(raw_output),
        )

        # ── JSON parsing ──────────────────────────────────────────────────────
        answer, confidence, cited_ids = self._parse_llm_output(raw_output)

        # ── Citation matching ─────────────────────────────────────────────────
        chunk_map: dict[str, Chunk] = {c.id: c for c in chunks}
        citations: list[Citation] = []
        for cid in cited_ids:
            if cid not in chunk_map:
                log.debug("RAGPipeline: cited chunk '{}' not in retrieved set", cid)
                continue
            chunk = chunk_map[cid]
            citations.append(
                Citation(
                    chunk_id=cid,
                    doc_id=chunk.doc_id,
                    snippet=chunk.text[:300],
                    score=chunk_score_map.get(cid, 0.0),
                )
            )

        total_ms = (time.perf_counter() - pipeline_start) * 1000.0

        log.info(
            "RAGPipeline: total {:.1f}ms | retrieval {:.1f}ms | generation {:.1f}ms | "
            "{} citations | confidence {:.2f}",
            total_ms,
            retrieval_ms,
            generation_ms,
            len(citations),
            confidence,
        )

        return RAGResponse(
            answer=answer,
            citations=citations,
            retrieval_latency_ms=round(retrieval_ms, 2),
            generation_latency_ms=round(generation_ms, 2),
            total_latency_ms=round(total_ms, 2),
            confidence=confidence,
            mode=retrieval_mode,
            num_chunks_retrieved=len(chunks),
            retrieved_chunk_ids=[c.id for c in chunks],
            context_used=context_str,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_llm_output(
        self, raw: str
    ) -> tuple[str, float, list[str]]:
        """Parse the LLM JSON output into (answer, confidence, cited_ids).

        Returns a degraded triple on any parse failure instead of crashing.
        """
        try:
            data = json.loads(raw)
            answer: str = str(data.get("answer", ""))
            confidence: float = float(data.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
            cited_ids: list[str] = [str(c) for c in data.get("citations", [])]
            return answer, confidence, cited_ids
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            log.warning("RAGPipeline: JSON parse failed ({}): {}", type(exc).__name__, exc)
            truncated = raw[:200]
            return (
                f"JSON parse failure - raw output: {truncated}",
                0.0,
                [],
            )
