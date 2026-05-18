"""API route definitions."""

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from src.api.schemas import AskRequest, AskResponse, CitationOut, HealthResponse
from src.pipeline.rag_pipeline import RAGResponse
from src.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter()

_VERSION = "0.1.0"


@router.get("/", response_class=JSONResponse)
async def root() -> dict:
    return {"service": "financial-rag-assistant", "docs": "/docs"}


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version=_VERSION)


@router.get("/ready")
async def ready(request: Request) -> JSONResponse:
    pipeline = request.app.state.pipeline
    llm_client = request.app.state.llm_client

    errors: list[str] = []

    # Check ChromaDB has documents
    try:
        count = pipeline.retriever.vector_store.count()
        if count == 0:
            errors.append("ChromaDB collection is empty")
    except Exception as exc:
        errors.append(f"ChromaDB unavailable: {exc}")

    # Check Ollama reachable
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(llm_client.host + "/api/tags")
            if resp.status_code != 200:
                errors.append(f"Ollama returned HTTP {resp.status_code}")
    except Exception as exc:
        errors.append(f"Ollama unreachable: {exc}")

    if errors:
        return JSONResponse(status_code=503, content={"status": "not ready", "errors": errors})
    return JSONResponse(status_code=200, content={"status": "ready"})


@router.post("/ask", response_model=AskResponse)
async def ask(request: Request, body: AskRequest) -> AskResponse:
    pipeline = request.app.state.pipeline

    try:
        result: RAGResponse = await pipeline.query(
            question=body.question,
            top_k=body.top_k,
            retrieval_mode=body.retrieval_mode,
            use_reranker=body.use_reranker,
        )
    except Exception as exc:
        log.error("ask: pipeline error: {}", exc)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

    citations = [
        CitationOut(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            snippet=c.snippet,
            score=c.score,
        )
        for c in result.citations
    ]

    return AskResponse(
        answer=result.answer,
        citations=citations,
        confidence=result.confidence,
        retrieval_latency_ms=result.retrieval_latency_ms,
        generation_latency_ms=result.generation_latency_ms,
        total_latency_ms=result.total_latency_ms,
        mode=result.mode,
        num_chunks_retrieved=result.num_chunks_retrieved,
    )
