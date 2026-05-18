"""Pydantic v2 request/response schemas for the RAG API."""

from typing import Annotated

from pydantic import BaseModel, Field, field_validator

_VALID_MODES = {"dense", "hybrid"}


class AskRequest(BaseModel):
    question: Annotated[str, Field(min_length=3)]
    top_k: Annotated[int, Field(default=5, ge=1, le=20)] = 5
    retrieval_mode: str = "hybrid"
    use_reranker: bool = False

    @field_validator("retrieval_mode")
    @classmethod
    def _validate_mode(cls, v: str) -> str:
        if v not in _VALID_MODES:
            raise ValueError(f"retrieval_mode must be one of {sorted(_VALID_MODES)}, got '{v}'")
        return v


class CitationOut(BaseModel):
    chunk_id: str
    doc_id: str
    snippet: str
    score: float


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    confidence: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    mode: str
    num_chunks_retrieved: int


class HealthResponse(BaseModel):
    status: str
    version: str
