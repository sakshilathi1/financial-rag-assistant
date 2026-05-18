"""Tests for the FastAPI service — fully mocked, no real LLM/embedding calls."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.pipeline.rag_pipeline import Citation, RAGResponse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_rag_response(**overrides) -> RAGResponse:
    defaults = dict(
        answer="Apple reported revenue of $383B.",
        citations=[
            Citation(
                chunk_id="chunk_001",
                doc_id="AAPL_10K_2023",
                snippet="Total net sales were $383.3 billion.",
                score=0.92,
            )
        ],
        retrieval_latency_ms=12.5,
        generation_latency_ms=340.0,
        total_latency_ms=352.5,
        confidence=0.95,
        mode="hybrid",
        num_chunks_retrieved=5,
        retrieved_chunk_ids=["chunk_001", "chunk_002"],
        context_used="Total net sales were $383.3 billion.",
    )
    defaults.update(overrides)
    return RAGResponse(**defaults)


@pytest.fixture()
def client():
    """Return a TestClient with pipeline fully mocked on app.state."""
    # Import here so patching happens before module-level lifespan runs
    from src.api.main import create_app

    mock_pipeline = MagicMock()
    mock_pipeline.query = AsyncMock(return_value=_make_rag_response())

    mock_llm = MagicMock()
    mock_llm.host = "http://localhost:11434"

    app = create_app()

    # Replace lifespan-created state with mocks before the client opens
    with TestClient(app, raise_server_exceptions=True) as c:
        app.state.pipeline = mock_pipeline
        app.state.llm_client = mock_llm
        yield c, mock_pipeline


# ---------------------------------------------------------------------------
# Route: GET /
# ---------------------------------------------------------------------------

def test_root(client):
    c, _ = client
    resp = c.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["service"] == "financial-rag-assistant"
    assert data["docs"] == "/docs"


# ---------------------------------------------------------------------------
# Route: GET /health
# ---------------------------------------------------------------------------

def test_health_returns_200(client):
    c, _ = client
    resp = c.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


# ---------------------------------------------------------------------------
# Route: GET /ready
# ---------------------------------------------------------------------------

def test_ready_when_ollama_up_and_chroma_populated(client):
    c, mock_pipeline = client

    mock_pipeline.retriever.vector_store.count.return_value = 2873

    with patch("src.api.routes.httpx.AsyncClient") as mock_ac:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_ac.return_value.__aenter__ = AsyncMock(return_value=MagicMock(
            get=AsyncMock(return_value=mock_resp)
        ))
        mock_ac.return_value.__aexit__ = AsyncMock(return_value=False)

        resp = c.get("/ready")

    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


def test_ready_503_when_chroma_empty(client):
    c, mock_pipeline = client

    mock_pipeline.retriever.vector_store.count.return_value = 0

    with patch("src.api.routes.httpx.AsyncClient") as mock_ac:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_ac.return_value.__aenter__ = AsyncMock(return_value=MagicMock(
            get=AsyncMock(return_value=mock_resp)
        ))
        mock_ac.return_value.__aexit__ = AsyncMock(return_value=False)

        resp = c.get("/ready")

    assert resp.status_code == 503
    data = resp.json()
    assert data["status"] == "not ready"
    assert any("empty" in e for e in data["errors"])


def test_ready_503_when_ollama_unreachable(client):
    c, mock_pipeline = client

    mock_pipeline.retriever.vector_store.count.return_value = 2873

    with patch("src.api.routes.httpx.AsyncClient") as mock_ac:
        import httpx as _httpx
        mock_ac.return_value.__aenter__ = AsyncMock(return_value=MagicMock(
            get=AsyncMock(side_effect=_httpx.ConnectError("refused"))
        ))
        mock_ac.return_value.__aexit__ = AsyncMock(return_value=False)

        resp = c.get("/ready")

    assert resp.status_code == 503
    data = resp.json()
    assert any("Ollama" in e for e in data["errors"])


# ---------------------------------------------------------------------------
# Route: POST /ask
# ---------------------------------------------------------------------------

def test_ask_returns_200_with_valid_schema(client):
    c, mock_pipeline = client
    mock_pipeline.query = AsyncMock(return_value=_make_rag_response())

    resp = c.post("/ask", json={"question": "What was Apple revenue in 2023?"})

    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert isinstance(data["citations"], list)
    assert data["citations"][0]["chunk_id"] == "chunk_001"
    assert data["mode"] == "hybrid"
    assert data["num_chunks_retrieved"] == 5
    assert 0.0 <= data["confidence"] <= 1.0


def test_ask_passes_request_fields_to_pipeline(client):
    c, mock_pipeline = client
    mock_pipeline.query = AsyncMock(return_value=_make_rag_response(mode="dense"))

    resp = c.post(
        "/ask",
        json={"question": "What is NVDA revenue?", "top_k": 3, "retrieval_mode": "dense"},
    )

    assert resp.status_code == 200
    mock_pipeline.query.assert_awaited_once_with(
        question="What is NVDA revenue?",
        top_k=3,
        retrieval_mode="dense",
        use_reranker=False,
    )


def test_ask_empty_question_returns_422(client):
    c, _ = client
    resp = c.post("/ask", json={"question": ""})
    assert resp.status_code == 422


def test_ask_short_question_returns_422(client):
    c, _ = client
    resp = c.post("/ask", json={"question": "Hi"})  # len=2, min_length=3
    assert resp.status_code == 422


def test_ask_invalid_retrieval_mode_returns_422(client):
    c, _ = client
    resp = c.post("/ask", json={"question": "What is revenue?", "retrieval_mode": "bm25_only"})
    assert resp.status_code == 422


def test_ask_top_k_out_of_range_returns_422(client):
    c, _ = client
    resp = c.post("/ask", json={"question": "What is revenue?", "top_k": 0})
    assert resp.status_code == 422
    resp2 = c.post("/ask", json={"question": "What is revenue?", "top_k": 21})
    assert resp2.status_code == 422


def test_ask_pipeline_error_returns_500(client):
    c, mock_pipeline = client
    mock_pipeline.query = AsyncMock(side_effect=RuntimeError("Ollama timeout"))

    resp = c.post("/ask", json={"question": "What is revenue?"})

    assert resp.status_code == 500
    assert "Pipeline error" in resp.json()["detail"]
