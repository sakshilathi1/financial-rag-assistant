"""FastAPI application entry point with lifespan startup/shutdown."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import router
from src.utils.config import load_config
from src.utils.logging import get_logger

log = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize shared infrastructure on startup; clean up on shutdown."""
    # Must load .env before load_config so ${OLLAMA_HOST} et al. interpolate.
    load_dotenv(_PROJECT_ROOT / ".env")
    log.info("API startup: loading config and initializing pipeline")

    cfg = load_config(_PROJECT_ROOT / "configs" / "default.yaml")

    from src.embeddings.embedder import Embedder
    from src.generation.llm_client import OllamaClient
    from src.pipeline.rag_pipeline import RAGPipeline
    from src.retrieval.retriever import Retriever
    from src.retrieval.vector_store import VectorStore

    emb_cfg = cfg.get("embeddings", {})
    embedder = Embedder(
        model_name=emb_cfg.get("model_name", "BAAI/bge-small-en-v1.5"),
        batch_size=int(emb_cfg.get("batch_size", 64)),
        normalize=bool(emb_cfg.get("normalize", True)),
    )
    vector_store = VectorStore(
        embedder=embedder,
        persist_dir=cfg["data_paths"]["chroma_dir"],
    )
    retriever = Retriever(vector_store=vector_store, mode="hybrid")
    llm_client = OllamaClient.from_config(cfg)
    pipeline = RAGPipeline(retriever=retriever, llm_client=llm_client)

    app.state.pipeline = pipeline
    app.state.llm_client = llm_client

    log.info("API startup complete")
    yield

    log.info("API shutdown")
    # Nothing to explicitly close for ChromaDB/httpx (connections are per-request)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Financial RAG Assistant",
        description="Retrieval-augmented generation over SEC 10-K filings",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.error("Unhandled exception: {}", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error": str(exc)},
        )

    app.include_router(router)
    return app


app = create_app()
