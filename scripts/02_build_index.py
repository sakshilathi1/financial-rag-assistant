"""Chunk, embed, and index all 10-K documents into ChromaDB.

Usage examples
--------------
# Index all tickers with fixed chunker (default):
    python scripts/02_build_index.py

# Semantic chunker, only AAPL and MSFT:
    python scripts/02_build_index.py --chunker semantic --tickers AAPL,MSFT

# Wipe the existing index first, then re-index:
    python scripts/02_build_index.py --reset
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path when running as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

from src.chunking.base import Chunk
from src.chunking.fixed_chunker import FixedChunker
from src.chunking.semantic_chunker import SemanticChunker
from src.embeddings.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.utils.config import load_config
from src.utils.logging import get_logger

log = get_logger(__name__)


def _dir_size_mb(path: str) -> float:
    """Return total size of all files under *path* in megabytes."""
    total = 0
    for p in Path(path).rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024 * 1024)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ChromaDB index from 10-K text files.")
    parser.add_argument(
        "--chunker",
        choices=["fixed", "semantic"],
        default="fixed",
        help="Chunking strategy (default: fixed)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing ChromaDB collection before indexing.",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated list of tickers to index (default: all from config).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(_PROJECT_ROOT / "configs" / "default.yaml")

    raw_dir = Path(cfg["data_paths"]["raw_dir"])
    chroma_dir = cfg["data_paths"]["chroma_dir"]

    # ── Ticker selection ───────────────────────────────────────────────────────
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = [str(t) for t in cfg.get("tickers", [])]

    log.info("Indexing tickers: {}", tickers)

    # ── Embedder + VectorStore ─────────────────────────────────────────────────
    emb_cfg = cfg.get("embeddings", {})
    embedder = Embedder(
        model_name=emb_cfg.get("model_name", "BAAI/bge-small-en-v1.5"),
        batch_size=int(emb_cfg.get("batch_size", 64)),
        normalize=bool(emb_cfg.get("normalize", True)),
    )

    vector_store = VectorStore(
        embedder=embedder,
        persist_dir=chroma_dir,
    )

    if args.reset:
        log.warning("--reset: deleting existing collection '{}'", vector_store.collection_name)
        vector_store.delete_collection()

    # ── Chunker ────────────────────────────────────────────────────────────────
    chunk_cfg = cfg.get("chunking", {})
    if args.chunker == "semantic":
        chunker = SemanticChunker(
            similarity_threshold=float(chunk_cfg.get("similarity_threshold", 0.7)),
            min_chunk_size=int(chunk_cfg.get("min_chunk_size", 100)),
        )
        log.info("Using SemanticChunker (threshold={})", chunk_cfg.get("similarity_threshold", 0.7))
    else:
        chunker = FixedChunker(
            chunk_size=int(chunk_cfg.get("chunk_size", 1000)),
            overlap=int(chunk_cfg.get("overlap", 200)),
            min_chunk_size=int(chunk_cfg.get("min_chunk_size", 100)),
        )
        log.info(
            "Using FixedChunker (chunk_size={}, overlap={})",
            chunk_cfg.get("chunk_size", 1000),
            chunk_cfg.get("overlap", 200),
        )

    # ── Per-document chunking ──────────────────────────────────────────────────
    start_time = time.perf_counter()
    results: list[dict] = []
    all_chunks: list[Chunk] = []

    for ticker in tickers:
        txt_path = raw_dir / f"{ticker}_10k.txt"
        if not txt_path.exists():
            log.warning("No text file found for {}: {} — skipping", ticker, txt_path)
            results.append({"ticker": ticker, "chunks": 0, "status": "MISSING"})
            continue

        text = txt_path.read_text(encoding="utf-8")
        word_count = len(text.split())
        log.info("{}: {} words — chunking …", ticker, f"{word_count:,}")

        chunks = chunker.chunk(text, doc_id=ticker)
        log.info("{}: {} chunks produced", ticker, len(chunks))

        all_chunks.extend(chunks)
        results.append({"ticker": ticker, "chunks": len(chunks), "status": "OK"})

    # ── Batch index ───────────────────────────────────────────────────────────
    if all_chunks:
        log.info("Embedding and indexing {} total chunks …", len(all_chunks))
        vector_store.add(all_chunks)
    else:
        log.warning("No chunks to index.")

    elapsed = time.perf_counter() - start_time

    # ── Report ─────────────────────────────────────────────────────────────────
    total_chunks = sum(r["chunks"] for r in results)
    chroma_mb = _dir_size_mb(chroma_dir) if Path(chroma_dir).exists() else 0.0

    print("\n" + "=" * 52)
    print(f"{'Indexing report':^52}")
    print("=" * 52)
    print(f"{'Ticker':<10} {'Chunks':>8}  {'Status'}")
    print("-" * 52)
    for r in results:
        print(f"{r['ticker']:<10} {r['chunks']:>8}  {r['status']}")
    print("-" * 52)
    print(f"{'TOTAL':<10} {total_chunks:>8}")
    print("=" * 52)
    print(f"Chunker      : {args.chunker}")
    print(f"Runtime      : {elapsed:.1f}s")
    print(f"ChromaDB dir : {chroma_dir}  ({chroma_mb:.1f} MB)")
    print(f"Collection   : {vector_store.collection_name}  ({vector_store.count()} docs)")
    print("=" * 52)


if __name__ == "__main__":
    main()
