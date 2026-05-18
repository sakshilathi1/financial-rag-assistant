#!/usr/bin/env python
"""CLI: Build dense + sparse retrieval indices from raw filing JSON files.

Example
-------
    python scripts/build_index.py \\
        --input-dir data/raw \\
        --output-dir data/index \\
        --model-name BAAI/bge-small-en-v1.5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessor import TextPreprocessor
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    if config_path.exists():
        with open(config_path) as fh:
            return yaml.safe_load(fh) or {}
    return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build dense + sparse retrieval indices from raw filings."
    )
    parser.add_argument(
        "--input-dir",
        default="data/raw",
        help="Directory containing *_filings.json files (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/index",
        help="Directory to write index files (default: data/index)",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Sentence-transformer model name (overrides config.yaml)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(Path(args.config))
    retrieval_cfg = config.get("retrieval", {})
    data_cfg = config.get("data", {})

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all filing JSON files
    filing_files = sorted(input_dir.glob("*_filings.json"))
    if not filing_files:
        logger.error("No *_filings.json files found in %s", input_dir)
        sys.exit(1)

    filings: list[dict] = []
    for fp in filing_files:
        with open(fp, encoding="utf-8") as fh:
            batch = json.load(fh)
        filings.extend(batch)
        logger.info("Loaded %d filings from %s", len(batch), fp.name)

    # Preprocess
    preprocessor = TextPreprocessor(
        chunk_size=data_cfg.get("chunk_size", 512),
        chunk_overlap=data_cfg.get("chunk_overlap", 64),
        min_chunk_length=data_cfg.get("min_chunk_length", 100),
    )
    logger.info("Chunking %d filings…", len(filings))
    chunks = preprocessor.process_filings(filings)
    logger.info("Produced %d chunks total.", len(chunks))

    if not chunks:
        logger.error("No chunks produced — check input data.")
        sys.exit(1)

    # Dense index
    dense_cfg = retrieval_cfg.get("dense", {})
    model_name = args.model_name or dense_cfg.get("model_name", "BAAI/bge-small-en-v1.5")
    dense = DenseRetriever(
        model_name=model_name,
        top_k=dense_cfg.get("top_k", 10),
        batch_size=dense_cfg.get("batch_size", 32),
    )
    logger.info("Building dense index with model=%s…", model_name)
    dense.build_index(chunks)
    dense.save_index(output_dir)

    # Sparse index
    sparse_cfg = retrieval_cfg.get("sparse", {})
    sparse = SparseRetriever(
        top_k=sparse_cfg.get("top_k", 10),
        b=sparse_cfg.get("b", 0.75),
        k1=sparse_cfg.get("k1", 1.5),
    )
    logger.info("Building sparse BM25 index…")
    sparse.build_index(chunks)
    sparse.save_index(output_dir)

    logger.info("Indices saved to %s", output_dir)


if __name__ == "__main__":
    main()
