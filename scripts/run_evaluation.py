#!/usr/bin/env python
"""CLI: Compare dense vs hybrid retrieval on synthetic QA pairs.

Example
-------
    python scripts/run_evaluation.py \\
        --index-dir data/index \\
        --output-dir results/ \\
        --n-queries 20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.evaluator import RAGBenchmark, RetrievalEvaluator
from src.generation.generator import AnswerGenerator
from src.pipeline.rag_pipeline import RAGPipeline
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
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
        description="Evaluate dense vs hybrid retrieval on synthetic QA pairs."
    )
    parser.add_argument(
        "--index-dir",
        default="data/index",
        help="Directory containing dense + sparse indices (default: data/index)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to write evaluation results (default: results/)",
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=20,
        help="Number of synthetic QA pairs to generate (default: 20)",
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
    eval_cfg = config.get("evaluation", {})
    gen_cfg = config.get("generation", {})

    index_dir = Path(args.index_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    k_values: list[int] = eval_cfg.get("k_values", [1, 3, 5, 10])

    # --- Load dense retriever ---
    dense_cfg = retrieval_cfg.get("dense", {})
    model_name = dense_cfg.get("model_name", "BAAI/bge-small-en-v1.5")
    dense = DenseRetriever(
        model_name=model_name,
        top_k=dense_cfg.get("top_k", 10),
        batch_size=dense_cfg.get("batch_size", 32),
    )
    logger.info("Loading dense index from %s…", index_dir)
    dense.load_index(index_dir)

    # --- Load sparse retriever ---
    sparse_cfg = retrieval_cfg.get("sparse", {})
    sparse = SparseRetriever(
        top_k=sparse_cfg.get("top_k", 10),
        b=sparse_cfg.get("b", 0.75),
        k1=sparse_cfg.get("k1", 1.5),
    )
    logger.info("Loading sparse index from %s…", index_dir)
    sparse.load_index(index_dir)

    # --- Build hybrid retriever ---
    hybrid_cfg = retrieval_cfg.get("hybrid", {})
    hybrid = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        top_k=hybrid_cfg.get("top_k", 10),
        rrf_k=hybrid_cfg.get("rrf_k", 60),
    )

    # --- Generator ---
    generator = AnswerGenerator(
        model=gen_cfg.get("model", "gpt-3.5-turbo"),
        max_tokens=gen_cfg.get("max_tokens", 512),
        temperature=gen_cfg.get("temperature", 0.0),
    )
    top_k_context = gen_cfg.get("top_k_context", 5)

    # --- Pipelines ---
    dense_pipeline = RAGPipeline(
        retriever=dense,
        generator=generator,
        config={"generation": {"top_k_context": top_k_context}},
    )
    hybrid_pipeline = RAGPipeline(
        retriever=hybrid,
        generator=generator,
        config={"generation": {"top_k_context": top_k_context}},
    )

    # --- Synthetic QA pairs ---
    benchmark = RAGBenchmark()
    chunks = dense.chunks
    logger.info("Generating %d synthetic QA pairs from %d chunks…", args.n_queries, len(chunks))
    qa_pairs = benchmark.create_qa_pairs_from_filings(chunks, n=args.n_queries)
    logger.info("Created %d QA pairs.", len(qa_pairs))

    # --- Run full benchmark ---
    logger.info("Running full benchmark…")
    report = benchmark.run_full_benchmark(
        dense_pipeline=dense_pipeline,
        hybrid_pipeline=hybrid_pipeline,
        qa_pairs=qa_pairs,
        k_values=k_values,
    )

    # --- Save JSON report ---
    json_out = output_dir / "evaluation_report.json"
    with open(json_out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    logger.info("JSON report saved → %s", json_out)

    # --- Save CSV comparison ---
    rows: list[dict] = []
    for retriever_name in ("dense", "hybrid"):
        m = report["retrieval"][retriever_name]
        for metric in ("recall", "precision", "ndcg"):
            for k, val in m.get(metric, {}).items():
                rows.append(
                    {
                        "retriever": retriever_name,
                        "metric": f"{metric}@{k}",
                        "value": val,
                    }
                )
        rows.append({"retriever": retriever_name, "metric": "mrr", "value": m.get("mrr", 0)})
        gen_m = report["generation"][retriever_name]
        rows.append(
            {
                "retriever": retriever_name,
                "metric": "rouge_l_f1",
                "value": gen_m.get("rouge_l_f1", 0),
            }
        )

    df = pd.DataFrame(rows)
    csv_out = output_dir / "evaluation_report.csv"
    df.to_csv(csv_out, index=False)
    logger.info("CSV report saved → %s", csv_out)

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
