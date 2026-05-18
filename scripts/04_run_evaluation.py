"""Run RAG evaluation across configurations and generate a comparison report.

Usage
-----
# Smoke test — 2 questions, dense only:
    python scripts/04_run_evaluation.py --max-questions 2 --configs fixed_dense

# Full eval — both configs:
    python scripts/04_run_evaluation.py --configs fixed_dense,fixed_hybrid
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

from src.embeddings.embedder import Embedder
from src.evaluation.eval_runner import EvalRunner
from src.evaluation.qa_generator import QAPair
from src.generation.llm_client import OllamaClient
from src.pipeline.rag_pipeline import RAGPipeline
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore
from src.utils.config import load_config
from src.utils.logging import get_logger

log = get_logger(__name__)

# Supported config names → retrieval mode
_CONFIGS: dict[str, str] = {
    "fixed_dense": "dense",
    "fixed_hybrid": "hybrid",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG evaluation pipeline.")
    parser.add_argument(
        "--configs",
        type=str,
        default="fixed_dense,fixed_hybrid",
        help="Comma-separated config names (default: fixed_dense,fixed_hybrid).",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Cap the number of questions per config (useful for smoke tests).",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Skip evaluation; read existing results_*.jsonl files and regenerate the report.",
    )
    return parser.parse_args()


_METRIC_KEYS = [
    "hit_at_1", "hit_at_5", "hit_at_10", "mrr", "recall_at_10",
    "faithfulness", "relevance", "citation_accuracy",
]


def _aggregate_jsonl(jsonl_path: Path) -> dict:
    """Read a results JSONL and return averaged metrics."""
    totals: dict[str, float] = {k: 0.0 for k in _METRIC_KEYS}
    count = 0
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            for k in _METRIC_KEYS:
                totals[k] += float(rec.get(k, 0.0))
            count += 1
    if count == 0:
        return {k: 0.0 for k in _METRIC_KEYS}
    return {k: round(v / count, 4) for k, v in totals.items()}


def _load_qa_pairs(eval_dir: Path) -> list[QAPair]:
    """Load Q&A pairs from qa_pairs.json, raising if the file doesn't exist."""
    qa_path = eval_dir / "qa_pairs.json"
    if not qa_path.exists():
        raise FileNotFoundError(
            f"Q&A pairs file not found: {qa_path}\n"
            "Run scripts/03_generate_eval_set.py first."
        )
    with qa_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    pairs = [QAPair.from_dict(d) for d in data]
    log.info("Loaded {} Q&A pairs from '{}'", len(pairs), qa_path)
    return pairs


def _generate_report(
    results_by_config: dict[str, dict],
    eval_dir: Path,
) -> Path:
    """Write a Markdown comparison table to data/eval/comparison_report.md."""
    report_path = eval_dir / "comparison_report.md"

    metric_cols = [
        ("Hit@1", "hit_at_1"),
        ("Hit@5", "hit_at_5"),
        ("Hit@10", "hit_at_10"),
        ("MRR", "mrr"),
        ("Recall@10", "recall_at_10"),
        ("Faithfulness", "faithfulness"),
        ("Relevance", "relevance"),
        ("Cite Acc.", "citation_accuracy"),
    ]

    header = "| Config | " + " | ".join(h for h, _ in metric_cols) + " |"
    sep = "| --- | " + " | ".join("---" for _ in metric_cols) + " |"
    rows = [header, sep]

    for config_name, metrics in sorted(results_by_config.items()):
        values = " | ".join(
            f"{metrics.get(key, 0.0):.3f}" for _, key in metric_cols
        )
        rows.append(f"| {config_name} | {values} |")

    body = "\n".join(
        [
            "# RAG Evaluation Results",
            "",
            "## Aggregate Metrics",
            "",
        ]
        + rows
        + [
            "",
            "## Notes",
            "- Hit@k: fraction of questions where ground-truth chunk is in top-k",
            "- MRR: Mean Reciprocal Rank",
            "- Faithfulness / Relevance: LLM-as-judge binary scores",
            "- Cite Acc.: fraction of cited chunk IDs present in retrieved set",
        ]
    )

    report_path.write_text(body, encoding="utf-8")
    log.info("Report written to '{}'", report_path)
    return report_path


async def main() -> None:
    args = _parse_args()
    cfg = load_config(_PROJECT_ROOT / "configs" / "default.yaml")

    eval_dir = Path(cfg["data_paths"]["eval_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)

    # ── Report-only mode ──────────────────────────────────────────────────
    if args.report_only:
        results_by_config: dict[str, dict] = {}
        for name in sorted(_CONFIGS):
            jsonl_path = eval_dir / f"results_{name}.jsonl"
            if jsonl_path.exists() and jsonl_path.stat().st_size > 0:
                results_by_config[name] = _aggregate_jsonl(jsonl_path)
                log.info("Loaded {} results from '{}'", name, jsonl_path)
            else:
                log.warning("No results file found for config '{}' — skipping.", name)
        if not results_by_config:
            log.error("No results files found — nothing to report.")
            return
        report_path = _generate_report(results_by_config, eval_dir)
        print("\n" + "=" * 56)
        print(f"{'Report Generated':^56}")
        print("=" * 56)
        for config_name, metrics in sorted(results_by_config.items()):
            print(f"\n{config_name}:")
            for key, val in metrics.items():
                print(f"  {key:<22} {val:.4f}")
        print(f"\nReport: {report_path}")
        print("=" * 56)
        return

    chroma_dir = cfg["data_paths"]["chroma_dir"]

    selected_configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    for name in selected_configs:
        if name not in _CONFIGS:
            raise ValueError(f"Unknown config '{name}'. Choose from: {list(_CONFIGS)}")

    # ── Load Q&A pairs ────────────────────────────────────────────────────
    qa_pairs = _load_qa_pairs(eval_dir)
    if not qa_pairs:
        log.error("No Q&A pairs found — aborting.")
        return

    # ── Shared infrastructure ─────────────────────────────────────────────
    emb_cfg = cfg.get("embeddings", {})
    embedder = Embedder(
        model_name=emb_cfg.get("model_name", "BAAI/bge-small-en-v1.5"),
        batch_size=int(emb_cfg.get("batch_size", 64)),
        normalize=bool(emb_cfg.get("normalize", True)),
    )
    vector_store = VectorStore(embedder=embedder, persist_dir=chroma_dir)
    llm_client = OllamaClient.from_config(cfg)

    results_by_config: dict[str, dict] = {}

    # ── Per-config evaluation ─────────────────────────────────────────────
    for config_name in selected_configs:
        retrieval_mode = _CONFIGS[config_name]
        output_path = eval_dir / f"results_{config_name}.jsonl"

        log.info("=== Config: {} (mode={}) ===", config_name, retrieval_mode)

        retriever = Retriever(vector_store=vector_store, mode=retrieval_mode)
        pipeline = RAGPipeline(retriever=retriever, llm_client=llm_client)

        runner = EvalRunner(
            pipeline=pipeline,
            llm_client=llm_client,
            output_path=output_path,
        )

        metrics = await runner.run(
            qa_pairs=qa_pairs,
            config_name=config_name,
            retrieval_mode=retrieval_mode,
            top_k=int(cfg.get("retrieval", {}).get("top_k", 5)),
            max_questions=args.max_questions,
        )

        results_by_config[config_name] = metrics

        log.info(
            "Config {} done — hit@5={:.3f} mrr={:.3f} faith={:.3f} rel={:.3f}",
            config_name,
            metrics.get("hit_at_5", 0),
            metrics.get("mrr", 0),
            metrics.get("faithfulness", 0),
            metrics.get("relevance", 0),
        )

    # ── Report ────────────────────────────────────────────────────────────
    report_path = _generate_report(results_by_config, eval_dir)

    print("\n" + "=" * 56)
    print(f"{'Evaluation Complete':^56}")
    print("=" * 56)
    for config_name, metrics in sorted(results_by_config.items()):
        print(f"\n{config_name}:")
        for key, val in metrics.items():
            print(f"  {key:<22} {val:.4f}")
    print(f"\nReport: {report_path}")
    print("=" * 56)


if __name__ == "__main__":
    asyncio.run(main())
