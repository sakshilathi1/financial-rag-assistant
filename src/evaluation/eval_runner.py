"""Resumable evaluation runner: processes Q&A pairs and appends results to JSONL."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from src.evaluation.generation_metrics import (
    citation_accuracy,
    faithfulness_score,
    relevance_score,
)
from src.evaluation.qa_generator import QAPair
from src.evaluation.retrieval_metrics import hit_at_k, mrr, recall_at_k
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.generation.llm_client import OllamaClient
    from src.pipeline.rag_pipeline import RAGPipeline

log = get_logger(__name__)


class EvalRunner:
    """Run pipeline queries over a Q&A set and record per-pair metrics to JSONL.

    Results are appended incrementally so the run survives interruption.
    On restart, already-processed pair IDs are skipped automatically.

    Args:
        pipeline: Fully configured :class:`~src.pipeline.rag_pipeline.RAGPipeline`.
        llm_client: Async LLM client used for LLM-as-judge scoring.
        output_path: Path to the JSONL file where results are written.
    """

    def __init__(
        self,
        pipeline: "RAGPipeline",
        llm_client: "OllamaClient",
        output_path: Path,
    ) -> None:
        self.pipeline = pipeline
        self.llm_client = llm_client
        self.output_path = output_path
        self._processed_ids: set[str] = self._load_processed_ids()

    # ------------------------------------------------------------------
    # Resumability
    # ------------------------------------------------------------------

    def _load_processed_ids(self) -> set[str]:
        """Return the set of pair_ids already present in the output JSONL."""
        if not self.output_path.exists():
            return set()
        ids: set[str] = set()
        with self.output_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    ids.add(record["pair_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        log.info("EvalRunner: {} pairs already processed in '{}'", len(ids), self.output_path)
        return ids

    def _append_result(self, record: dict) -> None:
        """Append a single result record as a JSONL line."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------

    async def run(
        self,
        qa_pairs: list[QAPair],
        config_name: str,
        retrieval_mode: str = "dense",
        top_k: int = 5,
        max_questions: Optional[int] = None,
    ) -> dict:
        """Evaluate *pipeline* against *qa_pairs* and return aggregate metrics.

        Args:
            qa_pairs: List of ground-truth Q&A pairs.
            config_name: Human-readable name for this run (stored in each record).
            retrieval_mode: ``"dense"`` or ``"hybrid"``.
            top_k: Number of chunks to retrieve per question.
            max_questions: Optional cap on the number of pairs to process.

        Returns:
            Dict of aggregate metric averages.
        """
        pending = [
            p for p in qa_pairs if p.pair_id not in self._processed_ids
        ]
        if max_questions is not None:
            pending = pending[:max_questions]

        log.info(
            "EvalRunner: {} pairs to process (config={}, mode={})",
            len(pending),
            config_name,
            retrieval_mode,
        )

        totals: dict[str, float] = {
            "hit_at_1": 0.0, "hit_at_5": 0.0, "hit_at_10": 0.0,
            "mrr": 0.0, "recall_at_10": 0.0,
            "faithfulness": 0.0, "relevance": 0.0, "citation_accuracy": 0.0,
        }
        processed = 0

        for pair in pending:
            log.debug("EvalRunner: processing pair '{}'", pair.pair_id)

            # ── Pipeline query ─────────────────────────────────────────────
            try:
                response = await self.pipeline.query(
                    question=pair.question,
                    top_k=top_k,
                    retrieval_mode=retrieval_mode,
                )
            except Exception as exc:
                log.warning("EvalRunner: pipeline error for '{}': {}", pair.pair_id, exc)
                continue

            retrieved_ids = response.retrieved_chunk_ids
            cited_ids = [c.chunk_id for c in response.citations]

            # ── Retrieval metrics ──────────────────────────────────────────
            h1 = float(hit_at_k(retrieved_ids, pair.ground_truth_chunk_id, 1))
            h5 = float(hit_at_k(retrieved_ids, pair.ground_truth_chunk_id, 5))
            h10 = float(hit_at_k(retrieved_ids, pair.ground_truth_chunk_id, 10))
            mrr_score = mrr(retrieved_ids, pair.ground_truth_chunk_id)
            r10 = recall_at_k(retrieved_ids, {pair.ground_truth_chunk_id}, 10)

            # ── Generation metrics ─────────────────────────────────────────
            faith = await faithfulness_score(
                response.answer, response.context_used, self.llm_client
            )
            rel = await relevance_score(
                pair.question, response.answer, self.llm_client
            )
            cite_acc = citation_accuracy(cited_ids, retrieved_ids)

            # ── Record ─────────────────────────────────────────────────────
            record = {
                "pair_id": pair.pair_id,
                "config": config_name,
                "question": pair.question,
                "expected_answer": pair.expected_answer,
                "ground_truth_chunk_id": pair.ground_truth_chunk_id,
                "answer": response.answer,
                "citations": cited_ids,
                "confidence": response.confidence,
                "retrieval_latency_ms": response.retrieval_latency_ms,
                "generation_latency_ms": response.generation_latency_ms,
                "total_latency_ms": response.total_latency_ms,
                "hit_at_1": h1,
                "hit_at_5": h5,
                "hit_at_10": h10,
                "mrr": mrr_score,
                "recall_at_10": r10,
                "faithfulness": faith,
                "relevance": rel,
                "citation_accuracy": cite_acc,
            }
            self._append_result(record)
            self._processed_ids.add(pair.pair_id)

            totals["hit_at_1"] += h1
            totals["hit_at_5"] += h5
            totals["hit_at_10"] += h10
            totals["mrr"] += mrr_score
            totals["recall_at_10"] += r10
            totals["faithfulness"] += faith
            totals["relevance"] += rel
            totals["citation_accuracy"] += cite_acc

            processed += 1

            if processed % 5 == 0:
                avgs = {k: v / processed for k, v in totals.items()}
                log.info(
                    "EvalRunner [{}/{}] hit@5={:.2f} mrr={:.2f} faith={:.2f} rel={:.2f}",
                    processed,
                    len(pending),
                    avgs["hit_at_5"],
                    avgs["mrr"],
                    avgs["faithfulness"],
                    avgs["relevance"],
                )

        if processed == 0:
            return {k: 0.0 for k in totals}

        return {k: round(v / processed, 4) for k, v in totals.items()}
