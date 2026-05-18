"""Generate Q&A evaluation pairs from 10-K text files.

Usage
-----
# All tickers (10 pairs each = 50 total), skip existing:
    python scripts/03_generate_eval_set.py

# Single ticker for smoke testing:
    python scripts/03_generate_eval_set.py --tickers AAPL

# Specify number of pairs per ticker:
    python scripts/03_generate_eval_set.py --num-pairs 5
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

from src.evaluation.qa_generator import (
    QAPair,
    generate_qa_for_ticker,
    load_ticker_qa,
    save_ticker_qa,
)
from src.generation.llm_client import OllamaClient
from src.utils.config import load_config
from src.utils.logging import get_logger

log = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Q&A evaluation pairs from 10-K text files."
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated tickers (default: all from config).",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=10,
        help="Q&A pairs per ticker (default: 10).",
    )
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    cfg = load_config(_PROJECT_ROOT / "configs" / "default.yaml")

    raw_dir = Path(cfg["data_paths"]["raw_dir"])
    eval_dir = Path(cfg["data_paths"]["eval_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)

    tickers: list[str] = (
        [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        if args.tickers
        else [str(t) for t in cfg.get("tickers", [])]
    )

    llm_client = OllamaClient.from_config(cfg)
    log.info("03_generate_eval_set: tickers={}, num_pairs={}", tickers, args.num_pairs)

    all_pairs: list[QAPair] = []

    for ticker in tickers:
        # ── Resumability: skip if already generated ───────────────────────
        existing = load_ticker_qa(ticker, eval_dir)
        if existing is not None:
            log.info("Skipping {} — {} pairs already on disk", ticker, len(existing))
            all_pairs.extend(existing)
            continue

        txt_path = raw_dir / f"{ticker}_10k.txt"
        if not txt_path.exists():
            log.warning("{}: text file not found, skipping", ticker)
            continue

        text = txt_path.read_text(encoding="utf-8")
        log.info("{}: generating {} Q&A pairs …", ticker, args.num_pairs)

        pairs = await generate_qa_for_ticker(
            ticker=ticker,
            text=text,
            llm_client=llm_client,
            num_pairs=args.num_pairs,
        )

        if pairs:
            save_ticker_qa(ticker, pairs, eval_dir)
            all_pairs.extend(pairs)
        else:
            log.warning("{}: no pairs generated", ticker)

    # ── Merge into qa_pairs.json ──────────────────────────────────────────
    merged_path = eval_dir / "qa_pairs.json"
    with merged_path.open("w", encoding="utf-8") as fh:
        json.dump([p.to_dict() for p in all_pairs], fh, indent=2)

    print(f"\nTotal Q&A pairs: {len(all_pairs)}")
    print(f"Saved to: {merged_path}")
    for ticker in tickers:
        count = sum(1 for p in all_pairs if p.source_doc == ticker)
        print(f"  {ticker}: {count} pairs")


if __name__ == "__main__":
    asyncio.run(main())
