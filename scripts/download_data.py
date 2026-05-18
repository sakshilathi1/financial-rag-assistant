#!/usr/bin/env python
"""CLI: Download SEC 10-K filings via EDGAR API.

Example
-------
    python scripts/download_data.py \\
        --tickers AAPL,MSFT,GOOGL \\
        --output-dir data/raw \\
        --limit 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.sec_loader import SECLoader

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download SEC 10-K filings.")
    parser.add_argument(
        "--tickers",
        required=True,
        help="Comma-separated list of ticker symbols, e.g. AAPL,MSFT,GOOGL",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Directory to write JSON files (default: data/raw)",
    )
    parser.add_argument(
        "--form-type",
        default="10-K",
        help="SEC form type to download (default: 10-K)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2,
        help="Max filings per company (default: 2)",
    )
    parser.add_argument(
        "--user-agent",
        default=os.getenv(
            "SEC_USER_AGENT", "Financial RAG Assistant contact@example.com"
        ),
        help="User-Agent header sent to SEC EDGAR",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        logger.error("No tickers provided.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = SECLoader(user_agent=args.user_agent)

    logger.info("Downloading %s filings for: %s", args.form_type, ", ".join(tickers))

    for ticker in tqdm(tickers, desc="Tickers"):
        try:
            filings = loader.load_filings(
                [ticker],
                form_type=args.form_type,
                limit_per_company=args.limit,
            )
        except Exception as exc:
            logger.error("Failed for %s: %s", ticker, exc)
            continue

        out_file = output_dir / f"{ticker.lower()}_filings.json"
        with open(out_file, "w", encoding="utf-8") as fh:
            json.dump(filings, fh, ensure_ascii=False, indent=2)

        logger.info("  Saved %d filing(s) → %s", len(filings), out_file)

    logger.info("Done. Files written to %s", output_dir)


if __name__ == "__main__":
    main()
