"""Download and parse SEC 10-K filings for target tickers.

Usage:
    python scripts/01_download_data.py

Downloads the most recent 10-K for each ticker in configs/default.yaml,
parses key sections, and prints a summary table.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.ingestion.document_parser import parse_10k_html
from src.ingestion.sec_downloader import download_10k
from src.utils.config import load_config
from src.utils.logging import get_logger

log = get_logger(__name__)

_COL = "{:<6}  {:<12}  {:<22}  {:>10}"


def _divider() -> str:
    return "-" * 56


def main() -> None:
    """Entry point: download, parse, and summarise 10-K filings."""
    cfg = load_config("configs/default.yaml")
    tickers: list[str] = cfg.get("tickers", ["AAPL", "MSFT", "GOOGL", "NVDA", "JPM"])
    raw_dir = Path(cfg["data_paths"]["raw_dir"])

    log.info("Starting 10-K download for {} tickers: {}", len(tickers), tickers)

    results: list[dict] = []

    for ticker in tickers:
        log.info("=" * 50)
        log.info("Processing: {}", ticker)
        try:
            metadata = download_10k(ticker, raw_dir=raw_dir)
            parse_result = parse_10k_html(metadata["html_path"], save_text=True)
            results.append(
                {
                    "ticker": ticker,
                    "status": "OK",
                    "filing_date": metadata["filing_date"],
                    "sections_found": parse_result["sections_found"],
                    "word_count": parse_result["word_count"],
                    "html_size_mb": metadata["html_size_bytes"] / 1e6,
                }
            )
        except Exception as exc:
            log.error("FAILED {}: {}", ticker, exc)
            results.append(
                {
                    "ticker": ticker,
                    "status": f"FAILED: {exc}",
                    "filing_date": "N/A",
                    "sections_found": [],
                    "word_count": 0,
                    "html_size_mb": 0.0,
                }
            )

    # ── Summary table ──────────────────────────────────────────────
    print("\n")
    print(_divider())
    print("  10-K Download & Parse Summary")
    print(_divider())
    print(_COL.format("Ticker", "Filed", "Sections Found", "Words"))
    print(_divider())
    for r in results:
        sections_str = (
            ",".join(r["sections_found"]) if r["sections_found"] else r["status"]
        )
        print(
            _COL.format(
                r["ticker"],
                r["filing_date"],
                sections_str[:22],
                f"{r['word_count']:,}",
            )
        )
    print(_divider())

    successes = sum(1 for r in results if r["status"] == "OK")
    log.info("Done. {}/{} tickers succeeded.", successes, len(tickers))

    if successes == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
