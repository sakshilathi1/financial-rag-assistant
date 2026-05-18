"""SEC EDGAR REST API client for downloading 10-K filings."""

import json
import os
import time
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

from src.utils.logging import get_logger

load_dotenv()

log = get_logger(__name__)

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_INDEX_URL = (
    "https://www.sec.gov/Archives/edgar/data"
    "/{int_cik}/{accession_nodash}/{accession_nodash}-index.json"
)
_DOC_URL = (
    "https://www.sec.gov/Archives/edgar/data"
    "/{int_cik}/{accession_nodash}/{filename}"
)

# SEC fair-access policy: stay well under 10 req/s
_REQUEST_DELAY_S = 0.15


def _headers() -> dict[str, str]:
    """Build SEC-compliant request headers from environment."""
    agent = os.environ.get("SEC_USER_AGENT", "").strip()
    if not agent:
        raise EnvironmentError(
            "SEC_USER_AGENT is not set. "
            "Add it to your .env file (e.g. 'Name email@example.com')."
        )
    return {"User-Agent": agent, "Accept-Encoding": "gzip, deflate"}


def _get(url: str, stream: bool = False, timeout: int = 60) -> requests.Response:
    """Rate-limited GET with SEC-compliant headers."""
    time.sleep(_REQUEST_DELAY_S)
    log.debug("GET {}", url)
    resp = requests.get(url, headers=_headers(), timeout=timeout, stream=stream)
    resp.raise_for_status()
    return resp


def get_cik(ticker: str) -> str:
    """Resolve a ticker symbol to its 10-digit zero-padded SEC CIK.

    Args:
        ticker: Stock ticker (e.g. "AAPL"). Case-insensitive.

    Returns:
        10-digit CIK string, zero-padded to 10 chars.

    Raises:
        ValueError: Ticker not found in SEC's company list.
        EnvironmentError: SEC_USER_AGENT env var not set.
    """
    resp = _get(_TICKERS_URL)
    data = resp.json()
    ticker_upper = ticker.upper()
    for entry in data.values():
        if entry.get("ticker", "").upper() == ticker_upper:
            return str(entry["cik_str"]).zfill(10)
    raise ValueError(f"CIK not found for ticker: {ticker!r}")


def get_latest_10k_metadata(cik: str, year: Optional[int] = None) -> dict:
    """Fetch metadata for the most recent 10-K filing for a given CIK.

    Args:
        cik: 10-digit zero-padded CIK string.
        year: Optional 4-digit year filter (matches filing date prefix).

    Returns:
        Dict with keys: accession_number, filing_date, cik, primary_document.

    Raises:
        ValueError: No 10-K found matching the criteria.
    """
    url = _SUBMISSIONS_URL.format(cik=cik)
    resp = _get(url)
    data = resp.json()

    filings = data["filings"]["recent"]
    forms: list[str] = filings.get("form", [])
    dates: list[str] = filings.get("filingDate", [])
    accessions: list[str] = filings.get("accessionNumber", [])
    primary_docs: list[str] = filings.get("primaryDocument", [""] * len(forms))

    for i, form in enumerate(forms):
        if form != "10-K":
            continue
        if year is not None and not dates[i].startswith(str(year)):
            continue
        return {
            "accession_number": accessions[i],
            "filing_date": dates[i],
            "cik": cik,
            "primary_document": primary_docs[i] if i < len(primary_docs) else "",
        }

    raise ValueError(f"No 10-K found for CIK={cik!r}, year={year}")


def _resolve_primary_doc(
    cik: str, accession_number: str, hint: str = ""
) -> str:
    """Find the primary HTML filename from the filing index, falling back to hint.

    Args:
        cik: 10-digit zero-padded CIK.
        accession_number: Formatted accession (e.g. "0000320193-23-000077").
        hint: primaryDocument value from submissions API (may be empty).

    Returns:
        Filename of the primary HTML document.

    Raises:
        ValueError: No HTML document located.
    """
    # Fast path: hint is already a valid HTML filename
    if hint and hint.lower().endswith((".htm", ".html")):
        return hint

    int_cik = int(cik)
    accession_nodash = accession_number.replace("-", "")
    index_url = _INDEX_URL.format(
        int_cik=int_cik, accession_nodash=accession_nodash
    )
    try:
        resp = _get(index_url)
        index_data = resp.json()
        items = index_data.get("directory", {}).get("item", [])
        # Prefer the document whose type is "10-K" and is HTML
        for doc in items:
            if doc.get("type") == "10-K" and doc["name"].lower().endswith(
                (".htm", ".html")
            ):
                return doc["name"]
        # Second pass: any HTML file
        for doc in items:
            if doc["name"].lower().endswith((".htm", ".html")):
                return doc["name"]
    except Exception as exc:
        log.warning("Could not parse filing index ({}): {}", index_url, exc)

    if hint:
        return hint
    raise ValueError(
        f"No primary HTML document found for accession {accession_number}"
    )


def download_10k(
    ticker: str,
    year: Optional[int] = None,
    raw_dir: str | Path = "data/raw",
) -> dict:
    """Download the most recent 10-K HTML for a ticker and persist to disk.

    Saves two files:
    - ``data/raw/{TICKER}_10k.html``  – raw filing HTML
    - ``data/raw/{TICKER}_metadata.json`` – CIK, accession, filing date

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL").
        year: Optional year filter; None → most recent available.
        raw_dir: Directory for saved files.

    Returns:
        Metadata dict (keys: accession_number, filing_date, cik,
        primary_document, html_path, html_size_bytes).

    Raises:
        ValueError: CIK or 10-K filing not found.
        requests.HTTPError: SEC API returned a non-2xx response.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    ticker_upper = ticker.upper()

    log.info("[{}] Resolving CIK ...", ticker_upper)
    cik = get_cik(ticker_upper)
    log.info("[{}] CIK = {}", ticker_upper, cik)

    metadata = get_latest_10k_metadata(cik, year=year)
    accession_number = metadata["accession_number"]
    accession_nodash = accession_number.replace("-", "")
    log.info(
        "[{}] Found 10-K: accession={}, filed={}",
        ticker_upper,
        accession_number,
        metadata["filing_date"],
    )

    primary_doc = _resolve_primary_doc(
        cik, accession_number, hint=metadata["primary_document"]
    )
    log.info("[{}] Primary document: {}", ticker_upper, primary_doc)

    int_cik = int(cik)
    doc_url = _DOC_URL.format(
        int_cik=int_cik,
        accession_nodash=accession_nodash,
        filename=primary_doc,
    )
    log.info("[{}] Downloading {} ...", ticker_upper, doc_url)

    resp = _get(doc_url, stream=True, timeout=120)
    chunks = []
    for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB chunks
        chunks.append(chunk)
    html_content = b"".join(chunks).decode("utf-8", errors="replace")

    html_path = raw_dir / f"{ticker_upper}_10k.html"
    html_path.write_text(html_content, encoding="utf-8")
    size_mb = html_path.stat().st_size / 1e6
    log.info("[{}] Saved HTML: {} ({:.1f} MB)", ticker_upper, html_path, size_mb)

    metadata["primary_document"] = primary_doc
    metadata["html_path"] = str(html_path)
    metadata["html_size_bytes"] = html_path.stat().st_size

    meta_path = raw_dir / f"{ticker_upper}_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log.info("[{}] Saved metadata: {}", ticker_upper, meta_path)

    return metadata
