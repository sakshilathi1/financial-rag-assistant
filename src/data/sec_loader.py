"""SEC EDGAR filing downloader.

Downloads 10-K (and other) filings via the public EDGAR API.
SEC requires a descriptive User-Agent header on every request.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_FILING_INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_path}/{accession_number}-index.json"
_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"


class SECLoader:
    """Download and parse SEC 10-K filings via the EDGAR API."""

    def __init__(
        self,
        user_agent: str = "Financial RAG Assistant contact@example.com",
        rate_limit_sleep: float = 0.1,
    ) -> None:
        self.user_agent = user_agent
        self.rate_limit_sleep = rate_limit_sleep
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Host": "data.sec.gov",
            }
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, url: str, timeout: int = 30, host_override: str | None = None) -> requests.Response:
        headers: dict[str, str] = {}
        if host_override:
            headers["Host"] = host_override
        try:
            resp = self.session.get(url, timeout=timeout, headers=headers)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error("GET %s failed: %s", url, exc)
            raise
        finally:
            time.sleep(self.rate_limit_sleep)
        return resp

    @staticmethod
    def _pad_cik(cik: str | int) -> str:
        """Zero-pad CIK to 10 digits as required by EDGAR."""
        return str(int(cik)).zfill(10)

    @staticmethod
    def _normalise_accession(accession_number: str) -> str:
        """Return accession number without dashes."""
        return accession_number.replace("-", "")

    # ------------------------------------------------------------------
    # Ticker → CIK resolution
    # ------------------------------------------------------------------

    def resolve_cik(self, ticker: str) -> str:
        """Resolve a ticker symbol to a CIK via the EDGAR company search."""
        url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&forms=10-K"
        try:
            resp = self._get(url, host_override="efts.sec.gov")
            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])
            if hits:
                entity_id = hits[0].get("_source", {}).get("entity_id", "")
                if entity_id:
                    return self._pad_cik(entity_id)
        except Exception as exc:
            logger.warning("Search-index resolution failed for %s: %s", ticker, exc)

        # Fallback: company tickers JSON published by SEC
        try:
            tickers_url = "https://www.sec.gov/files/company_tickers.json"
            resp = self._get(tickers_url, host_override="www.sec.gov")
            mapping = resp.json()
            ticker_upper = ticker.upper()
            for entry in mapping.values():
                if entry.get("ticker", "").upper() == ticker_upper:
                    return self._pad_cik(entry["cik_str"])
        except Exception as exc:
            logger.warning("company_tickers.json lookup failed: %s", exc)

        raise ValueError(f"Could not resolve ticker '{ticker}' to a CIK.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_company_filings(
        self,
        ticker_or_cik: str,
        form_type: str = "10-K",
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Return metadata for recent filings of *form_type* for a company.

        Parameters
        ----------
        ticker_or_cik:
            Ticker symbol (e.g. ``"AAPL"``) or numeric CIK string.
        form_type:
            SEC form type, e.g. ``"10-K"``.
        limit:
            Maximum number of filings to return.

        Returns
        -------
        list[dict]
            Each dict has keys: ``accession_number``, ``cik``, ``form_type``,
            ``filing_date``, ``period_of_report``.
        """
        # Resolve to padded CIK
        if ticker_or_cik.isdigit():
            cik = self._pad_cik(ticker_or_cik)
        else:
            cik = self.resolve_cik(ticker_or_cik)

        url = _SUBMISSIONS_URL.format(cik=cik)
        try:
            resp = self._get(url)
            data = resp.json()
        except Exception as exc:
            logger.error("Failed to fetch submissions for CIK %s: %s", cik, exc)
            return []

        filings_data = data.get("filings", {}).get("recent", {})
        form_types = filings_data.get("form", [])
        accessions = filings_data.get("accessionNumber", [])
        filing_dates = filings_data.get("filingDate", [])
        periods = filings_data.get("reportDate", [])

        results: list[dict[str, Any]] = []
        for i, ft in enumerate(form_types):
            if ft == form_type:
                results.append(
                    {
                        "accession_number": accessions[i],
                        "cik": cik,
                        "form_type": ft,
                        "filing_date": filing_dates[i] if i < len(filing_dates) else "",
                        "period_of_report": periods[i] if i < len(periods) else "",
                    }
                )
                if len(results) >= limit:
                    break

        return results

    def download_filing(self, accession_number: str, cik: str) -> str:
        """Download the primary document of a filing and return cleaned text.

        Parameters
        ----------
        accession_number:
            Filing accession number (with or without dashes).
        cik:
            Numeric CIK (will be stripped of leading zeros for URL path).

        Returns
        -------
        str
            Cleaned plain-text content of the primary filing document.
        """
        cik_int = str(int(cik))
        acc_clean = self._normalise_accession(accession_number)
        acc_dashed = f"{acc_clean[:10]}-{acc_clean[10:12]}-{acc_clean[12:]}"
        acc_path = acc_clean

        index_url = (
            f"{_ARCHIVES_BASE}/{cik_int}/{acc_path}/{acc_dashed}-index.json"
        )
        try:
            resp = self._get(index_url, host_override="www.sec.gov")
            index_data = resp.json()
        except Exception as exc:
            logger.warning("Could not fetch filing index %s: %s", index_url, exc)
            return ""

        # Find the primary document (10-K htm/txt)
        primary_doc: str | None = None
        for item in index_data.get("directory", {}).get("item", []):
            name: str = item.get("name", "")
            doc_type: str = item.get("type", "")
            if doc_type in ("10-K", "10-K/A") and name.endswith((".htm", ".html", ".txt")):
                primary_doc = name
                break

        if primary_doc is None:
            # Fallback: pick first .htm file
            for item in index_data.get("directory", {}).get("item", []):
                name = item.get("name", "")
                if name.endswith((".htm", ".html")) and not name.startswith("R"):
                    primary_doc = name
                    break

        if primary_doc is None:
            logger.warning("No primary document found for accession %s", accession_number)
            return ""

        doc_url = f"{_ARCHIVES_BASE}/{cik_int}/{acc_path}/{primary_doc}"
        try:
            resp = self._get(doc_url, host_override="www.sec.gov")
            raw_html = resp.text
        except Exception as exc:
            logger.warning("Failed to download document %s: %s", doc_url, exc)
            return ""

        return self._parse_html(raw_html)

    def load_filings(
        self,
        tickers: list[str],
        form_type: str = "10-K",
        limit_per_company: int = 2,
    ) -> list[dict[str, Any]]:
        """Download and parse filings for multiple companies.

        Parameters
        ----------
        tickers:
            List of ticker symbols.
        form_type:
            SEC form type.
        limit_per_company:
            Number of filings to download per company.

        Returns
        -------
        list[dict]
            Each dict has keys: ``company``, ``cik``, ``period``, ``text``,
            ``filing_url``, ``accession_number``.
        """
        results: list[dict[str, Any]] = []
        for ticker in tickers:
            logger.info("Processing ticker: %s", ticker)
            try:
                filings_meta = self.get_company_filings(
                    ticker, form_type=form_type, limit=limit_per_company
                )
            except Exception as exc:
                logger.error("Failed to get filings for %s: %s", ticker, exc)
                continue

            for meta in filings_meta:
                cik = meta["cik"]
                acc = meta["accession_number"]
                period = meta["period_of_report"]
                cik_int = str(int(cik))
                acc_clean = self._normalise_accession(acc)
                filing_url = f"{_ARCHIVES_BASE}/{cik_int}/{acc_clean}/"

                logger.info("  Downloading %s (%s)…", ticker, period)
                text = self.download_filing(acc, cik)
                if not text:
                    logger.warning("  Empty text for %s %s — skipping.", ticker, period)
                    continue

                results.append(
                    {
                        "company": ticker.upper(),
                        "cik": cik,
                        "period": period,
                        "text": text,
                        "filing_url": filing_url,
                        "accession_number": acc,
                    }
                )

        return results

    # ------------------------------------------------------------------
    # HTML parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_html(html: str) -> str:
        """Strip HTML tags and return clean plain text."""
        soup = BeautifulSoup(html, "lxml")

        # Remove script and style blocks
        for tag in soup(["script", "style", "head", "meta"]):
            tag.decompose()

        text = soup.get_text(separator=" ")
        # Collapse whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
