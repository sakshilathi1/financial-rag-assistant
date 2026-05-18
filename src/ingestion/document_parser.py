"""10-K HTML parser: extracts clean text and key sections from EDGAR filings."""

import re
import warnings
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from src.utils.logging import get_logger

# Suppress the benign warning that triggers when lxml parses iXBRL documents
# (which are technically XML but parse fine as HTML for our text-extraction needs).
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

log = get_logger(__name__)

# Matches an item-number header that begins a NEW line (not an in-text cross-reference).
# Used as the gap-measurement anchor: cross-refs like "(see Item 8)" are mid-sentence
# and do NOT start a line, so they are excluded.
_STANDALONE_ITEM = re.compile(r"^\s*item\s+\d", re.I | re.M)

# Ordered section definitions:
#   (name, strict_start_pattern, fallback_start_pattern, end_pattern)
# strict: requires section title keyword (may fail on iXBRL split-word artifacts).
# fallback: matches by item number only; more permissive.
# end: first standalone item header that closes this section.
_SECTION_DEFS: list[tuple[str, re.Pattern, re.Pattern, Optional[re.Pattern]]] = [
    (
        "business",
        re.compile(r"\bitem\s+1\b[\s.]*business\b", re.I),
        re.compile(r"\bitem\s+1\b(?!\s*a\b)", re.I),     # item 1, not item 1a
        re.compile(r"^\s*item\s+1a\b", re.I | re.M),
    ),
    (
        "risk_factors",
        re.compile(r"\bitem\s+1a\b[\s.]*risk\s+factors?\b", re.I),
        re.compile(r"\bitem\s+1a\b", re.I),
        re.compile(r"^\s*item\s+(?:1b|2)\b", re.I | re.M),
    ),
    (
        "mda",
        re.compile(
            r"\bitem\s+7\b[\s.]*management.{0,40}discussion", re.I | re.S
        ),
        re.compile(r"\bitem\s+7\b(?!\s*a\b)", re.I),     # item 7, not item 7a
        re.compile(r"^\s*item\s+(?:7a|8)\b", re.I | re.M),
    ),
    (
        "financial_statements",
        re.compile(r"\bitem\s+8\b[\s.]*financial\s+statements?\b", re.I),
        re.compile(r"\bitem\s+8\b", re.I),
        re.compile(r"^\s*item\s+9\b", re.I | re.M),
    ),
]

# Max characters extracted per section (guards against runaway extraction)
_MAX_SECTION_CHARS = 150_000

# Form-field checkbox characters that appear on 10-K cover pages
_CHECKBOX_CHARS = re.compile(r"[☒☐☑✓✗]")
# "Indicate by check mark" boilerplate – entire line
_CHECK_MARK_LINE = re.compile(r"^[^\n]*indicate\s+by\s+check\s+mark[^\n]*$", re.I | re.M)
# Item 15+ exhibits section opener
_ITEM_15 = re.compile(r"\bitem\s+15\b", re.I)
# "Item 1. Business" – canonical start of filing content
_ITEM1_BUSINESS = re.compile(r"item\s+1\.\s+business", re.I)
# Bare "Item 1." fallback
_ITEM1_BARE = re.compile(r"item\s+1\.", re.I)


def strip_boilerplate(text: str) -> str:
    """Remove cover-page legalese and structural noise from a 10-K plain-text file.

    Operations applied in order:
    1. Slice from the first ``Item 1. Business`` occurrence to strip the cover
       page.  If that phrase is absent, fall back to the *last* ``Item 1.``
       occurrence (the TOC entry is the first; the real section is the last).
    2. Strip checkbox characters (☒ ☐ ☑ ✓ ✗) used in EDGAR cover forms.
    3. Remove entire lines containing "Indicate by check mark" boilerplate.
    4. Drop everything from ``Item 15.`` onward (exhibits appendix).
    5. Collapse three or more consecutive blank lines into two.
    6. Drop lines that are pure whitespace or contain only a single character.

    Args:
        text: Raw plain text from ``_html_to_text`` or an existing ``.txt`` file.

    Returns:
        Cleaned text string.
    """
    # ── 1. Strip cover page ───────────────────────────────────────────────────
    matches = list(_ITEM1_BUSINESS.finditer(text))
    if matches:
        text = text[matches[0].start():]
    else:
        bare_matches = list(_ITEM1_BARE.finditer(text))
        if bare_matches:
            # Last occurrence is the real section (first is usually TOC)
            text = text[bare_matches[-1].start():]

    # ── 2. Form-field checkbox characters ────────────────────────────────────
    text = _CHECKBOX_CHARS.sub("", text)

    # ── 3. "Indicate by check mark" lines ────────────────────────────────────
    text = _CHECK_MARK_LINE.sub("", text)

    # ── 4. Drop Item 15+ exhibits ─────────────────────────────────────────────
    # Use the LAST occurrence — the TOC lists Item 15 near the top while the
    # actual exhibits/signatures appendix is at the very end.
    # Guard: only cut if the last hit is in the final 30 % of the document.
    # Some bank filings (e.g. JPM) embed financial tables inside their Item 15
    # exhibits section so the match appears at ~15 % — cutting there would lose
    # 85 % of substantive content.
    all_m15 = list(_ITEM_15.finditer(text))
    if all_m15:
        last_m15 = all_m15[-1]
        if last_m15.start() >= 0.70 * len(text):
            text = text[: last_m15.start()]

    # ── 5. Collapse 3+ blank lines → 2 ───────────────────────────────────────
    text = re.sub(r"\n{3,}", "\n\n", text)

    # ── 6. Drop whitespace-only or single-character lines ────────────────────
    lines = text.splitlines()
    lines = [ln for ln in lines if len(ln.strip()) > 1]
    text = "\n".join(lines)

    return text.strip()


def _html_to_text(html_path: Path) -> str:
    """Convert an EDGAR 10-K HTML file to clean plain text.

    Strips scripts, style blocks, and XBRL metadata containers.
    Collapses excessive whitespace while preserving paragraph breaks.

    Args:
        html_path: Path to the raw HTML file.

    Returns:
        Clean plain text string.
    """
    raw_html = html_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw_html, "lxml")

    # Remove non-content tags
    for tag in soup.find_all(["script", "style"]):
        tag.decompose()

    # Strip XBRL header/hidden sections (ix: namespace) – they hold metadata only
    for tag in soup.find_all(True):
        name = getattr(tag, "name", "") or ""
        if name.lower() in ("ix:header", "ix:hidden"):
            tag.decompose()

    text = soup.get_text(separator="\n")

    # Normalise whitespace: collapse runs of spaces/tabs but keep newlines
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Collapse runs of 3+ newlines to a double newline (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text


def _best_section_start(
    text: str,
    start_pat: re.Pattern,
    end_pat: Optional[re.Pattern] = None,
    min_gap: int = 1000,
) -> Optional[int]:
    """Return the character index of the actual section start (not a TOC entry).

    Two-pass strategy:
    1. *Primary*: measure the gap from each match to the next occurrence of
       ``end_pat`` (the opening header of the *following* section).  A match
       with no ``end_pat`` after it cannot be a real section start (it must be
       a cross-reference embedded in later content), so those matches are
       discarded.  Among the survivors, pick the one with the largest gap.
    2. *Fallback*: if no match produced an ``end_pat`` hit (e.g. Item 9 is
       absent in short filings), fall back to the largest gap measured against
       the next standalone item-number line.

    Args:
        text: Full plain-text document.
        start_pat: Compiled regex identifying this section's opening header.
        end_pat: Compiled regex identifying the *next* section's opening header,
                 used as the gap boundary.  If None, the standalone-item
                 fallback is used directly.
        min_gap: Minimum character gap to treat a candidate as genuine content
                 (filters out TOC stubs where the next section follows in < 1 kB).

    Returns:
        Best character index, or None if nothing passes the minimum gap.
    """
    # (gap, start_pos) pairs; split by whether end_pat was found after the match
    with_end: list[tuple[int, int]] = []
    without_end: list[tuple[int, int]] = []

    for m in start_pat.finditer(text):
        after = m.end()
        if end_pat:
            end_m = end_pat.search(text, after)
            if end_m is not None:
                with_end.append((end_m.start() - after, m.start()))
                continue
        # No specific end_pat hit → fall back to next standalone section header
        nxt = _STANDALONE_ITEM.search(text, after)
        gap = (nxt.start() - after) if nxt else (len(text) - after)
        without_end.append((gap, m.start()))

    # Prefer candidates bounded by the explicit end pattern
    pool = with_end if with_end else without_end
    if not pool:
        return None

    best_gap, best_pos = max(pool, key=lambda x: x[0])
    return best_pos if best_gap >= min_gap else None


def _extract_sections(text: str) -> dict[str, str]:
    """Locate and extract the four key 10-K sections from plain text.

    Tries a strict pattern first (requires section title keyword); falls back
    to an item-number-only pattern for iXBRL filings that split words across
    elements (e.g. "B USINESS", "RIS K FACTORS").

    Args:
        text: Full document plain text (from _html_to_text).

    Returns:
        Dict mapping section name → extracted text (may be empty if not found).
    """
    # Locate the start of each section
    found: list[tuple[str, int, Optional[re.Pattern]]] = []
    for name, strict_pat, fallback_pat, end_pat in _SECTION_DEFS:
        pos = _best_section_start(text, strict_pat, end_pat=end_pat)
        if pos is None:
            log.debug("Strict pattern miss for '{}', trying item-number fallback", name)
            pos = _best_section_start(text, fallback_pat, end_pat=end_pat)
        if pos is not None:
            found.append((name, pos, end_pat))
            log.debug("Section '{}' found at char {}", name, pos)
        else:
            log.warning("Section '{}' not found in document", name)

    # Sort by position so we can use the next section as a natural boundary
    found.sort(key=lambda x: x[1])

    sections: dict[str, str] = {}
    for i, (name, start_pos, end_pat) in enumerate(found):
        # Upper bound: start of the next located section
        if i + 1 < len(found):
            candidate_end = found[i + 1][1]
        else:
            candidate_end = len(text)

        # Tighten with the explicit end pattern if it fires earlier
        if end_pat:
            end_m = end_pat.search(text, start_pos + 50)
            if end_m and end_m.start() < candidate_end:
                candidate_end = end_m.start()

        raw = text[start_pos:candidate_end].strip()
        raw = raw[:_MAX_SECTION_CHARS]

        word_count = len(raw.split())
        if word_count < 50:
            log.warning(
                "Section '{}' has only {} words – likely a false positive, skipping",
                name,
                word_count,
            )
            continue

        sections[name] = raw
        log.info("Section '{}': {:,} words", name, word_count)

    return sections


def parse_10k_html(
    html_path: str | Path,
    save_text: bool = True,
) -> dict:
    """Parse a 10-K HTML file into structured sections.

    Writes ``{stem}_10k.txt`` alongside the HTML file when *save_text* is True.

    Args:
        html_path: Path to the raw 10-K HTML file.
        save_text: Whether to save the full clean text as a .txt file.

    Returns:
        Dict with keys:
        - ``html_path`` (str)
        - ``text_path`` (str | None)
        - ``sections`` (dict[str, str])
        - ``sections_found`` (list[str])
        - ``word_count`` (int)
    """
    html_path = Path(html_path)
    log.info("Parsing {}", html_path.name)

    text = _html_to_text(html_path)
    word_count = len(text.split())
    log.info("Extracted {:,} words from {}", word_count, html_path.name)

    text_path: Optional[Path] = None
    if save_text:
        # Derive ticker from filename like "AAPL_10k.html"
        text_path = html_path.parent / html_path.name.replace(".html", ".txt").replace(
            ".htm", ".txt"
        )
        text_path.write_text(text, encoding="utf-8")
        log.info("Saved plain text: {}", text_path)

    sections = _extract_sections(text)

    return {
        "html_path": str(html_path),
        "text_path": str(text_path) if text_path else None,
        "sections": sections,
        "sections_found": list(sections.keys()),
        "word_count": word_count,
    }
