"""LLM-driven Q&A pair generation from 10-K text chunks."""

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from src.chunking.fixed_chunker import FixedChunker
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.generation.llm_client import OllamaClient

log = get_logger(__name__)

_QA_PROMPT = """\
You are building a financial evaluation dataset from SEC 10-K filings.

Given the following text chunk from {company_name}'s ({ticker}) 10-K filing, generate one factual Q&A pair.

CRITICAL REQUIREMENTS — read carefully before generating:
1. The question MUST contain the company name "{company_name}" explicitly (not just a pronoun or "the company").
2. The question must be specific to facts UNIQUE to {company_name} — it must NOT be a generic question that could apply to any company's 10-K.
3. The question must be answerable ONLY from this specific chunk.
4. Ask about a concrete fact: a specific dollar amount, percentage, product name, date, or named risk unique to {company_name}.
5. The answer must be a single concise factual sentence — NOT a paraphrase of the question.
6. If the chunk contains no company-specific facts, respond with null.

BAD examples (too generic — reject these):
- "What was the company's revenue?" → missing company name, could apply anywhere
- "What risks does the company face?" → generic
- "What is the fiscal year?" → generic

GOOD examples:
- "What was {company_name}'s total net revenue in fiscal 2024?"
- "Which {company_name} product line generated the most revenue?"
- "What specific risk did {company_name} identify regarding [named factor]?"

Chunk [{chunk_id}]:
{chunk_text}

Respond with JSON only (no markdown, no extra text):
{{"question": "...", "answer": "..."}}

Or if no good company-specific question is possible:
null"""


@dataclass
class QAPair:
    """A ground-truth Q&A pair for retrieval and generation evaluation."""

    pair_id: str
    question: str
    expected_answer: str
    ground_truth_chunk_id: str
    source_doc: str
    section: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "QAPair":
        return cls(**d)


async def generate_qa_for_ticker(
    ticker: str,
    text: str,
    llm_client: "OllamaClient",
    num_pairs: int = 10,
    chunk_size: int = 1000,
    overlap: int = 200,
    seed: int = 42,
) -> list[QAPair]:
    """Generate *num_pairs* Q&A pairs from *text* for *ticker*.

    Chunks the text with :class:`~src.chunking.fixed_chunker.FixedChunker`,
    randomly samples chunks, and asks the LLM to produce one Q&A per chunk.
    Chunks that yield ``null`` from the LLM are skipped; additional chunks
    are sampled until *num_pairs* is reached or candidates are exhausted.

    Args:
        ticker: Uppercase ticker symbol (used as doc_id and for logging).
        text: Full stripped 10-K text.
        llm_client: Async LLM client for generation.
        num_pairs: Target number of Q&A pairs.
        chunk_size: FixedChunker chunk size in characters.
        overlap: FixedChunker overlap in characters.
        seed: Random seed for reproducible chunk sampling.

    Returns:
        List of :class:`QAPair` objects (may be fewer than *num_pairs* if
        the document doesn't yield enough useful chunks).
    """
    chunker = FixedChunker(
        chunk_size=chunk_size,
        overlap=overlap,
        min_chunk_size=200,
    )
    chunks = chunker.chunk(text, doc_id=ticker)
    if not chunks:
        log.warning("qa_generator: no chunks for ticker '{}'", ticker)
        return []

    rng = random.Random(seed)
    candidates = chunks.copy()
    rng.shuffle(candidates)

    pairs: list[QAPair] = []
    pair_idx = 0

    for chunk in candidates:
        if len(pairs) >= num_pairs:
            break

        # Map ticker → human-readable company name for the prompt.
        _COMPANY_NAMES = {
            "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet",
            "NVDA": "NVIDIA", "JPM": "JPMorgan Chase",
        }
        company_name = _COMPANY_NAMES.get(ticker, ticker)
        prompt = _QA_PROMPT.format(
            ticker=ticker,
            company_name=company_name,
            chunk_id=chunk.id,
            chunk_text=chunk.text[:1200],
        )

        try:
            raw = await llm_client.generate(prompt, max_tokens=256, temperature=0.2)
        except Exception as exc:
            log.warning("qa_generator: LLM error for chunk {}: {}", chunk.id, exc)
            continue

        raw = raw.strip()
        if not raw or raw.lower() == "null":
            log.debug("qa_generator: skipping chunk {} (null response)", chunk.id)
            continue

        try:
            data = json.loads(raw)
            question = str(data.get("question", "")).strip()
            answer = str(data.get("answer", "")).strip()
        except (json.JSONDecodeError, AttributeError):
            # Try to extract JSON from a response that has surrounding text
            try:
                start = raw.index("{")
                end = raw.rindex("}") + 1
                data = json.loads(raw[start:end])
                question = str(data.get("question", "")).strip()
                answer = str(data.get("answer", "")).strip()
            except (ValueError, json.JSONDecodeError):
                log.debug("qa_generator: JSON parse failed for chunk {}", chunk.id)
                continue

        if not question or not answer or len(question) < 10:
            log.debug("qa_generator: skipping empty Q&A for chunk {}", chunk.id)
            continue

        # Reject questions that don't mention the company name.
        if company_name.lower() not in question.lower() and ticker.lower() not in question.lower():
            log.debug(
                "qa_generator: rejecting generic question (no company name) for chunk {}",
                chunk.id,
            )
            continue

        pair_idx += 1
        pairs.append(
            QAPair(
                pair_id=f"{ticker}_{pair_idx:04d}",
                question=question,
                expected_answer=answer,
                ground_truth_chunk_id=chunk.id,
                source_doc=ticker,
                section=chunk.section,
            )
        )
        log.debug("qa_generator: generated pair {} for {}", pair_idx, ticker)

    log.info("qa_generator: {} pairs generated for {}", len(pairs), ticker)
    return pairs


def save_ticker_qa(
    ticker: str,
    pairs: list[QAPair],
    eval_dir: str | Path,
) -> Path:
    """Persist *pairs* for *ticker* to ``qa_{ticker}.json``.

    Args:
        ticker: Ticker symbol (used as filename key).
        pairs: List of :class:`QAPair` objects to save.
        eval_dir: Directory to write the file into.

    Returns:
        Path of the written file.
    """
    path = Path(eval_dir) / f"qa_{ticker}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump([p.to_dict() for p in pairs], fh, indent=2)
    log.info("qa_generator: saved {} pairs to '{}'", len(pairs), path)
    return path


def load_ticker_qa(
    ticker: str,
    eval_dir: str | Path,
) -> Optional[list[QAPair]]:
    """Load persisted Q&A pairs for *ticker*, or ``None`` if not found.

    Args:
        ticker: Ticker symbol.
        eval_dir: Directory to search for ``qa_{ticker}.json``.

    Returns:
        List of :class:`QAPair` objects, or ``None`` if the file doesn't exist.
    """
    path = Path(eval_dir) / f"qa_{ticker}.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    pairs = [QAPair.from_dict(d) for d in data]
    log.info("qa_generator: loaded {} pairs for {} from '{}'", len(pairs), ticker, path)
    return pairs
