"""Generation quality metrics: LLM-as-judge and programmatic scoring."""

from typing import TYPE_CHECKING

from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.generation.llm_client import OllamaClient

log = get_logger(__name__)

_FAITHFULNESS_PROMPT = """\
You are evaluating an AI assistant's answer for factual grounding.

Context passages:
{context}

Answer to evaluate:
{answer}

Does every factual claim in the answer appear in the context passages above?
Respond with a single digit: 1 if yes, 0 if no. No other text."""

_RELEVANCE_PROMPT = """\
You are evaluating whether an AI assistant's answer addresses the question asked.

Question: {question}
Answer: {answer}

Does the answer directly address the question?
Respond with a single digit: 1 if yes, 0 if no. No other text."""


def _parse_binary(raw: str) -> float:
    """Extract 0 or 1 from an LLM response that may contain surrounding text."""
    for ch in raw.strip():
        if ch in ("0", "1"):
            return float(ch)
    log.warning("generation_metrics: could not parse binary from '{}'", raw[:50])
    return 0.0


async def faithfulness_score(
    answer: str,
    context: str,
    llm_client: "OllamaClient",
) -> float:
    """Judge whether every claim in *answer* is grounded in *context*.

    Uses the LLM as a binary judge (1 = fully grounded, 0 = not grounded).
    Returns 0.0 on LLM errors.

    Args:
        answer: Generated answer string.
        context: Formatted context string passed to the RAG pipeline.
        llm_client: Async LLM client for judging.

    Returns:
        ``1.0`` or ``0.0``.
    """
    if not answer or not context:
        return 0.0

    prompt = _FAITHFULNESS_PROMPT.format(
        context=context[:3000],
        answer=answer[:1000],
    )
    try:
        raw = await llm_client.generate(prompt, max_tokens=50, temperature=0.0)
        return _parse_binary(raw)
    except Exception as exc:
        log.warning("faithfulness_score: LLM error: {}", exc)
        return 0.0


async def relevance_score(
    question: str,
    answer: str,
    llm_client: "OllamaClient",
) -> float:
    """Judge whether *answer* directly addresses *question*.

    Uses the LLM as a binary judge (1 = relevant, 0 = not relevant).
    Returns 0.0 on LLM errors.

    Args:
        question: The original question string.
        answer: Generated answer string.
        llm_client: Async LLM client for judging.

    Returns:
        ``1.0`` or ``0.0``.
    """
    if not answer or not question:
        return 0.0

    prompt = _RELEVANCE_PROMPT.format(question=question, answer=answer[:1000])
    try:
        raw = await llm_client.generate(prompt, max_tokens=50, temperature=0.0)
        return _parse_binary(raw)
    except Exception as exc:
        log.warning("relevance_score: LLM error: {}", exc)
        return 0.0


def citation_accuracy(
    cited_ids: list[str],
    retrieved_ids: list[str],
) -> float:
    """Fraction of cited chunk IDs that appear in the retrieved set.

    Purely programmatic — no LLM calls.

    Args:
        cited_ids: Chunk IDs cited by the LLM in its answer.
        retrieved_ids: Chunk IDs that were actually retrieved and passed
                       as context to the LLM.

    Returns:
        Value in [0, 1].  Returns ``0.0`` if *cited_ids* is empty.
    """
    if not cited_ids:
        return 0.0
    retrieved_set = set(retrieved_ids)
    valid = sum(1 for cid in cited_ids if cid in retrieved_set)
    return valid / len(cited_ids)
