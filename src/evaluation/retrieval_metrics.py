"""Pure retrieval metric functions — no I/O, no side effects."""


def hit_at_k(retrieved_ids: list[str], relevant_id: str, k: int) -> bool:
    """Return ``True`` if *relevant_id* appears in the top-*k* retrieved IDs.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs (best first).
        relevant_id: Ground-truth chunk ID to look for.
        k: Cut-off rank.

    Returns:
        ``True`` if found within the top-*k* results.
    """
    return relevant_id in retrieved_ids[:k]


def mrr(retrieved_ids: list[str], relevant_id: str) -> float:
    """Compute Mean Reciprocal Rank for a single query.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs (best first).
        relevant_id: Ground-truth chunk ID.

    Returns:
        ``1 / rank`` where rank is the 1-based position of *relevant_id*,
        or ``0.0`` if not found.
    """
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id == relevant_id:
            return 1.0 / rank
    return 0.0


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Recall at *k*: fraction of *relevant_ids* found in the top-*k* retrieved.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs (best first).
        relevant_ids: Set of all ground-truth relevant chunk IDs.
        k: Cut-off rank.

    Returns:
        Recall in [0, 1].  Returns ``0.0`` if *relevant_ids* is empty.
    """
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    found = top_k & relevant_ids
    return len(found) / len(relevant_ids)
