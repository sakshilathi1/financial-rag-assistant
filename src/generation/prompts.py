"""Prompt templates and context-formatting helpers for RAG generation."""

from src.chunking.base import Chunk

CITATION_PROMPT = """\
You are a financial analyst assistant. Answer the user's question using ONLY the context passages provided below. Do not use any external knowledge.

Rules:
1. Cite every factual claim with the chunk ID in square brackets, e.g. [AAPL_0012].
2. Only cite chunks that directly support the specific claim being made. Do not cite chunks just to fill the citations list.
3. If a passage is not relevant, ignore it.
4. If the context does not contain enough information to answer the question, respond exactly: "I don't know based on the provided context."
5. Be concise and precise - this is a professional financial context.

Confidence scoring:
- 0.9-1.0: All claims directly supported by cited passages with high specificity.
- 0.6-0.8: Most claims supported, some inference required, or one weak citation.
- 0.3-0.5: Partial answer, significant inference or uncertainty.
- 0.0-0.2: Used only when answering "I don't know based on the provided context."

EXAMPLE:

Context:
[AAPL_0042] (financial_statements)
Apple Inc. reported total net sales of $391.0 billion for fiscal 2024, a 2 percent increase from $383.3 billion in fiscal 2023.

[AAPL_0043] (financial_statements)
Services revenue grew 13 percent year-over-year, reaching $96.2 billion in fiscal 2024.

Question: What was Apple's revenue in fiscal 2024?

Output:
{{
  "answer": "Apple's total net sales were $391.0 billion in fiscal 2024, a 2% increase from $383.3 billion in fiscal 2023 [AAPL_0042]. Services revenue grew 13% to $96.2 billion [AAPL_0043].",
  "citations": ["AAPL_0042", "AAPL_0043"],
  "confidence": 0.95
}}

NOW ANSWER THE FOLLOWING:

Context:
{context}

Question: {question}

Respond with a JSON object and nothing else - no markdown fences, no extra text:
{{
  "answer": "<your answer with inline [chunk_id] citations>",
  "citations": ["chunk_id_1", "chunk_id_2"],
  "confidence": <float between 0.0 and 1.0>
}}"""


def format_context(chunks: list[Chunk]) -> str:
    """Render a list of chunks into a numbered context block for the prompt.

    Each chunk is prefixed with its ID so the model can cite it precisely.

    Args:
        chunks: Ordered list of retrieved :class:`~src.chunking.base.Chunk` objects.

    Returns:
        Multi-line string with one block per chunk, separated by blank lines.
    """
    parts: list[str] = []
    for chunk in chunks:
        header = f"[{chunk.id}]"
        if chunk.section:
            header += f" ({chunk.section})"
        parts.append(f"{header}\n{chunk.text}")
    return "\n\n".join(parts)


def build_citation_prompt(question: str, chunks: list[Chunk]) -> str:
    """Fill in the :data:`CITATION_PROMPT` template.

    Args:
        question: User's question string.
        chunks: Retrieved chunks to include as context.

    Returns:
        Complete prompt string ready to send to the LLM.
    """
    context = format_context(chunks)
    return CITATION_PROMPT.format(context=context, question=question)
