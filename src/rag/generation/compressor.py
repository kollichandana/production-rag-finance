"""Contextual compression — drop irrelevant sentences from retrieved chunks.

Reduces tokens fed to the LLM and combats lost-in-the-middle.
We use a simple but effective lexical-overlap heuristic. For higher quality
you can swap in a cross-encoder sentence-level scorer.
"""
from __future__ import annotations

import re

from rag.retrieval.embeddings import get_embedding_service
from rag.schemas import RetrievedChunk

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def compress_chunk(query: str, text: str, keep_sentences: int = 6, min_score: float = 0.25) -> str:
    """Keep the top-N sentences by embedding similarity to the query.

    Header lines like [Item 7. ...] are always preserved.
    """
    lines = text.split("\n\n", 1)
    header, body = ("", text)
    if len(lines) == 2 and lines[0].startswith("["):
        header, body = lines[0], lines[1]

    sentences = _split_sentences(body)
    if len(sentences) <= keep_sentences:
        return text

    embedder = get_embedding_service()
    qvec = embedder.embed_query(query)
    svecs = embedder.embed_passages(sentences)
    scored = [(s, embedder.cosine(qvec, v)) for s, v in zip(sentences, svecs, strict=False)]
    scored.sort(key=lambda x: x[1], reverse=True)
    selected = [s for s, score in scored[:keep_sentences] if score >= min_score]
    if not selected:
        selected = [s for s, _ in scored[:keep_sentences]]

    # Restore original order so the text reads naturally
    selected_set = set(selected)
    in_order = [s for s in sentences if s in selected_set]
    compressed_body = " ".join(in_order)
    return f"{header}\n\n{compressed_body}" if header else compressed_body


def reorder_for_attention(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Mitigate lost-in-the-middle by placing top chunks at the start and end.

    Order: [1, 3, 5, ..., 6, 4, 2]. The LLM attends most to first and last.
    """
    if len(chunks) <= 2:
        return chunks
    odds = chunks[::2]  # rank 1, 3, 5
    evens = chunks[1::2]  # rank 2, 4, 6
    return odds + evens[::-1]
