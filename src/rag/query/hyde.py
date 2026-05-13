"""HyDE — Hypothetical Document Embeddings.

Generate a fictitious answer paragraph, embed it, and retrieve against that.
The hypothetical answer is closer in embedding space to real answer passages
than the question itself is — particularly helpful for short queries.
"""
from __future__ import annotations

from loguru import logger

from rag.generation.llm_client import get_llm_client
from rag.generation.prompts.templates import HYDE_SYSTEM_PROMPT
from rag.retrieval.embeddings import get_embedding_service


def generate_hyde_text(query: str) -> str:
    try:
        result = get_llm_client().complete(
            messages=[{"role": "user", "content": query}],
            system=HYDE_SYSTEM_PROMPT,
            max_tokens=300,
            temperature=0.2,
            cache_system=True,
        )
        return result["text"].strip()
    except Exception as e:
        logger.warning(f"HyDE generation failed: {e}")
        return query


def hyde_query_vector(query: str) -> list[float]:
    """Embed a hypothetical answer instead of the literal query."""
    hyde_text = generate_hyde_text(query)
    combined = f"{query}\n\n{hyde_text}"
    return get_embedding_service().embed_query(combined)
