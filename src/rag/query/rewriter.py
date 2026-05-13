"""Query rewriting — expands acronyms, normalizes entity names."""
from __future__ import annotations

from loguru import logger

from rag.generation.llm_client import get_llm_client
from rag.generation.prompts.templates import QUERY_REWRITE_SYSTEM_PROMPT


def rewrite_query(query: str) -> str:
    try:
        result = get_llm_client().complete(
            messages=[{"role": "user", "content": query}],
            system=QUERY_REWRITE_SYSTEM_PROMPT,
            max_tokens=200,
            temperature=0.0,
            cache_system=True,
        )
        rewritten = result["text"].strip()
        if rewritten and rewritten.lower() != query.lower():
            logger.debug(f"Rewrote: '{query}' -> '{rewritten}'")
            return rewritten
        return query
    except Exception as e:
        logger.warning(f"Query rewrite failed, using original: {e}")
        return query
