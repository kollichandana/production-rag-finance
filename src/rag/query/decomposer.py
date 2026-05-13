"""Decompose complex queries into atomic sub-queries."""
from __future__ import annotations

import json

from loguru import logger

from rag.generation.llm_client import get_llm_client
from rag.generation.prompts.templates import DECOMPOSE_SYSTEM_PROMPT


def decompose(query: str) -> list[str]:
    try:
        result = get_llm_client().complete(
            messages=[{"role": "user", "content": query}],
            system=DECOMPOSE_SYSTEM_PROMPT,
            max_tokens=400,
            temperature=0.0,
            cache_system=True,
        )
        text = result["text"].strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return [query]
        payload = json.loads(text[start : end + 1])
        subs = payload.get("sub_questions", [])
        subs = [s.strip() for s in subs if isinstance(s, str) and s.strip()]
        if not subs:
            return [query]
        return subs[:4]
    except Exception as e:
        logger.warning(f"Decomposition failed: {e}")
        return [query]
