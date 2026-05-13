"""Multi-query expansion — generate N paraphrases of a query and union their retrievals.

Cheaper alternative to HyDE when you want recall boosts without the LLM
hallucinating financial vocabulary; works well as a complement.
"""
from __future__ import annotations

import json

from loguru import logger

from rag.generation.llm_client import get_llm_client

MULTI_QUERY_SYSTEM_PROMPT = """You generate 3 distinct paraphrases of a financial question to improve retrieval recall.

Rules:
- Each paraphrase must preserve the original meaning exactly.
- Use different but equivalent terminology (e.g. "revenue" / "net sales" / "total revenues").
- Each paraphrase should be a complete question.
- Output strict JSON: {"variants": ["...", "...", "..."]}
- Do not include the original question in the variants.
"""


def expand(query: str, n: int = 3) -> list[str]:
    try:
        result = get_llm_client().complete(
            messages=[{"role": "user", "content": query}],
            system=MULTI_QUERY_SYSTEM_PROMPT,
            max_tokens=300,
            temperature=0.4,
            cache_system=True,
        )
        text = result["text"].strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return [query]
        payload = json.loads(text[start : end + 1])
        variants = [v.strip() for v in payload.get("variants", []) if isinstance(v, str) and v.strip()]
        return [query] + variants[:n]
    except Exception as e:
        logger.warning(f"Multi-query expansion failed: {e}")
        return [query]
