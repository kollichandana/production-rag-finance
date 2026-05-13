"""Lightweight query router. Classifies intent + decides which retrieval tricks to apply."""
from __future__ import annotations

import json
from dataclasses import dataclass

from loguru import logger

from rag.generation.llm_client import get_llm_client
from rag.generation.prompts.templates import ROUTER_SYSTEM_PROMPT


@dataclass
class RouteDecision:
    category: str
    needs_decomposition: bool
    needs_table_data: bool
    in_scope: bool


def route(query: str) -> RouteDecision:
    try:
        result = get_llm_client().complete(
            messages=[{"role": "user", "content": query}],
            system=ROUTER_SYSTEM_PROMPT,
            max_tokens=120,
            temperature=0.0,
            cache_system=True,
        )
        text = result["text"].strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return RouteDecision("factual", False, False, True)
        payload = json.loads(text[start : end + 1])
        category = payload.get("category", "factual")
        return RouteDecision(
            category=category,
            needs_decomposition=bool(payload.get("needs_decomposition", False)),
            needs_table_data=bool(payload.get("needs_table_data", False)),
            in_scope=category != "out_of_scope",
        )
    except Exception as e:
        logger.warning(f"Router failed, defaulting: {e}")
        return RouteDecision("factual", False, False, True)
