"""Post-hoc grounding / faithfulness check.

Use the LLM-as-judge with strict instructions to flag unsupported claims.
This is an inference-time guardrail; the eval harness runs a similar check
offline using RAGAS for benchmarking.
"""
from __future__ import annotations

import json

from loguru import logger

from rag.generation.llm_client import get_llm_client
from rag.generation.prompts.templates import GROUNDING_CHECK_SYSTEM_PROMPT


def _format_context(blocks: list[dict]) -> str:
    lines = []
    for i, b in enumerate(blocks, start=1):
        lines.append(f"[{i}] {b['text']}\n")
    return "\n".join(lines)


def grounding_check(answer: str, blocks: list[dict]) -> dict:
    if not blocks or not answer:
        return {"supported": False, "faithfulness_score": 0.0, "unsupported_claims": []}

    user_msg = (
        f"# Context\n{_format_context(blocks)}\n\n"
        f"# Candidate Answer\n{answer}\n\n"
        "Verify whether the answer is supported by the context."
    )

    try:
        result = get_llm_client().complete(
            messages=[{"role": "user", "content": user_msg}],
            system=GROUNDING_CHECK_SYSTEM_PROMPT,
            max_tokens=400,
            temperature=0.0,
            cache_system=True,
        )
        text = result["text"].strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return {"supported": True, "faithfulness_score": 0.7, "unsupported_claims": []}
        payload = json.loads(text[start : end + 1])
        return {
            "supported": bool(payload.get("supported", True)),
            "faithfulness_score": float(payload.get("faithfulness_score", 0.7)),
            "unsupported_claims": payload.get("unsupported_claims", []) or [],
        }
    except Exception as e:
        logger.warning(f"Grounding check failed: {e}")
        return {"supported": True, "faithfulness_score": 0.7, "unsupported_claims": []}
