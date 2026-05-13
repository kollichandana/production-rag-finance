"""Answer generation with citations + grounded refusal."""
from __future__ import annotations

import re

from loguru import logger

from rag.generation.compressor import compress_chunk, reorder_for_attention
from rag.generation.llm_client import get_llm_client
from rag.generation.prompts.templates import (
    ANSWER_SYSTEM_PROMPT,
    build_answer_user_message,
)
from rag.schemas import Citation, RetrievedChunk

REFUSAL_TEXT = (
    "I don't have enough information in the retrieved filings to answer that confidently."
)


def _to_context_block(rc: RetrievedChunk) -> dict:
    md = rc.chunk.metadata or {}
    return {
        "company": md.get("company") or "N/A",
        "fiscal_year": md.get("fiscal_year"),
        "section": rc.chunk.section,
        "page": rc.chunk.page,
        "text": rc.chunk.text,
        "chunk_id": rc.chunk.chunk_id,
        "doc_id": rc.chunk.doc_id,
    }


def _extract_citation_indices(answer: str) -> set[int]:
    return {int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", answer)}


def build_citations(answer: str, blocks: list[dict]) -> list[Citation]:
    used = _extract_citation_indices(answer)
    citations: list[Citation] = []
    for idx in sorted(used):
        if 1 <= idx <= len(blocks):
            b = blocks[idx - 1]
            snippet = b["text"].split("\n\n", 1)[-1][:300].strip()
            citations.append(
                Citation(
                    chunk_id=b["chunk_id"],
                    doc_id=b["doc_id"],
                    company=b.get("company"),
                    page=b.get("page"),
                    section=b.get("section"),
                    snippet=snippet,
                )
            )
    return citations


def generate_answer(
    query: str,
    retrieved: list[RetrievedChunk],
    compress: bool = True,
    reorder: bool = True,
    min_chunks: int = 1,
) -> dict:
    if not retrieved or len(retrieved) < min_chunks:
        return {
            "answer": REFUSAL_TEXT,
            "citations": [],
            "blocks": [],
            "usage": {"input_tokens": 0, "output_tokens": 0, "cache_read": 0},
            "model": None,
            "refused": True,
        }

    working = retrieved
    if reorder:
        working = reorder_for_attention(working)

    blocks = []
    for rc in working:
        text = rc.chunk.text
        if compress and rc.chunk.chunk_type.value != "table":
            text = compress_chunk(query, text)
        block = _to_context_block(rc)
        block["text"] = text
        blocks.append(block)

    user_msg = build_answer_user_message(query, blocks)

    llm = get_llm_client()
    result = llm.complete(
        messages=[{"role": "user", "content": user_msg}],
        system=ANSWER_SYSTEM_PROMPT,
        cache_system=True,
    )

    answer = result["text"].strip()
    refused = REFUSAL_TEXT.lower().split(".")[0] in answer.lower()
    citations = build_citations(answer, blocks)

    if not citations and not refused and len(answer) > 80:
        # Model answered without citing — likely hallucinating. Flag low confidence.
        logger.warning("Answer has no citations; treating as low confidence")

    return {
        "answer": answer,
        "citations": citations,
        "blocks": blocks,
        "usage": {
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "cache_read": result["cache_read"],
            "cache_creation": result["cache_creation"],
        },
        "model": result["model"],
        "refused": refused,
    }
