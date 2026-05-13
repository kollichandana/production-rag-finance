"""Reciprocal Rank Fusion for combining dense + sparse result lists.

RRF score for an item: sum over rankers of 1 / (k + rank).
Defaults to k=60 per Cormack et al. — robust across many query types and a
much better fusion baseline than weighted score averaging because it sidesteps
the score-scale mismatch between dense (cosine) and BM25.
"""
from __future__ import annotations

from collections import defaultdict

from rag.schemas import RetrievedChunk


def reciprocal_rank_fusion(
    *ranked_lists: list[RetrievedChunk],
    k: int = 60,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    fused_scores: dict[str, float] = defaultdict(float)
    keep: dict[str, RetrievedChunk] = {}

    for results in ranked_lists:
        for rank, item in enumerate(results, start=1):
            cid = item.chunk.chunk_id
            fused_scores[cid] += 1.0 / (k + rank)
            if cid not in keep or item.score > keep[cid].score:
                keep[cid] = item

    ordered = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    fused = []
    for cid, score in ordered:
        rc = keep[cid]
        fused.append(
            RetrievedChunk(chunk=rc.chunk, score=float(score), retrieval_method="hybrid")
        )
    return fused[:top_k] if top_k else fused
