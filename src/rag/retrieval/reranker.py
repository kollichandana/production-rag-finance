"""Cross-encoder reranker using FastEmbed's reranker.

A bi-encoder retrieves cheaply but its similarity is symmetric and coarse.
A cross-encoder scores the (query, chunk) pair jointly and produces much
sharper relevance signals — at the cost of inference per candidate.

The pattern: retrieve ~20 candidates with hybrid search, then rerank top-N.
"""
from __future__ import annotations

from functools import lru_cache

from fastembed.rerank.cross_encoder import TextCrossEncoder
from loguru import logger

from rag.schemas import RetrievedChunk
from rag.settings import get_settings


class Reranker:
    def __init__(self, model_name: str | None = None) -> None:
        s = get_settings()
        self.model_name = model_name or s.reranker_model
        logger.info(f"Loading reranker: {self.model_name}")
        self._model = TextCrossEncoder(model_name=self.model_name)

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        texts = [c.chunk.text for c in candidates]
        scores = list(self._model.rerank(query, texts))
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            RetrievedChunk(chunk=c.chunk, score=float(s), retrieval_method="reranked")
            for c, s in scored[:top_k]
        ]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker()
