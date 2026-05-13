"""Orchestrates dense + sparse retrieval, fuses via RRF, optionally reranks.

Public surface for the rest of the pipeline. A single `HybridRetriever`
instance is shared across requests; the BM25 index is lazy-built on first
use from chunks scrolled out of Qdrant.
"""
from __future__ import annotations

import threading
from typing import Any

from loguru import logger

from rag.retrieval.fusion import reciprocal_rank_fusion
from rag.retrieval.reranker import get_reranker
from rag.retrieval.sparse import BM25Index
from rag.retrieval.vector_store import VectorStore
from rag.schemas import RetrievedChunk
from rag.settings import get_settings


class HybridRetriever:
    def __init__(self, store: VectorStore | None = None) -> None:
        self.settings = get_settings()
        self.store = store or VectorStore()
        self.bm25 = BM25Index()
        self._bm25_ready = False
        self._bm25_lock = threading.Lock()

    def _ensure_bm25(self) -> None:
        if self._bm25_ready:
            return
        with self._bm25_lock:
            if self._bm25_ready:
                return
            chunks = self.store.get_all_chunks()
            self.bm25.build(chunks)
            self._bm25_ready = True
            logger.info(f"BM25 index ready ({self.bm25.size} chunks)")

    def invalidate_sparse_index(self) -> None:
        self._bm25_ready = False

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        rerank: bool = True,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
        query_vector: list[float] | None = None,
    ) -> list[RetrievedChunk]:
        s = self.settings
        top_k = top_k or s.rerank_top_k
        dense_top_k = dense_top_k or s.dense_top_k
        sparse_top_k = sparse_top_k or s.sparse_top_k

        self._ensure_bm25()

        dense = self.store.dense_search(
            query, top_k=dense_top_k, filters=filters, query_vector=query_vector
        )
        sparse = self.bm25.search(query, top_k=sparse_top_k, filters=filters)

        fused = reciprocal_rank_fusion(dense, sparse, k=s.rrf_k, top_k=max(top_k * 4, 20))

        if rerank and fused:
            try:
                fused = get_reranker().rerank(query, fused, top_k=top_k)
            except Exception as e:
                logger.warning(f"Reranker failed, falling back to fused order: {e}")
                fused = fused[:top_k]
        else:
            fused = fused[:top_k]

        return fused
