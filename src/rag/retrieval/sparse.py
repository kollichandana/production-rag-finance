"""BM25 sparse retrieval. Builds an in-memory index from Qdrant chunks.

For production scale you'd front this with Elasticsearch or use Qdrant's
sparse vectors. For a single-process service over a few hundred filings
this is fast and adds no infra dependency.
"""
from __future__ import annotations

import re
import threading
from typing import Any

from loguru import logger
from rank_bm25 import BM25Okapi

from rag.schemas import Chunk, RetrievedChunk

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 1]


class BM25Index:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._tokenized: list[list[str]] = []

    def build(self, chunks: list[Chunk]) -> None:
        with self._lock:
            logger.info(f"Building BM25 over {len(chunks)} chunks")
            self._chunks = chunks
            self._tokenized = [_tokenize(c.text) for c in chunks]
            self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    def search(
        self,
        query: str,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        if self._bm25 is None or not self._chunks:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results: list[RetrievedChunk] = []
        for idx, score in ranked:
            if score <= 0:
                break
            chunk = self._chunks[idx]
            if filters and not self._matches(chunk, filters):
                continue
            results.append(
                RetrievedChunk(chunk=chunk, score=float(score), retrieval_method="sparse")
            )
            if len(results) >= top_k:
                break
        return results

    @staticmethod
    def _matches(chunk: Chunk, filters: dict[str, Any]) -> bool:
        merged = {
            "doc_id": chunk.doc_id,
            "section": chunk.section,
            "chunk_type": chunk.chunk_type.value if hasattr(chunk.chunk_type, "value") else chunk.chunk_type,
            **chunk.metadata,
        }
        for k, v in filters.items():
            if v is None:
                continue
            actual = merged.get(k)
            if isinstance(v, list):
                if actual not in v:
                    return False
            elif actual != v:
                return False
        return True

    @property
    def size(self) -> int:
        return len(self._chunks)
