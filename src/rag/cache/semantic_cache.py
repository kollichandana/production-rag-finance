"""In-memory semantic cache.

Keyed by embedding similarity rather than literal string equality so
paraphrased questions hit. For multi-instance deployments swap the dict
backend for Redis with a HNSW index — interface stays the same.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from rag.retrieval.embeddings import get_embedding_service
from rag.settings import get_settings


@dataclass
class _Entry:
    vector: np.ndarray
    payload: dict[str, Any]
    expires_at: float


class SemanticCache:
    def __init__(self, threshold: float | None = None, ttl_seconds: int | None = None) -> None:
        s = get_settings()
        self.threshold = threshold if threshold is not None else s.semantic_cache_threshold
        self.ttl = ttl_seconds if ttl_seconds is not None else s.cache_ttl_seconds
        self._store: dict[str, _Entry] = {}
        self._lock = threading.Lock()

    def _now(self) -> float:
        return time.time()

    def get(self, query: str) -> dict[str, Any] | None:
        embedder = get_embedding_service()
        qvec = np.asarray(embedder.embed_query(query))
        now = self._now()
        best_score, best_entry = 0.0, None
        with self._lock:
            expired_keys = []
            for k, entry in self._store.items():
                if entry.expires_at < now:
                    expired_keys.append(k)
                    continue
                score = float(np.dot(qvec, entry.vector) / ((np.linalg.norm(qvec) * np.linalg.norm(entry.vector)) or 1e-9))
                if score > best_score:
                    best_score = score
                    best_entry = entry
            for k in expired_keys:
                self._store.pop(k, None)
        if best_entry and best_score >= self.threshold:
            return {**best_entry.payload, "_cache_similarity": best_score}
        return None

    def put(self, query: str, payload: dict[str, Any]) -> None:
        embedder = get_embedding_service()
        qvec = np.asarray(embedder.embed_query(query))
        with self._lock:
            self._store[query] = _Entry(
                vector=qvec, payload=payload, expires_at=self._now() + self.ttl
            )

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def size(self) -> int:
        return len(self._store)


_cache_instance: SemanticCache | None = None


def get_cache() -> SemanticCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SemanticCache()
    return _cache_instance
