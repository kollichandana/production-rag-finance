"""FastEmbed-based embedding service (ONNX runtime, no torch dependency)."""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np
from fastembed import TextEmbedding
from loguru import logger

from rag.settings import get_settings


class EmbeddingService:
    """Wraps FastEmbed's TextEmbedding with passage/query prefix conventions for BGE models."""

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        logger.info(f"Loading embedding model: {self.model_name}")
        self._model = TextEmbedding(model_name=self.model_name, max_length=512)
        self._is_bge = "bge" in self.model_name.lower()

    def embed_passages(self, texts: Iterable[str]) -> list[list[float]]:
        texts = [t if t else " " for t in texts]
        # BGE models recommend "passage: " prefix during ingestion (some variants)
        # bge-small-en-v1.5 actually does NOT require prefix; instruction is at query side only
        embeddings = list(self._model.embed(texts))
        return [e.tolist() for e in embeddings]

    def embed_query(self, text: str) -> list[float]:
        # BGE-small-en-v1.5 uses the "Represent this sentence for searching relevant passages: " instruction
        if self._is_bge:
            text = f"Represent this sentence for searching relevant passages: {text}"
        embedding = next(self._model.embed([text]))
        return embedding.tolist()

    @property
    def dim(self) -> int:
        # bge-small-en-v1.5 = 384
        return 384

    @staticmethod
    def cosine(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
        a = np.asarray(a)
        b = np.asarray(b)
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
        return float(np.dot(a, b) / denom)


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
