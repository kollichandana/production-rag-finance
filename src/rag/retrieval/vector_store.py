"""Qdrant vector store wrapper. Supports local + cloud, batched upserts, filtered search."""
from __future__ import annotations

import uuid
from contextlib import suppress
from typing import Any

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from tqdm import tqdm

from rag.retrieval.embeddings import get_embedding_service
from rag.schemas import Chunk, RetrievedChunk
from rag.settings import get_settings


def _chunk_id_to_point_id(chunk_id: str) -> str:
    """Qdrant requires UUID or int point IDs. Stable hash chunk_id → UUID5."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class VectorStore:
    def __init__(self, collection: str | None = None) -> None:
        s = get_settings()
        self.collection = collection or s.qdrant_collection
        self.client = QdrantClient(
            url=s.qdrant_url,
            api_key=s.qdrant_api_key or None,
            timeout=60,
            prefer_grpc=False,
        )
        self.embedder = get_embedding_service()

    def ensure_collection(self, recreate: bool = False) -> None:
        collections = {c.name for c in self.client.get_collections().collections}
        if self.collection in collections and recreate:
            logger.warning(f"Recreating collection {self.collection}")
            self.client.delete_collection(self.collection)
            collections.discard(self.collection)
        if self.collection not in collections:
            logger.info(f"Creating collection {self.collection} (dim={self.embedder.dim})")
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(size=self.embedder.dim, distance=qm.Distance.COSINE),
                optimizers_config=qm.OptimizersConfigDiff(indexing_threshold=10000),
            )
            for field in ["doc_id", "company", "ticker", "fiscal_year", "section", "chunk_type"]:
                with suppress(Exception):
                    self.client.create_payload_index(
                        collection_name=self.collection,
                        field_name=field,
                        field_schema=qm.PayloadSchemaType.KEYWORD
                        if field != "fiscal_year"
                        else qm.PayloadSchemaType.INTEGER,
                    )

    def upsert_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        if not chunks:
            return
        for start in tqdm(range(0, len(chunks), batch_size), desc="Embedding"):
            batch = chunks[start : start + batch_size]
            vectors = self.embedder.embed_passages([c.text for c in batch])
            points = []
            for c, v in zip(batch, vectors, strict=False):
                points.append(
                    qm.PointStruct(
                        id=_chunk_id_to_point_id(c.chunk_id),
                        vector=v,
                        payload={
                            "chunk_id": c.chunk_id,
                            "doc_id": c.doc_id,
                            "text": c.text,
                            "chunk_type": c.chunk_type.value,
                            "section": c.section,
                            "page": c.page,
                            "token_count": c.token_count,
                            **{k: v for k, v in c.metadata.items() if v is not None},
                        },
                    )
                )
            self.client.upsert(collection_name=self.collection, points=points, wait=False)

    def _build_filter(self, filters: dict[str, Any] | None) -> qm.Filter | None:
        if not filters:
            return None
        must: list[qm.FieldCondition] = []
        for k, v in filters.items():
            if v is None:
                continue
            if isinstance(v, list):
                must.append(qm.FieldCondition(key=k, match=qm.MatchAny(any=v)))
            else:
                must.append(qm.FieldCondition(key=k, match=qm.MatchValue(value=v)))
        return qm.Filter(must=must) if must else None

    def dense_search(
        self,
        query: str,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
        query_vector: list[float] | None = None,
    ) -> list[RetrievedChunk]:
        vector = query_vector or self.embedder.embed_query(query)
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=top_k,
            query_filter=self._build_filter(filters),
            with_payload=True,
        )
        return [self._hit_to_retrieved(h, "dense") for h in hits]

    def get_all_chunks(self, batch_size: int = 256) -> list[Chunk]:
        """Used by BM25 to build sparse index in-memory."""
        chunks: list[Chunk] = []
        next_page = None
        while True:
            points, next_page = self.client.scroll(
                collection_name=self.collection,
                limit=batch_size,
                offset=next_page,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                payload = p.payload or {}
                chunks.append(
                    Chunk(
                        chunk_id=payload.get("chunk_id", str(p.id)),
                        doc_id=payload.get("doc_id", ""),
                        text=payload.get("text", ""),
                        chunk_type=payload.get("chunk_type", "text"),
                        section=payload.get("section"),
                        page=payload.get("page"),
                        token_count=payload.get("token_count", 0),
                        metadata={
                            k: v
                            for k, v in payload.items()
                            if k
                            not in {"chunk_id", "doc_id", "text", "chunk_type", "section", "page", "token_count"}
                        },
                    )
                )
            if next_page is None:
                break
        return chunks

    def count(self) -> int:
        try:
            return self.client.count(self.collection, exact=True).count
        except Exception:
            return 0

    @staticmethod
    def _hit_to_retrieved(hit, method: str) -> RetrievedChunk:
        p = hit.payload or {}
        chunk = Chunk(
            chunk_id=p.get("chunk_id", str(hit.id)),
            doc_id=p.get("doc_id", ""),
            text=p.get("text", ""),
            chunk_type=p.get("chunk_type", "text"),
            section=p.get("section"),
            page=p.get("page"),
            token_count=p.get("token_count", 0),
            metadata={
                k: v
                for k, v in p.items()
                if k not in {"chunk_id", "doc_id", "text", "chunk_type", "section", "page", "token_count"}
            },
        )
        return RetrievedChunk(chunk=chunk, score=float(hit.score), retrieval_method=method)
