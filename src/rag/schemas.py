"""Shared Pydantic schemas used across ingestion, retrieval, and the API."""
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    FILING_10K = "10-K"
    FILING_10Q = "10-Q"
    OTHER = "other"


class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    HEADING = "heading"


class DocumentMetadata(BaseModel):
    doc_id: str
    company: str
    ticker: str | None = None
    filing_type: DocumentType = DocumentType.FILING_10K
    fiscal_year: int | None = None
    filed_date: str | None = None
    source_path: str | None = None


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    chunk_type: ChunkType = ChunkType.TEXT
    section: str | None = None
    page: int | None = None
    token_count: int = 0
    parent_chunk_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    retrieval_method: str  # "dense", "sparse", "hybrid", "reranked"


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: dict[str, Any] | None = None
    use_hyde: bool = True
    use_decomposition: bool = False
    use_reranker: bool = True
    stream: bool = False


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    company: str | None = None
    page: int | None = None
    section: str | None = None
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    rewritten_query: str | None = None
    sub_queries: list[str] | None = None
    confidence: float
    faithfulness_score: float | None = None
    latency_ms: float
    token_usage: dict[str, int]
    model: str
    cached: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)
