"""Query route — wraps the RAGPipeline."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from loguru import logger

from rag.pipeline import get_pipeline
from rag.schemas import QueryRequest, QueryResponse


router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    if not payload.query or not payload.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")
    try:
        pipeline = get_pipeline()
        # Per-request override of common toggles
        pipeline.cfg.use_hyde = payload.use_hyde
        pipeline.cfg.use_decomposition = payload.use_decomposition
        pipeline.cfg.use_reranker = payload.use_reranker
        pipeline.cfg.top_k = payload.top_k
        return pipeline.run(payload.query, filters=payload.filters)
    except Exception as e:
        logger.exception(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
