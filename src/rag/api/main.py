"""FastAPI application — exposes /query, /health, /stats."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from rag.api.routes import query as query_routes
from rag.pipeline import get_pipeline
from rag.retrieval.vector_store import VectorStore
from rag.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming up RAG pipeline...")
    pipeline = get_pipeline()
    try:
        chunks_count = pipeline.retriever.store.count()
        logger.info(f"Vector store has {chunks_count} chunks")
    except Exception as e:
        logger.warning(f"Could not reach Qdrant on startup: {e}")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Production RAG over SEC 10-K Filings",
    description="Hybrid retrieval + reranking + HyDE + grounded generation with Claude.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.1f}"
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception):
    logger.exception(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "detail": str(exc)},
    )


app.include_router(query_routes.router, tags=["query"])


@app.get("/health")
def health():
    s = get_settings()
    qdrant_ok = False
    chunks = 0
    try:
        store = VectorStore()
        chunks = store.count()
        qdrant_ok = True
    except Exception as e:
        logger.warning(f"Qdrant health check failed: {e}")
    return {
        "status": "ok" if qdrant_ok else "degraded",
        "qdrant": qdrant_ok,
        "chunks": chunks,
        "collection": s.qdrant_collection,
        "model": s.generation_model,
    }


@app.get("/stats")
def stats():
    store = VectorStore()
    chunks = store.get_all_chunks()
    if not chunks:
        return {"chunks": 0, "documents": 0, "companies": []}
    companies = {c.metadata.get("company") for c in chunks if c.metadata.get("company")}
    docs = {c.doc_id for c in chunks}
    return {
        "chunks": len(chunks),
        "documents": len(docs),
        "companies": sorted(c for c in companies if c),
        "tables": sum(1 for c in chunks if c.chunk_type.value == "table"),
    }
