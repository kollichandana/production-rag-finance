"""End-to-end RAG orchestration.

Single entry point for the API and the eval harness. Composes:
    route → rewrite → (decompose | hyde) → hybrid retrieve → rerank
         → compress + reorder → generate → ground-check → cache

Every step is optional and toggleable so the same pipeline can serve the
production path and the naive baseline used in benchmarks.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from loguru import logger

from rag.cache.semantic_cache import get_cache
from rag.generation.answerer import REFUSAL_TEXT, generate_answer
from rag.guardrails.grounding import grounding_check
from rag.observability.tracer import flush, log_generation, trace
from rag.query.decomposer import decompose
from rag.query.hyde import hyde_query_vector
from rag.query.rewriter import rewrite_query
from rag.query.router import route
from rag.retrieval.embeddings import get_embedding_service
from rag.retrieval.hybrid import HybridRetriever
from rag.schemas import Citation, QueryResponse, RetrievedChunk


@dataclass
class PipelineConfig:
    use_router: bool = True
    use_rewrite: bool = True
    use_hyde: bool = True
    use_decomposition: bool = False
    use_reranker: bool = True
    use_compression: bool = True
    use_reorder: bool = True
    use_grounding_check: bool = True
    use_cache: bool = True
    top_k: int = 5
    confidence_floor: float = 0.4  # below this we refuse


class RAGPipeline:
    def __init__(
        self, retriever: HybridRetriever | None = None, config: PipelineConfig | None = None
    ) -> None:
        self.retriever = retriever or HybridRetriever()
        self.cfg = config or PipelineConfig()

    @classmethod
    def naive(cls, retriever: HybridRetriever | None = None) -> RAGPipeline:
        """Baseline: dense-only retrieval, no rewrite/HyDE/rerank/compression/grounding."""
        return cls(
            retriever,
            PipelineConfig(
                use_router=False,
                use_rewrite=False,
                use_hyde=False,
                use_decomposition=False,
                use_reranker=False,
                use_compression=False,
                use_reorder=False,
                use_grounding_check=False,
                use_cache=False,
                top_k=5,
            ),
        )

    def _retrieve_for_query(
        self, query: str, filters: dict[str, Any] | None
    ) -> list[RetrievedChunk]:
        query_vector = None
        if self.cfg.use_hyde:
            try:
                query_vector = hyde_query_vector(query)
            except Exception as e:
                logger.warning(f"HyDE failed, falling back to literal embed: {e}")

        if query_vector is None:
            query_vector = get_embedding_service().embed_query(query)

        return self.retriever.retrieve(
            query=query,
            top_k=self.cfg.top_k,
            filters=filters,
            rerank=self.cfg.use_reranker,
            query_vector=query_vector,
        )

    def run(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
    ) -> QueryResponse:
        start = time.perf_counter()
        cache = get_cache() if self.cfg.use_cache else None

        if cache:
            hit = cache.get(query)
            if hit:
                logger.info(f"Cache hit (sim={hit['_cache_similarity']:.3f})")
                payload = {k: v for k, v in hit.items() if k != "_cache_similarity"}
                resp = QueryResponse(**payload)
                resp.cached = True
                resp.latency_ms = (time.perf_counter() - start) * 1000
                return resp

        with trace("rag_query", metadata={"query": query, "filters": filters or {}}) as t:
            if self.cfg.use_router:
                decision = route(query)
                if not decision.in_scope:
                    return QueryResponse(
                        answer="This system only answers questions about SEC 10-K filings in its corpus.",
                        citations=[],
                        retrieved_chunks=[],
                        confidence=1.0,
                        latency_ms=(time.perf_counter() - start) * 1000,
                        token_usage={"input": 0, "output": 0},
                        model="router",
                    )
                use_decompose = self.cfg.use_decomposition or decision.needs_decomposition
            else:
                use_decompose = self.cfg.use_decomposition

            working_query = rewrite_query(query) if self.cfg.use_rewrite else query
            sub_queries = decompose(working_query) if use_decompose else [working_query]

            all_retrieved: dict[str, RetrievedChunk] = {}
            for sq in sub_queries:
                for rc in self._retrieve_for_query(sq, filters):
                    existing = all_retrieved.get(rc.chunk.chunk_id)
                    if not existing or rc.score > existing.score:
                        all_retrieved[rc.chunk.chunk_id] = rc

            retrieved = sorted(all_retrieved.values(), key=lambda r: r.score, reverse=True)[
                : self.cfg.top_k * (2 if use_decompose else 1)
            ]

            gen = generate_answer(
                query=working_query,
                retrieved=retrieved,
                compress=self.cfg.use_compression,
                reorder=self.cfg.use_reorder,
            )

            faithfulness = None
            confidence = 1.0
            if self.cfg.use_grounding_check and not gen["refused"]:
                gc = grounding_check(gen["answer"], gen["blocks"])
                faithfulness = gc["faithfulness_score"]
                confidence = float(faithfulness)
                if confidence < self.cfg.confidence_floor:
                    logger.warning(
                        f"Low faithfulness ({confidence:.2f}); converting to refusal"
                    )
                    gen["answer"] = REFUSAL_TEXT
                    gen["citations"] = []

            log_generation(
                t,
                name="answer",
                model=gen["model"] or "n/a",
                input_data={"query": query, "rewritten": working_query, "subs": sub_queries},
                output_data=gen["answer"],
                usage=gen["usage"],
            )

            latency_ms = (time.perf_counter() - start) * 1000
            response = QueryResponse(
                answer=gen["answer"],
                citations=gen["citations"] if isinstance(gen["citations"], list) else [Citation(**c) for c in gen["citations"]],
                retrieved_chunks=retrieved,
                rewritten_query=working_query if working_query != query else None,
                sub_queries=sub_queries if len(sub_queries) > 1 else None,
                confidence=confidence,
                faithfulness_score=faithfulness,
                latency_ms=latency_ms,
                token_usage={
                    "input": gen["usage"]["input_tokens"],
                    "output": gen["usage"]["output_tokens"],
                    "cache_read": gen["usage"].get("cache_read", 0),
                },
                model=gen["model"] or "n/a",
            )

            if cache and not gen["refused"]:
                cache.put(query, response.model_dump(mode="json"))

            flush()
            return response


_pipeline_singleton: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline_singleton
    if _pipeline_singleton is None:
        _pipeline_singleton = RAGPipeline()
    return _pipeline_singleton
