# Architecture

This document describes how the system is built and **why each design choice was made**. The "why" matters more than the "what" — that's what an interviewer or client will ask about.

## 1. Ingestion

```
PDF / HTML → parser → section-aware chunker → embeddings → Qdrant
                                            ↘ JSONL on disk (for eval / debugging)
```

### Parsers
- **PDF (`pdfplumber`)** — extracts both text and tables. Tables are kept separately and converted to Markdown so the LLM gets a structured representation.
- **HTML (`BeautifulSoup` + `lxml`)** — SEC EDGAR serves filings as HTML. We walk the DOM, detect headings (`Item 1A.`, `Part II`, etc.), and group paragraphs into sections.

### Chunking
We don't use embedding-based semantic chunking. Why:
- It's expensive (one embedding per sentence at ingest time).
- 10-Ks already have strong structural signals (`Item N.`) — those *are* the semantic boundaries.

Instead:
- Split by section first.
- Within sections, greedily pack paragraphs to ~512 tokens with ~64 tokens of paragraph-level overlap.
- Tables get their own chunks, with the section header prepended so they're self-contextualizing.
- Oversized paragraphs fall back to sentence splitting.

Each chunk carries `doc_id`, `company`, `ticker`, `fiscal_year`, `section`, `page`, `chunk_type`. These power metadata filtering at query time.

## 2. Retrieval

### Hybrid: Dense + Sparse + RRF

Dense alone misses exact-token matches that BM25 catches trivially (ticker symbols, exact dollar figures, specific named entities). BM25 alone misses paraphrased queries. We run both and fuse with **Reciprocal Rank Fusion** at `k=60`.

Why RRF over weighted-score fusion? Dense cosine scores live in `[0, 1]`, BM25 scores are unbounded — combining them with weights forces brittle calibration. RRF only uses ranks, so it sidesteps the scale mismatch.

### Cross-encoder rerank

Top-20 fused candidates → top-K reranked. The bi-encoder embedding model is symmetric and coarse; a cross-encoder scores `(query, chunk)` jointly and produces much sharper relevance signals. Cost: ~1ms per candidate on CPU for MiniLM-L-6.

### Why FastEmbed for both

- ONNX runtime, no `torch` dependency → smaller container, faster cold start on Streamlit Cloud.
- Same library for embeddings and reranking.
- We use `BAAI/bge-small-en-v1.5` (384-dim, query-side instruction prefix per BGE convention) — strong quality-per-MB tradeoff.

## 3. Query understanding

Four optional transformations, each toggleable from the UI/API:

1. **Router** — LLM classifies intent into `factual | comparative | analytical | summary | out_of_scope`. Out-of-scope queries get a deterministic refusal without retrieval.
2. **Query rewriter** — expands acronyms (R&D → research and development), normalizes entity names. Improves both dense and BM25 recall.
3. **HyDE** — generates a 2-4 sentence hypothetical answer, embeds the concatenation of the question + the hypothetical answer. The hypothetical text lives in the same vocabulary as real passages, so it lands closer to them in embedding space.
4. **Decomposition** — multi-part questions ("compare R&D between Apple and Microsoft") are split into atomic sub-queries; each is retrieved separately; results are unioned and re-fused.

## 4. Generation

### Prompt caching

The answer system prompt is large and stable across requests. We mark it with `cache_control: {"type": "ephemeral"}`, which lets Anthropic serve it from cache. On a hot system most generation requests hit the cache — measurable in the benchmark via `prompt_cache_read_ratio`.

### Contextual compression + reorder

Two transformations between retrieval and the model:

- **Compression** — for each retrieved chunk, keep only the sentences most similar to the query (by embedding cosine). Header lines are always preserved. Cuts tokens 40-60% on typical chunks.
- **Reorder** — place the highest-ranked chunks at the start AND end of the context window. Mitigates lost-in-the-middle (Liu et al., 2023).

### Citations

The system prompt enforces bracketed citations `[N]`. After generation we regex-extract those indices and resolve them back to the source chunks for the UI. Answers without citations are flagged as low-confidence in logs.

### Refusal

If the post-hoc grounding check returns faithfulness < 0.4, the answer is replaced with the canonical refusal. We'd rather say "I don't know" than hallucinate a number.

## 5. Caching

Two layers:

- **Anthropic prompt cache** (server-side, free token discount on cache hits).
- **Semantic cache** (in-process, embedding-keyed). Threshold defaults to 0.95 cosine similarity — paraphrases hit, unrelated queries don't.

For multi-instance deployments swap the dict-backed semantic cache for Redis with HNSW. The interface stays the same.

## 6. Observability

- **Loguru** structured logs (every step in the pipeline).
- **Langfuse** tracing (optional, no-op without creds). Spans for query, retrieval, rerank, generation, grounding.
- **X-Process-Time-Ms** header on every API response.
- `/health` reports Qdrant reachability + chunk count.

## 7. Evaluation

Two harnesses:

### Fast harness (`scripts/run_eval.py`)
Runs both naive and production pipelines on the bundled benchmark. Reports substring recall, citation rate, refusal rate, OOS accuracy, p50/p95 latency, prompt cache ratio. Fast enough to run in CI.

### RAGAS harness (`scripts/run_ragas.py`)
Optional, slower. Computes the canonical `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall` over a sample. These are the numbers a hiring manager will recognize, so we run them for README headline figures.

The naive baseline is exactly the same code as production, with all toggles flipped off. That keeps the comparison honest — no apples-to-oranges code path differences.

## 8. Why this stack

| Choice | Alternatives considered | Why we picked it |
|---|---|---|
| Qdrant | Pinecone, Weaviate, pgvector | Self-hostable, free tier in cloud, mature hybrid support, fast |
| FastEmbed | sentence-transformers, OpenAI embeddings | ONNX = no torch, fast cold starts, free, runs on Streamlit Cloud |
| BGE-small | OpenAI text-embedding-3-small | Free, open, good for English finance text, 384 dims |
| MiniLM-L-6 reranker | Cohere Rerank, BGE-reranker | Free, ONNX, ~22MB, good enough for our recall/precision needs |
| BM25 in-process | Elasticsearch | Zero extra infra for <1M chunks |
| Claude Sonnet 4.6 | GPT-4o, Gemini | Best citation-following + prompt caching ergonomics |
| Streamlit | Next.js + FastAPI | This is a backend project — frontend is a means, not the end |
| Langfuse | LangSmith, Arize | OSS / self-hostable |
