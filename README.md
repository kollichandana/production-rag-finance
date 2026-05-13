# Production RAG over SEC 10-K Filings

A production-grade Retrieval-Augmented Generation system that answers questions about US public company 10-K filings — built to demonstrate everything that separates a real RAG system from a tutorial: **hybrid retrieval, query understanding, reranking, grounded generation with citations, evaluation, observability, and a benchmark that proves the production pipeline beats the naive baseline**.

> **Live demo:** _add Streamlit Cloud URL after deploy_
> **Tech:** Python · Claude Sonnet 4.6 · Qdrant · FastEmbed (BGE) · BM25 · Cross-encoder reranker · FastAPI · Streamlit · RAGAS · Langfuse

---

## Why this exists

Most public RAG demos are a 50-line script that chunks a PDF, embeds it, and pipes the top-k chunks into a model. That isn't production. Production means:

- **Measurable quality** — eval harness, benchmark dataset, regression-aware metrics
- **Grounded answers** — citations, refusal when context is weak, post-hoc faithfulness check
- **Real retrieval** — dense + sparse + rerank, not just cosine similarity
- **Cost & latency awareness** — prompt caching, semantic cache, contextual compression
- **Operational signals** — health checks, structured logs, tracing

This repo implements all of that on a domain (SEC filings) where mistakes are easy to spot — exact numbers, exact section references, exact ticker symbols.

---

## Headline results — Naive vs. Production

Run on the bundled benchmark over the default 5-filing corpus (AAPL, MSFT, GOOGL, AMZN, META — FY2023 10-Ks). Reproduce with `make benchmark`. Generated table lives in [`docs/benchmarks.md`](docs/benchmarks.md).

| Metric                       | Naive RAG | Production RAG | Δ |
|------------------------------|-----------|----------------|---|
| Substring recall ↑           | _populated by `make benchmark`_ | | |
| Citation rate ↑              | | | |
| Avg faithfulness (LLM-judge) ↑ | | | |
| Out-of-scope refusal acc. ↑  | | | |
| p50 latency (ms)             | | | |
| p95 latency (ms)             | | | |
| Prompt-cache read ratio ↑    | | | |

The harness scores every benchmark item under both pipelines so the comparison is fully reproducible — not cherry-picked.

---

## Architecture

```
                    ┌──────────────────────────────┐
                    │         User query           │
                    └──────────────┬───────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │  1. Router (in-scope?)       │
                    │  2. Query rewriter           │
                    │  3. Decomposition (optional) │
                    │  4. HyDE — hypothetical doc  │
                    └──────────────┬───────────────┘
                                   ▼
              ┌────────────────────────────────────────────┐
              │              Hybrid retrieval              │
              │  Dense (FastEmbed BGE)  ◇  Sparse (BM25)   │
              │              ▼                ▼            │
              │       Reciprocal Rank Fusion (k=60)        │
              └────────────────────┬───────────────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │  Cross-encoder reranker      │
                    │  (MS MARCO MiniLM-L-6)       │
                    └──────────────┬───────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │  Context compression         │
                    │  + lost-in-the-middle reorder │
                    └──────────────┬───────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │  Claude Sonnet 4.6           │
                    │  (cached system + citations) │
                    └──────────────┬───────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │  Faithfulness check          │
                    │  → refuse if score < floor   │
                    └──────────────┬───────────────┘
                                   ▼
                              Answer + [1][2] citations
```

Wrapping the pipeline: **semantic cache**, **Langfuse tracing**, **structured logs**, **health/stats endpoints**.

Full design in [`docs/architecture.md`](docs/architecture.md). Decisions captured as ADRs in [`docs/adr/`](docs/adr/).

---

## What's actually production-grade about it

| Concern | What's built |
|---|---|
| Document parsing | `pdfplumber` + `BeautifulSoup`, table-aware extraction, section detection (`Item 1A.`, `Part II`, ...) |
| Chunking | Section-aware hierarchical chunker with token-bounded packing + paragraph-level overlap. Tables become their own chunks. |
| Embeddings | `BAAI/bge-small-en-v1.5` via FastEmbed (ONNX, no torch). Query-side instruction prefix for BGE convention. |
| Vector DB | Qdrant (local in dev, Qdrant Cloud in deploy). Payload indexes on `company`, `ticker`, `fiscal_year`, `section`, `chunk_type`. |
| Sparse | BM25 in-process via `rank_bm25` — zero infra dependency. Easy swap to Qdrant sparse vectors or Elasticsearch for scale. |
| Hybrid fusion | Reciprocal Rank Fusion (k=60). Sidesteps the dense/BM25 score-scale mismatch that breaks weighted-sum fusion. |
| Rerank | Cross-encoder via FastEmbed. Bi-encoder retrieves cheaply, cross-encoder scores `(query, chunk)` jointly. |
| Query understanding | Router · query rewriter (acronym expansion) · multi-query expansion · HyDE · decomposition for multi-hop |
| Generation | Claude Sonnet 4.6 with **prompt caching** on the system prompt, automatic fallback to Haiku 4.5 on 5xx/rate-limit, bounded retries via Tenacity |
| Citations | Bracketed `[N]` citations, parsed back into a structured `Citation` list with source company + section + page |
| Faithfulness | Post-hoc LLM-judge grounding check. Low-confidence answers are converted to refusal. |
| Out-of-scope handling | Router classifies, generation refuses cleanly |
| Caching | Two layers: Anthropic prompt cache (on the system prompt — huge token win on repeated requests) + in-process semantic cache on the query |
| Lost-in-the-middle | Top chunks placed at start AND end of context, lowest-ranked in the middle |
| Evaluation | Hand-curated benchmark + harness comparing **naive vs. production** on substring recall, citation rate, faithfulness, OOS accuracy, p50/p95 latency, cache hit rate |
| Optional RAGAS | `scripts/run_ragas.py` runs `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall` for the README headline numbers |
| Observability | Optional Langfuse tracing (no-ops if creds absent), Loguru structured logs, `X-Process-Time-Ms` response header |
| API | FastAPI with `/query`, `/health`, `/stats`, CORS, global exception handler |
| UI | Streamlit chat with retrieval-settings sidebar, per-query metrics, expandable citations, side-by-side naive/production toggle |
| Containerization | Multi-service `docker-compose.yml` (Qdrant + FastAPI + Streamlit) |
| Deployment | Streamlit Cloud-ready (reads `st.secrets` → env vars). Works with Qdrant Cloud free tier. |
| Tests | Unit tests for chunker, RRF, compressor |
| CI | GitHub Actions: lint + unit tests on every push |

---

## Quickstart

### 1. Local dev (Docker)

```bash
git clone <this-repo>
cd production-rag-finance
cp .env.example .env
# edit .env and add ANTHROPIC_API_KEY=sk-ant-...

make docker-up           # spins up Qdrant + API + Streamlit
python scripts/download_filings.py   # downloads 5 sample 10-Ks from SEC EDGAR
python scripts/ingest.py             # parses + chunks + embeds + upserts
make benchmark           # populates docs/benchmarks.md with naive-vs-production results

open http://localhost:8501  # Streamlit UI
open http://localhost:8000/docs  # FastAPI docs
```

### 2. Pure-local (no Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
make qdrant              # only Qdrant in Docker
cp .env.example .env     # add ANTHROPIC_API_KEY

python scripts/download_filings.py
make ingest
make ui                  # Streamlit at :8501
# or
make serve               # FastAPI at :8000
```

### 3. Deploy live demo (Streamlit Community Cloud + Qdrant Cloud)

See [`docs/deployment.md`](docs/deployment.md). 10-minute walkthrough.

---

## Project layout

```
production-rag-finance/
├── streamlit_app.py                  # Entry point for Streamlit Cloud
├── docker-compose.yml                # Qdrant + API + UI
├── Dockerfile / Dockerfile.streamlit
├── Makefile                          # make ingest / benchmark / serve / ui
├── src/rag/
│   ├── settings.py                   # pydantic-settings (env-driven)
│   ├── schemas.py                    # Pydantic shapes shared across layers
│   ├── pipeline.py                   # End-to-end orchestration + naive baseline
│   ├── ingestion/
│   │   ├── parsers/{pdf,html}_parser.py
│   │   ├── chunkers/semantic_chunker.py
│   │   └── pipeline.py
│   ├── retrieval/
│   │   ├── embeddings.py             # FastEmbed BGE
│   │   ├── vector_store.py           # Qdrant client wrapper
│   │   ├── sparse.py                 # BM25
│   │   ├── fusion.py                 # Reciprocal Rank Fusion
│   │   ├── reranker.py               # Cross-encoder
│   │   └── hybrid.py                 # Orchestrates dense + sparse + rerank
│   ├── query/
│   │   ├── router.py                 # Intent classification
│   │   ├── rewriter.py               # Acronym/entity expansion
│   │   ├── hyde.py                   # Hypothetical document embeddings
│   │   ├── decomposer.py             # Multi-hop decomposition
│   │   └── multi_query.py
│   ├── generation/
│   │   ├── llm_client.py             # Anthropic w/ caching + fallback
│   │   ├── compressor.py             # Sentence pruning + reorder
│   │   ├── answerer.py               # Generate + parse citations
│   │   └── prompts/templates.py      # All system prompts (cache-friendly)
│   ├── guardrails/grounding.py       # LLM-judge faithfulness
│   ├── cache/semantic_cache.py       # Embedding-keyed cache
│   ├── observability/tracer.py       # Langfuse (optional)
│   ├── eval/
│   │   ├── benchmark_dataset.py
│   │   ├── metrics.py
│   │   └── runner.py
│   └── api/main.py + routes/query.py
├── scripts/
│   ├── download_filings.py
│   ├── ingest.py
│   ├── run_eval.py                   # Naive vs production benchmark
│   └── run_ragas.py                  # Optional RAGAS metrics
├── tests/                            # Chunker, RRF, compressor
├── docs/
│   ├── architecture.md
│   ├── benchmarks.md                 # Auto-generated by `make benchmark`
│   ├── deployment.md
│   └── adr/                          # Architecture Decision Records
└── .github/workflows/ci.yml
```

---

## Roadmap

- [ ] Streaming responses in Streamlit & FastAPI
- [ ] Swap BM25 for Qdrant sparse vectors (single store, no in-process index)
- [ ] Fine-tune `bge-small` on a small finance corpus
- [ ] Multi-turn conversation memory with retrieval-augmented history
- [ ] Table-cell-level retrieval for financial figures
- [ ] LangGraph migration to support tool calls (e.g. SEC EDGAR lookup mid-answer)

---

## License

MIT. See [`LICENSE`](LICENSE).
