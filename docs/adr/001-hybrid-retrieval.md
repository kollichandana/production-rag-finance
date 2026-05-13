# ADR-001: Use hybrid (dense + sparse) retrieval with RRF, not dense-only

## Status
Accepted.

## Context
The corpus is SEC 10-K filings. Queries include both natural language ("what does the company say about supply chain risk") and exact-token lookups ("AAPL fiscal 2023 net sales"). Dense embeddings handle the former well but degrade on the latter — exact tickers and numerical strings don't always survive embedding similarity. BM25 handles the latter trivially.

## Decision
Run dense and BM25 in parallel and fuse the ranked lists with Reciprocal Rank Fusion at k=60.

## Why RRF, not weighted score fusion
Dense cosine scores live in [0, 1]; BM25 scores are unbounded. Calibrating a weighted sum requires a labeled set and is brittle across query types. RRF only consumes ranks, sidestepping the scale mismatch. Cormack et al. show k=60 is robust across many corpora.

## Consequences
- One extra dependency (`rank_bm25`).
- BM25 index is in-process; for >1M chunks we'd front it with Elasticsearch or move to Qdrant sparse vectors.
- Reranking is now necessary downstream because fused top-K is noisier than a pure dense rerank-able list.
