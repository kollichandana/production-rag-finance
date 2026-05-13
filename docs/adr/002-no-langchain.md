# ADR-002: No LangChain framework, hand-rolled pipeline

## Status
Accepted.

## Context
The default tutorial path for RAG is LangChain or LlamaIndex. Both abstract over retrieval, generation, and orchestration with high-level wrappers.

## Decision
We use the Anthropic SDK directly, the Qdrant client directly, FastEmbed directly, and orchestrate them in `rag/pipeline.py`. No framework.

## Why
- The cost of LangChain abstractions is loss of visibility into prompt construction, retry behavior, and error paths — exactly the things production breaks on.
- Token accounting, prompt caching, and fallback routing are easier to reason about when you write them yourself.
- For this project's goal (demonstrating ability to deliver production RAG), hand-rolled code shows the skill better than glued-together framework calls.

## Consequences
- More lines of code in this repo than a LangChain equivalent.
- Each feature (HyDE, decomposition, etc.) is a small module with a clear surface area.
- Tests are unit tests against real functions, not framework mocks.
