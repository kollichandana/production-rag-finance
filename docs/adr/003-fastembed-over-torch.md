# ADR-003: FastEmbed (ONNX) over sentence-transformers (PyTorch)

## Status
Accepted.

## Context
We need an embedding model and a cross-encoder reranker. Both have established PyTorch implementations via `sentence-transformers`.

## Decision
Use FastEmbed for both, which runs ONNX-quantized versions of the same models.

## Why
- ~5× smaller container image (no torch). This matters for Streamlit Community Cloud, where the working-set ceiling is 1 GB.
- Cold starts are seconds, not minutes.
- The model files are downloaded on first use and cached — no separate build step.
- Identical model weights to `sentence-transformers/all-MiniLM` / `BAAI/bge-small`, so quality is the same.

## Consequences
- We're tied to whatever models FastEmbed packages. If we ever need a custom or fine-tuned model that isn't ONNX-exported, we'd have to add the torch path.
- ONNX inference is CPU-only in FastEmbed (no GPU acceleration on CPU instances) — fine for the scale here.
