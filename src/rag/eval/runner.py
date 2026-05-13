"""Run a benchmark against one or more pipeline configurations and emit a report."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from rag.eval.benchmark_dataset import BenchmarkItem, build_default_benchmark
from rag.eval.metrics import BenchmarkResult, score_item
from rag.pipeline import RAGPipeline
from rag.retrieval.vector_store import VectorStore
from rag.settings import get_settings


def discover_available_companies(store: VectorStore | None = None) -> set[str]:
    store = store or VectorStore()
    chunks = store.get_all_chunks()
    return {c.metadata.get("company") for c in chunks if c.metadata.get("company")}


def run_benchmark(
    pipeline: RAGPipeline,
    items: list[BenchmarkItem],
    label: str,
) -> BenchmarkResult:
    result = BenchmarkResult(label=label)
    for item in tqdm(items, desc=f"Eval [{label}]"):
        try:
            response = pipeline.run(item.question)
            result.items.append(score_item(item, response))
        except Exception as e:
            logger.exception(f"Item {item.id} failed: {e}")
    return result


def run_comparison(
    items: list[BenchmarkItem] | None = None,
    output_path: Path | None = None,
) -> dict:
    s = get_settings()
    store = VectorStore()
    if store.count() == 0:
        raise RuntimeError("No chunks in vector store. Run ingestion first.")

    if items is None:
        companies = discover_available_companies(store)
        items = build_default_benchmark(companies)

    production = RAGPipeline()
    naive = RAGPipeline.naive(production.retriever)

    prod_result = run_benchmark(production, items, "production")
    naive_result = run_benchmark(naive, items, "naive")

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "n_items": len(items),
        "naive": naive_result.aggregate(),
        "production": prod_result.aggregate(),
        "items": {
            "naive": [r.__dict__ for r in naive_result.items],
            "production": [r.__dict__ for r in prod_result.items],
        },
    }

    output_path = output_path or (s.eval_data_dir / "latest_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    logger.info(f"Saved eval report to {output_path}")

    return report


def render_summary_markdown(report: dict) -> str:
    n = report["naive"]
    p = report["production"]

    def cell(d: dict, key: str) -> str:
        v = d.get(key)
        return "—" if v is None else str(v)

    rows = [
        ("Items", "n"),
        ("Substring recall (↑)", "substring_recall"),
        ("Citation rate (↑)", "citation_rate"),
        ("Avg faithfulness (↑)", "avg_faithfulness"),
        ("Out-of-scope accuracy (↑)", "out_of_scope_accuracy"),
        ("Refusal rate", "refusal_rate"),
        ("Latency p50 (ms)", "latency_p50_ms"),
        ("Latency p95 (ms)", "latency_p95_ms"),
        ("Prompt cache ratio (↑)", "prompt_cache_read_ratio"),
        ("Total input tokens", "total_input_tokens"),
        ("Total output tokens", "total_output_tokens"),
    ]

    md = "# Benchmark: naive vs production\n\n"
    md += f"Run: {report['timestamp']}\n\n"
    md += "| Metric | Naive RAG | Production RAG |\n|---|---|---|\n"
    for label, key in rows:
        md += f"| {label} | {cell(n, key)} | {cell(p, key)} |\n"
    return md
