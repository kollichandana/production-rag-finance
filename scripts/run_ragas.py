"""Optional: run RAGAS metrics over a sample of benchmark questions.

RAGAS gives the standard faithfulness / context_precision / context_recall /
answer_relevancy scores that recruiters know. We default it OFF because it
makes its own LLM calls (slow + cost). Run on demand for the README numbers.

Usage:
    python scripts/run_ragas.py --n 20

Requires ANTHROPIC_API_KEY. Uses Claude as the eval judge via LangChain shim.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loguru import logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=15)
    parser.add_argument("--out", default="data/eval/ragas_report.json")
    args = parser.parse_args()

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as e:
        logger.error(f"RAGAS not installed: {e}")
        sys.exit(1)

    from rag.eval.benchmark_dataset import build_default_benchmark
    from rag.eval.runner import discover_available_companies
    from rag.pipeline import RAGPipeline

    companies = discover_available_companies()
    items = build_default_benchmark(companies)[: args.n]

    pipeline = RAGPipeline()

    rows = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    for item in items:
        if item.category == "out_of_scope":
            continue
        try:
            resp = pipeline.run(item.question)
        except Exception as e:
            logger.warning(f"Skipping {item.id}: {e}")
            continue
        rows["question"].append(item.question)
        rows["answer"].append(resp.answer)
        rows["contexts"].append([rc.chunk.text for rc in resp.retrieved_chunks])
        rows["ground_truth"].append(item.ground_truth or item.question)

    ds = Dataset.from_dict(rows)
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {k: float(v) for k, v in result.to_pandas().mean(numeric_only=True).to_dict().items()}
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
