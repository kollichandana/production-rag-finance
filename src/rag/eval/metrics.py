"""Lightweight eval metrics. Doesn't require RAGAS — RAGAS bumps quality of the
faithfulness / context_precision scores but adds a heavy dependency. We compute
both: a fast harness for CI and an optional RAGAS run for the README numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rag.eval.benchmark_dataset import BenchmarkItem
from rag.schemas import QueryResponse


@dataclass
class ItemResult:
    item_id: str
    question: str
    category: str
    answer: str
    expected_substrings: list[str]
    substring_hits: int
    substring_total: int
    citation_count: int
    refused: bool
    faithfulness: float | None
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cache_read: int
    retrieved_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    label: str
    items: list[ItemResult] = field(default_factory=list)

    def aggregate(self) -> dict:
        if not self.items:
            return {"label": self.label, "n": 0}

        n = len(self.items)
        substring_hit_rate = (
            sum(r.substring_hits for r in self.items) / max(sum(r.substring_total for r in self.items), 1)
        )
        citation_rate = sum(1 for r in self.items if r.citation_count > 0) / n
        refusal_rate = sum(1 for r in self.items if r.refused) / n

        faithfulness_scores = [r.faithfulness for r in self.items if r.faithfulness is not None]
        avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else None

        oos_items = [r for r in self.items if r.category == "out_of_scope"]
        oos_correct = sum(1 for r in oos_items if r.refused or _looks_like_refusal(r.answer))
        oos_accuracy = (oos_correct / len(oos_items)) if oos_items else None

        latencies = sorted(r.latency_ms for r in self.items)
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)]

        total_input = sum(r.input_tokens for r in self.items)
        total_cache = sum(r.cache_read for r in self.items)
        cache_hit_rate = total_cache / total_input if total_input else 0.0

        return {
            "label": self.label,
            "n": n,
            "substring_recall": round(substring_hit_rate, 3),
            "citation_rate": round(citation_rate, 3),
            "refusal_rate": round(refusal_rate, 3),
            "out_of_scope_accuracy": round(oos_accuracy, 3) if oos_accuracy is not None else None,
            "avg_faithfulness": round(avg_faithfulness, 3) if avg_faithfulness is not None else None,
            "latency_p50_ms": round(p50, 1),
            "latency_p95_ms": round(p95, 1),
            "prompt_cache_read_ratio": round(cache_hit_rate, 3),
            "total_input_tokens": total_input,
            "total_output_tokens": sum(r.output_tokens for r in self.items),
        }


def _looks_like_refusal(answer: str) -> bool:
    keywords = ["don't have", "not in", "out of scope", "only answers", "cannot answer"]
    a = answer.lower()
    return any(k in a for k in keywords)


def score_item(item: BenchmarkItem, response: QueryResponse) -> ItemResult:
    answer_l = response.answer.lower()
    expected = [s for s in item.expected_substrings if s]
    hits = sum(1 for s in expected if s.lower() in answer_l)

    refused = _looks_like_refusal(response.answer)

    return ItemResult(
        item_id=item.id,
        question=item.question,
        category=item.category,
        answer=response.answer,
        expected_substrings=expected,
        substring_hits=hits,
        substring_total=len(expected) or 1,
        citation_count=len(response.citations),
        refused=refused,
        faithfulness=response.faithfulness_score,
        latency_ms=response.latency_ms,
        input_tokens=response.token_usage.get("input", 0),
        output_tokens=response.token_usage.get("output", 0),
        cache_read=response.token_usage.get("cache_read", 0),
        retrieved_chunk_ids=[rc.chunk.chunk_id for rc in response.retrieved_chunks],
    )
