"""Hand-curated benchmark questions for the 10-K corpus.

Each item carries ground-truth answer expectations + which doc(s) should be
retrieved + the category. We deliberately include hard cases: multi-hop,
table lookups, out-of-scope, and numerical precision tests.

This dataset is the contract our evals are graded against. Keep it small
(~30-60 items) so we can run it on every PR. Quality > quantity.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BenchmarkItem:
    id: str
    question: str
    expected_substrings: list[str] = field(default_factory=list)
    expected_companies: list[str] = field(default_factory=list)
    category: str = "factual"  # factual | comparative | analytical | summary | out_of_scope
    difficulty: str = "medium"  # easy | medium | hard
    ground_truth: str | None = None
    notes: str | None = None


# Default benchmark — populated dynamically with whatever filings are ingested.
# The harness builds a richer dataset by reading the corpus, but these
# corpus-agnostic checks always run.
GENERIC_BENCHMARK: list[BenchmarkItem] = [
    BenchmarkItem(
        id="oos-weather",
        question="What's the weather today in San Francisco?",
        category="out_of_scope",
        difficulty="easy",
        expected_substrings=["only", "filings", "scope"],
    ),
    BenchmarkItem(
        id="oos-recipe",
        question="Give me a recipe for chocolate cake.",
        category="out_of_scope",
        difficulty="easy",
        expected_substrings=["only", "filings"],
    ),
    BenchmarkItem(
        id="refusal-not-in-corpus",
        question="What was the population of Mars in 2050?",
        category="out_of_scope",
        difficulty="easy",
        expected_substrings=["don't have", "not", "scope", "filings"],
    ),
]


# Concrete questions for the default Big-Tech 10-K corpus we ship.
# These are templated against the companies most likely to be ingested.
APPLE_BENCHMARK: list[BenchmarkItem] = [
    BenchmarkItem(
        id="apple-revenue",
        question="What was Apple's total net sales in fiscal 2023?",
        expected_companies=["Apple"],
        expected_substrings=["$", "billion"],
        category="factual",
        difficulty="easy",
    ),
    BenchmarkItem(
        id="apple-segments",
        question="What are Apple's reportable operating segments?",
        expected_companies=["Apple"],
        expected_substrings=["Americas", "Europe", "Greater China", "Japan", "Rest of Asia Pacific"],
        category="factual",
        difficulty="easy",
    ),
    BenchmarkItem(
        id="apple-risk",
        question="What are the principal risk factors Apple discloses related to supply chain?",
        expected_companies=["Apple"],
        expected_substrings=["supplier", "concentration"],
        category="analytical",
        difficulty="medium",
    ),
]

MICROSOFT_BENCHMARK: list[BenchmarkItem] = [
    BenchmarkItem(
        id="msft-segments",
        question="What are Microsoft's three reportable operating segments?",
        expected_companies=["Microsoft"],
        expected_substrings=["Productivity", "Intelligent Cloud", "Personal Computing"],
        category="factual",
        difficulty="easy",
    ),
    BenchmarkItem(
        id="msft-rnd",
        question="How much did Microsoft spend on research and development in fiscal 2023?",
        expected_companies=["Microsoft"],
        expected_substrings=["$", "billion", "research"],
        category="factual",
        difficulty="medium",
    ),
]

COMPARATIVE_BENCHMARK: list[BenchmarkItem] = [
    BenchmarkItem(
        id="cmp-rnd-apple-msft",
        question="Compare R&D spending between Apple and Microsoft in their most recent fiscal year.",
        expected_companies=["Apple", "Microsoft"],
        expected_substrings=["research", "development", "$"],
        category="comparative",
        difficulty="hard",
    ),
]


def build_default_benchmark(available_companies: set[str]) -> list[BenchmarkItem]:
    """Return only benchmark items whose required companies are present."""
    items = list(GENERIC_BENCHMARK)
    pools = [APPLE_BENCHMARK, MICROSOFT_BENCHMARK, COMPARATIVE_BENCHMARK]
    for pool in pools:
        for item in pool:
            if all(c in available_companies or any(c.lower() in a.lower() for a in available_companies) for c in item.expected_companies):
                items.append(item)
    return items
