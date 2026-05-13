"""Run the benchmark and write a markdown summary to docs/benchmarks.md."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag.eval.runner import render_summary_markdown, run_comparison
from rag.settings import get_settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compare-naive", action="store_true", help="Run both naive and production pipelines"
    )
    parser.add_argument(
        "--out-json", default=None, help="Output path for the JSON report"
    )
    args = parser.parse_args()

    s = get_settings()
    out = Path(args.out_json) if args.out_json else s.eval_data_dir / "latest_report.json"
    report = run_comparison(output_path=out)
    print(json.dumps({"naive": report["naive"], "production": report["production"]}, indent=2))

    md = render_summary_markdown(report)
    md_path = Path(__file__).resolve().parent.parent / "docs" / "benchmarks.md"
    md_path.parent.mkdir(exist_ok=True, parents=True)
    md_path.write_text(md)
    print(f"\nWrote benchmark summary to {md_path}")


if __name__ == "__main__":
    main()
