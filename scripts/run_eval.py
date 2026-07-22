#!/usr/bin/env python
"""
Evaluate a model (Baseline or fine-tuned Student) served by Ollama against
Teacher References, then compare runs.

Examples:

    # Baseline (untuned base model built with scripts/build_modelfile.py)
    PYTHONPATH=pipeline python scripts/run_eval.py run \
        data/processed/training/databricks_ld_foundations/eval_references.jsonl \
        data/processed/eval/baseline \
        --model edge-slm-baseline

    # Fine-tuned Student, with the Sonnet judge (needs ANTHROPIC_API_KEY)
    PYTHONPATH=pipeline python scripts/run_eval.py run \
        data/processed/training/databricks_ld_foundations/eval_references.jsonl \
        data/processed/eval/tuned \
        --model edge-slm-study-notes --judge

    # Side-by-side comparison
    PYTHONPATH=pipeline python scripts/run_eval.py compare \
        data/processed/eval/baseline/metrics.json \
        data/processed/eval/tuned/metrics.json

The holdout references (holdout_references.jsonl) are spent once, at the very
end — do not iterate against them.
"""

from __future__ import annotations

import argparse
import json
import logging

from evaluation.judge import DEFAULT_JUDGE_MODEL, JudgeClient
from evaluation.run_eval import (
    DEFAULT_OLLAMA_URL,
    ollama_generate_fn,
    run_evaluation,
)


def cmd_run(args: argparse.Namespace) -> int:
    judge = None
    if args.judge:
        judge = JudgeClient(model=args.judge_model)
    metrics = run_evaluation(
        args.references_jsonl,
        args.output_dir,
        ollama_generate_fn(args.model, base_url=args.ollama_url),
        model_label=args.model,
        judge=judge,
        limit=args.limit,
    )
    print(json.dumps(metrics, indent=2))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    runs = [
        json.loads(open(path, encoding="utf-8").read()) for path in args.metrics
    ]
    keys = [
        "tasks",
        "json_valid_rate",
        "schema_valid_rate",
        "mean_groundedness",
        "judged_tasks",
        "judge_groundedness",
        "judge_completeness",
        "judge_schema_quality",
    ]
    width = max(len(k) for k in keys) + 2
    header = " " * width + "  ".join(f"{run['model']:>24}" for run in runs)
    print(header)
    for key in keys:
        row = f"{key:<{width}}"
        row += "  ".join(f"{_fmt(run.get(key)):>24}" for run in runs)
        print(row)
    return 0


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Evaluate one model")
    run.add_argument("references_jsonl")
    run.add_argument("output_dir")
    run.add_argument("--model", required=True, help="Ollama model name")
    run.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    run.add_argument("--judge", action="store_true")
    run.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    run.add_argument("--limit", type=int, default=None)
    run.set_defaults(func=cmd_run)

    compare = sub.add_parser("compare", help="Compare metrics.json files")
    compare.add_argument("metrics", nargs="+")
    compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
