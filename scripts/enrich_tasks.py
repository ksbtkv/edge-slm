#!/usr/bin/env python
"""
Run Teacher enrichment over a study_note_tasks.jsonl file.

Requires ANTHROPIC_API_KEY in the environment and `pip install -r
requirements-enrichment.txt`.

Examples:

    # Spot-check 10 tasks with realtime calls before committing to a batch
    PYTHONPATH=pipeline python scripts/enrich_tasks.py \
        data/processed/source_packs/databricks_ld_foundations/study_note_tasks.jsonl \
        data/processed/enrichment/databricks_ld_foundations \
        --sample 10

    # Full run via the Message Batches API (50% discount, resumable)
    PYTHONPATH=pipeline python scripts/enrich_tasks.py \
        data/processed/source_packs/databricks_ld_foundations/study_note_tasks.jsonl \
        data/processed/enrichment/databricks_ld_foundations
"""

from __future__ import annotations

import argparse
import logging
import sys

from enrichment.enrich import (
    DEFAULT_REJECT_THRESHOLD,
    EnrichmentAborted,
    run_enrichment,
)
from enrichment.teacher import DEFAULT_TEACHER_MODEL, TeacherClient, TeacherConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks_jsonl", help="Path to study_note_tasks.jsonl")
    parser.add_argument("output_dir", help="Enrichment output directory")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Enrich only the first N pending tasks, using realtime calls",
    )
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Use realtime calls for the full run instead of the Batch API",
    )
    parser.add_argument("--model", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument(
        "--reject-threshold",
        type=float,
        default=DEFAULT_REJECT_THRESHOLD,
        help="Abort if the reject fraction exceeds this (default 0.05)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    teacher = TeacherClient(TeacherConfig(model=args.model))
    try:
        summary = run_enrichment(
            args.tasks_jsonl,
            args.output_dir,
            teacher=teacher,
            sample=args.sample,
            use_batch=not args.no_batch,
            reject_threshold=args.reject_threshold,
        )
    except EnrichmentAborted as exc:
        print(f"ABORTED: {exc}", file=sys.stderr)
        return 1

    print(
        f"tasks={summary.total_tasks} already_done={summary.already_done} "
        f"new={summary.newly_enriched} retried={summary.retried} "
        f"rejected={summary.rejected} "
        f"tokens_in={summary.input_tokens} tokens_out={summary.output_tokens}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
