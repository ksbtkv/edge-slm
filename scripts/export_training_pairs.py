#!/usr/bin/env python
"""
Export Training Pairs and Teacher References from a completed enrichment run.

Example:

    PYTHONPATH=pipeline python scripts/export_training_pairs.py \
        data/processed/source_packs/databricks_ld_foundations/study_note_tasks.jsonl \
        data/processed/enrichment/databricks_ld_foundations \
        data/processed/training/databricks_ld_foundations
"""

from __future__ import annotations

import argparse

from training.export_pairs import DEFAULT_VAL_FRACTION, export_training_data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks_jsonl", help="Path to study_note_tasks.jsonl")
    parser.add_argument("enrichment_dir", help="Enrichment output directory")
    parser.add_argument("output_dir", help="Training data output directory")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=DEFAULT_VAL_FRACTION,
        help="Fraction of train tasks held for training-time validation loss",
    )
    args = parser.parse_args()

    counts = export_training_data(
        args.tasks_jsonl,
        args.enrichment_dir,
        args.output_dir,
        val_fraction=args.val_fraction,
    )
    print(" ".join(f"{key}={value}" for key, value in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
