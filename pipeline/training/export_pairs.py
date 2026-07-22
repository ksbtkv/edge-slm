"""
Export Training Pairs and Teacher References from an enrichment run.

Split boundary enforcement (see CONTEXT.md):

- train split  -> train.jsonl / valid.jsonl (chat-format Training Pairs)
- eval split   -> eval_references.jsonl   (never trained on)
- holdout split -> holdout_references.jsonl (never trained on; used once)

Every Training Pair uses the Canonical System Prompt as its system message
and the raw chunk content as the user message — never the verbose Teacher
prompt (ADR 0001). `valid.jsonl` is carved deterministically out of the
train split for training-time validation loss, so eval/holdout stay unseen.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from enrichment.enrich import load_tasks
from training.canonical_prompt import CANONICAL_SYSTEM_PROMPT

DEFAULT_VAL_FRACTION = 0.05

SPLIT_FILES = {
    "eval": "eval_references.jsonl",
    "holdout": "holdout_references.jsonl",
}


class ExportError(Exception):
    """Raised when export inputs are inconsistent."""


def training_pair(source_content: str, study_note: dict[str, Any]) -> dict[str, Any]:
    """One chat-format supervised example."""
    return {
        "messages": [
            {"role": "system", "content": CANONICAL_SYSTEM_PROMPT},
            {"role": "user", "content": source_content},
            {
                "role": "assistant",
                "content": json.dumps(study_note, ensure_ascii=False),
            },
        ]
    }


def export_training_data(
    tasks_path: str | Path,
    enrichment_dir: str | Path,
    output_dir: str | Path,
    *,
    val_fraction: float = DEFAULT_VAL_FRACTION,
) -> dict[str, int]:
    """
    Build train/valid pairs and eval/holdout references.

    Returns counts per output file. Tasks without a validated note (Rejects
    or not-yet-enriched) are skipped and counted.
    """
    notes_dir = Path(enrichment_dir) / "notes"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tasks = sorted(load_tasks(tasks_path), key=lambda t: t["task_id"])
    counts = {
        "train": 0,
        "valid": 0,
        "eval": 0,
        "holdout": 0,
        "skipped_unenriched": 0,
        "skipped_other_split": 0,
    }

    handles = {
        name: open(out / f"{name}.jsonl", "w", encoding="utf-8")
        for name in ("train", "valid")
    }
    for split, filename in SPLIT_FILES.items():
        handles[split] = open(out / filename, "w", encoding="utf-8")

    try:
        for task in tasks:
            task_id = task["task_id"]
            split = task.get("split")
            note_path = notes_dir / f"{task_id}.json"
            if not note_path.exists():
                counts["skipped_unenriched"] += 1
                continue
            record = json.loads(note_path.read_text(encoding="utf-8"))
            note = record["study_note"]

            if split == "train":
                bucket = (
                    "valid" if _val_bucket(task_id, val_fraction) else "train"
                )
                line = training_pair(task["source_content"], note)
            elif split in SPLIT_FILES:
                bucket = split
                line = {
                    "task_id": task_id,
                    "split": split,
                    "source_id": task.get("source_id"),
                    "topic_bucket_ids": task.get("topic_bucket_ids"),
                    "source_content": task["source_content"],
                    "reference_note": note,
                }
            else:
                counts["skipped_other_split"] += 1
                continue

            handles[bucket].write(json.dumps(line, ensure_ascii=False) + "\n")
            counts[bucket] += 1
    finally:
        for handle in handles.values():
            handle.close()

    if counts["train"] == 0:
        raise ExportError("no training pairs produced — is the enrichment done?")

    summary = {
        **counts,
        "canonical_prompt_sha256": hashlib.sha256(
            CANONICAL_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
    }
    (out / "export_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return counts


def _val_bucket(task_id: str, val_fraction: float) -> bool:
    """Deterministic assignment of a train task to the validation file."""
    digest = hashlib.sha256(task_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < val_fraction
