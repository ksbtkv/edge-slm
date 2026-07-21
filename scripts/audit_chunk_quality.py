"""
Audit study-note task quality for undersized or header-like chunks.

Scans an existing ``study_note_tasks.jsonl`` (or a pack directory containing
one) and reports tasks whose ``chunk_word_count`` falls below a threshold.

Usage:
    PYTHONPATH=pipeline python scripts/audit_chunk_quality.py \\
        data/processed/source_packs/my_pack/study_note_tasks.jsonl

    PYTHONPATH=pipeline python scripts/audit_chunk_quality.py \\
        data/processed/source_packs/my_pack --threshold 120
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "pipeline"))

from ingestion.chunking import ChunkingConfig


def resolve_tasks_path(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "study_note_tasks.jsonl"
        if not candidate.exists():
            raise FileNotFoundError(f"No study_note_tasks.jsonl under {path}")
        return candidate
    if not path.exists():
        raise FileNotFoundError(f"Tasks file not found: {path}")
    return path


def load_tasks(path: Path) -> list[dict]:
    tasks: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            tasks.append(json.loads(line))
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        help="Path to study_note_tasks.jsonl or a pack directory containing it",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=ChunkingConfig.min_words,
        help=f"Flag chunks with fewer than this many words (default: {ChunkingConfig.min_words})",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=120,
        help="Characters of source_content to show per flagged task",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max flagged task previews to print",
    )
    args = parser.parse_args()

    tasks_path = resolve_tasks_path(args.path)
    tasks = load_tasks(tasks_path)

    if not tasks:
        print(f"No tasks in {tasks_path}")
        return 0

    flagged = [
        task
        for task in tasks
        if int(task.get("chunk_word_count") or 0) < args.threshold
    ]

    by_source: dict[str, int] = defaultdict(int)
    for task in flagged:
        by_source[task.get("source_id") or "(unknown)"] += 1

    print(f"Tasks file: {tasks_path}")
    print(f"Total tasks: {len(tasks)}")
    print(f"Below threshold ({args.threshold} words): {len(flagged)}")
    if tasks:
        pct = 100.0 * len(flagged) / len(tasks)
        print(f"Percent flagged: {pct:.1f}%")

    if by_source:
        print("\nBy source_id:")
        for source_id, count in sorted(by_source.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {source_id}: {count}")

    if flagged:
        print(f"\nSample flagged tasks (up to {args.limit}):")
        for task in flagged[: args.limit]:
            content = task.get("source_content") or ""
            preview = content.replace("\n", " ")[: args.preview_chars]
            print(
                f"  - {task.get('task_id')} "
                f"words={task.get('chunk_word_count')} "
                f"source={task.get('source_id')}"
            )
            print(f"    preview: {preview!r}")

    return 1 if flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
