"""
Enrichment orchestration: run every study-note task through the Teacher.

Layout of the enrichment output directory:

- notes/<task_id>.json    validated study note + generation metadata
- rejects.jsonl           tasks whose responses failed validation after retry
- batch_state.json        in-flight batch id (makes batch runs resumable)
- run_summary.json        counts and token usage for the last completed run

The run is resumable at task granularity: a task with a note on disk is never
re-sent, so an interrupted run never re-pays for completed work.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

from enrichment.study_note_validation import (
    StudyNoteParseError,
    parse_study_note_response,
    validate_study_note,
)
from enrichment.teacher import TeacherClient, TeacherResult

logger = logging.getLogger(__name__)

DEFAULT_REJECT_THRESHOLD = 0.05

RETRY_FEEDBACK_TEMPLATE = """{prompt}

IMPORTANT: A previous attempt at this task produced a response that failed
validation with the following errors:

{errors}

Return a corrected response as valid JSON only, fixing every error above.
"""


class EnrichmentAborted(Exception):
    """Raised when the reject rate crosses the abort threshold."""


@dataclasses.dataclass
class EnrichmentSummary:
    """Outcome of one enrichment run."""

    total_tasks: int
    already_done: int
    newly_enriched: int
    retried: int
    rejected: int
    input_tokens: int
    output_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def load_tasks(tasks_path: str | Path) -> list[dict[str, Any]]:
    """Load study-note task records from a JSONL file."""
    tasks = []
    with open(tasks_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def run_enrichment(
    tasks_path: str | Path,
    output_dir: str | Path,
    *,
    teacher: TeacherClient | None = None,
    sample: int | None = None,
    use_batch: bool = True,
    reject_threshold: float = DEFAULT_REJECT_THRESHOLD,
    poll_interval_s: float = 30.0,
) -> EnrichmentSummary:
    """
    Enrich all tasks (or the first `sample` pending ones, realtime).

    Sample mode always uses realtime calls so results are immediate.
    """
    teacher = teacher or TeacherClient()
    out = Path(output_dir)
    notes_dir = out / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(tasks_path)
    by_id = {task["task_id"]: task for task in tasks}
    done_ids = {path.stem for path in notes_dir.glob("*.json")}
    rejected_ids = _load_reject_ids(out / "rejects.jsonl")
    pending = [
        task
        for task in tasks
        if task["task_id"] not in done_ids and task["task_id"] not in rejected_ids
    ]

    if sample is not None:
        pending = pending[:sample]
        use_batch = False

    stats = {"retried": 0, "rejected": 0, "input_tokens": 0, "output_tokens": 0}
    newly_enriched = 0

    if pending:
        if use_batch:
            results = _run_batch_phase(
                teacher,
                {t["task_id"]: t["prompt"] for t in pending},
                out,
                poll_interval_s=poll_interval_s,
            )
        else:
            results = [
                teacher.generate(t["task_id"], t["prompt"]) for t in pending
            ]

        retry_errors = _process_results(results, notes_dir, out, by_id, stats)
        newly_enriched += len(results) - len(retry_errors) - _failed_count(results)

        # One retry round with validation errors fed back to the Teacher.
        api_failures = {
            r.task_id: [r.error or "API error"] for r in results if not r.ok
        }
        retry_errors.update(api_failures)
        if retry_errors:
            stats["retried"] = len(retry_errors)
            retry_prompts = {
                task_id: RETRY_FEEDBACK_TEMPLATE.format(
                    prompt=by_id[task_id]["prompt"],
                    errors="\n".join(f"- {e}" for e in errors),
                )
                for task_id, errors in retry_errors.items()
            }
            if use_batch:
                retry_results = _run_batch_phase(
                    teacher, retry_prompts, out, poll_interval_s=poll_interval_s
                )
            else:
                retry_results = [
                    teacher.generate(task_id, prompt)
                    for task_id, prompt in retry_prompts.items()
                ]
            final_errors = _process_results(
                retry_results, notes_dir, out, by_id, stats, is_retry=True
            )
            for result in retry_results:
                if not result.ok:
                    final_errors[result.task_id] = [result.error or "API error"]
            for task_id, errors in final_errors.items():
                _write_reject(out / "rejects.jsonl", by_id[task_id], errors)
                stats["rejected"] += 1
            newly_enriched += len(retry_results) - len(final_errors)

    total_rejected = stats["rejected"] + len(rejected_ids)
    if tasks and total_rejected / len(tasks) > reject_threshold:
        raise EnrichmentAborted(
            f"reject rate {total_rejected}/{len(tasks)} exceeds threshold "
            f"{reject_threshold:.0%} — inspect rejects.jsonl before re-running"
        )

    summary = EnrichmentSummary(
        total_tasks=len(tasks),
        already_done=len(done_ids),
        newly_enriched=newly_enriched,
        retried=stats["retried"],
        rejected=stats["rejected"],
        input_tokens=stats["input_tokens"],
        output_tokens=stats["output_tokens"],
    )
    (out / "run_summary.json").write_text(
        json.dumps(summary.to_dict(), indent=2), encoding="utf-8"
    )
    return summary


def _run_batch_phase(
    teacher: TeacherClient,
    prompts: dict[str, str],
    out: Path,
    *,
    poll_interval_s: float,
) -> list[TeacherResult]:
    """Submit (or resume) one batch and collect its results."""
    state_path = out / "batch_state.json"
    batch_id = None
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if set(state.get("task_ids", [])) == set(prompts):
            batch_id = state["batch_id"]
            logger.info("Resuming batch %s", batch_id)
    if batch_id is None:
        batch_id = teacher.submit_batch(prompts)
        state_path.write_text(
            json.dumps({"batch_id": batch_id, "task_ids": sorted(prompts)}),
            encoding="utf-8",
        )
        logger.info("Submitted batch %s with %d tasks", batch_id, len(prompts))

    teacher.wait_for_batch(batch_id, poll_interval_s=poll_interval_s)
    results = list(teacher.batch_results(batch_id, prompts.keys()))
    state_path.unlink(missing_ok=True)
    return results


def _process_results(
    results: list[TeacherResult],
    notes_dir: Path,
    out: Path,
    by_id: dict[str, dict[str, Any]],
    stats: dict[str, int],
    *,
    is_retry: bool = False,
) -> dict[str, list[str]]:
    """Validate results, write notes, return {task_id: errors} for failures."""
    failures: dict[str, list[str]] = {}
    for result in results:
        if not result.ok:
            continue  # API-level failures handled by the caller
        stats["input_tokens"] += result.input_tokens or 0
        stats["output_tokens"] += result.output_tokens or 0
        try:
            note = parse_study_note_response(result.text or "")
            errors = validate_study_note(note)
        except StudyNoteParseError as exc:
            errors = [str(exc)]
            note = None
        if errors:
            failures[result.task_id] = errors
            continue
        task = by_id[result.task_id]
        record = {
            "task_id": result.task_id,
            "split": task.get("split"),
            "source_id": task.get("source_id"),
            "study_note": note,
            "teacher": {
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "retried": is_retry,
            },
        }
        (notes_dir / f"{result.task_id}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return failures


def _failed_count(results: list[TeacherResult]) -> int:
    return sum(1 for r in results if not r.ok)


def _load_reject_ids(rejects_path: Path) -> set[str]:
    if not rejects_path.exists():
        return set()
    ids = set()
    with open(rejects_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["task_id"])
    return ids


def _write_reject(
    rejects_path: Path, task: dict[str, Any], errors: list[str]
) -> None:
    entry = {
        "task_id": task["task_id"],
        "split": task.get("split"),
        "source_id": task.get("source_id"),
        "errors": errors,
    }
    with open(rejects_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
