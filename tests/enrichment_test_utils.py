"""Shared fakes and fixtures for enrichment/training/evaluation tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from enrichment.teacher import TeacherResult, batch_custom_id


def make_valid_note(**overrides: Any) -> dict[str, Any]:
    """A minimal study note that passes schema validation."""
    note = {
        "title": "Delta Lake basics",
        "summary": "Delta Lake adds ACID transactions to data lakes.",
        "key_concepts": [
            {
                "concept": "Delta Lake",
                "simple_explanation": "A storage layer with transactions.",
                "why_it_matters": "Reliable tables in a real project.",
            }
        ],
        "important_features_or_tools": [
            {
                "name": "readStream",
                "type": "API",
                "what_it_does": "Reads a table as a stream.",
                "when_to_use_it": "Incremental processing.",
                "important_parameters": [
                    {
                        "parameter": "checkpointLocation",
                        "meaning": "Where progress is stored.",
                        "example_value": "/tmp/checkpoint",
                    }
                ],
            }
        ],
        "practical_workflow": [
            {"step": 1, "action": "Create a table.", "reason": "Storage first."}
        ],
        "common_mistakes_or_confusions": [
            {"mistake": "No checkpoint.", "correction": "Always set one."}
        ],
        "project_usage_notes": ["Use Delta tables for pipeline outputs."],
    }
    note.update(overrides)
    return note


def make_task(task_id: str, split: str = "train", **overrides: Any) -> dict[str, Any]:
    task = {
        "task_id": task_id,
        "split": split,
        "source_id": f"src_{task_id}",
        "topic_bucket_ids": ["delta_lake"],
        "prompt": f"PROMPT for {task_id}",
        "source_content": (
            "Delta Lake supports readStream with a checkpointLocation option."
        ),
    }
    task.update(overrides)
    return task


def write_tasks(path: Path, tasks: list[dict[str, Any]]) -> Path:
    path.write_text(
        "".join(json.dumps(task) + "\n" for task in tasks), encoding="utf-8"
    )
    return path


class FakeTeacher:
    """
    Duck-typed TeacherClient: responses are scripted per task_id as a list,
    consumed one per call (so retries get the next scripted response).
    """

    def __init__(self, responses: dict[str, list[str | Exception]]) -> None:
        self.responses = {k: list(v) for k, v in responses.items()}
        self.generate_calls: list[str] = []
        self.batches: dict[str, dict[str, str]] = {}
        self._batch_counter = 0

    def _next(self, task_id: str) -> TeacherResult:
        script = self.responses.get(task_id)
        if not script:
            return TeacherResult(task_id=task_id, ok=False, error="unscripted")
        item = script.pop(0)
        if isinstance(item, Exception):
            return TeacherResult(task_id=task_id, ok=False, error=str(item))
        return TeacherResult(
            task_id=task_id, ok=True, text=item, input_tokens=10, output_tokens=20
        )

    def generate(self, task_id: str, prompt: str) -> TeacherResult:
        self.generate_calls.append(task_id)
        return self._next(task_id)

    def submit_batch(self, prompts: dict[str, str]) -> str:
        self._batch_counter += 1
        batch_id = f"batch_{self._batch_counter}"
        # Stored keyed by hashed custom_id, mirroring the real wire format.
        self.batches[batch_id] = {
            batch_custom_id(task_id): prompt for task_id, prompt in prompts.items()
        }
        return batch_id

    def wait_for_batch(self, batch_id: str, **_: Any) -> None:
        pass

    def batch_results(self, batch_id: str, task_ids: Iterable[str]):
        reverse = {batch_custom_id(task_id): task_id for task_id in task_ids}
        for custom_id in self.batches[batch_id]:
            yield self._next(reverse[custom_id])
