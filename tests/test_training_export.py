"""Stage 3: training-pair export, split enforcement, prompt contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from training.canonical_prompt import CANONICAL_SYSTEM_PROMPT
from training.export_pairs import ExportError, export_training_data, training_pair

from tests.enrichment_test_utils import make_task, make_valid_note, write_tasks

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _prepare(tmp_path: Path, tasks: list[dict], enriched_ids: set[str]) -> tuple:
    tasks_path = write_tasks(tmp_path / "tasks.jsonl", tasks)
    notes_dir = tmp_path / "enrichment/notes"
    notes_dir.mkdir(parents=True)
    for task in tasks:
        if task["task_id"] in enriched_ids:
            (notes_dir / f"{task['task_id']}.json").write_text(
                json.dumps(
                    {
                        "task_id": task["task_id"],
                        "split": task["split"],
                        "study_note": make_valid_note(title=task["task_id"]),
                    }
                ),
                encoding="utf-8",
            )
    return tasks_path, tmp_path / "enrichment"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_split_boundary_is_enforced(tmp_path: Path) -> None:
    tasks = [
        make_task("train_1", split="train"),
        make_task("train_2", split="train"),
        make_task("eval_1", split="eval"),
        make_task("holdout_1", split="holdout"),
    ]
    tasks_path, enrichment_dir = _prepare(
        tmp_path, tasks, {t["task_id"] for t in tasks}
    )
    out = tmp_path / "training"

    counts = export_training_data(
        tasks_path, enrichment_dir, out, val_fraction=0.0
    )

    assert counts["train"] == 2
    assert counts["eval"] == 1
    assert counts["holdout"] == 1

    trained_titles = {
        json.loads(pair["messages"][2]["content"])["title"]
        for pair in _read_jsonl(out / "train.jsonl")
    }
    assert trained_titles == {"train_1", "train_2"}

    eval_refs = _read_jsonl(out / "eval_references.jsonl")
    assert [ref["task_id"] for ref in eval_refs] == ["eval_1"]
    assert "messages" not in eval_refs[0]


def test_training_pair_uses_canonical_prompt_and_raw_content() -> None:
    pair = training_pair("some chunk content", make_valid_note())
    system, user, assistant = pair["messages"]
    assert system == {"role": "system", "content": CANONICAL_SYSTEM_PROMPT}
    assert user == {"role": "user", "content": "some chunk content"}
    assert json.loads(assistant["content"])["title"] == "Delta Lake basics"
    # The verbose Teacher prompt must never leak into training input.
    assert "output_schema" not in system["content"]
    assert "training data" not in system["content"]


def test_unenriched_tasks_are_skipped_and_counted(tmp_path: Path) -> None:
    tasks = [make_task("t1"), make_task("t2")]
    tasks_path, enrichment_dir = _prepare(tmp_path, tasks, {"t1"})

    counts = export_training_data(
        tasks_path, enrichment_dir, tmp_path / "out", val_fraction=0.0
    )

    assert counts["train"] == 1
    assert counts["skipped_unenriched"] == 1


def test_val_bucket_is_deterministic(tmp_path: Path) -> None:
    tasks = [make_task(f"t{i}") for i in range(40)]
    tasks_path, enrichment_dir = _prepare(
        tmp_path, tasks, {t["task_id"] for t in tasks}
    )

    counts_a = export_training_data(
        tasks_path, enrichment_dir, tmp_path / "a", val_fraction=0.2
    )
    counts_b = export_training_data(
        tasks_path, enrichment_dir, tmp_path / "b", val_fraction=0.2
    )

    assert counts_a == counts_b
    assert counts_a["valid"] > 0
    assert counts_a["train"] + counts_a["valid"] == 40
    assert _read_jsonl(tmp_path / "a/valid.jsonl") == _read_jsonl(
        tmp_path / "b/valid.jsonl"
    )


def test_export_fails_without_training_pairs(tmp_path: Path) -> None:
    tasks = [make_task("e1", split="eval")]
    tasks_path, enrichment_dir = _prepare(tmp_path, tasks, {"e1"})

    with pytest.raises(ExportError):
        export_training_data(tasks_path, enrichment_dir, tmp_path / "out")


def test_generated_modelfile_embeds_canonical_prompt_verbatim() -> None:
    """ADR 0001: training system prompt and Modelfile SYSTEM are byte-identical."""
    spec = importlib.util.spec_from_file_location(
        "build_modelfile", PROJECT_ROOT / "scripts/build_modelfile.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rendered = module.MODELFILE_TEMPLATE.format(
        from_ref="model.gguf", system_prompt=CANONICAL_SYSTEM_PROMPT
    )
    assert f'SYSTEM """\n{CANONICAL_SYSTEM_PROMPT}\n"""' in rendered
