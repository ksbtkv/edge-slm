"""Stage 2: study-note validation and enrichment orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from enrichment.enrich import EnrichmentAborted, run_enrichment
from enrichment.teacher import batch_custom_id
from enrichment.study_note_validation import (
    StudyNoteParseError,
    parse_study_note_response,
    validate_study_note,
)

from tests.enrichment_test_utils import (
    FakeTeacher,
    make_task,
    make_valid_note,
    write_tasks,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLD_EXAMPLE = (
    PROJECT_ROOT / "data/manifests/examples/study_note_example_delta_streaming.json"
)


# -- validation -------------------------------------------------------------


def test_gold_example_passes_validation() -> None:
    note = json.loads(GOLD_EXAMPLE.read_text(encoding="utf-8"))
    assert validate_study_note(note) == []


def test_valid_note_passes() -> None:
    assert validate_study_note(make_valid_note()) == []


def test_missing_title_and_bad_types_are_reported() -> None:
    note = make_valid_note(title="", key_concepts="not a list")
    errors = validate_study_note(note)
    assert any("'title'" in error for error in errors)
    assert any("'key_concepts' must be a list" in error for error in errors)


def test_unknown_top_level_keys_are_reported() -> None:
    errors = validate_study_note(make_valid_note(extra_key=1))
    assert any("unknown top-level keys" in error for error in errors)


def test_parse_tolerates_markdown_fences() -> None:
    note = make_valid_note()
    wrapped = f"Here you go:\n```json\n{json.dumps(note)}\n```"
    assert parse_study_note_response(wrapped) == note


def test_parse_strips_tool_call_wrappers() -> None:
    note = make_valid_note()
    payload = json.dumps(note)
    wrapped = f"</tool_call>\n\n<tool_call>\n\n{payload}\n</tool_call>"
    assert parse_study_note_response(wrapped) == note


def test_parse_strips_tool_call_around_fenced_json() -> None:
    note = make_valid_note()
    wrapped = (
        "<tool_call>\njunk\n</tool_call>\n"
        f"```json\n{json.dumps(note)}\n```"
    )
    assert parse_study_note_response(wrapped) == note


def test_parse_rejects_non_json() -> None:
    with pytest.raises(StudyNoteParseError):
        parse_study_note_response("no json here at all")


# -- orchestration ----------------------------------------------------------


def _note_json(**overrides: object) -> str:
    return json.dumps(make_valid_note(**overrides))


def test_realtime_run_writes_notes(tmp_path: Path) -> None:
    tasks_path = write_tasks(
        tmp_path / "tasks.jsonl", [make_task("t1"), make_task("t2", split="eval")]
    )
    teacher = FakeTeacher({"t1": [_note_json()], "t2": [_note_json()]})

    summary = run_enrichment(
        tasks_path, tmp_path / "out", teacher=teacher, use_batch=False
    )

    assert summary.newly_enriched == 2
    assert summary.rejected == 0
    record = json.loads(
        (tmp_path / "out/notes/t2.json").read_text(encoding="utf-8")
    )
    assert record["split"] == "eval"
    assert validate_study_note(record["study_note"]) == []


def test_invalid_response_is_retried_with_error_feedback(tmp_path: Path) -> None:
    tasks_path = write_tasks(tmp_path / "tasks.jsonl", [make_task("t1")])
    teacher = FakeTeacher({"t1": ['{"title": "only a title"}', _note_json()]})

    summary = run_enrichment(
        tasks_path, tmp_path / "out", teacher=teacher, use_batch=False
    )

    assert summary.retried == 1
    assert summary.rejected == 0
    assert (tmp_path / "out/notes/t1.json").exists()


def test_double_failure_becomes_reject_and_can_abort(tmp_path: Path) -> None:
    tasks_path = write_tasks(tmp_path / "tasks.jsonl", [make_task("t1")])
    teacher = FakeTeacher({"t1": ["not json", "still not json"]})

    with pytest.raises(EnrichmentAborted):
        run_enrichment(tasks_path, tmp_path / "out", teacher=teacher, use_batch=False)

    rejects = (tmp_path / "out/rejects.jsonl").read_text(encoding="utf-8")
    assert json.loads(rejects)["task_id"] == "t1"


def test_completed_tasks_are_never_resent(tmp_path: Path) -> None:
    tasks_path = write_tasks(
        tmp_path / "tasks.jsonl", [make_task("t1"), make_task("t2")]
    )
    teacher = FakeTeacher({"t1": [_note_json()], "t2": [_note_json()]})
    run_enrichment(tasks_path, tmp_path / "out", teacher=teacher, use_batch=False)

    resumed = FakeTeacher({})
    summary = run_enrichment(
        tasks_path, tmp_path / "out", teacher=resumed, use_batch=False
    )

    assert resumed.generate_calls == []
    assert summary.already_done == 2
    assert summary.newly_enriched == 0


def test_sample_mode_limits_to_first_pending(tmp_path: Path) -> None:
    tasks_path = write_tasks(
        tmp_path / "tasks.jsonl", [make_task("t1"), make_task("t2")]
    )
    teacher = FakeTeacher({"t1": [_note_json()]})

    summary = run_enrichment(
        tasks_path, tmp_path / "out", teacher=teacher, sample=1
    )

    assert teacher.generate_calls == ["t1"]
    assert summary.newly_enriched == 1


def test_batch_run_processes_and_cleans_state(tmp_path: Path) -> None:
    tasks_path = write_tasks(
        tmp_path / "tasks.jsonl", [make_task("t1"), make_task("t2")]
    )
    teacher = FakeTeacher({"t1": [_note_json()], "t2": [_note_json()]})

    summary = run_enrichment(
        tasks_path, tmp_path / "out", teacher=teacher, use_batch=True
    )

    assert summary.newly_enriched == 2
    assert not (tmp_path / "out/batch_state.json").exists()
    assert len(teacher.batches) == 1


def test_batch_custom_id_fits_api_limit_for_long_task_ids() -> None:
    """The Batches API rejects custom_id over 64 chars; pack task ids reach ~98."""
    long_id = (
        "databricks_ld_foundations__"
        "doc_data_governance_unity_catalog_manage_privileges__c0003"
    )
    assert len(long_id) > 64
    custom_id = batch_custom_id(long_id)
    assert len(custom_id) <= 64
    assert custom_id == batch_custom_id(long_id)  # deterministic
    assert custom_id != batch_custom_id(long_id + "x")


def test_batch_round_trip_with_long_task_ids(tmp_path: Path) -> None:
    long_id = "x" * 98
    tasks_path = write_tasks(tmp_path / "tasks.jsonl", [make_task(long_id)])
    teacher = FakeTeacher({long_id: [_note_json()]})

    summary = run_enrichment(
        tasks_path, tmp_path / "out", teacher=teacher, use_batch=True
    )

    assert summary.newly_enriched == 1
    assert (tmp_path / f"out/notes/{long_id}.json").exists()


def test_api_failure_goes_through_retry_round(tmp_path: Path) -> None:
    tasks_path = write_tasks(tmp_path / "tasks.jsonl", [make_task("t1")])
    teacher = FakeTeacher({"t1": [RuntimeError("rate limited"), _note_json()]})

    summary = run_enrichment(
        tasks_path, tmp_path / "out", teacher=teacher, use_batch=False
    )

    assert summary.retried == 1
    assert (tmp_path / "out/notes/t1.json").exists()
