"""Stage 5: groundedness scoring, judge parsing, eval orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from evaluation.groundedness import (
    extract_claimed_names,
    groundedness_score,
    name_is_grounded,
)
from evaluation.judge import JudgeClient
from evaluation.run_eval import evaluate_task, run_evaluation

from tests.enrichment_test_utils import make_valid_note

SOURCE = (
    "Delta Lake supports readStream with a checkpointLocation option. "
    "Time travel lets you query old versions of a Delta Lake table."
)


# -- groundedness -----------------------------------------------------------


def test_extracts_feature_parameter_and_concept_names() -> None:
    names = extract_claimed_names(make_valid_note())
    assert "readStream" in names
    assert "checkpointLocation" in names
    assert "Delta Lake" in names


def test_grounded_names_score_one() -> None:
    result = groundedness_score(make_valid_note(), SOURCE)
    assert result["score"] == 1.0
    assert result["ungrounded"] == []


def test_invented_name_is_flagged() -> None:
    note = make_valid_note()
    note["important_features_or_tools"][0]["name"] = "autoMagicLoader"
    result = groundedness_score(note, SOURCE)
    assert result["score"] < 1.0
    assert result["ungrounded"] == ["autoMagicLoader"]


def test_word_level_fallback_matches_paraphrased_names() -> None:
    assert name_is_grounded("Delta Lake time travel", SOURCE)
    assert not name_is_grounded("Unity Catalog lineage", SOURCE)


# -- judge ------------------------------------------------------------------


class FakeAnthropicMessages:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self._text)],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )


def _fake_judge(text: str) -> tuple[JudgeClient, FakeAnthropicMessages]:
    messages = FakeAnthropicMessages(text)
    client = SimpleNamespace(messages=messages)
    return JudgeClient(client=client), messages


def test_judge_returns_rubric_scores() -> None:
    judge, _ = _fake_judge(
        '{"groundedness": 5, "completeness": 4, "schema_quality": 5, '
        '"rationale": "Faithful and complete."}'
    )
    verdict = judge.score(
        source=SOURCE,
        reference_note=make_valid_note(),
        candidate_note=make_valid_note(),
    )
    assert verdict["groundedness"] == 5
    assert "error" not in verdict


def test_judge_rejects_out_of_range_scores() -> None:
    judge, _ = _fake_judge(
        '{"groundedness": 9, "completeness": 4, "schema_quality": 5, '
        '"rationale": "bad"}'
    )
    verdict = judge.score(
        source=SOURCE,
        reference_note=make_valid_note(),
        candidate_note=make_valid_note(),
    )
    assert "error" in verdict


# -- orchestration ----------------------------------------------------------


def _reference(task_id: str = "e1") -> dict:
    return {
        "task_id": task_id,
        "split": "eval",
        "source_content": SOURCE,
        "reference_note": make_valid_note(),
    }


def test_evaluate_task_all_tiers_on_valid_output() -> None:
    judge, _ = _fake_judge(
        '{"groundedness": 5, "completeness": 5, "schema_quality": 5, '
        '"rationale": "ok"}'
    )
    result = evaluate_task(
        _reference(), lambda content: json.dumps(make_valid_note()), judge=judge
    )
    assert result["json_valid"] and result["schema_valid"]
    assert result["groundedness"]["score"] == 1.0
    assert result["judge"]["completeness"] == 5


def test_evaluate_task_on_garbage_output() -> None:
    result = evaluate_task(_reference(), lambda content: "not json")
    assert not result["json_valid"]
    assert not result["schema_valid"]
    assert result["groundedness"] is None


def test_evaluate_task_sanitizes_tool_call_wrappers() -> None:
    note = make_valid_note()
    wrapped = f"</tool_call>\n\n{json.dumps(note)}"
    result = evaluate_task(_reference(), lambda content: wrapped)
    assert result["json_valid"] and result["schema_valid"]
    assert result["raw_output"] == wrapped


def test_run_evaluation_writes_metrics_and_caches(tmp_path: Path) -> None:
    references_path = tmp_path / "eval_references.jsonl"
    references_path.write_text(
        json.dumps(_reference("e1")) + "\n" + json.dumps(_reference("e2")) + "\n",
        encoding="utf-8",
    )
    calls: list[str] = []

    def generate(content: str) -> str:
        calls.append(content)
        return json.dumps(make_valid_note())

    metrics = run_evaluation(
        references_path, tmp_path / "run", generate, model_label="fake-model"
    )
    assert metrics["tasks"] == 2
    assert metrics["json_valid_rate"] == 1.0
    assert metrics["schema_valid_rate"] == 1.0
    assert metrics["mean_groundedness"] == 1.0
    assert len(calls) == 2

    # Cached results mean the generator is not called again on re-run.
    metrics_again = run_evaluation(
        references_path, tmp_path / "run", generate, model_label="fake-model"
    )
    assert len(calls) == 2
    assert metrics_again["tasks"] == 2
