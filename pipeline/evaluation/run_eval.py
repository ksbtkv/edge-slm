"""
Evaluation orchestration: generate Student outputs and score all three tiers.

Generation goes through Ollama's chat API using the Canonical System Prompt,
so the model is evaluated exactly as it is deployed. Run this twice — once
against the Baseline model and once against the fine-tuned Student — and
compare the two metrics.json files (scripts/run_eval.py --compare).

Per-task results are cached in <output_dir>/outputs/, so an interrupted run
resumes where it stopped and judge scores are never re-bought.
"""

from __future__ import annotations

import json
import logging
import statistics
import urllib.request
from pathlib import Path
from typing import Any, Callable

from enrichment.study_note_validation import (
    StudyNoteParseError,
    parse_study_note_response,
    validate_study_note,
)
from evaluation.groundedness import groundedness_score
from evaluation.judge import RUBRIC_KEYS, JudgeClient
from training.canonical_prompt import CANONICAL_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434"

GenerateFn = Callable[[str], str]


def ollama_generate_fn(
    model: str, *, base_url: str = DEFAULT_OLLAMA_URL, timeout_s: float = 600.0
) -> GenerateFn:
    """Chat with a local Ollama model under the Canonical System Prompt."""

    def generate(source_content: str) -> str:
        payload = json.dumps(
            {
                "model": model,
                "stream": False,
                "options": {"temperature": 0.2},
                "messages": [
                    {"role": "system", "content": CANONICAL_SYSTEM_PROMPT},
                    {"role": "user", "content": source_content},
                ],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
        return body["message"]["content"]

    return generate


def load_references(path: str | Path) -> list[dict[str, Any]]:
    references = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                references.append(json.loads(line))
    return references


def evaluate_task(
    reference: dict[str, Any],
    generate: GenerateFn,
    *,
    judge: JudgeClient | None = None,
) -> dict[str, Any]:
    """All three tiers for one task."""
    raw = generate(reference["source_content"])
    result: dict[str, Any] = {
        "task_id": reference["task_id"],
        "raw_output": raw,
        "json_valid": False,
        "schema_valid": False,
        "schema_errors": None,
        "groundedness": None,
        "judge": None,
    }
    try:
        note = parse_study_note_response(raw)
    except StudyNoteParseError as exc:
        result["schema_errors"] = [str(exc)]
        return result
    result["json_valid"] = True
    result["note"] = note

    errors = validate_study_note(note)
    result["schema_valid"] = not errors
    result["schema_errors"] = errors or None

    result["groundedness"] = groundedness_score(
        note, reference["source_content"]
    )

    if judge is not None:
        result["judge"] = judge.score(
            source=reference["source_content"],
            reference_note=reference["reference_note"],
            candidate_note=note,
        )
    return result


def run_evaluation(
    references_path: str | Path,
    output_dir: str | Path,
    generate: GenerateFn,
    *,
    model_label: str,
    judge: JudgeClient | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Evaluate every reference task; returns aggregate metrics."""
    out = Path(output_dir)
    outputs_dir = out / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    references = load_references(references_path)
    if limit is not None:
        references = references[:limit]

    results = []
    for i, reference in enumerate(references):
        cache_path = outputs_dir / f"{reference['task_id']}.json"
        if cache_path.exists():
            results.append(json.loads(cache_path.read_text(encoding="utf-8")))
            continue
        logger.info(
            "[%d/%d] evaluating %s", i + 1, len(references), reference["task_id"]
        )
        result = evaluate_task(reference, generate, judge=judge)
        cache_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        results.append(result)

    metrics = aggregate_metrics(results, model_label=model_label)
    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def aggregate_metrics(
    results: list[dict[str, Any]], *, model_label: str
) -> dict[str, Any]:
    total = len(results)
    grounded_scores = [
        r["groundedness"]["score"] for r in results if r.get("groundedness")
    ]
    metrics: dict[str, Any] = {
        "model": model_label,
        "tasks": total,
        "json_valid_rate": _rate(results, "json_valid"),
        "schema_valid_rate": _rate(results, "schema_valid"),
        "mean_groundedness": (
            round(statistics.mean(grounded_scores), 4) if grounded_scores else None
        ),
    }
    judged = [
        r["judge"]
        for r in results
        if r.get("judge") and "error" not in r["judge"]
    ]
    metrics["judged_tasks"] = len(judged)
    for key in RUBRIC_KEYS:
        metrics[f"judge_{key}"] = (
            round(statistics.mean(v[key] for v in judged), 3) if judged else None
        )
    return metrics


def _rate(results: list[dict[str, Any]], key: str) -> float | None:
    if not results:
        return None
    return round(sum(1 for r in results if r.get(key)) / len(results), 4)
