"""
Validation of Teacher responses against the study-note output schema.

A response is accepted only if it parses as JSON and conforms to the
structure defined in `ingestion.study_notes_schema.STUDY_NOTES_OUTPUT_SCHEMA`.
Validation errors are returned as human-readable strings so they can be fed
back to the Teacher on retry.
"""

from __future__ import annotations

import json
from typing import Any

_SCALAR_EXAMPLE_TYPES = (str, int, float, bool)


class StudyNoteParseError(Exception):
    """Raised when a response cannot be parsed into a JSON object at all."""


def parse_study_note_response(text: str) -> dict[str, Any]:
    """
    Parse a Teacher response into a dict.

    Tolerates markdown code fences and prose around the JSON object, even
    though the prompt forbids them: we extract the outermost {...} span.
    """
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise StudyNoteParseError("response contains no JSON object")
        try:
            parsed = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise StudyNoteParseError(f"invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise StudyNoteParseError("top-level JSON value is not an object")
    return parsed


def validate_study_note(note: dict[str, Any]) -> list[str]:
    """
    Validate a parsed study note. Returns a list of errors; empty means valid.
    """
    errors: list[str] = []

    errors += _require_nonempty_str(note, "title")
    errors += _require_nonempty_str(note, "summary")

    errors += _require_list_of_objects(
        note,
        "key_concepts",
        required_str_fields=("concept", "simple_explanation", "why_it_matters"),
    )
    errors += _validate_features(note)
    errors += _validate_workflow(note)
    errors += _require_list_of_objects(
        note,
        "common_mistakes_or_confusions",
        required_str_fields=("mistake", "correction"),
    )
    errors += _validate_usage_notes(note)

    unknown = set(note) - {
        "title",
        "summary",
        "key_concepts",
        "important_features_or_tools",
        "practical_workflow",
        "common_mistakes_or_confusions",
        "project_usage_notes",
    }
    if unknown:
        errors.append(f"unknown top-level keys: {sorted(unknown)}")

    return errors


def _require_nonempty_str(note: dict[str, Any], key: str) -> list[str]:
    value = note.get(key)
    if not isinstance(value, str) or not value.strip():
        return [f"'{key}' must be a non-empty string"]
    return []


def _require_list_of_objects(
    note: dict[str, Any],
    key: str,
    *,
    required_str_fields: tuple[str, ...],
) -> list[str]:
    value = note.get(key)
    if not isinstance(value, list):
        return [f"'{key}' must be a list"]
    errors: list[str] = []
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            errors.append(f"'{key}[{i}]' must be an object")
            continue
        for field in required_str_fields:
            if not isinstance(item.get(field), str) or not item[field].strip():
                errors.append(f"'{key}[{i}].{field}' must be a non-empty string")
    return errors


def _validate_features(note: dict[str, Any]) -> list[str]:
    key = "important_features_or_tools"
    value = note.get(key)
    if not isinstance(value, list):
        return [f"'{key}' must be a list"]
    errors: list[str] = []
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            errors.append(f"'{key}[{i}]' must be an object")
            continue
        for field in ("name", "type", "what_it_does", "when_to_use_it"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                errors.append(f"'{key}[{i}].{field}' must be a non-empty string")
        params = item.get("important_parameters")
        if params is None:
            continue
        if not isinstance(params, list):
            errors.append(f"'{key}[{i}].important_parameters' must be a list")
            continue
        for j, param in enumerate(params):
            if not isinstance(param, dict):
                errors.append(
                    f"'{key}[{i}].important_parameters[{j}]' must be an object"
                )
                continue
            for field in ("parameter", "meaning"):
                if not isinstance(param.get(field), str) or not param[field].strip():
                    errors.append(
                        f"'{key}[{i}].important_parameters[{j}].{field}'"
                        " must be a non-empty string"
                    )
            example = param.get("example_value")
            if example is not None and not isinstance(
                example, _SCALAR_EXAMPLE_TYPES
            ):
                errors.append(
                    f"'{key}[{i}].important_parameters[{j}].example_value'"
                    " must be a scalar or null"
                )
    return errors


def _validate_workflow(note: dict[str, Any]) -> list[str]:
    key = "practical_workflow"
    value = note.get(key)
    if not isinstance(value, list):
        return [f"'{key}' must be a list"]
    errors: list[str] = []
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            errors.append(f"'{key}[{i}]' must be an object")
            continue
        if not isinstance(item.get("step"), int):
            errors.append(f"'{key}[{i}].step' must be an integer")
        for field in ("action", "reason"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                errors.append(f"'{key}[{i}].{field}' must be a non-empty string")
    return errors


def _validate_usage_notes(note: dict[str, Any]) -> list[str]:
    key = "project_usage_notes"
    value = note.get(key)
    if not isinstance(value, list):
        return [f"'{key}' must be a list"]
    return [
        f"'{key}[{i}]' must be a non-empty string"
        for i, item in enumerate(value)
        if not isinstance(item, str) or not item.strip()
    ]
