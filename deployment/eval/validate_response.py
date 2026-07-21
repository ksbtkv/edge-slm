"""
Validate study-note JSON responses from the deployed model.

Ported from Stage 2 generate_instruction_pairs.py (extract_json / validate_shape)
so Stage 4 eval uses the same required-key contract as training-pair generation.
"""
from __future__ import annotations

import json
import re
from typing import Any

REQUIRED_KEYS = {
    "title",
    "summary",
    "key_concepts",
    "important_features_or_tools",
    "practical_workflow",
    "common_mistakes_or_confusions",
    "project_usage_notes",
}


def extract_json(raw: str) -> dict[str, Any]:
    """Parse a model response as JSON, tolerating markdown fences or commentary."""
    raw = raw.strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
        raise json.JSONDecodeError("top-level JSON is not an object", raw, 0)
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(raw[start : end + 1])

    raise json.JSONDecodeError("no JSON object found", raw, 0)


def validate_shape(obj: Any) -> str | None:
    """Return an error string if shape is invalid, else None."""
    if not isinstance(obj, dict):
        return "top-level JSON is not an object"
    missing = REQUIRED_KEYS - obj.keys()
    if missing:
        return f"missing required keys: {sorted(missing)}"
    return None


def validate_response(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    """
    Parse and validate a raw model response.

    Returns (parsed_obj, None) on success, or (None, error_message) on failure.
    """
    try:
        obj = extract_json(raw)
    except json.JSONDecodeError as e:
        return None, f"json parse error: {e}"
    err = validate_shape(obj)
    if err:
        return None, err
    return obj, None
