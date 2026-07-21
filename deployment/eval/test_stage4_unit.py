#!/usr/bin/env python3
"""Lightweight unit checks for Stage 4 prompt + response validation (no Ollama)."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

DEPLOY = Path(__file__).resolve().parents[1]
EVAL = DEPLOY / "eval"
SCRIPTS = DEPLOY / "scripts"
sys.path.insert(0, str(EVAL))

from validate_response import REQUIRED_KEYS, extract_json, validate_response, validate_shape


def test_validate_shape_accepts_required_keys() -> None:
    obj = {k: [] if k != "title" and k != "summary" else "x" for k in REQUIRED_KEYS}
    assert validate_shape(obj) is None


def test_validate_shape_rejects_missing() -> None:
    err = validate_shape({"title": "t"})
    assert err and "missing" in err


def test_extract_json_tolerates_fences() -> None:
    raw = 'Here you go:\n```json\n{"title": "T", "summary": "S", "key_concepts": [], "important_features_or_tools": [], "practical_workflow": [], "common_mistakes_or_confusions": [], "project_usage_notes": []}\n```\n'
    obj = extract_json(raw)
    assert obj["title"] == "T"
    parsed, err = validate_response(raw)
    assert err is None and parsed is not None


def test_build_system_prompt_matches_content_marker_split() -> None:
    schema_path = DEPLOY / "prompts" / "study_notes_schema.py"
    spec = importlib.util.spec_from_file_location("sns", schema_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    content_marker = "Content to summarise:\n"
    full = mod.build_study_notes_prompt(content="CHUNK_TEXT_XYZ")
    instruction = full[: full.find(content_marker)].rstrip()

    out = subprocess.check_output(
        [sys.executable, str(SCRIPTS / "build_system_prompt.py"), "--dry-run"],
        cwd=str(DEPLOY),
        text=True,
    ).rstrip()
    assert out == instruction
    assert "CHUNK_TEXT_XYZ" not in out


def test_build_system_prompt_check_against_alpaca() -> None:
    # Synthesize a Stage 2-style Alpaca line from the vendored schema
    schema_path = DEPLOY / "prompts" / "study_notes_schema.py"
    spec = importlib.util.spec_from_file_location("sns", schema_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    content_marker = "Content to summarise:\n"
    full = mod.build_study_notes_prompt(content="hello world")
    instruction = full[: full.find(content_marker)].rstrip()
    rec = {
        "instruction": instruction,
        "input": "hello world",
        "output": json.dumps({"title": "t"}),
    }
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        fh.write(json.dumps(rec) + "\n")
        path = fh.name
    try:
        subprocess.check_call(
            [
                sys.executable,
                str(SCRIPTS / "build_system_prompt.py"),
                "--check",
                path,
                "--dry-run",
            ],
            cwd=str(DEPLOY),
        )
    finally:
        Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    tests = [
        test_validate_shape_accepts_required_keys,
        test_validate_shape_rejects_missing,
        test_extract_json_tolerates_fences,
        test_build_system_prompt_matches_content_marker_split,
        test_build_system_prompt_check_against_alpaca,
    ]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"All {len(tests)} checks passed.")
