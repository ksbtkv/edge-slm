#!/usr/bin/env python3
"""
Build Stage 4 system_prompt.txt from the Stage 1 study-notes schema.

Mirrors Stage 2 split_instruction(): everything before CONTENT_MARKER becomes
the fixed instruction used as the Ollama / Open WebUI system prompt. Chunk text
is supplied at inference time as the user message (Alpaca `input`).

Usage:
    python scripts/build_system_prompt.py
    python scripts/build_system_prompt.py --check   # compare to a sample Alpaca instruction
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = DEPLOY_ROOT / "prompts"
SCHEMA_PATH = PROMPTS_DIR / "study_notes_schema.py"
OUTPUT_PATH = PROMPTS_DIR / "system_prompt.txt"

CONTENT_MARKER = "Content to summarise:\n"


def _load_schema_module():
    """Import the vendored study_notes_schema without requiring PYTHONPATH."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "study_notes_schema", SCHEMA_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load schema module from {SCHEMA_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def split_instruction(prompt: str) -> str:
    """Same marker logic as Stage 2 generate_instruction_pairs.split_instruction."""
    idx = prompt.find(CONTENT_MARKER)
    if idx == -1:
        raise ValueError(
            f"CONTENT_MARKER {CONTENT_MARKER!r} not found in prompt; "
            "template format changed"
        )
    return prompt[:idx].rstrip()


def build_system_prompt() -> str:
    mod = _load_schema_module()
    # Placeholder content — we only keep the instruction half before the marker.
    full = mod.build_study_notes_prompt(content="__PLACEHOLDER__")
    instruction = split_instruction(full)
    if "__PLACEHOLDER__" in instruction:
        raise RuntimeError(
            "Placeholder leaked into instruction; CONTENT_MARKER split failed"
        )
    return instruction


def check_against_alpaca(alpaca_path: Path) -> None:
    """Verify system prompt matches the `instruction` field of a Stage 2 record."""
    line = alpaca_path.read_text(encoding="utf-8").splitlines()[0]
    rec = json.loads(line)
    expected = (rec.get("instruction") or "").strip()
    actual = build_system_prompt().strip()
    if expected != actual:
        print("MISMATCH: system prompt != Alpaca instruction", file=sys.stderr)
        print(f"  alpaca instruction length: {len(expected)}", file=sys.stderr)
        print(f"  built system prompt length: {len(actual)}", file=sys.stderr)
        # Show first differing region
        for i, (a, b) in enumerate(zip(expected, actual)):
            if a != b:
                print(f"  first diff at char {i}: {a!r} vs {b!r}", file=sys.stderr)
                print(f"  context expected: {expected[max(0,i-40):i+40]!r}", file=sys.stderr)
                print(f"  context actual:   {actual[max(0,i-40):i+40]!r}", file=sys.stderr)
                break
        else:
            print(
                f"  length/prefix mismatch (one is prefix of the other)",
                file=sys.stderr,
            )
        sys.exit(1)
    print(f"OK: system prompt matches instruction in {alpaca_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output path (default: {OUTPUT_PATH})",
    )
    ap.add_argument(
        "--check",
        type=Path,
        metavar="ALPACA_JSONL",
        help="Compare built prompt to first record's instruction field",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompt to stdout instead of writing a file",
    )
    args = ap.parse_args()

    if not SCHEMA_PATH.is_file():
        sys.exit(f"[FATAL] Missing vendored schema: {SCHEMA_PATH}")

    prompt = build_system_prompt()

    if args.check:
        if not args.check.is_file():
            sys.exit(f"[FATAL] Alpaca file not found: {args.check}")
        check_against_alpaca(args.check)

    if args.dry_run:
        print(prompt)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(prompt + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({len(prompt)} chars)")


if __name__ == "__main__":
    main()
