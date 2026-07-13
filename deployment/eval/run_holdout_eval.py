#!/usr/bin/env python3
"""
Holdout eval against a local Ollama model (Stage 4).

Reads Alpaca-style instruction_pairs JSONL (or any JSONL with `input` / `output`),
calls Ollama /api/chat with the Stage 4 system prompt + passage as user message,
and scores valid-JSON + required-key presence.

Usage:
    python eval/run_holdout_eval.py \\
        --dataset path/to/instruction_pairs.eval.jsonl \\
        --model databricks-study-notes \\
        --system-prompt prompts/system_prompt.txt \\
        --output eval/report.json

Gate (default): --min-valid-rate 0.80
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
DEPLOY_ROOT = EVAL_DIR.parent
sys.path.insert(0, str(EVAL_DIR))

from validate_response import validate_response  # noqa: E402


def load_records(path: Path, *, split: str | None, limit: int | None) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f"[FATAL] {path}:{lineno} invalid JSON: {e}")
            if split and rec.get("split") and rec.get("split") != split:
                continue
            if not (rec.get("input") or "").strip():
                continue
            records.append(rec)
            if limit is not None and len(records) >= limit:
                break
    if not records:
        sys.exit(f"[FATAL] no records loaded from {path}")
    return records


def ollama_chat(
    *,
    base_url: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
    timeout_s: float,
) -> str:
    payload = {
        "model": model,
        "stream": False,
        "options": {"temperature": temperature},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama request failed: {e}") from e
    msg = body.get("message") or {}
    content = msg.get("content")
    if not content:
        raise RuntimeError(f"Empty Ollama response: {body!r}")
    return content


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--model", default="databricks-study-notes")
    ap.add_argument(
        "--system-prompt",
        type=Path,
        default=DEPLOY_ROOT / "prompts" / "system_prompt.txt",
    )
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    ap.add_argument("--split", default="holdout", help="Filter on record.split (empty=all)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--min-valid-rate", type=float, default=0.80)
    ap.add_argument(
        "--output",
        type=Path,
        default=EVAL_DIR / "report.json",
    )
    ap.add_argument(
        "--results",
        type=Path,
        default=EVAL_DIR / "results.jsonl",
        help="Per-example results JSONL",
    )
    args = ap.parse_args()

    if not args.system_prompt.is_file():
        sys.exit(
            f"[FATAL] Missing system prompt: {args.system_prompt}\n"
            "Run: python scripts/build_system_prompt.py"
        )
    system = args.system_prompt.read_text(encoding="utf-8").strip()
    split = args.split or None
    records = load_records(args.dataset, split=split, limit=args.limit)
    print(f"Evaluating {len(records)} records on model={args.model}")

    results: list[dict] = []
    n_valid = 0
    t0 = time.time()

    for i, rec in enumerate(records, 1):
        task_id = rec.get("task_id") or f"row-{i}"
        user = rec["input"].strip()
        gold = (rec.get("output") or "").strip()
        row: dict = {
            "task_id": task_id,
            "source_id": rec.get("source_id"),
            "split": rec.get("split"),
            "ok": False,
            "error": None,
            "latency_s": None,
        }
        started = time.time()
        try:
            raw = ollama_chat(
                base_url=args.ollama_url,
                model=args.model,
                system=system,
                user=user,
                temperature=args.temperature,
                timeout_s=args.timeout,
            )
            row["latency_s"] = round(time.time() - started, 2)
            row["raw_preview"] = raw[:500]
            obj, err = validate_response(raw)
            if err:
                row["error"] = err
            else:
                row["ok"] = True
                n_valid += 1
                row["title"] = obj.get("title") if obj else None
                if gold:
                    try:
                        gold_obj = json.loads(gold)
                        row["gold_title"] = gold_obj.get("title")
                    except json.JSONDecodeError:
                        pass
        except Exception as e:  # noqa: BLE001 — collect per-example failures
            row["latency_s"] = round(time.time() - started, 2)
            row["error"] = str(e)
        results.append(row)
        status = "OK" if row["ok"] else f"FAIL ({row['error']})"
        print(f"[{i}/{len(records)}] {task_id}: {status}")

    total = len(results)
    rate = n_valid / total if total else 0.0
    report = {
        "model": args.model,
        "dataset": str(args.dataset),
        "split": split,
        "n_total": total,
        "n_valid_json": n_valid,
        "valid_json_rate": round(rate, 4),
        "min_valid_rate": args.min_valid_rate,
        "passed_gate": rate >= args.min_valid_rate,
        "elapsed_s": round(time.time() - t0, 1),
        "failures": [
            {"task_id": r["task_id"], "error": r["error"]}
            for r in results
            if not r["ok"]
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with args.results.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    md_path = args.output.with_suffix(".md")
    md_path.write_text(
        "\n".join(
            [
                f"# Holdout eval — `{args.model}`",
                "",
                f"- Dataset: `{args.dataset}`",
                f"- Split filter: `{split}`",
                f"- Valid JSON: **{n_valid}/{total}** ({rate:.1%})",
                f"- Gate (>= {args.min_valid_rate:.0%}): "
                f"**{'PASS' if report['passed_gate'] else 'FAIL'}**",
                f"- Elapsed: {report['elapsed_s']}s",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print(f"Wrote {args.output} and {md_path}")
    if not report["passed_gate"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
