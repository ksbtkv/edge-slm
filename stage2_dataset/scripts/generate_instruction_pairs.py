"""
Stage 2: LLM enrichment of study_note_tasks.jsonl into Alpaca-style instruction pairs.

Teacher model runs via vLLM batch inference (tensor-parallel across the Ohm
GPU allocation). Each task's `prompt` field already embeds the fixed
instruction template + the chunk content (`source_content`) + the target JSON
schema, so it is sent to the model as-is. The stored Alpaca record splits
that back into:

    instruction = fixed template (identical across all tasks)
    input       = source_content (the chunk text)
    output      = the model's validated JSON response

Usage (single node, N GPUs):

    python generate_instruction_pairs.py \
        --input data/study_note_tasks.jsonl \
        --output data/instruction_pairs.jsonl \
        --failures data/failures.jsonl \
        --model meta-llama/Llama-3.3-70B-Instruct \
        --tensor-parallel-size 4

Requires HF_TOKEN in the environment if the model repo is gated.
"""

import argparse
import json
import re
import sys
from pathlib import Path

CONTENT_MARKER = "Content to summarise:\n"

REQUIRED_KEYS = {
    "title",
    "summary",
    "key_concepts",
    "important_features_or_tools",
    "practical_workflow",
    "common_mistakes_or_confusions",
    "project_usage_notes",
}

REPAIR_TEMPLATE = (
    "Your previous response was not valid JSON matching the required schema.\n"
    "Error: {error}\n\n"
    "Return ONLY the corrected JSON object for this schema and content. "
    "No markdown fences, no commentary.\n\n"
    "Schema:\n{schema}\n\nContent:\n{content}"
)


def load_tasks(path: Path) -> list[dict]:
    tasks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def split_instruction(prompt: str) -> str:
    """Return the fixed template portion of a task's prompt (everything
    before the embedded chunk content)."""
    idx = prompt.find(CONTENT_MARKER)
    if idx == -1:
        raise ValueError("CONTENT_MARKER not found in prompt; template format changed")
    return prompt[:idx].rstrip()


def extract_json(raw: str) -> dict:
    """Parse a model response as JSON, tolerating markdown fences or leading
    or trailing commentary around a single JSON object."""
    raw = raw.strip()
    try:
        return json.loads(raw)
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


def validate_shape(obj: dict) -> str | None:
    """Cheap structural validation only (required top-level keys present).
    Deeper groundedness/hallucination checking against source_content is a
    separate, not-yet-built pass -- see stage2_dataset/README.md."""
    if not isinstance(obj, dict):
        return "top-level JSON is not an object"
    missing = REQUIRED_KEYS - obj.keys()
    if missing:
        return f"missing required keys: {sorted(missing)}"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--failures", required=True, type=Path)
    ap.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    ap.add_argument("--tensor-parallel-size", type=int, default=4)
    ap.add_argument("--max-repair-attempts", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=None, help="process only the first N tasks (smoke test)")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams  # deferred: heavy import, HPC-only dep

    tasks = load_tasks(args.input)
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"Loaded {len(tasks)} tasks from {args.input}", file=sys.stderr)

    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size)
    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    prompts = [t["prompt"] for t in tasks]
    outputs = llm.generate(prompts, sampling)
    raw_by_task = [o.outputs[0].text for o in outputs]
    last_raw_by_task = list(raw_by_task)  # updated as repair passes run

    results: dict[str, dict] = {}
    pending_repair: list[tuple[int, str]] = []  # (task_index, error)

    for i, (task, raw) in enumerate(zip(tasks, raw_by_task)):
        try:
            obj = extract_json(raw)
            err = validate_shape(obj)
            if err:
                pending_repair.append((i, err))
                continue
            results[task["task_id"]] = obj
        except json.JSONDecodeError as e:
            pending_repair.append((i, str(e)))

    print(f"First pass: {len(results)} valid, {len(pending_repair)} need repair", file=sys.stderr)

    for attempt in range(1, args.max_repair_attempts + 1):
        if not pending_repair:
            break
        repair_prompts = []
        for i, error in pending_repair:
            task = tasks[i]
            schema_str = json.dumps(task["expected_output_schema"], indent=2)
            repair_prompts.append(
                REPAIR_TEMPLATE.format(error=error, schema=schema_str, content=task["source_content"])
            )
        repair_outputs = llm.generate(repair_prompts, sampling)
        still_pending = []
        for (i, _), out in zip(pending_repair, repair_outputs):
            raw = out.outputs[0].text
            last_raw_by_task[i] = raw
            task = tasks[i]
            try:
                obj = extract_json(raw)
                err = validate_shape(obj)
                if err:
                    still_pending.append((i, err))
                    continue
                results[task["task_id"]] = obj
            except json.JSONDecodeError as e:
                still_pending.append((i, str(e)))
        print(f"Repair pass {attempt}: {len(pending_repair) - len(still_pending)} recovered, {len(still_pending)} still failing", file=sys.stderr)
        pending_repair = still_pending

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.failures.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(args.output, "w", encoding="utf-8") as out_f:
        for task in tasks:
            obj = results.get(task["task_id"])
            if obj is None:
                continue
            record = {
                "instruction": split_instruction(task["prompt"]),
                "input": task["source_content"],
                "output": json.dumps(obj, ensure_ascii=False),
                "task_id": task["task_id"],
                "source_id": task["source_id"],
                "source_title": task["source_title"],
                "original_url": task["original_url"],
                "topic_bucket_ids": task["topic_bucket_ids"],
                "split": task["split"],
                "chunk_id": task["chunk_id"],
                "document_id": task["document_id"],
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    with open(args.failures, "w", encoding="utf-8") as fail_f:
        for i, error in pending_repair:
            task = tasks[i]
            fail_f.write(json.dumps({
                "task_id": task["task_id"],
                "source_id": task["source_id"],
                "error": error,
                "raw_output": last_raw_by_task[i],
            }, ensure_ascii=False) + "\n")

    print(f"Done. {written}/{len(tasks)} instruction pairs written to {args.output}", file=sys.stderr)
    if pending_repair:
        print(f"{len(pending_repair)} tasks failed validation after {args.max_repair_attempts} repair attempts -> {args.failures}", file=sys.stderr)


if __name__ == "__main__":
    main()
