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

Hardened end-to-end for Ohm's shared, non-interactive, preemption-risk job
environment (ohm-infrav2-architecture-slides.pdf; SOW Section 4.2 names
Pawsey turnaround as the project's single highest-impact HPC risk):

  - Fail-fast schema check on ALL loaded tasks, before any GPU work starts
    (the README's field-mapping assumption was only verified on a sample --
    this makes it a hard, whole-dataset precondition instead).
  - Startup diagnostics: torch/CUDA/vLLM versions, per-GPU name/memory, and
    GPU memory actually used after the model loads.
  - Disk-quota preflight against the 50 GiB /home/jovyan default.
  - Generation runs in configurable batches (--batch-size), not one giant
    blocking call. Results are checkpointed (atomic write-then-rename)
    after EVERY batch, in both the first pass and every repair pass -- so
    a kill/OOM/preemption mid-pass loses at most one batch, not the whole
    run.
  - Graceful shutdown: SIGTERM/SIGINT set a stop flag that's checked
    between batches, so a preempted job finishes its current batch,
    checkpoints, and exits cleanly instead of being hard-killed mid-write.
  - Optional soft --time-budget-minutes: stop cleanly and checkpoint before
    an external time limit would kill the job uncleanly.
  - Resume: task_ids already in --output are skipped on the next run, so
    resubmitting after any of the above only regenerates what's missing.

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
import logging
import re
import shutil
import signal
import sys
import time
from pathlib import Path

try:
    from tqdm import tqdm  # ships in the Ohm base image (see deck: "Progress: tqdm")
except ImportError:  # pragma: no cover - defensive fallback only
    class _NullPbar:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def update(self, n=1): pass

    def tqdm(*a, **k):
        return _NullPbar()

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

log = logging.getLogger("stage2")


# --------------------------------------------------------------------------
# Diagnostics / preflight
# --------------------------------------------------------------------------

def log_environment(tensor_parallel_size: int) -> None:
    log.info("=== Environment diagnostics ===")
    log.info("Python %s", sys.version.split()[0])
    try:
        import torch
        log.info("torch %s", torch.__version__)
        if not torch.cuda.is_available():
            log.error(
                "No CUDA device visible to torch. Confirm the job was submitted with "
                "--gpus >= 1 and --image pytorch-llm."
            )
            sys.exit(1)
        n_gpus = torch.cuda.device_count()
        log.info("CUDA devices visible: %d", n_gpus)
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            log.info("  GPU %d: %s, %.1f GiB", i, props.name, props.total_memory / 1024**3)
        if n_gpus < tensor_parallel_size:
            log.error(
                "Requested --tensor-parallel-size %d but only %d GPU(s) visible to this job.",
                tensor_parallel_size, n_gpus,
            )
            sys.exit(1)
    except ImportError:
        log.warning("torch not importable for diagnostics yet (vllm import will pull it in next).")
    try:
        import vllm
        log.info("vllm %s", vllm.__version__)
    except Exception as e:  # pragma: no cover - diagnostic only
        log.warning("Could not read vllm version: %s", e)


def log_gpu_memory_after_load() -> None:
    try:
        import torch
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            log.info("GPU %d memory after model load: %.1f GiB allocated, %.1f GiB reserved", i, alloc, reserved)
    except Exception as e:  # pragma: no cover - diagnostic only
        log.warning("Could not read post-load GPU memory: %s", e)


def check_disk_space(min_free_gb: float) -> None:
    """Ohm's default home-dir quota is 50 GiB, shared by model caches,
    datasets, and checkpoints. Fail loudly here instead of mid-download."""
    home = Path.home()
    usage = shutil.disk_usage(home)
    free_gb = usage.free / 1024**3
    log.info("Free space on %s: %.1f GiB (Ohm's default home quota is 50 GiB total)", home, free_gb)
    if free_gb < min_free_gb:
        log.error(
            "Only %.1f GiB free, below the %.1f GiB safety margin needed for the teacher "
            "model download plus working files. Check `du -h -d 1 ~/.cache/huggingface` and "
            "`du -h -d 1 ~/.cache` for cleanup targets, then retry.",
            free_gb, min_free_gb,
        )
        sys.exit(1)


# --------------------------------------------------------------------------
# Core parsing / validation
# --------------------------------------------------------------------------

def load_tasks(path: Path) -> list[dict]:
    if not path.exists():
        log.error("Input file not found: %s", path)
        sys.exit(1)
    tasks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def validate_tasks_schema(tasks: list[dict]) -> None:
    """Fail fast, before any GPU work, if the upstream handoff format has
    drifted. README.md notes the field-mapping assumption was only
    'verified identical template across sampled tasks' -- this checks it
    across the WHOLE dataset instead of trusting the sample."""
    missing_marker = [t.get("task_id", "?") for t in tasks if CONTENT_MARKER not in t.get("prompt", "")]
    if missing_marker:
        log.error(
            "%d/%d tasks are missing the expected content marker %r in their `prompt` field -- "
            "the upstream template has likely changed. First few task_ids: %s",
            len(missing_marker), len(tasks), CONTENT_MARKER, missing_marker[:5],
        )
        sys.exit(1)

    required_fields = {"task_id", "source_id", "source_title", "original_url",
                        "topic_bucket_ids", "split", "chunk_id", "document_id",
                        "source_content", "expected_output_schema"}
    for t in tasks:
        missing = required_fields - t.keys()
        if missing:
            log.error(
                "Task %s is missing required field(s) %s -- cannot build an output record for it.",
                t.get("task_id", "?"), sorted(missing),
            )
            sys.exit(1)

    mismatched = [
        t["task_id"] for t in tasks
        if set(t.get("expected_output_schema", {}).keys()) and
        set(t["expected_output_schema"].keys()) != REQUIRED_KEYS
    ]
    if mismatched:
        log.warning(
            "%d tasks have an expected_output_schema whose keys differ from the hardcoded "
            "REQUIRED_KEYS used for validation. Validation will still only check REQUIRED_KEYS. "
            "First few: %s",
            len(mismatched), mismatched[:5],
        )
    log.info("Schema check passed for all %d tasks.", len(tasks))


def split_instruction(prompt: str) -> str:
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
    if not isinstance(obj, dict):
        return "top-level JSON is not an object"
    missing = REQUIRED_KEYS - obj.keys()
    if missing:
        return f"missing required keys: {sorted(missing)}"
    return None


def build_record(task: dict, obj: dict) -> dict:
    return {
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


# --------------------------------------------------------------------------
# Checkpointing / resume
# --------------------------------------------------------------------------

def load_existing_records(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    records[rec["task_id"]] = rec
    return records


def write_records_atomic(path: Path, records: dict[str, dict]) -> None:
    """Write-then-rename so a crash mid-write never leaves a truncated or
    corrupt checkpoint on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in records.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_failures(path: Path, pending: list[tuple[int, str]], tasks: list[dict], last_raw: dict[int, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for i, error in pending:
            task = tasks[i]
            f.write(json.dumps({
                "task_id": task["task_id"],
                "source_id": task["source_id"],
                "error": error,
                "raw_output": last_raw.get(i, ""),
            }, ensure_ascii=False) + "\n")
    tmp.replace(path)


# --------------------------------------------------------------------------
# Batched generation with checkpointing, graceful stop, and a time budget
# --------------------------------------------------------------------------

def generate_in_batches(llm, sampling, prompts: list[str], batch_size: int,
                         deadline: float | None, stop_state: dict, desc: str):
    """Yields (batch_start, raw_texts) for each completed batch. Stops
    early -- without raising -- if a stop was requested or the soft time
    budget was exceeded; the caller can tell how much was actually
    processed from the batches it received."""
    n = len(prompts)
    with tqdm(total=n, desc=desc, unit="task", file=sys.stderr) as pbar:
        for start in range(0, n, batch_size):
            if stop_state["stop"]:
                log.warning("Stop requested -- halting %s after %d/%d tasks", desc, start, n)
                return
            if deadline is not None and time.monotonic() > deadline:
                log.warning("Time budget exceeded -- halting %s after %d/%d tasks", desc, start, n)
                return
            batch_prompts = prompts[start:start + batch_size]
            t0 = time.monotonic()
            outputs = llm.generate(batch_prompts, sampling)
            dt = time.monotonic() - t0
            raws = [o.outputs[0].text for o in outputs]
            rate = len(batch_prompts) / dt if dt > 0 else float("inf")
            log.info("%s batch [%d:%d] done in %.1fs (%.2f tasks/s)", desc, start, start + len(batch_prompts), dt, rate)
            pbar.update(len(batch_prompts))
            yield start, raws


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--failures", required=True, type=Path)
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-AWQ")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--quantization", default="awq", help="pass '' to disable (e.g. for an unquantized checkpoint)")
    ap.add_argument("--max-repair-attempts", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                     help="fraction of GPU memory vLLM may reserve; lower this if sharing the A100 or hitting OOM")
    ap.add_argument("--max-model-len", type=int, default=None,
                     help="cap vLLM's KV-cache context length; leave unset to use the model's default")
    ap.add_argument("--min-free-gb", type=float, default=25.0,
                     help="abort before downloading the model if less than this much home-dir quota is free")
    ap.add_argument("--batch-size", type=int, default=64,
                     help="tasks per vLLM generate() call; results checkpoint to disk after every batch")
    ap.add_argument("--time-budget-minutes", type=float, default=None,
                     help="stop cleanly and checkpoint once this many minutes have elapsed, instead of "
                          "running until an external limit kills the job uncleanly")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N tasks (smoke test)")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                     help="disable resume: regenerate every task even if --output already has results for it")
    ap.set_defaults(resume=True)
    args = ap.parse_args()

    log_environment(args.tensor_parallel_size)
    check_disk_space(args.min_free_gb)

    stop_state = {"stop": False}

    def _handle_signal(signum, _frame):
        log.warning("Received signal %s -- finishing the current batch, checkpointing, then exiting.", signum)
        stop_state["stop"] = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    deadline = None
    if args.time_budget_minutes is not None:
        deadline = time.monotonic() + args.time_budget_minutes * 60
        log.info("Soft time budget: %.1f minutes", args.time_budget_minutes)

    from vllm import LLM, SamplingParams  # deferred: heavy import, HPC-only dep

    tasks = load_tasks(args.input)
    if args.limit:
        tasks = tasks[: args.limit]
    log.info("Loaded %d tasks from %s", len(tasks), args.input)
    validate_tasks_schema(tasks)

    existing = load_existing_records(args.output) if args.resume else {}
    if existing:
        before = len(tasks)
        done_ids = existing.keys()
        tasks = [t for t in tasks if t["task_id"] not in done_ids]
        log.info(
            "Resume: %d records already in %s, skipping them; %d/%d tasks remaining",
            len(existing), args.output, len(tasks), before,
        )

    if not tasks:
        log.info("Nothing left to do -- all requested tasks are already in %s.", args.output)
        return

    try:
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            quantization=args.quantization or None,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )
    except Exception as e:
        log.error("Failed to initialize the vLLM engine: %s", e)
        log.error(
            "If this looks like a CUDA-kernel or AWQ compatibility error, see the unverified-risk "
            "notes in stage2_dataset/README.md (AWQ kernels on Ohm's A100 driver/CUDA 12.4 stack). "
            "If autoawq is compiling from source, that means its prebuilt wheel didn't match this "
            "image -- check `ohm images` and the job logs for the exact CUDA/driver version."
        )
        raise
    log_gpu_memory_after_load()
    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    results: dict[str, dict] = {}
    pending_repair: list[tuple[int, str]] = []
    last_raw_by_task: dict[int, str] = {}

    def checkpoint():
        write_records_atomic(args.output, {**existing, **results})
        write_failures(args.failures, pending_repair, tasks, last_raw_by_task)

    try:
        # ---- First pass ----
        prompts = [t["prompt"] for t in tasks]
        n_done = 0
        for start, raws in generate_in_batches(llm, sampling, prompts, args.batch_size, deadline, stop_state, "first pass"):
            for offset, raw in enumerate(raws):
                i = start + offset
                task = tasks[i]
                last_raw_by_task[i] = raw
                try:
                    obj = extract_json(raw)
                    err = validate_shape(obj)
                    if err:
                        pending_repair.append((i, err))
                    else:
                        results[task["task_id"]] = build_record(task, obj)
                except json.JSONDecodeError as e:
                    pending_repair.append((i, str(e)))
            n_done = start + len(raws)
            checkpoint()

        if n_done < len(tasks):
            log.warning(
                "First pass stopped early: %d/%d tasks processed. The remaining %d were never "
                "attempted and will be regenerated on the next resumed run.",
                n_done, len(tasks), len(tasks) - n_done,
            )
        log.info("First pass complete: %d valid, %d need repair, %d not attempted",
                  len(results), len(pending_repair), len(tasks) - n_done)

        # ---- Repair passes ----
        for attempt in range(1, args.max_repair_attempts + 1):
            if not pending_repair:
                break
            if stop_state["stop"] or (deadline is not None and time.monotonic() > deadline):
                log.warning("Skipping repair attempt %d (stop requested or time budget exceeded).", attempt)
                break

            current = pending_repair
            error_by_idx = dict(current)
            idx_order = [i for i, _ in current]
            repair_prompts = []
            for i, error in current:
                task = tasks[i]
                schema_str = json.dumps(task["expected_output_schema"], indent=2)
                repair_prompts.append(
                    REPAIR_TEMPLATE.format(error=error, schema=schema_str, content=task["source_content"])
                )

            pending_repair = []
            n_repaired_this_attempt = 0
            for start, raws in generate_in_batches(llm, sampling, repair_prompts, args.batch_size,
                                                     deadline, stop_state, f"repair {attempt}"):
                for offset, raw in enumerate(raws):
                    i = idx_order[start + offset]
                    last_raw_by_task[i] = raw
                    task = tasks[i]
                    try:
                        obj = extract_json(raw)
                        err = validate_shape(obj)
                        if err:
                            pending_repair.append((i, err))
                        else:
                            results[task["task_id"]] = build_record(task, obj)
                    except json.JSONDecodeError as e:
                        pending_repair.append((i, str(e)))
                n_repaired_this_attempt = start + len(raws)
                checkpoint()

            # Anything never attempted this round (interrupted mid-attempt)
            # goes back into pending_repair with its known error/raw output
            # so failures.jsonl still accounts for it.
            if n_repaired_this_attempt < len(idx_order):
                for i in idx_order[n_repaired_this_attempt:]:
                    pending_repair.append((i, error_by_idx[i]))
                checkpoint()

            log.info("Repair attempt %d: %d/%d still failing afterward", attempt, len(pending_repair), len(current))
    finally:
        # Belt-and-braces: persist current state even on an exception raised
        # between checkpoints (e.g. inside the per-item processing loop).
        checkpoint()

    total_records = len(existing) + len(results)
    log.info("Done. %d instruction pairs total in %s (%d new this run)", total_records, args.output, len(results))
    if pending_repair:
        log.info(
            "%d tasks still failing/unattempted -> %s. Resubmit the same command to retry just "
            "these (resume skips everything already in --output).",
            len(pending_repair), args.failures,
        )


if __name__ == "__main__":
    main()