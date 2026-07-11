# Stage 2 — Instruction-pair generation

Turns `study_note_tasks.jsonl` (Stage 1.5 output, 792 tasks from the
2026-07-09 Databricks handoff) into an Alpaca-style instruction dataset for
LoRA fine-tuning of the Stage 0 student model (Qwen2.5-3B-Instruct).

Teacher model: **Llama-3.3-70B-Instruct**, run on Ohm via vLLM batch
inference across the full 4×A100 allocation. If your HF access is pinned to
Llama-3.1-70B-Instruct instead, pass `--model meta-llama/Meta-Llama-3.1-70B-Instruct`.

## Design decisions settled here

These were open in the top-level CLAUDE.md (Open Blockers, Stage 2 design
questions) and are now settled for this implementation:

- **Alpaca field mapping**: `study_note_tasks.jsonl`'s `prompt` field is a
  fixed instruction template + `"Content to summarise:\n"` + `source_content`
  (verified identical template across sampled tasks). So:
  - `instruction` = the fixed template (everything before the content marker)
  - `input` = `source_content` (the chunk text)
  - `output` = the teacher's validated JSON response

  The model is still called with the *full* `prompt` (template + content
  together) for generation — the instruction/input split only applies to how
  the result is stored.

- **Validation depth**: JSON-parse (tolerating markdown fences / stray text
  around the object) plus a structural check that all seven required
  top-level keys from `expected_output_schema` are present. **Not done**:
  groundedness/hallucination checking against `source_content` — that's a
  separate pass, not yet built. Don't treat `instruction_pairs.jsonl` as
  hallucination-free.

- **Retry/repair strategy**: failed tasks get up to `--max-repair-attempts`
  (default 2) re-prompts asking the model to correct its own invalid output,
  batched the same way as the first pass. Tasks still failing after that are
  written to `failures.jsonl` with the error and raw output, not silently
  dropped.

## Usage

```bash
# 1. Get the Stage 1.5 handoff data into place
mkdir -p data
cp /path/to/databricks_ld_foundations_20260709/study_note_tasks.jsonl data/

# 2. Smoke test on Ohm before spending the full allocation
ohm test --command "python scripts/generate_instruction_pairs.py \
  --input data/study_note_tasks.jsonl \
  --output data/instruction_pairs.smoke.jsonl \
  --failures data/failures.smoke.jsonl \
  --limit 10"

# 3. Full run
bash ohm/submit_teacher_gen.sh
```

Output: `data/instruction_pairs.jsonl` (one Alpaca record per successfully
validated task) and `data/failures.jsonl` (tasks that never produced valid
JSON, for manual review or a future repair pass).

## Not yet confirmed

- `peft`/`trl`/`vllm` availability on the Ohm `pytorch-llm` base image —
  `requirements.txt` here is installed via `ohm submit --install`, but if
  vLLM's CUDA build doesn't match the image's CUDA 12.4 exactly, pin a
  specific `vllm==` version rather than the current floor.
- Whether `ohm submit` forwards `HF_TOKEN` from the shell environment. If
  Llama-3.3-70B-Instruct is gated on your HF account, run
  `huggingface-cli login` inside an interactive Ohm session first so the
  token is cached under `/home/jovyan/.cache/huggingface` before submitting
  the batch job — see comments in `ohm/submit_teacher_gen.sh`.
- Storage: `study_note_tasks.jsonl` is 8.9MB; `instruction_pairs.jsonl` will
  be similar or larger. Check `/home/jovyan` quota (50GiB shared) isn't
  already under pressure from the HF model cache before this job downloads
  ~140GB of Llama-70B weights.
