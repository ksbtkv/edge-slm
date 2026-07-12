# Stage 2 — Instruction-pair generation

Turns `study_note_tasks.jsonl` (Stage 1.5 output, 792 tasks from the
2026-07-09 Databricks handoff) into an Alpaca-style instruction dataset for
LoRA fine-tuning of the Stage 0 student model (Qwen2.5-3B-Instruct).

Teacher model: **Qwen2.5-32B-Instruct-AWQ** (4-bit AWQ, ~20GB), run on Ohm
via vLLM batch inference on a single A100. Originally scoped for
Llama-3.3-70B-Instruct per the top-level CLAUDE.md, but that model's bf16
weights (~140GB) don't fit Ohm's 50GiB default home-dir quota, and 70B-class
quantized builds were still tight against it — Qwen2.5-32B-Instruct-AWQ was
chosen instead as a teacher model that comfortably fits quota at good
quality. Not gated on Hugging Face, so no HF_TOKEN/approval step needed.

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

Run all of this from a terminal **inside an already-running Ohm JupyterLab
session** — `ohm test` executes in your current notebook pod, it doesn't
submit anything itself (see the compute-compatibility notes below).

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

# 4. Find the actual job name (a suffix is appended on submit) and follow it
ohm list
ohm logs stage2-teacher-gen-<suffix>
```

Output: `data/instruction_pairs.jsonl` (one Alpaca record per successfully
validated task) and `data/failures.jsonl` (tasks that never produced valid
JSON, for manual review or a future repair pass).

## Fault tolerance on Ohm

The Scope of Work flags Pawsey/Ohm turnaround time as the project's highest
HPC risk (a job can run long, get preempted, or fail partway through with no
interactive access to debug it live). `generate_instruction_pairs.py` is
written defensively around that risk:

- **Fail-fast schema check.** Before any GPU work, every loaded task is
  checked for the content marker and required fields. The README's field-
  mapping assumption was only verified on a sample -- this enforces it
  across the whole dataset and exits in seconds if the upstream template
  has drifted, instead of discovering it after a full generation pass.
- **Startup diagnostics.** Logs Python/torch/CUDA/vLLM versions, per-GPU
  name and memory, and GPU memory actually used after the model loads.
  Exits immediately with a clear message if the GPU count doesn't match
  `--tensor-parallel-size`.
- **Disk preflight.** Checks free space under the home-dir quota (default
  50GiB) before downloading the ~19.3GB model and aborts with the specific
  `du` command to run if there isn't enough headroom.
- **Batched, checkpointed generation.** Prompts are sent to vLLM in
  `--batch-size` chunks (default 64), not one giant blocking call. Results
  are checkpointed (atomic write-then-rename) after **every batch**, in
  both the first pass and every repair pass -- a kill mid-run loses at
  most one batch, not the whole job.
- **Graceful shutdown.** SIGTERM/SIGINT set a stop flag checked between
  batches, so a preempted job finishes its current batch, checkpoints, and
  exits cleanly rather than being hard-killed mid-write.
- **Soft time budget.** `--time-budget-minutes` stops generation cleanly
  and checkpoints before an external limit would kill the job uncleanly.
- **Resume.** Re-running the same command skips any `task_id` already
  present in `--output`, so resubmitting after any interruption above only
  regenerates what's missing. Pass `--no-resume` to force a clean rerun.

## Pawsey/Ohm compute compatibility (checked 2026-07-11)

Cross-checked against the actual platform deck
(`ohm-infrav2-architecture-slides.pdf`, dated 2026-07-03), not just the
informal Ohm notes in the top-level CLAUDE.md.

- **`requirements.txt` is version-pinned, not floored**, specifically because
  of this: unpinned `vllm>=0.6.3` resolves to whatever's newest on PyPI right
  now (0.24.0), which requires `torch==2.11.0` — nothing like the Ohm
  `pytorch-llm` image's PyTorch 2.5.1. Installing that would force a large,
  unnecessary torch/torchvision/torchaudio reinstall on the job node and burn
  quota. `vllm==0.6.6.post1` is the pinned version confirmed (via PyPI
  metadata) to require `torch==2.5.1` — an exact match to the base image, so
  `pip install -r requirements.txt` should not touch torch at all.
  `autoawq==0.2.9` has no hard torch pin, so it won't fight this either.
- **Model size confirmed via the HF API**: `Qwen/Qwen2.5-32B-Instruct-AWQ` is
  5 safetensors shards totaling **19.34GB**, not gated. Comfortably inside
  the 50GiB `/home/jovyan` quota alongside the dataset files, but check
  `du -h -d 1 ~/.cache/huggingface` isn't already under pressure from other
  cached models before this job downloads.
- **GPU/image request is valid but above the job default** — `--image
  pytorch-llm` is one of the two allowed images. `--gpus 1 --cpu 16 --memory
  96Gi` matches the *interactive notebook* "1 GPU" profile (deck slide 5),
  but submitted jobs actually default to CPU 8 / memory 48Gi if left
  unspecified (deck slide 12) — a separate, smaller default from the
  notebook profile table. The explicit 16 CPU/96GiB request is intentional
  headroom for model loading + batch generation, not a misreading of the
  defaults, but it's worth knowing it's above what's strictly needed.
  Single-GPU, single-node (`--tensor-parallel-size 1`), so the "multi-node
  distributed training is still an early workflow" caveat doesn't apply.
- **`ohm test` runs in your current notebook pod, not as a separate
  submission** — it's for catching path/import/argument bugs before
  requesting a GPU job, run from a JupyterLab terminal you already have
  open. `ohm submit` is the actual job scheduler call. Submitted job names
  get a random suffix appended — use `ohm list` to find the exact name
  before `ohm logs`/`ohm status`/`ohm cancel`.
- **Not verified from here** (no access to an actual Ohm session): whether
  the AWQ CUDA kernels vLLM ships actually run correctly on Ohm's specific
  A100 driver/CUDA 12.4 stack, and whether `autoawq`'s prebuilt wheel matches
  this environment closely enough to avoid a from-source build. Run the
  `ohm test --limit 10` smoke test below before committing the full job —
  if `autoawq` tries to compile from source, that'll show up immediately as
  a very slow `--install` step.
- Whether `ohm submit` forwards `HF_TOKEN` from the shell environment — moot
  for the current model (not gated), but relevant again if you swap back to
  a gated checkpoint. See comments in `ohm/submit_teacher_gen.sh`.
- **Teacher-model quality tradeoff**: Qwen2.5-32B-Instruct-AWQ is meaningfully
  smaller than the originally-planned Llama-3.3-70B-Instruct. Worth spot-
  checking a sample of `instruction_pairs.jsonl` output quality (JSON
  structure aside) before committing to a full 792-task run, since this
  swap wasn't in the original CLAUDE.md decision record.