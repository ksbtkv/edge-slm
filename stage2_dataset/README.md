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

## Pawsey/Ohm compute compatibility (checked 2026-07-11)

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
- **GPU/image request matches Ohm's documented profiles exactly**:
  `--gpus 1 --cpu 16 --memory 96Gi` is the "1 GPU" profile as specified in
  the Ohm quick reference, `--image pytorch-llm` is one of the two allowed
  images. Single-GPU, single-node (`--tensor-parallel-size 1`), so the
  "multi-node distributed training is early/unpolished" caveat doesn't apply.
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
