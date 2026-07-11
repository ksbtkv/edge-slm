#!/usr/bin/env bash
# Submit Stage 2 teacher-model enrichment to Ohm.
#
# Prereqs:
#   - data/study_note_tasks.jsonl copied into this workdir (from the
#     Databricks handoff zip, e.g. databricks_ld_foundations_20260709.zip)
#   - Qwen/Qwen2.5-32B-Instruct-AWQ is not gated, so no HF_TOKEN is needed.
#     If you swap back to a gated model (e.g. a Llama checkpoint), export
#     HF_TOKEN in this shell first -- ohm submit may not forward env vars by
#     default (confirm with `ohm submit --help`); if it doesn't, run
#     `huggingface-cli login` inside an interactive Ohm session first so the
#     token is cached under /home/jovyan/.cache/huggingface before submitting.
#
# Smoke-test first with --limit before spending the full GPU allocation:
#   ohm test --command "python scripts/generate_instruction_pairs.py \
#     --input data/study_note_tasks.jsonl \
#     --output data/instruction_pairs.smoke.jsonl \
#     --failures data/failures.smoke.jsonl \
#     --limit 10"

set -euo pipefail

ohm submit \
  --name stage2-teacher-gen \
  --image pytorch-llm \
  --gpus 1 --cpu 16 --memory 96Gi \
  --workdir stage2_dataset \
  --install requirements.txt \
  --command "python scripts/generate_instruction_pairs.py \
    --input data/study_note_tasks.jsonl \
    --output data/instruction_pairs.jsonl \
    --failures data/failures.jsonl \
    --model Qwen/Qwen2.5-32B-Instruct-AWQ \
    --tensor-parallel-size 1"
