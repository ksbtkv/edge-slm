#!/usr/bin/env bash
# Submit Stage 2 teacher-model enrichment to Ohm.
#
# Prereqs:
#   - data/study_note_tasks.jsonl copied into this workdir (from the
#     Databricks handoff zip, e.g. databricks_ld_foundations_20260709.zip)
#   - HF_TOKEN exported in this shell if meta-llama/Llama-3.3-70B-Instruct
#     is gated for your account (ohm submit does not forward env vars by
#     default -- confirm with `ohm submit --help`; if it doesn't, run
#     `huggingface-cli login` inside an interactive Ohm session first so the
#     token is cached under /home/jovyan/.cache/huggingface before submitting)
#
# Smoke-test first with --limit before spending the full 4-GPU allocation:
#   ohm test --command "python scripts/generate_instruction_pairs.py \
#     --input data/study_note_tasks.jsonl \
#     --output data/instruction_pairs.smoke.jsonl \
#     --failures data/failures.smoke.jsonl \
#     --limit 10"

set -euo pipefail

ohm submit \
  --name stage2-teacher-gen \
  --image pytorch-llm \
  --gpus 4 --cpu 32 --memory 192Gi \
  --workdir stage2_dataset \
  --install requirements.txt \
  --command "python scripts/generate_instruction_pairs.py \
    --input data/study_note_tasks.jsonl \
    --output data/instruction_pairs.jsonl \
    --failures data/failures.jsonl \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 4"
