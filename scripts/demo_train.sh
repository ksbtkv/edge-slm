#!/usr/bin/env bash
#
# DEMO ONLY — a fast, REAL mini fine-tune (~2-3 min) so an audience can watch the
# model actually learn (validation loss drops live) WITHOUT the multi-hour full run.
#
# This does NOT produce the deployable model — it trains a few layers for a few
# dozen iterations. For real output quality in the demo, use the PRE-TRAINED model
# (`ollama run edge-slm-databricks`) or the released adapter. See docs/demo_runbook.md.
#
# Requires: pip install -r requirements-training-local.txt (mlx-lm), Apple Silicon.
#
# Usage:  scripts/demo_train.sh [training_data_dir] [output_dir]
set -euo pipefail

DATA="${1:-data/processed/training/databricks_ld_foundations}"
OUT="${2:-data/processed/adapters/demo_run}"
MODEL="${EDGE_SLM_STUDENT_MLX:-mlx-community/Qwen3-4B-Instruct-2507-4bit}"

echo ">> DEMO mini-train (~2-3 min). Watch 'Val loss' fall. This is NOT the deployed model."
echo ">> Deployed model: ollama run edge-slm-databricks"
echo

# 40 iters / 6 layers / small val set -> ~2-3 min on an M-series Mac, with the
# validation loss printed at iter 1, 20, and 40 so the descent is visible live.
python -m mlx_lm lora \
  --model "$MODEL" \
  --train \
  --data "$DATA" \
  --adapter-path "$OUT" \
  --fine-tune-type lora \
  --num-layers 6 \
  --batch-size 1 \
  --iters 40 \
  --steps-per-eval 20 \
  --val-batches 6 \
  --save-every 40 \
  --grad-checkpoint \
  --mask-prompt

echo
echo ">> Done. Loss dropped in ~2-3 min. The FULL run (~600 iters) reaches the deployed"
echo ">> quality — see docs/mlx_loss_curve.svg — but you never wait for that live."
