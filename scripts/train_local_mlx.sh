#!/usr/bin/env bash
#
# Local Trainer Backend: QLoRA fine-tune of the Student on Apple Silicon (MLX).
#
# Consumes the train.jsonl / valid.jsonl exported by export_training_pairs.py
# (chat format is understood natively by mlx_lm) and emits a LoRA adapter.
# Local runs are development artifacts; the Pawsey run is canonical (ADR 0003).
#
# Requires: pip install -r requirements-training-local.txt
#
# Usage:
#   scripts/train_local_mlx.sh <training_data_dir> <adapter_output_dir> [extra mlx_lm.lora args]
#
# Example:
#   scripts/train_local_mlx.sh \
#     data/processed/training/databricks_ld_foundations \
#     data/processed/adapters/databricks_ld_foundations_mlx

set -euo pipefail

DATA_DIR="${1:?usage: train_local_mlx.sh <training_data_dir> <adapter_output_dir>}"
ADAPTER_DIR="${2:?usage: train_local_mlx.sh <training_data_dir> <adapter_output_dir>}"
shift 2

# 4-bit base keeps QLoRA within a 16GB M-series budget.
MODEL="${EDGE_SLM_STUDENT_MLX:-mlx-community/Qwen3-4B-Instruct-2507-4bit}"

# ~678 pairs / batch 1 * 2 epochs ≈ 1400 iters. Batch 1 + grad checkpointing
# keeps peak memory low; raise --batch-size on machines with more RAM.
python -m mlx_lm lora \
  --model "$MODEL" \
  --train \
  --data "$DATA_DIR" \
  --adapter-path "$ADAPTER_DIR" \
  --batch-size 1 \
  --iters 1400 \
  --learning-rate 1e-4 \
  --num-layers 16 \
  --grad-checkpoint \
  --save-every 200 \
  --steps-per-eval 200 \
  "$@"

echo "Adapter written to $ADAPTER_DIR"
echo "Fuse for deployment with:"
echo "  python -m mlx_lm fuse --model $MODEL --adapter-path $ADAPTER_DIR --save-path <fused_dir>"
