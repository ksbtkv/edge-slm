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

# Iters: a full run of the ~640-pair set overfits past ~600 iters — validation
# loss bottoms at iter ~600 (0.854) then rises to 0.990 by 1400 while train loss
# keeps falling (see docs/mlx_loss_curve.svg). So the default is 600, and we save
# + eval every 100 so the best checkpoint is easy to pick. Raise --iters (append
# e.g. `--iters 1400`) only if you add more data or regularisation.
# Batch 1 + grad checkpointing keeps peak memory ~5.5 GB (fits a 16 GB Mac).
python -m mlx_lm lora \
  --model "$MODEL" \
  --train \
  --data "$DATA_DIR" \
  --adapter-path "$ADAPTER_DIR" \
  --batch-size 1 \
  --iters 600 \
  --learning-rate 1e-4 \
  --num-layers 16 \
  --grad-checkpoint \
  --save-every 100 \
  --steps-per-eval 100 \
  "$@"

echo "Adapter written to $ADAPTER_DIR"
echo "Fuse for deployment with:"
echo "  python -m mlx_lm fuse --model $MODEL --adapter-path $ADAPTER_DIR --save-path <fused_dir>"
