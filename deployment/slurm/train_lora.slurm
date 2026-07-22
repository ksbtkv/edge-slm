#!/bin/bash -l
#SBATCH --job-name=edge-slm-lora
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --account=<PAWSEY_PROJECT_ACCOUNT>
#SBATCH --output=logs/%x-%j.out

# Canonical Run: LoRA fine-tune of Qwen3-4B-Instruct on Setonix (ADR 0003).
#
# Before submitting:
#   1. Replace <PAWSEY_PROJECT_ACCOUNT> with your project code.
#   2. Copy the repo and the exported training data to $MYSCRATCH.
#   3. Build the environment once (ROCm PyTorch wheel — Setonix GPUs are
#      AMD MI250X; do NOT install the CUDA wheel):
#        python -m venv $MYSOFTWARE/envs/edge-slm
#        source $MYSOFTWARE/envs/edge-slm/bin/activate
#        pip install --index-url https://download.pytorch.org/whl/rocm6.2 torch
#        pip install -r requirements-training-hpc.txt
#
# Submit from the repo root on Setonix:
#   sbatch deployment/slurm/train_lora.slurm

module load rocm 2>/dev/null || true
source "$MYSOFTWARE/envs/edge-slm/bin/activate"

REPO_ROOT="${SLURM_SUBMIT_DIR:?}"
DATA_DIR="$REPO_ROOT/data/processed/training/databricks_ld_foundations"
OUTPUT_DIR="$REPO_ROOT/data/processed/adapters/databricks_ld_foundations_trl"

mkdir -p "$REPO_ROOT/logs"

cd "$REPO_ROOT"
PYTHONPATH=pipeline python -m training.train_trl \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR"
