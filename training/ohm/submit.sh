#!/usr/bin/env bash
# Stage 3 fine-tuning — submit a GPU job on the Ohm platform.
#
# Run from a JupyterLab terminal inside the training/ directory, AFTER
# `ohm/test.sh` passes. Override any knob via env var, e.g.:
#   NAME=qwen-run2 GPUS=1 CONFIG=config/qwen2.5_3b_qlora.yaml EXTRA="--merge-adapter" ./ohm/submit.sh
set -euo pipefail

NAME="${NAME:-qwen25-3b-lora}"
GPUS="${GPUS:-1}"          # 1x A100 is plenty for 3B QLoRA
CPU="${CPU:-16}"          # matches the 1-GPU notebook profile
MEM="${MEM:-96Gi}"       # matches the 1-GPU notebook profile
WORKDIR="${WORKDIR:-training}"
CONFIG="${CONFIG:-config/qwen2.5_3b_qlora.yaml}"
EXTRA="${EXTRA:-}"        # e.g. "--merge-adapter" to also emit the fp16 model

ohm submit \
  --name "${NAME}" \
  --image pytorch-llm \
  --gpus "${GPUS}" \
  --cpu "${CPU}" \
  --memory "${MEM}" \
  --workdir "${WORKDIR}" \
  --install requirements.txt \
  --command "python train.py --config ${CONFIG} ${EXTRA}"

echo
echo "Submitted '${NAME}'. Track it:"
echo "  ohm list                       # exact job name (gets a short suffix)"
echo "  ohm status ${NAME}-<suffix>"
echo "  ohm logs   ${NAME}-<suffix>"
echo "Outputs land under \$WORKDIR/outputs/ in your /home/jovyan volume."
