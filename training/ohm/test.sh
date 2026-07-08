#!/usr/bin/env bash
# Stage 3 smoke test on the Ohm platform.
#
# Runs the trainer's --dry-run in the CURRENT notebook pod (no GPU job): it
# validates env, config, dataset format, and tokenisation, then exits without
# loading the model. This is the cheap "catch path/import/argument problems
# before requesting a GPU job" step the platform recommends.
#
# Run from a JupyterLab terminal inside the training/ directory.
set -euo pipefail

CONFIG="${CONFIG:-config/qwen2.5_3b_qlora.yaml}"

ohm test --command "python train.py --config ${CONFIG} --dry-run"

# For a tiny REAL training step (needs a GPU notebook profile), instead run:
#   ohm test --command "python train.py --config ${CONFIG} --smoke --epochs 1"
