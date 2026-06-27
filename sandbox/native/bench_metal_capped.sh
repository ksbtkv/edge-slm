#!/usr/bin/env bash
# Apple-Metal benchmark UNDER a simulated VRAM cap.
#
# Apple Silicon shares one unified memory pool between CPU and GPU. The sysctl
# `iogpu.wired_limit_mb` bounds how much of it the GPU may wire — so setting it to
# e.g. 8192 makes the GPU behave as if it had an 8 GB VRAM budget. This lets the
# Mac reproduce the target laptop's discrete-VRAM HEADROOM constraint (2-8 GB),
# the decisive metric in both Stage 0 reports.
#
# What this DOES simulate: capacity — does the model + KV cache fit in N GB of GPU
#   memory? If it doesn't, Metal allocation fails / layers spill, exactly like a
#   real dGPU that's out of VRAM.
# What it does NOT simulate: speed — M-series unified bandwidth (~400 GB/s) and the
#   40-core GPU are unlike a laptop dGPU's GDDR6, so tok/s is NOT transferable.
#
# Requires sudo (system-wide setting). The cap is ALWAYS reset on exit.
#
# Usage:
#   sandbox/native/bench_metal_capped.sh 8           # 8 GB cap, label "metal-cap8"
#   sandbox/native/bench_metal_capped.sh 4 clean4    # 4 GB cap, custom label
set -euo pipefail

CAP_GB="${1:?usage: bench_metal_capped.sh <cap_gb> [label]}"
LABEL="${2:-metal-cap${CAP_GB}}"
CAP_MB=$(( CAP_GB * 1024 ))
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

reset_cap() {
  echo "resetting iogpu.wired_limit_mb -> 0 (default)"
  sudo sysctl -w iogpu.wired_limit_mb=0 >/dev/null
}
trap reset_cap EXIT

echo "previous: $(sysctl -n iogpu.wired_limit_mb) MB"
echo "setting GPU wired-memory cap -> ${CAP_MB} MB (${CAP_GB} GB) ..."
sudo sysctl -w iogpu.wired_limit_mb="${CAP_MB}" >/dev/null
echo "now: $(sysctl -n iogpu.wired_limit_mb) MB"
echo

# Run the standard Metal benchmark; the harness records the active wired limit.
bash "${HERE}/run_metal_bench.sh" "${LABEL}"
