#!/usr/bin/env bash
# Native Apple-Metal benchmark runner.
#
# Docker on macOS is CPU-only (no Apple GPU passthrough), so to measure REAL
# M-series GPU (Metal) performance the runtime must run natively. This script
# drives the same harness against a native Ollama install.
#
# Constraint note: macOS has no cgroups, so the 16 GB system-RAM target cannot be
# HARD-capped natively on a >16 GB machine. The harness instead records absolute
# footprint / peak / swap so you can judge "would this fit in 16 GB?". For a hard
# 16 GB cap use the Parallels VM (vm/parallels-setup.md) — but that loses Metal.
#
# Usage:
#   sandbox/native/run_metal_bench.sh clean
#   MODELS=qwen2.5:3b sandbox/native/run_metal_bench.sh clean
set -euo pipefail

LABEL="${1:-clean}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS="${HERE}/../harness"

command -v ollama >/dev/null 2>&1 || { echo "ollama not installed (brew install ollama)"; exit 1; }
curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1 || {
  echo "Ollama API not reachable on :11434 — start it with 'brew services start ollama'"; exit 1; }

# Use an isolated venv so we don't touch system python.
VENV="${VENV:-${HERE}/.venv}"
if [ ! -d "${VENV}" ]; then
  echo "creating venv at ${VENV}"
  python3 -m venv "${VENV}"
  "${VENV}/bin/pip" install -q --upgrade pip
  "${VENV}/bin/pip" install -q -r "${HARNESS}/requirements.txt"
fi

echo "running Metal benchmark (label=${LABEL})..."
cd "${HARNESS}"
"${VENV}/bin/python" stage0_bench.py --label "${LABEL}" --out "${HERE}/../results"
echo "done -> sandbox/results/${LABEL}/results.json"
