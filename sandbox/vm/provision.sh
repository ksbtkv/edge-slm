#!/usr/bin/env bash
# Provision a Linux (Debian/Ubuntu ARM64) Parallels guest as an Edge SLM sandbox.
# Installs Ollama + Python harness, pulls the primary model, runs a smoke test.
# Idempotent-ish: safe to re-run.
set -euo pipefail

MODEL="${MODEL:-qwen2.5:3b}"
DEST="${DEST:-$HOME/edge-slm}"

log() { printf '\n\033[36m==> %s\033[0m\n' "$*"; }

log "Checking OS"
if ! grep -qiE 'debian|ubuntu' /etc/os-release 2>/dev/null; then
  echo "This script targets Debian/Ubuntu guests. Adapt the package steps for your distro." >&2
fi

log "Installing base packages (python3, pip, venv, curl)"
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip python3-venv curl

log "Installing Ollama"
if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
else
  echo "ollama already installed: $(ollama --version)"
fi

log "Starting Ollama service"
# The installer sets up a systemd unit on most distros; fall back to nohup.
if systemctl list-unit-files 2>/dev/null | grep -q '^ollama.service'; then
  sudo systemctl enable --now ollama
else
  pgrep -x ollama >/dev/null || (nohup ollama serve >/tmp/ollama.log 2>&1 &)
fi

log "Waiting for Ollama API"
for _ in $(seq 1 30); do
  if curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; then break; fi
  sleep 1
done

log "Staging harness into ${DEST}"
mkdir -p "${DEST}"
# If this script sits next to the harness (shared folder), copy it; else clone.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "${SCRIPT_DIR}/../harness" ]; then
  cp -r "${SCRIPT_DIR}/../harness" "${DEST}/harness"
else
  echo "harness/ not found next to this script — copy sandbox/harness into ${DEST}/ manually." >&2
fi

log "Python deps"
python3 -m pip install --user -r "${DEST}/harness/requirements.txt"

log "Pulling model: ${MODEL}"
ollama pull "${MODEL}"

log "Smoke test (1 prompt)"
cd "${DEST}/harness"
MODELS="${MODEL}" python3 stage0_bench.py --label smoke || {
  echo "smoke test failed — check /tmp/ollama.log and the harness output above" >&2
  exit 1
}

log "Done. Run a full benchmark with:"
echo "    cd ${DEST}/harness && python3 stage0_bench.py --label clean"
echo "  RAM cap is the VM's configured memory (set to 16384 MB in Parallels)."
