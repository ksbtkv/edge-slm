#!/usr/bin/env bash
# End-to-end smoke test against a running Ollama instance.
#
# Usage (from deployment/):
#   ./scripts/smoke_test.sh
#   MODEL_NAME=databricks-study-notes OLLAMA_URL=http://127.0.0.1:11434 ./scripts/smoke_test.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${DEPLOY_ROOT}"

MODEL_NAME="${MODEL_NAME:-databricks-study-notes}"
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-prompts/system_prompt.txt}"

if [[ ! -f "${SYSTEM_PROMPT}" ]]; then
  python3 scripts/build_system_prompt.py --output "${SYSTEM_PROMPT}"
fi

USER_MSG=$(cat <<'EOF'
Delta Lake tables can be used as both sources and sinks for Spark Structured
Streaming in Databricks. For simple streaming writes, use append mode with a
checkpoint location and write to a table using toTable. A checkpoint stores the
streaming query progress so the stream can resume after failure. When reading
from a Delta table that may receive updates or deletes, consider options such as
skipChangeCommits or change data feed. For complex sinks with upserts, use
foreachBatch with MERGE INTO and track idempotency with txnAppId and txnVersion.
EOF
)

SYSTEM_JSON=$(python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))' < "${SYSTEM_PROMPT}")
USER_JSON=$(python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))' <<< "${USER_MSG}")

echo "=== Checking Ollama at ${OLLAMA_URL} ==="
curl -sf "${OLLAMA_URL}/api/tags" >/dev/null \
  || { echo "[FATAL] Ollama not reachable at ${OLLAMA_URL}" >&2; exit 1; }

echo "=== Generating study notes with model=${MODEL_NAME} ==="
RESP=$(curl -sf "${OLLAMA_URL}/api/chat" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL_NAME}\",\"stream\":false,\"options\":{\"temperature\":0.2},\"messages\":[{\"role\":\"system\",\"content\":${SYSTEM_JSON}},{\"role\":\"user\",\"content\":${USER_JSON}}]}")

CONTENT=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["message"]["content"])' <<< "${RESP}")

echo "--- raw response (first 800 chars) ---"
python3 -c 'import sys; t=sys.stdin.read(); print(t[:800])' <<< "${CONTENT}"
echo "--------------------------------------"

export SMOKE_CONTENT="${CONTENT}"
python3 - <<'PY'
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path("eval").resolve()))
from validate_response import validate_response

raw = os.environ["SMOKE_CONTENT"]
obj, err = validate_response(raw)
if err:
    print(f"[FAIL] {err}", file=sys.stderr)
    sys.exit(1)
print(f"[OK] Valid study-note JSON — title={obj.get('title')!r}")
PY

echo "Smoke test PASSED."
