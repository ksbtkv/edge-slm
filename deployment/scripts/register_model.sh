#!/usr/bin/env bash
# Register the quantized GGUF with Ollama using the Stage 4 Modelfile.
#
# Prerequisites:
#   - ollama installed and running (host or Docker)
#   - models/databricks-study-notes-q4.gguf present
#   - prompts/system_prompt.txt built (run build_system_prompt.py first)
#
# Usage (from deployment/):
#   ./scripts/register_model.sh
#   MODEL_NAME=databricks-study-notes GGUF=models/databricks-study-notes-q4.gguf ./scripts/register_model.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${DEPLOY_ROOT}"

MODEL_NAME="${MODEL_NAME:-databricks-study-notes}"
GGUF="${GGUF:-models/databricks-study-notes-q4.gguf}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-prompts/system_prompt.txt}"
TEMPERATURE="${TEMPERATURE:-0.2}"
NUM_CTX="${NUM_CTX:-4096}"
TOP_P="${TOP_P:-0.9}"
OLLAMA_BIN="${OLLAMA_BIN:-ollama}"

if [[ ! -f "${GGUF}" ]]; then
  echo "[FATAL] GGUF not found: ${GGUF}" >&2
  echo "Run ./scripts/convert_to_gguf.sh --merged <path> first." >&2
  exit 1
fi

if [[ ! -f "${SYSTEM_PROMPT}" ]]; then
  echo "Building system prompt..."
  python3 scripts/build_system_prompt.py --output "${SYSTEM_PROMPT}"
fi

# Resolve absolute GGUF path for the Modelfile FROM line (Ollama resolves
# relative paths against the Modelfile directory).
GGUF_ABS="$(cd "$(dirname "${GGUF}")" && pwd)/$(basename "${GGUF}")"
MODELFILE_OUT="${DEPLOY_ROOT}/Modelfile.generated"

# Escape triple-quotes in the system prompt for Modelfile embedding.
SYSTEM_BODY="$(cat "${SYSTEM_PROMPT}")"

cat > "${MODELFILE_OUT}" <<EOF
FROM ${GGUF_ABS}

PARAMETER temperature ${TEMPERATURE}
PARAMETER num_ctx ${NUM_CTX}
PARAMETER top_p ${TOP_P}

SYSTEM """
${SYSTEM_BODY}
"""
EOF

# Also refresh the committed Modelfile template with a relative path for docs.
cat > "${DEPLOY_ROOT}/Modelfile" <<EOF
# Template Modelfile — register_model.sh writes Modelfile.generated with an
# absolute FROM path. Prefer: ./scripts/register_model.sh
FROM ./models/databricks-study-notes-q4.gguf

PARAMETER temperature ${TEMPERATURE}
PARAMETER num_ctx ${NUM_CTX}
PARAMETER top_p ${TOP_P}

SYSTEM """
${SYSTEM_BODY}
"""
EOF

echo "Creating Ollama model '${MODEL_NAME}' from ${MODELFILE_OUT}..."
"${OLLAMA_BIN}" create "${MODEL_NAME}" -f "${MODELFILE_OUT}"
echo "DONE. Verify with: ${OLLAMA_BIN} list"
echo "Test with: ${OLLAMA_BIN} run ${MODEL_NAME} \"Paste a short Databricks passage here\""
