#!/usr/bin/env bash
# Convert Stage 3 merged/ HuggingFace model → quantized GGUF for Ollama.
#
# Prerequisites:
#   - Python 3.10+ with numpy / torch (for llama.cpp convert script)
#   - A C++ toolchain to build llama-quantize (or use a prebuilt binary)
#
# Usage (from deployment/):
#   ./scripts/convert_to_gguf.sh \
#       --merged ../training/outputs/qwen2.5-3b-lora/merged \
#       --quant Q4_K_M
#
# Environment overrides:
#   LLAMA_CPP_DIR   path to a local llama.cpp checkout (default: .tools/llama.cpp)
#   LLAMA_CPP_REF   git tag/commit to pin (default: b4030 — adjust as needed)
#   OUT_NAME        output basename without extension (default: databricks-study-notes-q4)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${DEPLOY_ROOT}"

MERGED=""
QUANT="${QUANT:-Q4_K_M}"
OUT_NAME="${OUT_NAME:-databricks-study-notes-q4}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-${DEPLOY_ROOT}/.tools/llama.cpp}"
LLAMA_CPP_REF="${LLAMA_CPP_REF:-b4030}"

usage() {
  cat <<EOF
Usage: $0 --merged PATH [--quant Q4_K_M|Q5_K_M] [--out-name NAME]

Converts a Stage 3 merged fp16 HuggingFace directory to a quantized GGUF file
under deployment/models/.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --merged) MERGED="$2"; shift 2 ;;
    --quant) QUANT="$2"; shift 2 ;;
    --out-name) OUT_NAME="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[FATAL] Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${MERGED}" ]]; then
  echo "[FATAL] --merged PATH is required" >&2
  usage
  exit 1
fi
if [[ ! -d "${MERGED}" ]]; then
  echo "[FATAL] Merged model directory not found: ${MERGED}" >&2
  echo "Hint: copy training/outputs/qwen2.5-3b-lora/merged from Ohm, or run" >&2
  echo "      python scripts/merge_adapter.py ..." >&2
  exit 1
fi
if [[ ! -f "${MERGED}/config.json" ]]; then
  echo "[FATAL] ${MERGED}/config.json missing — not a HuggingFace model dir?" >&2
  exit 1
fi

mkdir -p models .tools

ensure_llama_cpp() {
  if [[ -d "${LLAMA_CPP_DIR}/.git" ]]; then
    echo "Using existing llama.cpp at ${LLAMA_CPP_DIR}"
    return
  fi
  echo "Cloning llama.cpp (${LLAMA_CPP_REF}) into ${LLAMA_CPP_DIR}..."
  git clone --depth 1 --branch "${LLAMA_CPP_REF}" \
    https://github.com/ggerganov/llama.cpp.git "${LLAMA_CPP_DIR}" \
    || git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "${LLAMA_CPP_DIR}"
}

ensure_llama_cpp

CONVERT_PY=""
for candidate in \
  "${LLAMA_CPP_DIR}/convert_hf_to_gguf.py" \
  "${LLAMA_CPP_DIR}/convert-hf-to-gguf.py"
do
  if [[ -f "${candidate}" ]]; then
    CONVERT_PY="${candidate}"
    break
  fi
done
if [[ -z "${CONVERT_PY}" ]]; then
  echo "[FATAL] convert_hf_to_gguf.py not found under ${LLAMA_CPP_DIR}" >&2
  exit 1
fi

FP16_GGUF="${DEPLOY_ROOT}/models/${OUT_NAME}-f16.gguf"
OUT_GGUF="${DEPLOY_ROOT}/models/${OUT_NAME}.gguf"

echo "=== Step 1: HF → GGUF (f16) ==="
python3 "${CONVERT_PY}" "${MERGED}" --outfile "${FP16_GGUF}" --outtype f16

QUANTIZE_BIN=""
for candidate in \
  "${LLAMA_CPP_DIR}/build/bin/llama-quantize" \
  "${LLAMA_CPP_DIR}/llama-quantize" \
  "$(command -v llama-quantize || true)"
do
  if [[ -n "${candidate}" && -x "${candidate}" ]]; then
    QUANTIZE_BIN="${candidate}"
    break
  fi
done

if [[ -z "${QUANTIZE_BIN}" ]]; then
  echo "=== Building llama-quantize ==="
  cmake -S "${LLAMA_CPP_DIR}" -B "${LLAMA_CPP_DIR}/build" -DGGML_NATIVE=OFF
  cmake --build "${LLAMA_CPP_DIR}/build" --target llama-quantize -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
  QUANTIZE_BIN="${LLAMA_CPP_DIR}/build/bin/llama-quantize"
fi

echo "=== Step 2: Quantize → ${QUANT} ==="
"${QUANTIZE_BIN}" "${FP16_GGUF}" "${OUT_GGUF}" "${QUANT}"

# Drop the large f16 intermediate to save disk (optional keep via KEEP_F16=1)
if [[ "${KEEP_F16:-0}" != "1" ]]; then
  rm -f "${FP16_GGUF}"
fi

ls -lh "${OUT_GGUF}"
echo "DONE. GGUF ready at ${OUT_GGUF}"
echo "Next: ./scripts/register_model.sh"
