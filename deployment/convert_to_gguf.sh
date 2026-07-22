#!/usr/bin/env bash
#
# Merge a LoRA adapter into the Student base weights and produce a quantised
# GGUF for Ollama. Works with adapters from either Trainer Backend.
#
# Prerequisites:
#   - llama.cpp checked out and built (for convert_hf_to_gguf.py + llama-quantize):
#       git clone https://github.com/ggml-org/llama.cpp && cmake -B llama.cpp/build llama.cpp && cmake --build llama.cpp/build -t llama-quantize
#   - For a TRL adapter: pip install -r requirements-training-hpc.txt (peft merge)
#   - For an MLX adapter: pip install -r requirements-training-local.txt (mlx fuse)
#
# Usage:
#   deployment/convert_to_gguf.sh mlx <adapter_dir> <output_dir>
#   deployment/convert_to_gguf.sh trl <adapter_dir> <output_dir>

set -euo pipefail

BACKEND="${1:?usage: convert_to_gguf.sh <mlx|trl> <adapter_dir> <output_dir>}"
ADAPTER_DIR="${2:?usage: convert_to_gguf.sh <mlx|trl> <adapter_dir> <output_dir>}"
OUTPUT_DIR="${3:?usage: convert_to_gguf.sh <mlx|trl> <adapter_dir> <output_dir>}"

LLAMA_CPP="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
BASE_MODEL_HF="${EDGE_SLM_STUDENT_HF:-Qwen/Qwen3-4B-Instruct-2507}"
QUANT="${EDGE_SLM_QUANT:-Q4_K_M}"

MERGED_DIR="$OUTPUT_DIR/merged"
mkdir -p "$OUTPUT_DIR"

case "$BACKEND" in
  mlx)
    # Fuse to full-precision HF weights so llama.cpp can convert them.
    python -m mlx_lm fuse \
      --model "$BASE_MODEL_HF" \
      --adapter-path "$ADAPTER_DIR" \
      --save-path "$MERGED_DIR" \
      --dequantize
    ;;
  trl)
    python - "$ADAPTER_DIR" "$MERGED_DIR" <<'PY'
import sys
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

adapter_dir, merged_dir = sys.argv[1], sys.argv[2]
model = AutoPeftModelForCausalLM.from_pretrained(adapter_dir, torch_dtype="bfloat16")
merged = model.merge_and_unload()
merged.save_pretrained(merged_dir)
AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(merged_dir)
PY
    ;;
  *)
    echo "unknown backend: $BACKEND (expected mlx or trl)" >&2
    exit 1
    ;;
esac

F16_GGUF="$OUTPUT_DIR/edge-slm-study-notes.f16.gguf"
FINAL_GGUF="$OUTPUT_DIR/edge-slm-study-notes.$QUANT.gguf"

python "$LLAMA_CPP/convert_hf_to_gguf.py" "$MERGED_DIR" \
  --outfile "$F16_GGUF" --outtype f16
"$LLAMA_CPP/build/bin/llama-quantize" "$F16_GGUF" "$FINAL_GGUF" "$QUANT"

echo "Quantised GGUF: $FINAL_GGUF"
echo "Next:"
echo "  PYTHONPATH=pipeline python scripts/build_modelfile.py --from-ref $FINAL_GGUF --output deployment/Modelfile.study-notes"
echo "  ollama create edge-slm-study-notes -f deployment/Modelfile.study-notes"
