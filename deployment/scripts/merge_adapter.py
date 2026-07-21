#!/usr/bin/env python3
"""
Merge a Stage 3 LoRA adapter into the base model (fp16) for GGUF conversion.

Only needed when the Ohm training run omitted --merge-adapter. Prefer
re-running Stage 3 with merge_adapter: true when possible.

Usage:
    python scripts/merge_adapter.py \\
        --base-model Qwen/Qwen2.5-3B-Instruct \\
        --adapter ../../training/outputs/qwen2.5-3b-lora/adapter \\
        --output ../../training/outputs/qwen2.5-3b-lora/merged
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HuggingFace base model id or local path",
    )
    ap.add_argument(
        "--adapter",
        type=Path,
        required=True,
        help="Path to Stage 3 adapter/ directory",
    )
    ap.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the merged fp16 model",
    )
    args = ap.parse_args()

    if not args.adapter.is_dir():
        sys.exit(f"[FATAL] Adapter directory not found: {args.adapter}")

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        sys.exit(
            f"[FATAL] Missing dependency: {e}\n"
            "Install peft + transformers + torch, or run merge on Ohm instead."
        )

    print(f"Loading base model {args.base_model} (fp16)...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Merging adapter from {args.adapter}...")
    merged = PeftModel.from_pretrained(base, str(args.adapter)).merge_and_unload()

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {args.output}...")
    merged.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    print(f"DONE. Merged fp16 model at {args.output}")


if __name__ == "__main__":
    main()
