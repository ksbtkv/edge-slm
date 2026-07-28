"""
Cross-platform fine-tuning dispatcher.

One command, three hardware targets. Detects the machine and routes to the right
LoRA backend, all consuming the identical exported chat JSONL and emitting the
same adapter format (ADR 0003):

    Apple Silicon (Mac)        -> MLX QLoRA        (scripts/train_local_mlx.sh)
    NVIDIA GPU (Win/Linux)     -> TRL QLoRA 4-bit  (training.train_trl --load-in-4bit)
    AMD ROCm (Pawsey Setonix)  -> TRL LoRA bf16    (training.train_trl)

Usage:

    PYTHONPATH=pipeline python -m training.train \
        --data-dir data/processed/training/databricks_ld_foundations \
        --output-dir data/processed/adapters/run1

    # force a backend, or bf16 on a big NVIDIA card:
    ... --backend cuda --bf16
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # pipeline/training/train.py -> repo


def detect_backend() -> str:
    """Return one of: mlx | cuda | rocm | cpu | no-torch."""
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "mlx"
    try:
        import torch
    except ImportError:
        return "no-torch"
    if torch.cuda.is_available():
        return "rocm" if getattr(torch.version, "hip", None) else "cuda"
    return "cpu"


def main() -> int:
    ap = argparse.ArgumentParser(description="Cross-platform LoRA fine-tuning dispatcher")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--backend", choices=["auto", "mlx", "cuda", "rocm"],
                    default="auto", help="override auto-detection")
    ap.add_argument("--bf16", action="store_true",
                    help="on NVIDIA, use bf16 LoRA instead of 4-bit QLoRA (needs a big card)")
    args, extra = ap.parse_known_args()

    backend = args.backend if args.backend != "auto" else detect_backend()

    reason = {
        "mlx": "Apple Silicon detected -> MLX (Metal) QLoRA",
        "cuda": "NVIDIA CUDA GPU detected -> TRL QLoRA (4-bit)",
        "rocm": "AMD ROCm GPU detected -> TRL LoRA (bf16)",
    }.get(backend)
    if backend == "no-torch":
        sys.exit("[FATAL] PyTorch not installed and this is not Apple Silicon. "
                 "Install the target requirements: requirements-training-windows.txt "
                 "(NVIDIA) or requirements-training-hpc.txt (ROCm).")
    if backend == "cpu":
        sys.exit("[FATAL] no GPU detected. Fine-tuning a 4B model on CPU is not "
                 "practical — use a Mac (MLX), an NVIDIA laptop, or the HPC path.")
    print(f"[dispatch] {reason}", flush=True)

    if backend == "mlx":
        cmd = ["bash", str(REPO_ROOT / "scripts" / "train_local_mlx.sh"),
               args.data_dir, args.output_dir, *extra]
        env = os.environ.copy()
    else:  # cuda / rocm -> the TRL backend
        cmd = [sys.executable, "-m", "training.train_trl",
               "--data-dir", args.data_dir, "--output-dir", args.output_dir]
        if backend == "cuda" and not args.bf16:
            cmd.append("--load-in-4bit")
        cmd += extra
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT / "pipeline") + os.pathsep + env.get("PYTHONPATH", "")

    print(f"[dispatch] running: {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=REPO_ROOT, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
