"""
TRL Trainer Backend: LoRA / QLoRA fine-tune of the Student with TRL + PEFT.

Consumes the train.jsonl / valid.jsonl exported by `export_pairs.py` and emits a
LoRA adapter. One file, two hardware targets:

  * HPC (Pawsey Setonix, AMD MI250X / ROCm) — bf16 LoRA, the Canonical Run
    (ADR 0003). Launch via deployment/slurm/train_lora.slurm. Default behaviour.
  * Local Windows / Linux with an NVIDIA GPU — pass `--load-in-4bit` for QLoRA
    (bitsandbytes 4-bit) so a 4B student fits a small (8 GB) laptop dGPU.

Apple Silicon uses the separate MLX backend (scripts/train_local_mlx.sh); the
`training.train` dispatcher picks the right one automatically.

Usage:

    # HPC / any bf16-capable GPU (Pawsey ROCm, or a big NVIDIA card)
    PYTHONPATH=pipeline python -m training.train_trl \
        --data-dir data/processed/training/databricks_ld_foundations \
        --output-dir data/processed/adapters/databricks_ld_foundations_trl

    # Local Windows/Linux NVIDIA laptop (QLoRA, fits ~8 GB VRAM)
    PYTHONPATH=pipeline python -m training.train_trl \
        --data-dir ... --output-dir ... --load-in-4bit
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

DEFAULT_STUDENT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

log = logging.getLogger("training.train_trl")


def log_environment(load_in_4bit: bool) -> None:
    """First thing a batch/remote log must show — fail-loud diagnostics."""
    import torch
    accel = "cpu"
    if torch.cuda.is_available():
        accel = "rocm" if getattr(torch.version, "hip", None) else "cuda"
    log.info("torch=%s | accelerator=%s | cuda_build=%s | hip_build=%s",
             torch.__version__, accel, torch.version.cuda,
             getattr(torch.version, "hip", None))
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            log.info("  gpu[%d]=%s (%.0f GB)", i, p.name, p.total_memory / 1024**3)
    for mod in ("transformers", "trl", "peft", "datasets", "bitsandbytes"):
        try:
            log.info("  %-13s=%s", mod, __import__(mod).__version__)
        except Exception as e:  # noqa: BLE001
            log.info("  %-13s=NOT INSTALLED (%s)", mod, e)
    if load_in_4bit and accel != "cuda":
        sys.exit(
            "[FATAL] --load-in-4bit needs an NVIDIA GPU with CUDA + bitsandbytes. "
            f"Detected accelerator '{accel}'. On Apple Silicon use the MLX backend; "
            "on AMD/ROCm (Pawsey) omit --load-in-4bit to train in bf16.")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    parser = argparse.ArgumentParser(
        description="TRL+PEFT LoRA/QLoRA trainer (HPC bf16 or local NVIDIA 4-bit)")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=DEFAULT_STUDENT_MODEL)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--load-in-4bit", action="store_true",
        help="QLoRA via bitsandbytes 4-bit (NVIDIA/CUDA only) — for small local dGPUs")
    args = parser.parse_args()

    log_environment(args.load_in_4bit)

    # Heavy imports deferred so `--help` works without the training stack.
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    data_dir = Path(args.data_dir)
    for split in ("train.jsonl", "valid.jsonl"):
        if not (data_dir / split).is_file():
            sys.exit(f"[FATAL] missing {data_dir / split} — run the export first")
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.jsonl"),
            "valid": str(data_dir / "valid.jsonl"),
        },
    )
    log.info("dataset: %d train / %d valid",
             len(dataset["train"]), len(dataset["valid"]))

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # bf16 by default (ROCm/HPC). --load-in-4bit loads a quantized base for QLoRA
    # on a small NVIDIA card; we then pass the model object (not a name) to TRL.
    model_arg = args.model
    optim = "adamw_torch"
    if args.load_in_4bit:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        log.info("loading 4-bit base for QLoRA: %s", args.model)
        model_arg = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            ),
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model_arg = prepare_model_for_kbit_training(
            model_arg, use_gradient_checkpointing=True)
        optim = "paged_adamw_8bit"  # offloads optimizer state to host RAM

    training_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_seq_length,
        bf16=True,
        optim=optim,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        seed=args.seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model_arg,
        args=training_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    log.info("adapter written to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 — a submitted job is debugged from logs only
        log.exception("TRAINING FAILED — full traceback follows")
        raise SystemExit(1)
