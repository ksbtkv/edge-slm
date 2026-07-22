"""
HPC Trainer Backend: LoRA fine-tune of the Student with TRL + PEFT.

Consumes the train.jsonl / valid.jsonl exported by `export_pairs.py` and
emits a LoRA adapter. This is the Canonical Run backend (ADR 0003), intended
for Pawsey Setonix GPU nodes (ROCm PyTorch); it also runs on CUDA machines
unchanged. Launch via deployment/slurm/train_lora.slurm.

Usage:

    PYTHONPATH=pipeline python -m training.train_trl \
        --data-dir data/processed/training/databricks_ld_foundations \
        --output-dir data/processed/adapters/databricks_ld_foundations_trl
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_STUDENT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    args = parser.parse_args()

    # Heavy imports deferred so `--help` works without the training stack.
    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    data_dir = Path(args.data_dir)
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.jsonl"),
            "valid": str(data_dir / "valid.jsonl"),
        },
    )

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    training_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_seq_length,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        seed=args.seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=args.model,
        args=training_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Adapter written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
