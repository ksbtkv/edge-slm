#!/usr/bin/env python3
"""
Stage 3 — LoRA/QLoRA fine-tuning for the Edge SLM pipeline (Ohm GPU Platform).

Fine-tunes an instruction model (default: Qwen2.5-3B-Instruct) with 4-bit QLoRA
on the Databricks study-note instruction dataset, and writes a LoRA adapter
(optionally a merged fp16 model for Stage 4 GGUF quantization).

Design constraints (from the SOW + the Ohm platform):
  * Runs as a SUBMITTED job with NO interactive access. Every failure must be
    loud and fully logged — you debug from `ohm logs <job>` alone. So: log the
    full environment up front, validate inputs before touching the GPU, and let
    any exception print a complete traceback and exit non-zero.
  * Fully PARAMETERIZED — base model, dataset, LoRA rank, LR, epochs, etc. come
    from a YAML config and/or CLI flags. Changing a run never edits this file.
  * A cheap SMOKE path (`--epochs 1` or `--smoke`) so `ohm test` can catch
    path/import/argument breakage in the notebook pod before a real GPU job.

Target env: PyTorch 2.5.1 / CUDA 12.4 (Ohm `pytorch-llm` image), 1x A100.

Usage (see README):
    python train.py --config config/qwen2.5_3b_qlora.yaml
    python train.py --config config/qwen2.5_3b_qlora.yaml --smoke      # tiny
    python train.py --config config/qwen2.5_3b_qlora.yaml --epochs 1   # ohm test
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ------------------------------------------------------------------------- #
# Args + config are parsed BEFORE importing torch/transformers so that
# HF_HOME (cache location — the #1 Ohm quota killer) is set first, and so a
# bad config fails instantly without paying heavy import cost.
# ------------------------------------------------------------------------- #

DEFAULTS = {
    "base_model": "Qwen/Qwen2.5-3B-Instruct",
    "dataset": "data/sample_alpaca.jsonl",
    "dataset_format": "alpaca",          # alpaca | chat
    "output_dir": "outputs/qwen2.5-3b-lora",
    "hf_home": None,                     # set to redirect HF cache off the quota
    # LoRA
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    # Quantization
    "load_in_4bit": True,
    # Optimisation
    "learning_rate": 2e-4,
    "epochs": 3,
    "max_steps": -1,                     # >0 overrides epochs (smoke / debugging)
    "per_device_batch_size": 2,
    "grad_accum": 8,
    "max_seq_len": 2048,
    "warmup_ratio": 0.03,
    "weight_decay": 0.0,
    "lr_scheduler": "cosine",
    "seed": 42,
    # Checkpointing / logging
    "logging_steps": 10,
    "save_steps": 200,
    "save_total_limit": 3,
    "resume": False,
    # Output
    "merge_adapter": False,              # also save a merged fp16 model (Stage 4)
    "report_to": "none",
}


@dataclass
class TrainConfig:
    base_model: str = DEFAULTS["base_model"]
    dataset: str = DEFAULTS["dataset"]
    dataset_format: str = DEFAULTS["dataset_format"]
    output_dir: str = DEFAULTS["output_dir"]
    hf_home: str | None = DEFAULTS["hf_home"]
    lora_r: int = DEFAULTS["lora_r"]
    lora_alpha: int = DEFAULTS["lora_alpha"]
    lora_dropout: float = DEFAULTS["lora_dropout"]
    lora_target_modules: list[str] = field(
        default_factory=lambda: list(DEFAULTS["lora_target_modules"]))
    load_in_4bit: bool = DEFAULTS["load_in_4bit"]
    learning_rate: float = DEFAULTS["learning_rate"]
    epochs: int = DEFAULTS["epochs"]
    max_steps: int = DEFAULTS["max_steps"]
    per_device_batch_size: int = DEFAULTS["per_device_batch_size"]
    grad_accum: int = DEFAULTS["grad_accum"]
    max_seq_len: int = DEFAULTS["max_seq_len"]
    warmup_ratio: float = DEFAULTS["warmup_ratio"]
    weight_decay: float = DEFAULTS["weight_decay"]
    lr_scheduler: str = DEFAULTS["lr_scheduler"]
    seed: int = DEFAULTS["seed"]
    logging_steps: int = DEFAULTS["logging_steps"]
    save_steps: int = DEFAULTS["save_steps"]
    save_total_limit: int = DEFAULTS["save_total_limit"]
    resume: bool = DEFAULTS["resume"]
    merge_adapter: bool = DEFAULTS["merge_adapter"]
    report_to: str = DEFAULTS["report_to"]


log = logging.getLogger("stage3.train")


def build_config() -> tuple[TrainConfig, bool]:
    """Resolve config from (defaults <- YAML <- CLI). Returns (config, smoke)."""
    ap = argparse.ArgumentParser(description="Stage 3 QLoRA fine-tuning")
    ap.add_argument("--config", help="YAML file with any TrainConfig fields")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny real run (few steps, no merge) — full path on a GPU pod")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate env/config/data/tokenisation then exit; no model "
                         "load, no GPU needed — ideal for `ohm test` on any pod")
    # The commonly-tuned knobs are also plain CLI flags so a run can be changed
    # without editing YAML. Anything not passed keeps its YAML/default value.
    ap.add_argument("--base-model")
    ap.add_argument("--dataset")
    ap.add_argument("--output-dir")
    ap.add_argument("--hf-home")
    ap.add_argument("--lora-r", type=int)
    ap.add_argument("--lora-alpha", type=int)
    ap.add_argument("--learning-rate", type=float)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--per-device-batch-size", type=int)
    ap.add_argument("--grad-accum", type=int)
    ap.add_argument("--max-seq-len", type=int)
    ap.add_argument("--no-4bit", action="store_true", help="disable 4-bit (bf16 LoRA)")
    ap.add_argument("--merge-adapter", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    values = dict(DEFAULTS)
    if args.config:
        import yaml
        cfg_path = Path(args.config)
        if not cfg_path.is_file():
            sys.exit(f"[FATAL] --config not found: {cfg_path}")
        loaded = yaml.safe_load(cfg_path.read_text()) or {}
        unknown = set(loaded) - set(DEFAULTS)
        if unknown:
            sys.exit(f"[FATAL] unknown config keys: {sorted(unknown)}")
        values.update(loaded)

    # CLI overrides (only when explicitly provided)
    cli = {
        "base_model": args.base_model, "dataset": args.dataset,
        "output_dir": args.output_dir, "hf_home": args.hf_home,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "learning_rate": args.learning_rate, "epochs": args.epochs,
        "max_steps": args.max_steps,
        "per_device_batch_size": args.per_device_batch_size,
        "grad_accum": args.grad_accum, "max_seq_len": args.max_seq_len,
    }
    for k, v in cli.items():
        if v is not None:
            values[k] = v
    if args.no_4bit:
        values["load_in_4bit"] = False
    if args.merge_adapter:
        values["merge_adapter"] = True
    if args.resume:
        values["resume"] = True

    cfg = TrainConfig(**values)

    if args.smoke:
        # Deterministic, fast, and self-contained: a handful of steps, no merge.
        cfg.max_steps = 5
        cfg.per_device_batch_size = 1
        cfg.grad_accum = 1
        cfg.save_steps = 1_000_000
        cfg.merge_adapter = False
    return cfg, args.smoke, args.dry_run


# ------------------------------------------------------------------------- #
# Environment logging — the first thing a failed job's log must show.
# ------------------------------------------------------------------------- #
def log_environment(cfg: TrainConfig) -> None:
    import torch
    log.info("=" * 68)
    log.info("Stage 3 fine-tuning — environment")
    log.info("  python           : %s", sys.version.split()[0])
    log.info("  torch            : %s", torch.__version__)
    log.info("  cuda available   : %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            log.info("  gpu[%d]           : %s (%.0f GB)", i, p.name,
                     p.total_memory / 1024**3)
    for mod in ("transformers", "peft", "accelerate", "datasets", "bitsandbytes"):
        try:
            m = __import__(mod)
            log.info("  %-16s : %s", mod, getattr(m, "__version__", "?"))
        except Exception as e:  # noqa: BLE001 — report, don't crash the banner
            log.info("  %-16s : NOT INSTALLED (%s)", mod, e)
    log.info("  HF_HOME          : %s", os.environ.get("HF_HOME", "(default)"))
    log.info("=" * 68)


# ------------------------------------------------------------------------- #
# Dataset -> tokenised, completion-only-masked examples.
# ------------------------------------------------------------------------- #
def load_records(path: str, fmt: str) -> list[dict]:
    p = Path(path)
    if not p.is_file():
        sys.exit(f"[FATAL] dataset not found: {p}")
    records = []
    with p.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                sys.exit(f"[FATAL] {p}:{lineno} is not valid JSON: {e}")
    if not records:
        sys.exit(f"[FATAL] dataset is empty: {p}")
    log.info("loaded %d records from %s (format=%s)", len(records), p, fmt)
    return records


def record_to_messages(rec: dict, fmt: str) -> list[dict]:
    """Normalise a dataset record to chat messages [user, assistant]."""
    if fmt == "chat":
        msgs = rec.get("messages")
        if not msgs:
            sys.exit("[FATAL] chat format requires a 'messages' field")
        return msgs
    # alpaca: instruction (+ optional input) -> user ; output -> assistant
    instruction = (rec.get("instruction") or "").strip()
    user_input = (rec.get("input") or "").strip()
    output = (rec.get("output") or rec.get("response") or "").strip()
    if not instruction or not output:
        sys.exit("[FATAL] alpaca records need non-empty 'instruction' and 'output'")
    user = instruction if not user_input else f"{instruction}\n\n{user_input}"
    return [{"role": "user", "content": user},
            {"role": "assistant", "content": output}]


def build_dataset(records, fmt, tokenizer, max_len):
    """Tokenise with completion-only labels (prompt tokens masked to -100)."""
    examples = []
    skipped = 0
    for rec in records:
        messages = record_to_messages(rec, fmt)
        # Prompt = everything up to (and including) the assistant turn header.
        prompt_ids = tokenizer.apply_chat_template(
            messages[:-1], add_generation_prompt=True, tokenize=True)
        full_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=True)
        if len(full_ids) > max_len:
            full_ids = full_ids[:max_len]
        labels = list(full_ids)
        mask_to = min(len(prompt_ids), len(full_ids))
        for i in range(mask_to):
            labels[i] = -100
        if all(t == -100 for t in labels):  # completion truncated away entirely
            skipped += 1
            continue
        examples.append({
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        })
    if not examples:
        sys.exit("[FATAL] every example was empty after tokenisation — check "
                 "max_seq_len and the dataset")
    if skipped:
        log.warning("skipped %d example(s) whose completion exceeded max_seq_len",
                    skipped)
    log.info("prepared %d training examples", len(examples))
    from datasets import Dataset
    return Dataset.from_list(examples)


class PadCollator:
    """Pad input_ids/attention_mask/labels to the batch max (labels pad = -100)."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        import torch
        maxlen = max(len(b["input_ids"]) for b in batch)

        def pad(seq, value):
            return seq + [value] * (maxlen - len(seq))

        return {
            "input_ids": torch.tensor([pad(b["input_ids"], self.pad_id) for b in batch]),
            "attention_mask": torch.tensor([pad(b["attention_mask"], 0) for b in batch]),
            "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
        }


# ------------------------------------------------------------------------- #
# Train
# ------------------------------------------------------------------------- #
def run(cfg: TrainConfig, smoke: bool, dry_run: bool) -> None:
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
        Trainer, TrainingArguments, set_seed,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    set_seed(cfg.seed)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps(asdict(cfg), indent=2))

    # --dry-run validates the parts that break most often (config, dataset
    # format, chat template, tokeniser) without loading the model or a GPU.
    if dry_run:
        log.info("[dry-run] loading tokenizer: %s", cfg.base_model)
        tok = AutoTokenizer.from_pretrained(cfg.base_model)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        recs = load_records(cfg.dataset, cfg.dataset_format)
        build_dataset(recs, cfg.dataset_format, tok, cfg.max_seq_len)
        log.info("[dry-run] OK — env, config, dataset, and tokenisation all valid. "
                 "No model was loaded; submit a real job to train.")
        return

    if cfg.load_in_4bit and not torch.cuda.is_available():
        sys.exit("[FATAL] --load_in_4bit needs CUDA; none available. Use "
                 "--no-4bit for a CPU smoke test, or run on a GPU node.")

    log.info("loading tokenizer: %s", cfg.base_model)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = load_records(cfg.dataset, cfg.dataset_format)
    train_ds = build_dataset(records, cfg.dataset_format, tokenizer, cfg.max_seq_len)

    log.info("loading base model (4bit=%s): %s", cfg.load_in_4bit, cfg.base_model)
    model_kwargs = {"torch_dtype": torch.bfloat16}
    if cfg.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model, attn_implementation="flash_attention_2", **model_kwargs)
    except Exception as e:  # flash-attn optional; fall back to SDPA
        log.warning("flash_attention_2 unavailable (%s); using sdpa", e)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model, attn_implementation="sdpa", **model_kwargs)

    model.config.use_cache = False
    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()

    model = get_peft_model(model, LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules, bias="none", task_type="CAUSAL_LM",
    ))
    trainable, total = model.get_nb_trainable_parameters()
    log.info("LoRA trainable params: %s / %s (%.4f%%)",
             f"{trainable:,}", f"{total:,}", 100 * trainable / total)

    args = TrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=torch.cuda.is_available(),
        optim="paged_adamw_8bit" if cfg.load_in_4bit else "adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=cfg.report_to,
        seed=cfg.seed,
        logging_first_step=True,
    )

    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )

    log.info("starting training (%s)...", "SMOKE" if smoke else "full")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=cfg.resume or None)
    log.info("training finished in %.1f s", time.time() - t0)

    adapter_dir = out / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    log.info("saved LoRA adapter -> %s", adapter_dir)

    if cfg.merge_adapter:
        log.info("merging adapter into base (fp16) for Stage 4 GGUF...")
        merge_adapter(cfg, tokenizer, out)

    log.info("DONE. adapter at %s", adapter_dir)


def merge_adapter(cfg, tokenizer, out: Path) -> None:
    """Reload base in fp16 (not 4-bit) and merge the adapter for GGUF export."""
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, torch_dtype=torch.float16, device_map="auto")
    merged = PeftModel.from_pretrained(base, out / "adapter").merge_and_unload()
    merged_dir = out / "merged"
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    log.info("saved merged fp16 model -> %s", merged_dir)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        stream=sys.stdout,
    )
    cfg, smoke, dry_run = build_config()

    # Set HF cache BEFORE any transformers import (quota management on Ohm).
    if cfg.hf_home:
        os.environ["HF_HOME"] = cfg.hf_home

    try:
        log_environment(cfg)
        log.info("effective config:\n%s", json.dumps(asdict(cfg), indent=2))
        run(cfg, smoke, dry_run)
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 — a submitted job must log the full trace
        log.exception("TRAINING FAILED — full traceback follows")
        sys.exit(1)


if __name__ == "__main__":
    main()
