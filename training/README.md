# Stage 3 — LoRA/QLoRA fine-tuning on the Ohm GPU Platform

Fine-tunes the Stage 0 model (**Qwen2.5-3B-Instruct**) with 4-bit **QLoRA** on the
Databricks study-note instruction dataset, and produces a LoRA adapter (optionally a
merged fp16 model for Stage 4 GGUF quantization).

This is the **HPC leg** of the pipeline. Upstream, the team's ingestion + instruction-pair
work produces the training data; this stage consumes it and trains. It runs as a
**submitted job on Ohm** (Kubernetes + JupyterHub, `ohm` CLI), with no interactive access.

## Target platform (confirmed from the Ohm architecture slides)

| | |
| --- | --- |
| GPU | 1–4× **NVIDIA A100** (1 GPU is plenty for 3B QLoRA) |
| Runtime | PyTorch 2.5.1, CUDA 12.4 (image `pytorch-llm`) |
| Submit | `ohm submit` / smoke via `ohm test` / monitor via `ohm list|status|logs` |
| Storage | `/home/jovyan`, **50 GiB** quota (watch the HuggingFace cache) |

## Layout

```
training/
├── train.py              # the trainer (parameterized, fail-loud, dry-run/smoke modes)
├── requirements.txt      # peft/bitsandbytes/transformers pinned for torch 2.5.1
├── config/
│   └── qwen2.5_3b_qlora.yaml   # all hyperparameters (override on CLI too)
├── data/
│   └── sample_alpaca.jsonl     # tiny sample for the smoke test
└── ohm/
    ├── test.sh           # `ohm test` — dry-run validation, no GPU
    └── submit.sh         # `ohm submit` — the real GPU job
```

## Workflow

From a JupyterLab terminal, with this `training/` directory in your `/home/jovyan`:

```bash
# 1. Smoke test in the current pod — validates config/data/tokenizer/imports, no GPU.
./ohm/test.sh
#    (equivalently: ohm test --command "python train.py --config config/qwen2.5_3b_qlora.yaml --dry-run")

# 2. Submit the real fine-tuning job (1x A100).
./ohm/submit.sh
#    override anything: NAME=run2 EXTRA="--merge-adapter" ./ohm/submit.sh

# 3. Monitor (job name gets a short suffix — `ohm list` shows the exact name).
ohm list
ohm status qwen25-3b-lora-<suffix>
ohm logs   qwen25-3b-lora-<suffix>

# 4. Fetch results — they're already in your home volume:
#    training/outputs/qwen2.5-3b-lora/adapter/   (LoRA adapter)
#    training/outputs/qwen2.5-3b-lora/merged/    (fp16 model, if --merge-adapter → Stage 4)
```

## Parameterization

Nothing about a run requires editing `train.py`. Change hyperparameters in
`config/qwen2.5_3b_qlora.yaml`, or override on the CLI (`python train.py --help`):
`--base-model --dataset --lora-r --lora-alpha --learning-rate --epochs --max-steps
--per-device-batch-size --grad-accum --max-seq-len --no-4bit --merge-adapter --resume`.

## Data format

Default is **Alpaca** JSONL — one object per line with `instruction`, optional `input`,
and `output`. The trainer renders these through the model's chat template and trains
**completion-only** (prompt tokens are masked). To consume the team's chat-style records
instead, set `dataset_format: chat` (each record has a `messages` list). Point `dataset:`
at the real study-note pairs once Stage 2 produces them.

## Storage / quota guardrails (the #1 Ohm gotcha)

The HuggingFace cache under `~/.cache/huggingface` counts against your 50 GiB quota and is
the most common source of "mysterious" write/download failures. If it fills up, redirect it:
set `hf_home:` in the config (or `--hf-home`) to a scratch path. Also prune old
`outputs/*/checkpoint-*` between runs (`save_total_limit` already caps them at 3).

## Design notes

- **Fail-loud, log-everything:** a submitted job is debugged only from `ohm logs`. The
  script logs the full environment (torch/CUDA/GPU/library versions), validates the dataset
  before touching the GPU, and prints a complete traceback + non-zero exit on any error.
- **`--dry-run`** does env+config+data+tokenisation with no model load (any pod, even CPU);
  **`--smoke`** does a few real training steps (GPU pod). Run the dry-run before every
  submit.
- **QLoRA** (4-bit nf4, bf16 compute, double quant) keeps a 3B fine-tune well within a
  single A100. Use `--no-4bit` only for a CPU dry/smoke.
