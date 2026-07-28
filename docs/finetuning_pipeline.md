# Fine-tuning pipeline — one contract, runs everywhere

The same LoRA fine-tune of the Qwen3-4B student runs **locally on a Mac**, **locally on a
Windows/Linux NVIDIA laptop**, and **on HPC (Pawsey Setonix)** — from one data contract,
producing one adapter format that feeds one deploy + eval path. Local runs are the
development path; the Pawsey run is the canonical, deployed result (ADR 0003).

```mermaid
flowchart LR
    D[train.jsonl / valid.jsonl<br/>chat messages, canonical prompt] --> DISP{training.train<br/>auto-detect hardware}
    DISP -->|Apple Silicon| MLX[MLX QLoRA 4-bit<br/>train_local_mlx.sh]
    DISP -->|NVIDIA CUDA| CUDA[TRL QLoRA 4-bit<br/>train_trl.py --load-in-4bit]
    DISP -->|AMD ROCm / Pawsey| ROCM[TRL LoRA bf16<br/>train_trl.py]
    MLX --> A[LoRA adapter]
    CUDA --> A
    ROCM --> A
    A --> M[merge → GGUF → Ollama] --> E[eval harness]
```

## Backend matrix

| Target | Hardware | Backend | Quant | Command |
| --- | --- | --- | --- | --- |
| **Mac** | Apple Silicon (Metal) | MLX `mlx_lm lora` | 4-bit | `scripts/train_local_mlx.sh` |
| **Windows / Linux** | NVIDIA dGPU (incl. ~8 GB laptops) | TRL + PEFT + bitsandbytes | 4-bit QLoRA | `scripts/train_local_windows.ps1` |
| **HPC (canonical)** | Pawsey Setonix, AMD MI250X | TRL + PEFT | bf16 LoRA | `deployment/slurm/train_lora.slurm` |

All three go through the unified dispatcher, or can be called directly.

## Shared data contract

Both backends consume the **identical** exported JSONL (`data/processed/training/.../{train,valid}.jsonl`)
— chat format, one object per line:

```json
{"messages": [
  {"role": "system", "content": "You turn Databricks learning content into structured study notes. ..."},
  {"role": "user", "content": "<source content>"},
  {"role": "assistant", "content": "<structured study notes>"}
]}
```

The system prompt is the canonical study-note prompt (pinned by `canonical_prompt_sha256` in
`export_summary.json`), so every backend trains on exactly the same objective.

## Run it

**Unified (auto-detects the machine):**
```bash
PYTHONPATH=pipeline python -m training.train \
  --data-dir data/processed/training/databricks_ld_foundations \
  --output-dir data/processed/adapters/run1
```
It prints which backend it chose and why, then runs it.

**Mac (explicit):**
```bash
pip install -r requirements-training-local.txt        # mlx-lm
scripts/train_local_mlx.sh \
  data/processed/training/databricks_ld_foundations \
  data/processed/adapters/databricks_ld_foundations_mlx
```

**Windows NVIDIA (explicit):**
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-training-windows.txt
.\scripts\train_local_windows.ps1 `
  data\processed\training\databricks_ld_foundations `
  data\processed\adapters\databricks_ld_foundations_win
```

**HPC / Pawsey (canonical):**
```bash
# ROCm torch wheel + requirements-training-hpc.txt (see the SLURM header), then:
sbatch deployment/slurm/train_lora.slurm
```

## Presentation demo script

A tight 3-part live demo that tells the "runs everywhere" story:

1. **One contract** — show one `train.jsonl` line (system/user/assistant) and note both
   backends read it unchanged.
2. **Runs anywhere** — run the unified dispatcher on the **Mac** live; it prints
   `Apple Silicon detected -> MLX (Metal) QLoRA` and starts training. Show the same command's
   detection output captured on a **Windows NVIDIA** box (`-> TRL QLoRA 4-bit`) and the
   **Pawsey** SLURM submission (`-> TRL LoRA bf16`).
3. **One deploy + eval** — show the resulting adapter → merged GGUF → `ollama run` producing
   structured study notes, scored by the eval harness against the baseline.

Have a **pre-trained Mac adapter** ready (a full run takes a while) so step 3 is instant; kick
off the live Mac run in step 2 just to show it starting.

## Notes / honesty

- **Canonical run:** adapters from different backends are not bit-identical (different
  kernels/quantization), so reported/deployed results come from the **Pawsey** run; local runs
  are for development and demos.
- **Windows VRAM:** QLoRA of a 4B model fits ~8 GB with `--batch-size 1 --grad-accum 16
  --max-seq-length 2048` (the `.ps1` defaults). A machine with **no NVIDIA GPU** can't train
  locally — use the Mac path or HPC.
- **Ohm vs Pawsey:** the earlier Ohm platform slides describe NVIDIA A100/CUDA; the canonical
  HPC target here is Pawsey Setonix (AMD/ROCm). Confirm which is the real deployment cluster
  before the canonical run.
