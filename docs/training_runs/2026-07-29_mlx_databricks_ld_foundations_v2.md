# Training run: local MLX QLoRA v2 — `databricks_ld_foundations_le4096`

**Status:** completed (development artifact; not the Canonical Run — see ADR 0003)  
**Date:** 2026-07-29  
**Host:** Apple Silicon laptop (≈16 GB unified memory)  
**Motivation:** Free-tier v2 showed inference hardening (tool_call strip + higher `num_predict`/`num_ctx`) did **not** recover tuned JSON; all 28 invalids still leaked `<tool_call>`. Retrain with earlier stop and lower LR.

**Command:**

```bash
scripts/train_local_mlx.sh \
    data/processed/training/databricks_ld_foundations_le4096 \
    data/processed/adapters/databricks_ld_foundations_mlx_v2 \
    --iters 600 \
    --learning-rate 5e-5 \
    --max-seq-length 4096
```

## Data filter

Sibling of the original export (untouched): `data/processed/training/databricks_ld_foundations_le4096`.

Pairs whose Qwen3 chat-template token length exceeded 4096 were dropped (`filter_summary.json`):

| Split | Kept | Dropped | Max len before filter |
|---|---:|---:|---:|
| train | 625 | 15 | 4862 |
| valid | 32 | 2 | 4385 |

## Configuration

| Setting | Value |
|---|---|
| Base model | `mlx-community/Qwen3-4B-Instruct-2507-4bit` |
| Fine-tune type | LoRA (QLoRA on 4-bit base) |
| Train / valid pairs | 625 / 32 (filtered) |
| Iters | **600** |
| Batch size | 1 |
| Learning rate | **`5e-5`** (was `1e-4`) |
| LoRA layers | 16 |
| LoRA rank / scale / dropout | 8 / 20.0 / 0.0 |
| Max sequence length | 4096 |
| Grad checkpointing | on |
| Save / eval every | 200 iters |
| Trainable params | 0.182% (7.340M / 4022.468M) |

Full mlx-lm config: `data/processed/adapters/databricks_ld_foundations_mlx_v2/adapter_config.json`.

## Metrics

Validation loss (every 200 iters; 25 val batches):

| Iter | Val loss | Train loss (at report) | Notes |
|---:|---:|---:|---|
| 1 | 1.555 | — | cold start |
| 200 | 0.790 | 0.844 | |
| 400 | 0.793 | 0.836 | slight rise |
| **600** | **0.776** | 0.794 | **best / final** |

Throughput (steady state): ~0.026–0.034 it/s, ~75–88 tokens/s.  
Peak memory: ~8.87 GB.  
Tokens trained by iter 600: ~1.69M.  
Log: `logs/mlx_v2_train.log`.

`adapters.safetensors` = iter-600 weights (`0000600_adapters.safetensors`).

## Deployment + free-tier eval

| Artifact | Path / name |
|---|---|
| Adapter | `data/processed/adapters/databricks_ld_foundations_mlx_v2` |
| GGUF (Q4_K_M) | `data/processed/gguf_mlx_v2/edge-slm-study-notes.Q4_K_M.gguf` |
| Ollama tuned | `edge-slm-study-notes` (recreated from v2 GGUF) |

Free-tier eval on `eval_references.jsonl` (85 tasks; no `--judge`, no holdout) → `data/processed/eval/tuned_mlx_v2/`:

| Metric | baseline_v2 | tuned_mlx_v2 | prior tuned_iter600_v2 |
|---|---:|---:|---:|
| json_valid_rate | 0.9882 | **0.8353** | 0.6706 |
| schema_valid_rate | 0.0 | **0.7412** | 0.6588 |
| mean_groundedness | 1.0 | 0.8529 | 0.8241 |

Failure breakdown (`tuned_mlx_v2/outputs/`): **63** fully OK, **14** invalid JSON (**13/14** still tool_call-contaminated), **8** valid JSON / wrong schema.

Versus prior inference-only v2: JSON +16.5 pp, schema +8.2 pp, tool-contaminated invalids 28 → 13. Still short of baseline JSON (~0.99) and the aspirational ≥0.9 bar, but clearly recovered vs 0.67 while schema strengthened.

Metrics: `data/processed/eval/tuned_mlx_v2/metrics.json` (compare vs `baseline_v2`).
