# Training run: local MLX QLoRA — `databricks_ld_foundations`

**Status:** completed (development artifact; not the Canonical Run — see ADR 0003)  
**Date:** 2026-07-22 → 2026-07-23  
**Host:** Apple Silicon laptop (≈16 GB unified memory)  
**Command:**

```bash
scripts/train_local_mlx.sh \
    data/processed/training/databricks_ld_foundations \
    data/processed/adapters/databricks_ld_foundations_mlx \
    --max-seq-length 4096
```

## Configuration

| Setting | Value |
|---|---|
| Base model | `mlx-community/Qwen3-4B-Instruct-2507-4bit` |
| Fine-tune type | LoRA (QLoRA on 4-bit base) |
| Train / valid pairs | 640 / 34 (`export_summary.json`) |
| Iters | 1400 (~2 epochs at batch size 1) |
| Batch size | 1 |
| Learning rate | `1e-4` (constant) |
| LoRA layers | 16 |
| LoRA rank / scale / dropout | 8 / 20.0 / 0.0 |
| Max sequence length | 4096 |
| Grad checkpointing | on |
| Save / eval every | 200 iters |
| Trainable params | 0.182% (7.340M / 4022.468M) |

Full mlx-lm config snapshot: `data/processed/adapters/databricks_ld_foundations_mlx/adapter_config.json`.

## Metrics

Validation loss (every 200 iters; 25 val batches):

| Iter | Val loss | Train loss (at report) | Notes |
|---:|---:|---:|---|
| 1 | 1.483 | — | cold start |
| 200 | 0.816 | 0.829 | |
| 400 | 0.785 | 0.887 | |
| **600** | **0.757** | 0.751 | **best val** |
| 800 | 0.798 | 0.668 | val rose |
| 1000 | 0.780 | 0.637 | |
| 1200 | 0.788 | 0.639 | |
| 1400 | 0.832 | 0.471 | final; most overfit |

Throughput (steady state): ~0.03–0.04 it/s, ~90–100 tokens/s.  
Peak memory: ~8.87 GB.  
Tokens trained by iter 1400: ~4.00M.

### Interpretation

Train loss fell steadily through the run while validation loss bottomed at **iter 600** and then rose. That is overfitting on a small pair set — expected for 1400 iters without early stopping. Prefer the iter-600 adapter for fuse / eval / deployment experiments.

`adapters.safetensors` in the adapter dir is the **final** (iter 1400) weights. Numbered checkpoints:

```text
data/processed/adapters/databricks_ld_foundations_mlx/0000200_adapters.safetensors
… 
data/processed/adapters/databricks_ld_foundations_mlx/0000600_adapters.safetensors   ← preferred
…
data/processed/adapters/databricks_ld_foundations_mlx/0001400_adapters.safetensors
data/processed/adapters/databricks_ld_foundations_mlx/adapters.safetensors            ← = 1400
```

## Warnings observed

mlx-lm repeatedly warned that some sequences exceed 4096 tokens (longest observed in-log ≈ 4866) and truncated them. Training still completed; long answers may lose a trailing portion. Follow-up options: pre-split long pairs, filter outliers, or raise `--max-seq-length` if memory allows.

## Recommended next steps

1. Stage the preferred checkpoint for fuse (mlx-lm reads `adapters.safetensors`):

```bash
ADAPTER_SRC=data/processed/adapters/databricks_ld_foundations_mlx
ADAPTER_600=data/processed/adapters/databricks_ld_foundations_mlx_iter600
mkdir -p "$ADAPTER_600"
cp "$ADAPTER_SRC/adapter_config.json" "$ADAPTER_600/"
cp "$ADAPTER_SRC/0000600_adapters.safetensors" "$ADAPTER_600/adapters.safetensors"
```

2. Merge + quantise (requires built `llama.cpp`; see `deployment/convert_to_gguf.sh`):

```bash
deployment/convert_to_gguf.sh mlx \
    data/processed/adapters/databricks_ld_foundations_mlx_iter600 \
    data/processed/gguf_iter600
```

3. Register in Ollama and evaluate against `eval_references.jsonl` (baseline vs tuned) per `docs/finetuning.md` Stage 5–6.

4. Keep Pawsey TRL+PEFT as the Canonical Run for reported results and the Deployed Model (ADR 0003).

## Deployment + evaluation

Artifacts from the preferred iter-600 checkpoint:

| Artifact | Path / name |
|---|---|
| Adapter (staged) | `data/processed/adapters/databricks_ld_foundations_mlx_iter600` |
| GGUF (Q4_K_M) | `data/processed/gguf_iter600/edge-slm-study-notes.Q4_K_M.gguf` |
| Ollama baseline | `edge-slm-baseline` (`FROM qwen3:4b-instruct-2507-q4_K_M`) |
| Ollama tuned | `edge-slm-study-notes` |

Free-tier eval on `eval_references.jsonl` (85 tasks; no `--judge`, no holdout):

| Metric | edge-slm-baseline | edge-slm-study-notes (iter600) |
|---|---:|---:|
| json_valid_rate | 0.9882 | 0.7412 |
| schema_valid_rate | 0.0 | 0.6706 |
| mean_groundedness | 1.0 | 0.8425 |

Metrics files: `data/processed/eval/baseline/metrics.json`, `data/processed/eval/tuned_iter600/metrics.json`.

Tuned gains schema validity (0 → 0.67) at the cost of lower JSON validity and groundedness vs baseline. LLM-as-judge and holdout were **not** run.

### Eval failure analysis

Failure breakdown on tuned iter-600 outputs (`data/processed/eval/tuned_iter600/outputs/`, 85 tasks):

| Bucket | Count | Notes |
|---:|---:|---|
| Fully OK (JSON + schema) | 57 | |
| Invalid JSON | 22 | **21/22** contain `<tool_call>` / `</tool_call>` junk (Qwen3 tool-call contamination); the JSON body is often truncated or internally broken after that |
| Valid JSON, wrong schema | 6 | e.g. `project_usage_notes` not a list; unknown keys such as `important_concepts` |

Sample `raw_output` prefix (task `…abac_common_patterns__c0001`):

```text
</tool_call>

<tool_call>

{"title": "Data Classification and Access Control in Databricks Runtime 18.1+", ...
```

Priority before more training: stop tool-call leakage and recover JSON (parser sanitize + output token budget), then re-eval free tiers.

### Inference hardening (2026-07-23)

Without retraining:

1. **Parser** — `sanitize_study_note_response_text` strips `<tool_call>` / `</tool_call>` tags before `{…}` extraction in `parse_study_note_response`; the eval path applies the same helper before parse (raw cache still stores the model’s original text).
2. **Ollama budget** — Modelfiles regenerated with `PARAMETER num_ctx 8192` and `PARAMETER num_predict 4096` (was `num_ctx 4096`, no `num_predict`). Eval `ollama_generate_fn` passes the same options. Models recreated: `edge-slm-baseline`, `edge-slm-study-notes`.

Parser-only re-score of cached `tuned_iter600` `raw_output` did **not** move rates (still 0.7412 / 0.6706): most invalid bodies were broken or truncated *inside* the JSON, not merely wrapped.

### Free-tier re-eval (v2)

Fresh generate into `data/processed/eval/baseline_v2` and `data/processed/eval/tuned_iter600_v2` (no `--judge`, no holdout):

| Metric | edge-slm-baseline (v2) | edge-slm-study-notes iter600 (v2) | prior baseline | prior tuned |
|---|---:|---:|---:|---:|
| json_valid_rate | 0.9882 | **0.6706** | 0.9882 | 0.7412 |
| schema_valid_rate | 0.0 | 0.6588 | 0.0 | 0.6706 |
| mean_groundedness | 1.0 | 0.8241 | 1.0 | 0.8425 |

Metrics files: `data/processed/eval/baseline_v2/metrics.json`, `data/processed/eval/tuned_iter600_v2/metrics.json`.

Tuned v2 failure breakdown (`tuned_iter600_v2/outputs/`): **56** fully OK, **28** invalid JSON (all 28 still contain `<tool_call>` / `</tool_call>`), **1** valid JSON / wrong schema. Raising `num_predict` / `num_ctx` and parser strip did **not** recover JSON — leakage worsened vs prior (21 → 28 tool-contaminated invalids). When JSON does parse, schema is almost always correct (56/57).

**Implication (plan step 5):** inference hardening alone is insufficient; next iteration should be a short local retrain (stop at best val ≈600, consider lower LR / fewer iters) or proceed to Pawsey Canonical Run with the same early-stop discipline — not more judge/holdout spend yet.
