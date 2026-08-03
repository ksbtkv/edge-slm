# Edge SLM — Project Status Report

**Audience:** Visagio (client) and UWA supervisors  
**Repository:** [github.com/ksbtkv/edge_slm](https://github.com/ksbtkv/edge_slm)  
**As of:** 2026-08-03  
**Latest commit at write time:** `b6f662d`

This is the living stakeholder snapshot. Slide-oriented excerpts live in
`docs/client_presentation.md` and must stay consistent with this file.
Domain vocabulary: `CONTEXT.md`. How-to detail: `docs/ingestion.md`,
`docs/finetuning.md`, `docs/gui.md`. Decisions: `docs/adr/`.

---

## 1. Executive summary

The Edge SLM project is building an **offline, edge-deployable small language
model** that turns Databricks learning content into structured **Study Notes** —
starting with **Databricks Data Engineering Foundations**.

**Where we are:** the full lifecycle is implemented in this repo. Data
preparation, Teacher enrichment, and training-pair export have been run.
Local MLX LoRA runs and free-tier eval exist as **development evidence**. The
**Canonical Run** (Pawsey), **Holdout Run**, and a production-ready
**Deployed Model** have **not** been completed.

---

## 2. Project objective

| Goal | Description |
|---|---|
| Target use case | Offline edge model: paste content → structured Study Note |
| First domain | Databricks Data Engineering Foundations |
| First task type | Documentation/transcript chunk → structured study notes (JSON) |
| Design principle | Scriptable, provenance-preserving pipeline suitable for future domains |

---

## 3. Status conventions

| Column | Allowed values |
|---|---|
| **Implemented** | `Not started` / `Partial` / `Yes` — code exists in this repo |
| **Executed** | `Not run` / `Local-only` / `Tracked` / `Canonical` / `Holdout` |

**Executed grades:**

- **Local-only** — produced on a machine; artifact not in git (and no committed run note)
- **Tracked** — citable from git (`docs/training_runs/`, tracked `data/processed/training/`, or equivalent committed evidence)
- **Canonical** — Pawsey Trainer Backend run designated for reported results and the Deployed Model (ADR 0003)
- **Holdout** — single final evaluation on holdout references; run once at the end

---

## 4. Pipeline status

| Stage | Implemented | Executed | Deliverable |
|---|---|---|---|
| **0 — Acquisition** | Yes | Local-only | Transcripts + doc exports under `data/raw/` (gitignored) |
| **1 — Ingestion** | Yes | Tracked | `Document` JSON contract + tests (`docs/ingestion.md`) |
| **1.5 — Source pack + chunking** | Yes | Tracked | 792 tasks documented in `docs/ingestion.md` (pack JSONL itself gitignored) |
| **2 — Teacher enrichment** | Yes | Tracked | Validated notes → evidenced by tracked export (`export_summary.json`: 5 rejects / 792 tasks) |
| **3 — Training-pair export** | Yes | Tracked | `train.jsonl` / `valid.jsonl` + eval/holdout references under `data/processed/training/` |
| **4 — LoRA fine-tuning** | Yes | Tracked | Local MLX adapters (dev). **Canonical (Pawsey): Not run** |
| **5 — Evaluation** | Yes | Tracked | Free-tier baseline-vs-tuned (no Judge, no Holdout). **Holdout: Not run** |
| **6 — Deployment** | Yes | Tracked (Ollama) / Not run (Open WebUI) | Local GGUF → Ollama path in MLX run notes; Open WebUI compose ready. **Not a production Deployed Model** |

Stage 0 model-selection paperwork remains external to this repo (project PDFs).

---

## 5. Operator and end-user UIs

| UI | Implemented | Executed | Role |
|---|---|---|---|
| **Streamlit Pipeline Runner** (`GUI/app.py`) | Yes | Not run | Operator tooling for local prepare → enrich → export → MLX smoke → handover |
| **Open WebUI** (`deployment/open-webui/`) | Yes | Not run | End-user front-end for the Deployed Model (paste → Study Note) |

How-to: [`docs/gui.md`](gui.md).

---

## 6. Evidence summary

- **Source pack (local):** 792 study-note tasks from 303 enabled sources; train/eval/holdout 678/86/28. Details: `docs/ingestion.md` (pack snapshot).
- **Enrichment → export (tracked):** `data/processed/training/databricks_ld_foundations/` — 640 train / 34 valid pairs; 85 eval + 28 holdout Teacher References; 5 tasks skipped unenriched.
- **MLX v1 (tracked, dev):** [2026-07-22 run note](training_runs/2026-07-22_mlx_databricks_ld_foundations.md) — best val at iter 600; later iters overfit. Not the Canonical Run.
- **MLX v2 (tracked, dev):** [2026-07-29 run note](training_runs/2026-07-29_mlx_databricks_ld_foundations_v2.md) — 600 iters, LR `5e-5`, ≤4096 filter; val **0.776**.
- **Free-tier eval (tracked narrative, dev — not Holdout):** tuned MLX v2 vs baseline — JSON **0.835** / schema **0.741** (prior tuned iter600: 0.671 / 0.659). Full tables in the v2 run note.
- **Streamlit Pipeline Runner:** code in git (`GUI/app.py`); no tracked run narrative — Executed **Not run**. See [`docs/gui.md`](gui.md).
- **Open WebUI:** compose in git; not evidenced in `docs/training_runs/` — Executed **Not run**. See [`docs/gui.md`](gui.md).
- **Judge / Holdout:** not spent.
- **Pawsey Canonical Run:** not run.

---

## 7. Non-claims

- No **Canonical Run** results yet — local MLX metrics are **development only**.
- No **Holdout** numbers yet — free-tier eval on the eval split is not final.
- No claim of a **production-ready / client-shipped Deployed Model** until that path is Executed beyond local Ollama experiments.

---

## 8. Next (ordered)

1. **Pawsey Canonical Run** — TRL+PEFT LoRA on Setonix (ADR 0003)
2. **Tracked eval compare** — baseline vs Canonical-tuned (structural, groundedness; Judge as needed)
3. **Deployment path** — merge → GGUF → Ollama → Open WebUI for the Canonical adapter
4. **Holdout Run** — once, after iteration stops; never used to tune

Optional data-prep follow-ups (not on the critical path): deferred long transcripts; `pl_associate_*` transcript cleanup if groundedness suffers — see `docs/ingestion.md`.

---

## 9. Links

| Doc | Role |
|---|---|
| [`CONTEXT.md`](../CONTEXT.md) | Domain glossary |
| [`docs/ingestion.md`](ingestion.md) | Stages 0–1.5 how-to + pack snapshot |
| [`docs/finetuning.md`](finetuning.md) | Stages 2–6 how-to + recorded runs table |
| [`docs/gui.md`](gui.md) | Streamlit Pipeline Runner + Open WebUI how-to |
| [`docs/adr/`](adr/) | Locked decisions (incl. [ADR 0004](adr/0004-stakeholder-status-report.md) status conventions) |
| [`docs/training_runs/`](training_runs/) | Dated MLX run narratives |
| [`docs/client_presentation.md`](client_presentation.md) | Talk deck excerpt (must sync with this report) |
