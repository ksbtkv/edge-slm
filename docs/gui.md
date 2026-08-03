# Operator and end-user UIs

Two browser surfaces ship with this repo. They serve different audiences.

| UI | Role | Path |
|---|---|---|
| **Streamlit Pipeline Runner** | Operator tooling — upload sources and drive a local prepare → enrich → export → MLX smoke-train → handover flow | `GUI/app.py` |
| **Open WebUI** | End-user front-end for the **Deployed Model** — paste content, receive a Study Note (via host Ollama) | `deployment/open-webui/docker-compose.yml` |

Stakeholder status for both: `docs/project_status_report.md`. Domain vocabulary (Deployed Model, Study Note): `CONTEXT.md`. Full Stage 6 merge/GGUF/Ollama steps: `docs/finetuning.md`.

---

## Streamlit Pipeline Runner

Use this when you want an interactive local run of pipeline stages without typing every CLI command. It does **not** replace the Databricks curated source-pack workflow in `docs/ingestion.md`, and it is **not** the end-user Deployed Model UI.

### Install and start

`streamlit` is **not** pinned in any `requirements*.txt` yet. Install it into your project environment, then from the repo root:

```bash
pip install streamlit
streamlit run GUI/app.py
```

### Workflow (app steps)

Runs write under `data/gui_runs/`.

1. **Prepare files** — upload PDF/PPTX/text/Markdown/audio/video (and optional YouTube URL); builds a folder pack / study-note tasks (`scripts/build_folder_pack.py`).
2. **Generate examples** — sample Teacher enrichment (`scripts/enrich_tasks.py`; needs `ANTHROPIC_API_KEY` / UI key field).
3. **Create dataset** — export Training Pairs (`scripts/export_training_pairs.py`).
4. **Test training** — short local MLX smoke train (`scripts/train_local_mlx.sh`, 10 iters).
5. **Final report** — pipeline integrity / handover report in the app’s “Final report” view.

**Caveat:** the GUI’s final-report step is a **pipeline integrity / handover** check. It is **not** the full model-quality evaluation harness (`scripts/run_eval.py` — structural, groundedness, Judge). Use Stage 5 in `docs/finetuning.md` for reported eval metrics.

Ad-hoc folder packs can also be built from the CLI (`scripts/build_folder_pack.py`); see `docs/ingestion.md`.

---

## Open WebUI (Deployed Model)

Use this after a GGUF is registered in Ollama as `edge-slm-study-notes`. Supported interaction (CONTEXT.md): **paste Databricks content, receive a structured Study Note**. Free-form Q&A is out of scope for this phase.

### Prerequisites

1. Merge + quantise the adapter (`deployment/convert_to_gguf.sh`) — details in `docs/finetuning.md` Stage 6.
2. Build the Modelfile and create the Ollama model:

```bash
PYTHONPATH=pipeline python scripts/build_modelfile.py \
    --from-ref data/processed/gguf/edge-slm-study-notes.Q4_K_M.gguf \
    --output deployment/Modelfile.study-notes
ollama create edge-slm-study-notes -f deployment/Modelfile.study-notes
```

(Adjust the GGUF path to your actual artifact.)

### Start

```bash
docker compose -f deployment/open-webui/docker-compose.yml up -d
```

Open [http://localhost:3000](http://localhost:3000) and select **edge-slm-study-notes**. The compose file points at the host Ollama (`host.docker.internal:11434`) and disables signup for a single-purpose local deployment.
