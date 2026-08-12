# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Edge SLM Wizard: a wizard that lets a non-technical End User pick a local small language model, install it, optionally fine-tune it on their own task and data, evaluate it, and run it — entirely on their own machine. Domain vocabulary (End User, Task Template, Model Catalog, Trainer Backend, Teacher, Judge, Student, HPC Submission, ...) is defined in `CONTEXT.md` — read it before making product-shaped changes, since these terms are used verbatim throughout the docs and should be used verbatim in code/comments too.

**Transitional state:** the repo just pivoted (see `docs/adr/0005`) from a prior single-client, single-hardcoded-task pipeline (Databricks L&D "study notes" via Visagio) to this generalized wizard. The domain model and ADRs 0005/0006 describe the target shape; most of the actual code in `pipeline/`, `scripts/`, `docs/ingestion.md`, and `docs/finetuning.md` still reflects the old Databricks-specific pipeline that the wizard is meant to reuse and generalize (Task Templates replacing the one hardcoded task). Don't assume the code already implements Task Templates / Model Catalog / Entry Points — check before relying on it. ADRs 0001-0004 are marked superseded but kept as prior art (canonical-prompt pattern, model licensing trade-offs, dual-backend rationale, status-reporting conventions) for decisions the new product will re-derive rather than invent from scratch.

## Commands

Install (pick the surfaces you need; `requirements.txt` is core/stdlib-only):

```bash
pip install -r requirements-dev.txt          # core + pytest
pip install -r requirements-ingestion-all.txt  # all ingestion formats (PDF, PPTX, media)
pip install -r requirements-enrichment.txt   # Teacher/Judge calls — needs ANTHROPIC_API_KEY
pip install -r requirements-training-local.txt  # MLX (Apple Silicon only)
pip install -r requirements-training-hpc.txt    # TRL+PEFT (install ROCm/CUDA torch separately first)
```

Tests (must set `PYTHONPATH=pipeline`; pytest reads `pytest.ini`, `testpaths = tests`):

```bash
PYTHONPATH=pipeline pytest                          # fast tests only (default)
PYTHONPATH=pipeline pytest tests/test_ingestion_fast.py::test_name -v   # single test
RUN_SLOW_INGESTION=1 PYTHONPATH=pipeline pytest     # opt-in: real PDF/PPTX/audio/video fixtures
PYTHONPATH=pipeline python -m tests.smoke_test_ingestion   # manual smoke report (needs data/raw/ fixtures)
```

Test markers (`pytest.ini`): `slow` (real sample files / extraction models), `media` (audio/video transcription), `optional_dependency` (needs optional ingestion deps).

Pipeline CLI scripts are invoked the same way throughout the docs — `PYTHONPATH=pipeline python scripts/<name>.py ...` — because `pipeline/` is not an installed package, it's put on `sys.path` per-invocation (see `tests/conftest.py` for the equivalent in tests).

Run the Streamlit operator GUI (not pinned in any requirements file yet):

```bash
pip install streamlit
streamlit run GUI/app.py
```

No lint config exists in this repo currently.

## Architecture

The pipeline is a strict linear stage sequence, each stage's output file is the next stage's input — there is no shared mutable state or database. Full how-to detail lives in `docs/ingestion.md` (stages 0-1.5) and `docs/finetuning.md` (stages 2-6); this section is the map between those docs and the code so you know which module to open.

```
source files -> Document/Section -> TextChunk -> study_note_tasks.jsonl
                                                        |
                              Teacher enrichment -> Training Pairs -> LoRA -> eval -> deploy
```

**Stage 1 — Ingestion (`pipeline/ingestion/`).** Every format ingestor is a pure function `(path, **kwargs) -> Document`. `schema.py` defines the `Document`/`Section` contract: a `Section` is a faithful structural unit (page, slide, transcript segment) with optional flat provenance fields — deliberately *not* chunked, since fixed-size windowing is a downstream concern. `dispatch.py` is the single entry point (`ingest(path)`); it routes by extension via a central registry (`_wire_default_ingestors`) rather than decorator self-registration, specifically so a format can never go silently missing and each `*_ingestor.py` stays independently importable/testable. Optional-dependency ingestors (PDF, PPTX, audio/video) are wired through a lazy proxy so `import dispatch` never requires every backend installed. `quality.py` holds shared ingest-time filters (e.g. dropping thin markdown sections).

**Stage 1.5 — Chunking + source pack (`pipeline/ingestion/chunking.py`, `source_pack.py`, `source_manifest.py`).** `chunking.py` turns `Document.sections` into model-sized `TextChunk`s (target/max/min word counts, overlap) — this is where the Section→Chunk boundary that Stage 1 deliberately avoided actually happens. `source_pack.py` builds the full dataset artifact from a curated manifest (`build_source_pack`) or an ad-hoc folder (`build_source_pack_from_folder`), producing `study_note_tasks.jsonl` — the handoff artifact into enrichment. `study_notes_schema.py` currently hardcodes the one Databricks study-note task's prompt/output schema; this is the piece a future Task Template abstraction generalizes.

**Stage 2 — Teacher enrichment (`pipeline/enrichment/`, `scripts/enrich_tasks.py`).** Calls the Teacher (Claude Haiku via the Batch API) over each task, validates every response against the study-note schema (`study_note_validation.py`), retries once on validation failure, and rejects persistent failures. Resumable at task granularity via `batch_state.json`.

**Stage 3 — Training-pair export (`pipeline/training/export_pairs.py`).** Converts validated Teacher notes into chat-format Training Pairs. Every pair uses the **Canonical System Prompt** (`pipeline/training/canonical_prompt.py`) — a short, fixed, byte-identical-at-inference system prompt — never the verbose Teacher prompt (locked contract, see `docs/adr/0001`: changing it invalidates trained models). Also emits `eval_references.jsonl` and `holdout_references.jsonl`, which must never be trained on.

**Stage 4 — Fine-tuning: two Trainer Backends behind one contract (`docs/adr/0003`, generalized by `docs/adr/0006`).** Both consume the same `train.jsonl`/`valid.jsonl` and emit a LoRA adapter. Local MLX (`scripts/train_local_mlx.sh`, Apple Silicon only — CUDA/ROCm stack doesn't run on the dev machine) is the fully-usable local dev path. Pawsey Setonix TRL+PEFT (`pipeline/training/train_trl.py`, `deployment/slurm/train_lora.slurm`) is the Canonical Run backend for reported results. The two adapters are not bit-identical — don't treat local MLX runs as canonical.

**Stage 5 — Evaluation (`pipeline/evaluation/`, `scripts/run_eval.py`).** Three tiers, always run for both baseline and tuned models served identically via Ollama: structural (JSON/schema validity, free), groundedness (`groundedness.py` — claims must trace back to the source chunk, free/deterministic), and Judge (`judge.py` — Claude Sonnet, deliberately a different/stronger model than the Teacher, scores against the Teacher Reference; needs `ANTHROPIC_API_KEY`). `holdout_references.jsonl` is spent exactly once, at the end, never used to tune.

**Stage 6 — Deployment (`deployment/`).** `convert_to_gguf.sh` merges the LoRA adapter and quantizes; `scripts/build_modelfile.py` generates the Ollama Modelfile embedding the Canonical System Prompt; `deployment/open-webui/` is the end-user chat front-end pointed at the host Ollama.

**Two UIs, two audiences** (`docs/gui.md`): the Streamlit Pipeline Runner (`GUI/app.py`) is operator tooling driving prepare -> enrich -> export -> MLX smoke-train -> handover; Open WebUI (`deployment/open-webui/`) is the end-user front-end for the deployed model. The Streamlit app's "final report" is a pipeline-integrity/handover check, not the model-quality eval harness — use `scripts/run_eval.py` for that.

**Git-tracked vs local-only data** (see the table in `docs/ingestion.md`): `pipeline/`, `tests/`, `scripts/`, `data/manifests/`, and `data/processed/training/` are tracked; `data/raw/`, other `data/processed/*` subdirs, `logs/`, and `data/gui_runs/` are gitignored (large/generated).

## Key conventions worth knowing before editing

- Ingestion `Document`/`Section` schema is JSON-native-only (str/int/float/bool/None/list/dict) so it round-trips through plain `json.dumps` — don't add a field of a richer type without stringifying at the boundary.
- `raw_text` on a `Section` must be byte-faithful to the extractor's output (no cleaning applied) or set to `None` — a half-cleaned `raw_text` silently breaks downstream re-cleaning that depends on it being faithful.
- Adding an ingestion format means two visible touches in `dispatch.py`'s `_wire_default_ingestors` (a new ingestor module + one `_register_ingestor` line) — no decorator magic, on purpose.
- The Canonical System Prompt (`pipeline/training/canonical_prompt.py`) is a locked train/inference contract; changing it requires retraining, not just editing the Modelfile.
