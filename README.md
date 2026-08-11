# Edge SLM Wizard

A wizard that lets a non-technical End User choose a local small language model, install it, optionally fine-tune it on their own task and data, evaluate it, and run it — entirely on their own machine.

**Operator / Deployed Model UIs:** [`docs/gui.md`](docs/gui.md).

This repository carries the full lifecycle: data preparation (`docs/ingestion.md`), Teacher enrichment, fine-tuning, evaluation, and end-user deployment (`docs/finetuning.md`). Domain vocabulary lives in `CONTEXT.md`; key decisions in `docs/adr/`.

Grew out of the Visagio/Databricks study-notes club project, archived at [`edge_slm-visagio-archive`](https://github.com/ksbtkv/edge_slm-visagio-archive) — see [`docs/adr/0005`](docs/adr/0005-repo-split-from-visagio-project.md) for why this is a separate repo rather than a pivot in place.
