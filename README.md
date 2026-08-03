# 26W-ADV-VisagioSLM
In this project, we will be working with Visagio on a Small Language Model (SLM) leveraging access to the Pawsey Supercomputing Research Centre. The project will involve fine-tuning open-source SLMs and edge models on domain-specific data using parameter-efficient techniques such as LoRA and QLoRA, running at scale on HPC infrastructure.

**Project status (stakeholders):** [`docs/project_status_report.md`](docs/project_status_report.md) — living Implemented/Executed snapshot for Visagio and UWA supervisors. Talk deck excerpt: [`docs/client_presentation.md`](docs/client_presentation.md) (must sync with the status report). Operator / Deployed Model UIs: [`docs/gui.md`](docs/gui.md).

This repository carries the full lifecycle: data preparation (`docs/ingestion.md`), Teacher enrichment, fine-tuning, evaluation, and end-user deployment (`docs/finetuning.md`). Domain vocabulary lives in `CONTEXT.md`; key decisions in `docs/adr/`.
