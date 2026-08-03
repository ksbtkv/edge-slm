# Living stakeholder status report with Implemented/Executed evidence grades

Client and supervisors need one honest as-of snapshot, but June’s long status narrative and the July presentation drifted apart and blurred “code exists” with “we ran it” and “these are final numbers.” We decided that `docs/project_status_report.md` is the living stakeholder source of truth: each pipeline stage reports **Implemented** (`Not started` / `Partial` / `Yes`) and **Executed** (`Not run` / `Local-only` / `Tracked` / `Canonical` / `Holdout`); the body stays status plus short evidence bullets (dev metrics allowed only when labeled non-final); schemas and commands stay in stage docs and `docs/training_runs/`; `docs/client_presentation.md` is a talk excerpt that must sync whenever status, evidence, or next steps change. This keeps dual-audience claims auditable without maintaining two independent status stories or treating local MLX free-tier scores as Canonical/Holdout results (ADR 0003).

## Considered Options

- **Code-complete equals done** — rejected; hides the gap between implemented stages and runs that actually happened.
- **Dated frozen editions only** — rejected as the sole model; useful historically, but stakeholders need one place that is current.
- **Presentation as the only stakeholder doc** — rejected; slide excerpts and deep status drift too easily.
- **Soft-trim the June report (keep schema appendices)** — rejected; duplicates stage docs and recreates conflicting truths in one file.
