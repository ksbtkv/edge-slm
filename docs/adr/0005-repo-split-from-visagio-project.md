# Split into a new repo instead of pivoting the Visagio pipeline in place

The Visagio/Databricks study-notes pipeline (`UWA-AI-Club/26W-VisagioSLM`, archived here as `edge_slm-visagio-archive`) was built for one client, one hardcoded task, and one stakeholder-reporting cadence. The new goal — a wizard that lets a non-technical End User pick any local model and any task — is a different product for a different audience, not an extension of that one. Continuing in place would mean carrying four ADRs and a CONTEXT.md scoped to a deliverable this product isn't trying to be.

We mirror-cloned the full history of `26W-VisagioSLM` into this repo rather than using GitHub's native Fork, because the source repo has `allow_forking` disabled at the org level. The club repo is handed off and finished as-is; this repo continues independently, with no ongoing sync back.
