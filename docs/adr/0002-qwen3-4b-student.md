---
status: superseded by ADR-0005 (repo split) — kept as prior art on model licensing/size trade-offs for the new product's Model Catalog; Qwen3-4B is no longer "the" student, just a potential catalog entry.
---

# Qwen3-4B-Instruct as the student model, replacing the Qwen2.5-3B stand-in

The deployment stand-in Modelfile pointed at qwen2.5:3b, but Qwen2.5-3B specifically ships under the Qwen Research License (non-commercial) — unlike its 7B sibling — which is unacceptable for a client project with Visagio. We chose Qwen3-4B-Instruct: Apache 2.0, stronger at similar size, good structured-JSON behaviour, ~2.5GB at 4-bit quantisation, and well supported by LoRA tooling and Ollama.

## Considered Options

- **Qwen2.5-3B** — rejected on licence grounds despite being the original stand-in.
- **Llama-3.2-3B-Instruct** — viable fallback if Qwen3 tooling friction appears; Llama licence is workable but less clean than Apache 2.0.
- **Smaller (~1.5–2B) models** — rejected; edge budget allows 4B at 4-bit and the structured-output task benefits from the extra capacity.
