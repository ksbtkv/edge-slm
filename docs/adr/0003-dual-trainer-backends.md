---
status: superseded by ADR-0005 (repo split) and generalized by ADR-0006 (HPC Submission) — kept as the prior art ADR-0006 builds on; "Pawsey is the project mandate" / Pawsey-as-canonical no longer applies to the new product.
---

# Dual trainer backends: MLX locally, TRL+PEFT on Pawsey; Pawsey run is canonical

Full local fine-tuning must be possible, but the development machine is a 16GB Apple M3 where the CUDA/ROCm stack (TRL + PEFT + bitsandbytes) does not run. We decided on two trainer backends behind one contract: both consume the identical exported training-pairs JSONL and both emit a LoRA adapter that feeds the same merge → GGUF → Ollama path and the same eval harness. Locally, MLX-LM runs QLoRA on 4-bit Qwen3-4B; on Pawsey, TRL + PEFT runs on ROCm. Adapters from the two backends are not bit-identical, so the Pawsey run is designated the Canonical Run for reported results and the deployed model — Pawsey is the project mandate — while local MLX runs are a fully-usable development path.

## Considered Options

- **Single TRL+PEFT stack, smoke-test-only locally** — rejected once full local training became a requirement; MPS cannot run QLoRA (no bitsandbytes) and bf16 LoRA on a 4B model does not fit 16GB.
- **Shrink the student to ~1.7B so one stack runs everywhere** — rejected; sacrifices output quality for tooling uniformity.
- **Rented cloud GPU as the "local" path** — rejected; breaks the offline/owned-hardware development story.
