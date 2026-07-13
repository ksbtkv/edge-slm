# 26W-ADV-VisagioSLM

Visagio / UWA AI Winter Project — offline edge Small Language Model (SLM) for
Databricks Data Engineering Foundations study notes.

Fine-tuning uses LoRA / QLoRA on the Ohm GPU platform (Pawsey). Deployment is a
local Open WebUI + Ollama stack.

## Pipeline stages (team branches)

| Stage | Branch / path | Deliverable |
|-------|---------------|-------------|
| 1 — Ingestion | `stage1-ingestion` | `study_note_tasks.jsonl` |
| 2 — Instruction pairs | `stage2/instruction-pairs-v2` | Alpaca JSONL on Ohm |
| 3 — QLoRA fine-tune | `training/` on this branch | `adapter/` + `merged/` |
| 4 — Open WebUI deploy | [`deployment/`](deployment/README.md) | Offline laptop chat UI |

## Stage 4 quick start

See **[deployment/README.md](deployment/README.md)** for converting Stage 3
`merged/` → GGUF → Ollama → Open WebUI (`http://localhost:3000`).

```bash
cd deployment
python scripts/build_system_prompt.py
./scripts/convert_to_gguf.sh --merged ../training/outputs/qwen2.5-3b-lora/merged
./scripts/register_model.sh
cp .env.example .env && docker compose up -d
```

## Stage 3 quick start

See **[training/README.md](training/README.md)**. Production runs set
`merge_adapter: true` so Stage 4 always receives a merged fp16 model.
