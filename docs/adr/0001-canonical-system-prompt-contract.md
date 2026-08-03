# Canonical System Prompt is a locked train/inference contract

The study-note tasks store a ~700-word teacher prompt with the full JSON schema embedded — designed for Claude, not for the 4B student. We decided the student is trained on a short fixed system prompt (the rules, without the schema dump) with the raw chunk as the user message, and that exact byte-identical prompt ships in the deployed Ollama Modelfile. The schema and rules are baked into the weights by fine-tuning rather than repeated at inference.

## Consequences

- Changing the Canonical System Prompt after training invalidates the model — it requires retraining, not just editing the Modelfile.
- The verbose `prompt` field in `study_note_tasks.jsonl` is used only for Teacher enrichment; it must never be used as the student's training input.
- Half of the 4096-token inference context stays free for user content instead of restating the schema.
