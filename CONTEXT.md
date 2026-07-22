# Edge SLM

An offline, edge-deployable small language model that turns Databricks learning content into structured study notes. This repo carries the full lifecycle: source ingestion, teacher enrichment, fine-tuning, evaluation, and end-user deployment.

## Language

### Data preparation

**Source Pack**:
A curated, manifest-driven collection of ingested sources for one domain, built into chunk-level tasks.

**Study-Note Task**:
One model-sized chunk plus its generation prompt and provenance; a line in `study_note_tasks.jsonl`.
_Avoid_: chunk task, sample

**Study Note**:
The structured JSON output (title, summary, key concepts, …) produced from one chunk's content.

### Enrichment

**Teacher**:
The frontier model (Claude Haiku) that generates reference study notes from task content. One teacher for all splits.
_Avoid_: oracle, generator LLM

**Enrichment**:
Running every study-note task through the Teacher and validating the response against the study-note schema.

**Teacher Reference**:
A validated Teacher output for an eval or holdout task. Used only to judge the student; never trained on.
_Avoid_: gold label (the client's hand-written example is the "gold example")

**Reject**:
A task whose Teacher response still fails schema validation after one error-feedback retry. Excluded from the dataset and logged for inspection.

### Fine-tuning

**Training Pair**:
One supervised example exported for fine-tuning: Canonical System Prompt + raw chunk content as input, validated Teacher study note as target. Built only from train-split tasks.
_Avoid_: instruction pair

**Canonical System Prompt**:
The short fixed system prompt (rules without the schema dump) used verbatim in both training pairs and the deployed model. Byte-identical across training and inference — a locked contract.

**Student**:
The small base model being fine-tuned to imitate the Teacher on this task.
_Avoid_: edge model (ambiguous with the deployed artifact)

**Trainer Backend**:
One of the interchangeable fine-tuning implementations (local MLX, Pawsey TRL+PEFT). All backends consume the same Training Pairs and emit a LoRA adapter for the same downstream path.

**Canonical Run**:
The training run whose adapter is used for reported results and the Deployed Model — produced on Pawsey. Local runs are development artifacts.

### Evaluation

**Baseline**:
The untuned Student run under the same Canonical System Prompt. Every eval metric is reported as tuned-vs-Baseline.

**Judge**:
The frontier model (Claude Sonnet — deliberately distinct from and stronger than the Teacher) that rubric-scores Student outputs against Teacher References.
_Avoid_: evaluator, grader

**Holdout Run**:
The single, final evaluation on the 28 holdout tasks after all iteration is finished. Run once; its numbers are never used to tune anything.

### Deployment

**Deployed Model**:
The merged, quantised GGUF of the fine-tuned Student, served by Ollama with the Canonical System Prompt in its Modelfile, behind Open WebUI. Supports exactly one interaction: paste content, receive a Study Note.
