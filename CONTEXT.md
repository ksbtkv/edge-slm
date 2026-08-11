# Edge SLM Wizard

A wizard that lets a non-technical End User choose a local small language model, install it, optionally fine-tune it on their own task and data, evaluate it, and run it — entirely on their own machine. Grew out of the Visagio/Databricks study-notes pipeline (`edge_slm-visagio-archive`), whose pipeline plumbing it reuses but whose single hardcoded task it replaces with open-ended Task Templates.

## Language

### Product & audience

**End User**:
The non-technical person operating the wizard end-to-end — picks a model, optionally supplies a task and data, gets a model running locally.
_Avoid_: final user, customer, operator

**Advanced Mode**:
The escape hatch that exposes ML-level choices (Trainer Backend selection, HPC Submission, quantization, etc.) hidden from the default End User flow.
_Avoid_: expert mode, power-user mode

### Task definition

**Task Template**:
A reusable definition of what the End User is fine-tuning for: a Canonical System Prompt plus an Output Schema, with an optional Eval Rubric. Ships with built-in templates; Advanced Mode lets a user define their own.
_Avoid_: task type, use case

**Canonical System Prompt**:
The fixed system prompt for a Task Template, used verbatim in Training Pairs and at inference — a locked contract between training and deployment.

**Output Schema**:
The structured shape a Task Template's fine-tuned model must produce. Mandatory part of every Task Template.

**Eval Rubric**:
An optional, task-specific scoring guide for the Judge. When a Task Template omits one, evaluation falls back to a generic quality/coherence score.

### Entry points

**Entry Point**:
One of the ways an End User starts working with a model, distinguished by how much raw material they bring. v1 ships three:

- **Try It**: no data, no fine-tuning — install and run a stock Model Catalog entry as-is, to sanity-check it on this machine.
- **Task + Raw Data**: End User supplies unstructured content; the wizard runs Teacher enrichment over it to produce Training Pairs.
- **Prepared Dataset**: End User already has a trainer-ready dataset; skips straight to fine-tuning.

_Avoid_: Task-only (considered and deferred to roadmap — no dataset at all, wizard would have to synthesize training data from a task description alone; not solved for v1)

**Customize It**:
The full install → data → fine-tune → evaluate → inference flow, as opposed to Try It.

### Models & training

**Model Catalog**:
The curated list of base models the wizard supports. Each entry has known hardware requirements, a quantization path, and a LoRA configuration validated in advance. End Users choose from this list, not an arbitrary model ID — the wizard only promises "this will work on your machine" for models it has actually vetted.
_Avoid_: model registry, model zoo

**Base Model**:
A Model Catalog entry before any fine-tuning — what a Try It flow installs and runs as-is.

**Student**:
A Base Model once it enters a Customize It flow to be fine-tuned.
_Avoid_: edge model

**Teacher**:
The frontier cloud model that generates reference outputs from a Task + Raw Data entry point, validated against the Task Template's Output Schema.
_Avoid_: oracle, generator LLM

**Judge**:
The frontier cloud model — deliberately distinct from and stronger than the Teacher — that scores a Student's outputs, against the Eval Rubric when one exists, or a generic quality check otherwise.
_Avoid_: evaluator, grader

**Training Pair**:
One supervised example for fine-tuning: Canonical System Prompt + input, targeting a validated Teacher output (or a Prepared Dataset's own example).

**Trainer Backend**:
One of the interchangeable fine-tuning implementations. v1 ships Local MLX (macOS/Apple Silicon); HPC Submission is the second, Advanced-Mode-only backend.

### Advanced Mode: HPC

**HPC Submission**:
A general capability for offloading pipeline work to Pawsey, not a fine-tuning-specific mechanism — v1 wires up fine-tuning only, but the abstraction is shaped so enrichment can plug in later without a rewrite. Lives behind Advanced Mode; assumes the End User already holds Pawsey credentials and an allocation — the wizard submits and monitors jobs, it does not provision access. Requires explicit one-time consent before an End User's data first leaves their machine for it.
_Avoid_: Pawsey backend, remote training
