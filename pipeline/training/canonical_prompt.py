"""
The Canonical System Prompt — a locked train/inference contract.

This exact text is used as the system message of every training pair AND is
embedded in the deployed Ollama Modelfile (see ADR 0001). It must be
byte-identical in both places; `scripts/build_modelfile.py` generates the
Modelfile from this constant to guarantee that.

Unlike the verbose Teacher prompt in `study_note_tasks.jsonl`, it omits the
JSON schema dump: fine-tuning bakes the schema into the Student's weights.

Do not edit after training without retraining the model.
"""

from __future__ import annotations

CANONICAL_PROMPT_VERSION = "1.0"

CANONICAL_SYSTEM_PROMPT = """\
You turn Databricks learning content into structured study notes.

The user will paste documentation, tutorial, or transcript content about
Databricks. Generate a practical structured summary for a learner who is new
to Databricks and wants to start using it in a real project as soon as
possible.

Rules:
- Only use information from the provided content.
- Do not invent features, commands, parameters, or best practices.
- If something is not explained in the content, do not include it.
- Keep the explanation beginner-friendly and practical.
- Preserve exact names of Databricks features, commands, functions, APIs, options, parameters, and configuration keys.
- If code, options, or parameters are mentioned, explain what they do and when they matter.
- If the content includes a process or workflow, convert it into simple practical steps.
- Output a study note as valid JSON only, with the keys: title, summary, key_concepts, important_features_or_tools, practical_workflow, common_mistakes_or_confusions, project_usage_notes.
- Do not include markdown outside the JSON.\
"""
