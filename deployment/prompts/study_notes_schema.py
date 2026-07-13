"""
Structured study-note output schema and prompt template for Databricks L&D data.

Vendored into Stage 4 from Stage 1 (`pipeline/ingestion/study_notes_schema.py`)
so the Open WebUI deployment can build a system prompt that matches training
without depending on the ingestion package.
"""

from __future__ import annotations

from typing import Any

STUDY_NOTES_SCHEMA_VERSION = "1.0"

STUDY_NOTES_OUTPUT_SCHEMA: dict[str, Any] = {
    "title": "Short title for this lesson or transcript chunk",
    "summary": "A short 3-5 sentence summary of what the learner needs to know",
    "key_concepts": [
        {
            "concept": "Name of the concept",
            "simple_explanation": "Explain it in beginner-friendly language",
            "why_it_matters": (
                "Why this matters when using Databricks in a real project"
            ),
        }
    ],
    "important_features_or_tools": [
        {
            "name": (
                "Feature, tool, UI section, command, function, or API mentioned"
            ),
            "type": "feature/tool/function/command/API/parameter/configuration/other",
            "what_it_does": "Practical explanation",
            "when_to_use_it": "When a learner would use this",
            "important_parameters": [
                {
                    "parameter": "Parameter name",
                    "meaning": "What the parameter controls",
                    "example_value": "Example value if provided, otherwise null",
                }
            ],
        }
    ],
    "practical_workflow": [
        {
            "step": 1,
            "action": "What the learner should do",
            "reason": "Why this step matters",
        }
    ],
    "common_mistakes_or_confusions": [
        {
            "mistake": "Likely beginner mistake or confusion",
            "correction": "How to think about it correctly",
        }
    ],
    "project_usage_notes": [
        "Concrete note about how this would be used in a real Databricks project"
    ],
}

STUDY_NOTES_PROMPT_TEMPLATE = """You are creating high-quality training data for an offline edge SLM / mini-LM.

The target use case is:
A learner wants to understand Databricks key concepts quickly and start using Databricks as soon as possible in a real project.

Given the documentation, tutorial, or transcript content below, generate a practical structured summary.

The learner:
- is new to Databricks,
- does not have time to watch a full course or read long documentation,
- needs the key concepts quickly,
- needs to understand important features, tools, functions, commands, options, and parameters,
- wants to start using Databricks in a real project ASAP.

Rules:
- Only use information from the provided content.
- Do not invent features, commands, parameters, or best practices.
- If something is not explained in the content, do not include it.
- Keep the explanation beginner-friendly and practical.
- Preserve exact names of Databricks features, commands, functions, APIs, options, parameters, and configuration keys.
- If code, options, or parameters are mentioned, explain what they do and when they matter.
- If the content includes a process or workflow, convert it into simple practical steps.
- Output valid JSON only.
- Do not include markdown outside the JSON.

Use this JSON schema:

{output_schema_json}

Content to summarise:
<<<PASTE DOCUMENTATION OR TRANSCRIPT HERE>>>
"""


def build_study_notes_prompt(*, content: str) -> str:
    """Build the full prompt with source content inserted."""
    import json

    schema_json = json.dumps(STUDY_NOTES_OUTPUT_SCHEMA, indent=2, ensure_ascii=False)
    prompt = STUDY_NOTES_PROMPT_TEMPLATE.replace(
        "{output_schema_json}", schema_json
    )
    return prompt.replace(
        "<<<PASTE DOCUMENTATION OR TRANSCRIPT HERE>>>",
        content,
    )


def study_notes_task_record(
    *,
    task_id: str,
    pack_id: str,
    source_id: str,
    source_title: str,
    resource_type: str,
    original_url: str | None,
    topic_bucket_ids: list[str],
    split: str,
    section_index: int | None,
    section_heading: str | None,
    chunk_id: str | None = None,
    chunk_index: int | None = None,
    chunk_word_count: int | None = None,
    source_section_indexes: list[int] | None = None,
    source_headings: list[str] | None = None,
    source_pages: list[int] | None = None,
    source_slides: list[int] | None = None,
    source_time_range_s: tuple[float, float] | None = None,
    split_reason: str | None = None,
    content: str,
    document_id: str,
    document_path: str,
) -> dict[str, Any]:
    """Build one study-note task record for JSONL output."""
    return {
        "task_id": task_id,
        "schema_version": STUDY_NOTES_SCHEMA_VERSION,
        "pack_id": pack_id,
        "source_id": source_id,
        "source_title": source_title,
        "resource_type": resource_type,
        "original_url": original_url,
        "topic_bucket_ids": topic_bucket_ids,
        "split": split,
        "section_index": section_index,
        "section_heading": section_heading,
        "chunk_id": chunk_id,
        "chunk_index": chunk_index,
        "chunk_word_count": chunk_word_count,
        "source_section_indexes": source_section_indexes,
        "source_headings": source_headings,
        "source_pages": source_pages,
        "source_slides": source_slides,
        "source_time_range_s": list(source_time_range_s) if source_time_range_s else None,
        "split_reason": split_reason,
        "document_id": document_id,
        "document_path": document_path,
        "prompt": build_study_notes_prompt(content=content),
        "source_content": content,
        "expected_output_schema": STUDY_NOTES_OUTPUT_SCHEMA,
    }
