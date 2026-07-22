"""
Tier 3 — LLM-as-judge.

Claude Sonnet (deliberately distinct from and stronger than the Teacher, see
CONTEXT.md) scores a Student study note against the Teacher Reference given
the source chunk, on a fixed rubric.
"""

from __future__ import annotations

import json
from typing import Any

from enrichment.study_note_validation import parse_study_note_response

DEFAULT_JUDGE_MODEL = "claude-sonnet-4-5"

JUDGE_PROMPT_TEMPLATE = """You are grading a small language model that turns \
Databricks learning content into structured JSON study notes. You are given \
the SOURCE content, a REFERENCE study note produced by a strong model, and \
the CANDIDATE study note produced by the small model.

Score the CANDIDATE on this rubric, each 1-5 (5 best):

- groundedness: every claim, feature name, command, and parameter in the \
CANDIDATE is supported by the SOURCE. Inventions or details absent from the \
SOURCE lower this score.
- completeness: the CANDIDATE captures the important concepts, features, and \
workflows of the SOURCE about as well as the REFERENCE does. Missing major \
points lowers this score.
- schema_quality: the CANDIDATE uses the study-note structure well — \
beginner-friendly explanations in the right fields, practical steps, exact \
Databricks names preserved.

The REFERENCE is a guide, not ground truth: a CANDIDATE that covers the \
source differently but faithfully can still score 5.

Return JSON only:
{{"groundedness": <1-5>, "completeness": <1-5>, "schema_quality": <1-5>, "rationale": "<2-3 sentences>"}}

SOURCE:
<<<
{source}
>>>

REFERENCE:
<<<
{reference}
>>>

CANDIDATE:
<<<
{candidate}
>>>
"""

RUBRIC_KEYS = ("groundedness", "completeness", "schema_quality")


class JudgeClient:
    """Sonnet-backed judge. Construct with a client for testing."""

    def __init__(
        self,
        model: str = DEFAULT_JUDGE_MODEL,
        *,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def score(
        self,
        *,
        source: str,
        reference_note: dict[str, Any],
        candidate_note: dict[str, Any],
    ) -> dict[str, Any]:
        """Returns rubric scores, or {"error": ...} if judging failed."""
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            source=source,
            reference=json.dumps(reference_note, ensure_ascii=False, indent=2),
            candidate=json.dumps(candidate_note, ensure_ascii=False, indent=2),
        )
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(
                block.text
                for block in message.content
                if getattr(block, "type", "") == "text"
            )
            verdict = parse_study_note_response(text)
        except Exception as exc:
            return {"error": str(exc)}
        for key in RUBRIC_KEYS:
            value = verdict.get(key)
            if not isinstance(value, int) or not 1 <= value <= 5:
                return {"error": f"judge returned invalid '{key}': {value!r}"}
        return verdict
