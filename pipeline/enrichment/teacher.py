"""
Thin client around the Anthropic API for Teacher enrichment.

Two call modes:

- realtime: one `messages.create` call per task (used for `--sample` spot
  checks and prompt iteration)
- batch: the Message Batches API (50% discount) for full enrichment runs

The `anthropic` package is imported lazily so the rest of the pipeline does
not require it. Tests inject a fake object with the same surface.
"""

from __future__ import annotations

import dataclasses
import hashlib
import time
from typing import Any, Iterable, Iterator

DEFAULT_TEACHER_MODEL = "claude-haiku-4-5"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.2


@dataclasses.dataclass
class TeacherConfig:
    """Sampling configuration for the Teacher model."""

    model: str = DEFAULT_TEACHER_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE


def batch_custom_id(task_id: str) -> str:
    """
    Deterministic batch custom_id for a task.

    The Batches API caps custom_id at 64 characters and task ids can be
    longer (up to ~98 in the Databricks pack), so we send a truncated
    SHA-256 (128 bits — collision-free at any realistic dataset size) and
    map results back via `batch_results(task_ids=...)`.
    """
    return hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:32]


@dataclasses.dataclass
class TeacherResult:
    """One Teacher response for one task."""

    task_id: str
    ok: bool
    text: str | None = None
    error: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class TeacherClient:
    """Anthropic-backed Teacher. Construct with a client for testing."""

    def __init__(
        self,
        config: TeacherConfig | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config or TeacherConfig()
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def _params(self, prompt: str) -> dict[str, Any]:
        return {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

    # -- realtime ---------------------------------------------------------

    def generate(self, task_id: str, prompt: str) -> TeacherResult:
        try:
            message = self.client.messages.create(**self._params(prompt))
        except Exception as exc:  # API errors surface as a failed result
            return TeacherResult(task_id=task_id, ok=False, error=str(exc))
        return TeacherResult(
            task_id=task_id,
            ok=True,
            text=_message_text(message),
            input_tokens=getattr(message.usage, "input_tokens", None),
            output_tokens=getattr(message.usage, "output_tokens", None),
        )

    # -- batch ------------------------------------------------------------

    def submit_batch(self, prompts: dict[str, str]) -> str:
        """Submit one batch; keys are task_ids. Returns the batch id."""
        requests = [
            {"custom_id": batch_custom_id(task_id), "params": self._params(prompt)}
            for task_id, prompt in prompts.items()
        ]
        batch = self.client.messages.batches.create(requests=requests)
        return batch.id

    def batch_status(self, batch_id: str) -> str:
        return self.client.messages.batches.retrieve(batch_id).processing_status

    def wait_for_batch(
        self,
        batch_id: str,
        *,
        poll_interval_s: float = 30.0,
        timeout_s: float = 24 * 3600,
    ) -> None:
        deadline = time.monotonic() + timeout_s
        while True:
            if self.batch_status(batch_id) == "ended":
                return
            if time.monotonic() > deadline:
                raise TimeoutError(f"batch {batch_id} did not end in time")
            time.sleep(poll_interval_s)

    def batch_results(
        self, batch_id: str, task_ids: Iterable[str]
    ) -> Iterator[TeacherResult]:
        """Collect results, mapping hashed custom_ids back to task_ids."""
        reverse = {batch_custom_id(task_id): task_id for task_id in task_ids}
        for entry in self.client.messages.batches.results(batch_id):
            task_id = reverse[entry.custom_id]
            result = entry.result
            if result.type == "succeeded":
                message = result.message
                yield TeacherResult(
                    task_id=task_id,
                    ok=True,
                    text=_message_text(message),
                    input_tokens=getattr(message.usage, "input_tokens", None),
                    output_tokens=getattr(message.usage, "output_tokens", None),
                )
            else:
                yield TeacherResult(
                    task_id=task_id,
                    ok=False,
                    error=f"batch result type: {result.type}",
                )


def _message_text(message: Any) -> str:
    return "".join(
        block.text for block in message.content if getattr(block, "type", "") == "text"
    )
