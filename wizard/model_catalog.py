"""
The Model Catalog (see `CONTEXT.md`): the curated list of base models the
wizard supports. Each entry has known hardware requirements and an Ollama
reference the wizard has actually vetted — the wizard only promises "this
will work on your machine" for models on this list, never an arbitrary
model ID an End User might type in.

v1 is deliberately small. Growing it is a cheap, low-risk way to add wizard
value later without touching the Try It flow itself.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCatalogEntry:
    name: str
    display_name: str
    ollama_ref: str
    description: str
    approx_download_gb: float
    min_ram_gb: int


CATALOG: list[ModelCatalogEntry] = [
    ModelCatalogEntry(
        name="qwen3-4b",
        display_name="Qwen3 4B Instruct",
        ollama_ref="qwen3:4b-instruct-2507-q4_K_M",
        description=(
            "Apache 2.0, strong structured-output behaviour. Recommended default."
        ),
        approx_download_gb=2.5,
        min_ram_gb=8,
    ),
    ModelCatalogEntry(
        name="llama3.2-3b",
        display_name="Llama 3.2 3B Instruct",
        ollama_ref="llama3.2:3b",
        description="Meta's Llama licence; lighter fallback if Qwen3 gives you trouble.",
        approx_download_gb=2.0,
        min_ram_gb=8,
    ),
]


def get_entry(name: str) -> ModelCatalogEntry:
    for entry in CATALOG:
        if entry.name == name:
            return entry
    known = ", ".join(entry.name for entry in CATALOG)
    raise KeyError(f"Unknown Model Catalog entry {name!r}. Known entries: {known}")
