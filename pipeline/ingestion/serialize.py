"""
Central serialization for the Edge SLM ingestion pipeline.

`save_document` and `load_document` are the only sanctioned way Documents cross
the disk boundary. Keeping them here (not duplicated per ingestor) means the
on-disk format has exactly one definition.

Two things make this module load-bearing rather than boilerplate:

1. The version gate. `load_document` checks the stored `schema_version` against
   the current `schema.SCHEMA_VERSION` *before* reconstructing anything, and
   raises `SchemaVersionError` on mismatch. This is the loud failure on schema
   drift: a stale chunk-based "0.1" file does not silently load as an empty or
   half-populated Document — it crashes with a clear message naming both
   versions and the path.

2. Reconstruction. `dataclasses.asdict` flattens nested `Section` objects to
   plain dicts on save, so load must rebuild them by hand. This is exactly where
   a dropped field or a stray key would hide, so reconstruction validates keys
   explicitly (unknown keys and missing required keys both fail loudly) rather
   than trusting the file.

A `raw_text` loaded from disk is trusted as byte-faithful, per the schema
contract — this module does not (and cannot) re-verify that invariant; it is the
ingestor's responsibility to have honoured it on write.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from ingestion.schema import SCHEMA_VERSION, Document, Section


class SchemaError(Exception):
    """Base error for a serialized document that does not match the contract."""


class SchemaVersionError(SchemaError):
    """Raised when a file's schema_version differs from the current version.

    Catch this specifically if you want to handle stale files (e.g. trigger a
    re-ingest) rather than crash the whole run.
    """


# Field-set introspection, computed once. Used to reject unknown keys (drift)
# and to report missing required keys clearly.
_SECTION_FIELDS = {f.name for f in dataclasses.fields(Section)}
_DOCUMENT_FIELDS = {f.name for f in dataclasses.fields(Document)}


def _required_field_names(cls: type) -> set[str]:
    """Names of dataclass fields with no default and no default_factory."""
    required: set[str] = set()
    for f in dataclasses.fields(cls):
        has_default = f.default is not dataclasses.MISSING
        has_factory = f.default_factory is not dataclasses.MISSING  # type: ignore[misc]
        if not has_default and not has_factory:
            required.add(f.name)
    return required


_SECTION_REQUIRED = _required_field_names(Section)
_DOCUMENT_REQUIRED = _required_field_names(Document)


def save_document(document: Document, output_path: str | Path) -> Path:
    """
    Write a Document to disk as UTF-8 JSON.

    Returns the path written. Parent directories are created if missing. The
    JSON is indented and non-ASCII-preserving for human inspection during the
    proof-of-concept; revisit if corpus size makes that wasteful.
    """
    if not isinstance(document, Document):
        raise TypeError(
            f"save_document expects a Document, got {type(document).__name__}"
        )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    payload = dataclasses.asdict(document)

    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    return output_file


def load_document(input_path: str | Path) -> Document:
    """
    Read a Document from disk, enforcing the schema version on the way in.

    Raises:
        FileNotFoundError: path does not exist.
        SchemaVersionError: stored schema_version != current SCHEMA_VERSION.
        SchemaError:        structurally invalid (not an object, unknown keys,
                            missing required keys, malformed sections).
    """
    input_file = Path(input_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Document file not found: {input_file}")

    raw = input_file.read_text(encoding="utf-8")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        raise SchemaError(f"{input_file} is not valid JSON: {error}") from error

    if not isinstance(data, dict):
        raise SchemaError(
            f"{input_file} does not contain a JSON object at the top level "
            f"(got {type(data).__name__})."
        )

    # --- Version gate: first, before any reconstruction. ---
    found_version = data.get("schema_version")
    if found_version != SCHEMA_VERSION:
        raise SchemaVersionError(
            f"Schema version mismatch loading {input_file}: file is "
            f"{found_version!r}, this code expects {SCHEMA_VERSION!r}. "
            f"Re-ingest the source to produce a {SCHEMA_VERSION!r} document."
        )

    return _document_from_dict(data, source=str(input_file))


def _section_from_dict(data: Any, *, position: int) -> Section:
    """Rebuild one Section, failing loudly on unknown or missing keys."""
    if not isinstance(data, dict):
        raise SchemaError(
            f"sections[{position}] must be an object, got {type(data).__name__}."
        )

    unknown = set(data) - _SECTION_FIELDS
    if unknown:
        raise SchemaError(
            f"sections[{position}] has unknown keys: {sorted(unknown)}. "
            f"Allowed: {sorted(_SECTION_FIELDS)}."
        )

    missing = _SECTION_REQUIRED - set(data)
    if missing:
        raise SchemaError(
            f"sections[{position}] is missing required keys: {sorted(missing)}."
        )

    # Section.__post_init__ still runs its own value guards (index, text type).
    return Section(**data)


def _document_from_dict(data: dict[str, Any], *, source: str) -> Document:
    """Rebuild a Document from a version-checked dict, validating keys."""
    unknown = set(data) - _DOCUMENT_FIELDS
    if unknown:
        raise SchemaError(
            f"{source} has unknown document keys: {sorted(unknown)}. "
            f"Allowed: {sorted(_DOCUMENT_FIELDS)}."
        )

    missing = _DOCUMENT_REQUIRED - set(data)
    if missing:
        raise SchemaError(
            f"{source} is missing required document keys: {sorted(missing)}."
        )

    sections_raw = data["sections"]
    if not isinstance(sections_raw, list):
        raise SchemaError(
            f"{source}: 'sections' must be a list, got "
            f"{type(sections_raw).__name__}."
        )

    sections = [
        _section_from_dict(section, position=index)
        for index, section in enumerate(sections_raw)
    ]

    payload = {key: value for key, value in data.items() if key != "sections"}

    # Document.__post_init__ runs its own guards. Stored schema_version and
    # created_at flow through here, preserving the original values rather than
    # re-stamping.
    return Document(sections=sections, **payload)