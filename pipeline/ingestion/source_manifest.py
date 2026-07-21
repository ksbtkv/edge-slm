"""
Manifest model for curated Databricks L&D source packs.

A source pack is defined by a JSON manifest listing local files (or
transcripts) with provenance URLs, topic buckets, priority, and split hints.
This module validates and normalizes manifests; ingestion is handled by
`source_pack.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MANIFEST_VERSION = "1.0"

VALID_SPLITS = frozenset({"train", "eval", "holdout", "unassigned"})
VALID_RESOURCE_TYPES = frozenset(
    {
        "video_transcript",
        "documentation",
        "tutorial",
        "article",
        "course_outline",
        "playlist",
        "certification_page",
        "training_portal",
    }
)


class ManifestError(Exception):
    """Raised when a manifest fails validation."""


@dataclass
class TopicBucket:
    """A thematic grouping used for dataset creation and filtering."""

    id: str
    label: str
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
        }


@dataclass
class SourceRecord:
    """One curated source entry in a pack manifest."""

    source_id: str
    title: str
    resource_type: str
    local_path: str
    topic_bucket_ids: list[str]
    original_url: str | None = None
    description: str = ""
    priority: int = 100
    split: str = "unassigned"
    notes: str = ""
    enabled: bool = True

    def resolve_local_path(self, project_root: Path | None = None) -> Path:
        """Resolve manifest local_path against an optional project root."""
        return _resolve_local_path(self.local_path, project_root=project_root)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "title": self.title,
            "resource_type": self.resource_type,
            "original_url": self.original_url,
            "local_path": self.local_path,
            "topic_bucket_ids": list(self.topic_bucket_ids),
            "description": self.description,
            "priority": self.priority,
            "split": self.split,
            "notes": self.notes,
            "enabled": self.enabled,
        }


@dataclass
class SourceManifest:
    """Validated, normalized source-pack manifest."""

    manifest_version: str
    pack_id: str
    title: str
    description: str
    domain: str
    topic_buckets: list[TopicBucket]
    sources: list[SourceRecord]
    manifest_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "pack_id": self.pack_id,
            "title": self.title,
            "description": self.description,
            "domain": self.domain,
            "topic_buckets": [bucket.to_dict() for bucket in self.topic_buckets],
            "sources": [source.to_dict() for source in self.sources],
        }

    @property
    def topic_bucket_ids(self) -> set[str]:
        return {bucket.id for bucket in self.topic_buckets}

    def enabled_sources(self) -> list[SourceRecord]:
        return [source for source in self.sources if source.enabled]


def load_manifest(
    path: str | Path,
    *,
    project_root: Path | None = None,
    require_local_files: bool = False,
) -> SourceManifest:
    """Load and validate a manifest from disk."""
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ManifestError(f"{manifest_path} is not valid JSON: {error}") from error

    return parse_manifest(
        data,
        manifest_path=manifest_path,
        project_root=project_root,
        require_local_files=require_local_files,
    )


def parse_manifest(
    data: Any,
    *,
    manifest_path: Path | None = None,
    project_root: Path | None = None,
    require_local_files: bool = False,
) -> SourceManifest:
    """Parse and validate a manifest dict, optionally resolving local paths."""
    if not isinstance(data, dict):
        raise ManifestError("Manifest must be a JSON object at the top level.")

    version = _require_str(data, "manifest_version", context="manifest")
    if version != MANIFEST_VERSION:
        raise ManifestError(
            f"Unsupported manifest_version {version!r}; "
            f"expected {MANIFEST_VERSION!r}."
        )

    pack_id = _require_str(data, "pack_id", context="manifest")
    title = _require_str(data, "title", context="manifest")
    description = _optional_str(data.get("description")) or ""
    domain = _require_str(data, "domain", context="manifest")

    topic_buckets = _parse_topic_buckets(data.get("topic_buckets"))
    bucket_ids = {bucket.id for bucket in topic_buckets}

    sources = _parse_sources(
        data.get("sources"),
        bucket_ids=bucket_ids,
        project_root=project_root,
        require_local_files=require_local_files,
    )

    return SourceManifest(
        manifest_version=version,
        pack_id=pack_id,
        title=title,
        description=description,
        domain=domain,
        topic_buckets=topic_buckets,
        sources=sources,
        manifest_path=manifest_path,
    )


def default_topic_buckets() -> list[TopicBucket]:
    """Topic buckets from the client Databricks L&D source-pack scope."""
    return [
        TopicBucket(
            id="databricks_basics",
            label="Databricks basics",
            description=(
                "workspace, notebooks, compute, clusters/serverless, jobs, workflows"
            ),
        ),
        TopicBucket(
            id="lakehouse_delta",
            label="Lakehouse and Delta Lake",
            description=(
                "managed/external tables, Delta table operations, ACID/transaction "
                "log basics, time travel, optimisation concepts"
            ),
        ),
        TopicBucket(
            id="spark_sql_pyspark",
            label="Spark SQL and PySpark",
            description=(
                "DataFrames, reading/writing files, transformations, tables/views, "
                "common SQL commands"
            ),
        ),
        TopicBucket(
            id="ingestion_incremental",
            label="Data ingestion and incremental processing",
            description=(
                "Structured Streaming, Auto Loader, checkpoints, triggers, "
                "streaming reads/writes"
            ),
        ),
        TopicBucket(
            id="production_pipelines",
            label="Production pipelines",
            description=(
                "Lakeflow / Delta Live Tables concepts, jobs monitoring, "
                "scheduling, orchestration"
            ),
        ),
        TopicBucket(
            id="governance_security",
            label="Governance and security",
            description=(
                "Unity Catalog, catalogs/schemas/tables/volumes, access control, "
                "permissions, lineage concepts"
            ),
        ),
    ]


def _parse_topic_buckets(raw: Any) -> list[TopicBucket]:
    if not isinstance(raw, list) or not raw:
        raise ManifestError("'topic_buckets' must be a non-empty list.")

    buckets: list[TopicBucket] = []
    seen_ids: set[str] = set()

    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ManifestError(f"topic_buckets[{index}] must be an object.")

        bucket_id = _require_str(item, "id", context=f"topic_buckets[{index}]")
        if bucket_id in seen_ids:
            raise ManifestError(f"Duplicate topic bucket id: {bucket_id!r}.")
        seen_ids.add(bucket_id)

        label = _require_str(item, "label", context=f"topic_buckets[{index}]")
        description = _optional_str(item.get("description")) or ""

        buckets.append(
            TopicBucket(id=bucket_id, label=label, description=description)
        )

    return buckets


def _parse_sources(
    raw: Any,
    *,
    bucket_ids: set[str],
    project_root: Path | None,
    require_local_files: bool,
) -> list[SourceRecord]:
    if not isinstance(raw, list) or not raw:
        raise ManifestError("'sources' must be a non-empty list.")

    sources: list[SourceRecord] = []
    seen_ids: set[str] = set()

    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ManifestError(f"sources[{index}] must be an object.")

        source_id = _require_str(item, "source_id", context=f"sources[{index}]")
        if source_id in seen_ids:
            raise ManifestError(f"Duplicate source_id: {source_id!r}.")
        seen_ids.add(source_id)

        title = _require_str(item, "title", context=f"sources[{index}]")
        resource_type = _require_str(
            item, "resource_type", context=f"sources[{index}]"
        )
        if resource_type not in VALID_RESOURCE_TYPES:
            raise ManifestError(
                f"sources[{index}].resource_type {resource_type!r} is not supported. "
                f"Allowed: {sorted(VALID_RESOURCE_TYPES)}."
            )

        local_path = _require_str(item, "local_path", context=f"sources[{index}]")
        resolved = _resolve_local_path(local_path, project_root=project_root)

        enabled = item.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ManifestError(f"sources[{index}].enabled must be a boolean.")

        if enabled and require_local_files and not resolved.exists():
            raise ManifestError(
                f"sources[{index}] local_path does not exist: {resolved}"
            )

        topic_bucket_ids = item.get("topic_bucket_ids")
        if not isinstance(topic_bucket_ids, list) or not topic_bucket_ids:
            raise ManifestError(
                f"sources[{index}].topic_bucket_ids must be a non-empty list."
            )

        normalized_buckets: list[str] = []
        for bucket_index, bucket_id in enumerate(topic_bucket_ids):
            if not isinstance(bucket_id, str) or not bucket_id.strip():
                raise ManifestError(
                    f"sources[{index}].topic_bucket_ids[{bucket_index}] "
                    f"must be a non-empty string."
                )
            if bucket_id not in bucket_ids:
                raise ManifestError(
                    f"sources[{index}] references unknown topic bucket "
                    f"{bucket_id!r}."
                )
            normalized_buckets.append(bucket_id)

        split = _optional_str(item.get("split")) or "unassigned"
        if split not in VALID_SPLITS:
            raise ManifestError(
                f"sources[{index}].split {split!r} is invalid. "
                f"Allowed: {sorted(VALID_SPLITS)}."
            )

        priority = item.get("priority", 100)
        if not isinstance(priority, int):
            raise ManifestError(f"sources[{index}].priority must be an integer.")

        sources.append(
            SourceRecord(
                source_id=source_id,
                title=title,
                resource_type=resource_type,
                original_url=_optional_str(item.get("original_url")),
                local_path=local_path,
                topic_bucket_ids=normalized_buckets,
                description=_optional_str(item.get("description")) or "",
                priority=priority,
                split=split,
                notes=_optional_str(item.get("notes")) or "",
                enabled=enabled,
            )
        )

    return sources


def _resolve_local_path(
    local_path: str,
    *,
    project_root: Path | None,
) -> Path:
    path = Path(local_path)
    if path.is_absolute():
        return path
    if project_root is not None:
        return (project_root / path).resolve()
    return path.resolve()


def _require_str(data: dict[str, Any], key: str, *, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{context}.{key} must be a non-empty string.")
    return value.strip()


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ManifestError("Expected a string or null.")
    stripped = value.strip()
    return stripped or None
