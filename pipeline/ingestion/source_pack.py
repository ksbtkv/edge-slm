"""
Build a Databricks L&D source pack from a validated manifest.

Ingests each enabled local source through `dispatch.ingest`, enriches document
metadata, and writes deterministic pack artifacts:

- manifest.normalized.json
- documents/<source_id>.json
- source_pack.json
- study_note_tasks.jsonl
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

from ingestion.chunking import ChunkingConfig, chunk_document
from ingestion.dispatch import ingest, supported_extensions
from ingestion.schema import Document
from ingestion.serialize import save_document
from ingestion.folder_manifest import count_unsupported_files, manifest_from_folder
from ingestion.source_manifest import (
    SourceManifest,
    SourceRecord,
    load_manifest,
)
from ingestion.study_notes_schema import study_notes_task_record

logger = logging.getLogger(__name__)

SOURCE_PACK_VERSION = "1.0"


class SourcePackError(Exception):
    """Raised when source-pack construction fails."""


@dataclasses.dataclass
class IngestedSource:
    """One ingested source with its serialized document path."""

    source: SourceRecord
    document: Document
    document_path: Path
    skipped: bool = False
    skip_reason: str | None = None


def build_source_pack(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    project_root: str | Path | None = None,
    require_local_files: bool = True,
    skip_missing_files: bool = False,
    chunking_config: ChunkingConfig | None = None,
) -> dict[str, Any]:
    """
    Build a source pack from a manifest file.

    Returns the pack index dict (also written to source_pack.json).
    """
    manifest_file = Path(manifest_path)
    root = Path(project_root) if project_root is not None else _infer_project_root(
        manifest_file
    )

    manifest = load_manifest(
        manifest_file,
        project_root=root,
        require_local_files=require_local_files and not skip_missing_files,
    )

    return build_source_pack_from_manifest(
        manifest,
        output_dir,
        project_root=root,
        skip_missing_files=skip_missing_files,
        chunking_config=chunking_config,
    )


def build_source_pack_from_manifest(
    manifest: SourceManifest,
    output_dir: str | Path,
    *,
    project_root: Path | None = None,
    skip_missing_files: bool = False,
    chunking_config: ChunkingConfig | None = None,
) -> dict[str, Any]:
    """
    Build a source pack from an in-memory manifest.

    Returns the pack index dict (also written to source_pack.json).
    """
    output_root = Path(output_dir)
    documents_dir = output_root / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)

    ingested: list[IngestedSource] = []
    for source in sorted(manifest.enabled_sources(), key=lambda s: (s.priority, s.source_id)):
        ingested.append(
            _ingest_source(
                source,
                documents_dir=documents_dir,
                skip_missing_files=skip_missing_files,
                project_root=project_root,
            )
        )

    normalized_manifest_path = output_root / "manifest.normalized.json"
    _write_json(normalized_manifest_path, manifest.to_dict())

    study_note_tasks, source_chunk_counts, source_low_quality_counts = (
        _build_study_note_tasks(
            manifest=manifest,
            ingested=ingested,
            output_root=output_root,
            chunking_config=chunking_config,
        )
    )
    tasks_path = output_root / "study_note_tasks.jsonl"
    _write_jsonl(tasks_path, study_note_tasks)

    pack_index = _build_pack_index(
        manifest=manifest,
        ingested=ingested,
        output_root=output_root,
        normalized_manifest_path=normalized_manifest_path,
        tasks_path=tasks_path,
        source_chunk_counts=source_chunk_counts,
        source_low_quality_counts=source_low_quality_counts,
    )
    pack_index_path = output_root / "source_pack.json"
    _write_json(pack_index_path, pack_index)

    logger.info(
        "Built source pack %r: %d sources, %d tasks -> %s",
        manifest.pack_id,
        len(ingested),
        len(study_note_tasks),
        output_root,
    )

    return pack_index


def build_source_pack_from_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    recursive: bool = True,
    pack_id: str | None = None,
    title: str | None = None,
    domain: str | None = None,
    topic_bucket_ids: list[str] | None = None,
    chunking_config: ChunkingConfig | None = None,
) -> dict[str, Any]:
    """
    Scan a local folder, build a manifest, and produce a full source pack.

    Unsupported files in the tree are skipped silently; a summary count is
    logged at info level. Raises ``SourcePackError`` when no ingestible files
    are found.
    """
    folder = Path(input_dir).resolve()
    skipped_unsupported = count_unsupported_files(folder, recursive=recursive)
    if skipped_unsupported:
        logger.info(
            "Skipped %d unsupported file(s) under %s",
            skipped_unsupported,
            folder,
        )

    manifest = manifest_from_folder(
        folder,
        pack_id=pack_id,
        title=title,
        domain=domain,
        topic_bucket_ids=topic_bucket_ids,
        recursive=recursive,
    )

    if not manifest.sources:
        raise SourcePackError(
            f"No ingestible files found under {folder}. "
            f"Supported extensions: {supported_extensions()}."
        )

    return build_source_pack_from_manifest(
        manifest,
        output_dir,
        project_root=None,
        chunking_config=chunking_config,
    )


def _ingest_source(
    source: SourceRecord,
    *,
    documents_dir: Path,
    skip_missing_files: bool,
    project_root: Path | None = None,
) -> IngestedSource:
    local_path = source.resolve_local_path(project_root)

    if not local_path.exists():
        if skip_missing_files:
            logger.warning(
                "Skipping missing source %r (%s)", source.source_id, local_path
            )
            placeholder = Document(
                document_id=f"skipped_{source.source_id}",
                source_type="local_file",
                source_path=str(local_path),
                modality="text",
                content_type="plain_text",
                ingestor="source_pack",
                method="skipped",
                sections=[],
                title=source.title,
            )
            return IngestedSource(
                source=source,
                document=placeholder,
                document_path=documents_dir / f"{source.source_id}.json",
                skipped=True,
                skip_reason=f"local file not found: {local_path}",
            )
        raise SourcePackError(
            f"Source {source.source_id!r} local_path does not exist: {local_path}"
        )

    document = ingest(local_path, title=source.title)
    document = _enrich_document(document, source)

    document_path = documents_dir / f"{source.source_id}.json"
    save_document(document, document_path)

    return IngestedSource(
        source=source,
        document=document,
        document_path=document_path,
    )


def _enrich_document(document: Document, source: SourceRecord) -> Document:
    """Attach Databricks source-pack provenance to format_metadata."""
    metadata = dict(document.format_metadata)
    metadata["source_pack"] = {
        "source_id": source.source_id,
        "resource_type": source.resource_type,
        "original_url": source.original_url,
        "topic_bucket_ids": list(source.topic_bucket_ids),
        "priority": source.priority,
        "split": source.split,
        "description": source.description,
        "notes": source.notes,
    }
    return dataclasses.replace(document, format_metadata=metadata)


def _build_study_note_tasks(
    *,
    manifest: SourceManifest,
    ingested: list[IngestedSource],
    output_root: Path,
    chunking_config: ChunkingConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    tasks: list[dict[str, Any]] = []
    source_chunk_counts: dict[str, int] = {}
    source_low_quality_counts: dict[str, int] = {}
    effective_config = chunking_config or ChunkingConfig()

    for item in ingested:
        if item.skipped or item.document.section_count == 0:
            source_chunk_counts[item.source.source_id] = 0
            source_low_quality_counts[item.source.source_id] = 0
            continue

        rel_document_path = str(
            item.document_path.relative_to(output_root)
            if item.document_path.is_relative_to(output_root)
            else item.document_path
        )

        chunks = chunk_document(item.document, config=chunking_config)
        below_min = sum(
            1 for chunk in chunks if chunk.word_count < effective_config.min_words
        )
        source_chunk_counts[item.source.source_id] = len(chunks)
        source_low_quality_counts[item.source.source_id] = below_min
        if below_min:
            logger.info(
                "Source %s: %d chunks, %d below min_words=%d (kept)",
                item.source.source_id,
                len(chunks),
                below_min,
                effective_config.min_words,
            )

        for chunk in chunks:
            task_id = f"{manifest.pack_id}__{item.source.source_id}__c{chunk.chunk_index:04d}"
            section_index = (
                chunk.source_section_indexes[0]
                if len(chunk.source_section_indexes) == 1
                else None
            )
            section_heading = chunk.source_headings[0] if len(chunk.source_headings) == 1 else None
            tasks.append(
                study_notes_task_record(
                    task_id=task_id,
                    pack_id=manifest.pack_id,
                    source_id=item.source.source_id,
                    source_title=item.source.title,
                    resource_type=item.source.resource_type,
                    original_url=item.source.original_url,
                    topic_bucket_ids=item.source.topic_bucket_ids,
                    split=item.source.split,
                    section_index=section_index,
                    section_heading=section_heading,
                    chunk_id=chunk.chunk_id,
                    chunk_index=chunk.chunk_index,
                    chunk_word_count=chunk.word_count,
                    source_section_indexes=chunk.source_section_indexes,
                    source_headings=chunk.source_headings,
                    source_pages=chunk.source_pages,
                    source_slides=chunk.source_slides,
                    source_time_range_s=chunk.source_time_range_s,
                    split_reason=chunk.split_reason,
                    content=chunk.text,
                    document_id=item.document.document_id,
                    document_path=rel_document_path,
                )
            )

    return tasks, source_chunk_counts, source_low_quality_counts


def _build_pack_index(
    *,
    manifest: SourceManifest,
    ingested: list[IngestedSource],
    output_root: Path,
    normalized_manifest_path: Path,
    tasks_path: Path,
    source_chunk_counts: dict[str, int],
    source_low_quality_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    source_entries: list[dict[str, Any]] = []
    topic_bucket_counts: dict[str, int] = {bucket.id: 0 for bucket in manifest.topic_buckets}
    low_quality_counts = source_low_quality_counts or {}

    for item in ingested:
        section_count = item.document.section_count
        word_count = item.document.total_word_count

        for bucket_id in item.source.topic_bucket_ids:
            topic_bucket_counts[bucket_id] = topic_bucket_counts.get(bucket_id, 0) + 1

        if item.skipped:
            # No document file is written for skipped sources.
            rel_doc_path = None
        else:
            rel_doc_path = (
                str(item.document_path.relative_to(output_root))
                if item.document_path.is_relative_to(output_root)
                else str(item.document_path)
            )

        source_entries.append(
            {
                "source_id": item.source.source_id,
                "title": item.source.title,
                "resource_type": item.source.resource_type,
                "original_url": item.source.original_url,
                "local_path": item.source.local_path,
                "topic_bucket_ids": item.source.topic_bucket_ids,
                "priority": item.source.priority,
                "split": item.source.split,
                "enabled": item.source.enabled,
                "skipped": item.skipped,
                "skip_reason": item.skip_reason,
                "document_id": item.document.document_id,
                "document_path": rel_doc_path,
                "section_count": section_count,
                "chunk_count": source_chunk_counts.get(item.source.source_id, 0),
                "low_quality_chunk_count": low_quality_counts.get(
                    item.source.source_id, 0
                ),
                "word_count": word_count,
            }
        )

    return {
        "source_pack_version": SOURCE_PACK_VERSION,
        "pack_id": manifest.pack_id,
        "title": manifest.title,
        "description": manifest.description,
        "domain": manifest.domain,
        "topic_buckets": [bucket.to_dict() for bucket in manifest.topic_buckets],
        "topic_bucket_source_counts": topic_bucket_counts,
        "source_count": len(source_entries),
        "ingested_source_count": sum(1 for item in ingested if not item.skipped),
        "skipped_source_count": sum(1 for item in ingested if item.skipped),
        "total_section_count": sum(
            item.document.section_count for item in ingested if not item.skipped
        ),
        "total_word_count": sum(
            item.document.total_word_count for item in ingested if not item.skipped
        ),
        "total_chunk_count": sum(source_chunk_counts.values()),
        "study_note_task_count": sum(source_chunk_counts.values()),
        "artifacts": {
            "manifest_normalized": str(
                normalized_manifest_path.relative_to(output_root)
            ),
            "study_note_tasks": str(tasks_path.relative_to(output_root)),
            "documents_dir": "documents",
        },
        "sources": source_entries,
    }


def _infer_project_root(manifest_path: Path) -> Path:
    """Walk up from manifest path looking for a project root marker."""
    for parent in [manifest_path.parent, *manifest_path.parents]:
        if (parent / "pipeline").is_dir() and (parent / "tests").is_dir():
            return parent
    return manifest_path.parent.parent.resolve()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
