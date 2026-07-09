from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from ingestion.chunking import ChunkingConfig, chunk_document
from ingestion.schema import Document, Section
from ingestion.serialize import load_document
from ingestion.source_manifest import (
    MANIFEST_VERSION,
    ManifestError,
    SourceManifest,
    default_topic_buckets,
    load_manifest,
    parse_manifest,
)
from ingestion.source_pack import build_source_pack
from ingestion.study_notes_schema import (
    STUDY_NOTES_OUTPUT_SCHEMA,
    build_study_notes_prompt,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _minimal_manifest_dict(
    *,
    sources: list[dict],
    project_root: Path | None = None,
) -> dict:
    return {
        "manifest_version": MANIFEST_VERSION,
        "pack_id": "test_pack",
        "title": "Test Pack",
        "description": "Test source pack",
        "domain": "Test Domain",
        "topic_buckets": [bucket.to_dict() for bucket in default_topic_buckets()],
        "sources": sources,
    }


def _write_manifest(tmp_path: Path, payload: dict) -> Path:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def test_manifest_rejects_duplicate_source_ids(tmp_path: Path) -> None:
    payload = _minimal_manifest_dict(
        sources=[
            {
                "source_id": "dup",
                "title": "One",
                "resource_type": "documentation",
                "local_path": "a.md",
                "topic_bucket_ids": ["databricks_basics"],
            },
            {
                "source_id": "dup",
                "title": "Two",
                "resource_type": "documentation",
                "local_path": "b.md",
                "topic_bucket_ids": ["databricks_basics"],
            },
        ]
    )

    with pytest.raises(ManifestError, match="Duplicate source_id"):
        parse_manifest(payload, project_root=tmp_path)


def test_manifest_rejects_unknown_topic_bucket(tmp_path: Path) -> None:
    payload = _minimal_manifest_dict(
        sources=[
            {
                "source_id": "src1",
                "title": "One",
                "resource_type": "documentation",
                "local_path": "a.md",
                "topic_bucket_ids": ["not_a_real_bucket"],
            }
        ]
    )

    with pytest.raises(ManifestError, match="unknown topic bucket"):
        parse_manifest(payload, project_root=tmp_path)


def test_manifest_rejects_invalid_split(tmp_path: Path) -> None:
    payload = _minimal_manifest_dict(
        sources=[
            {
                "source_id": "src1",
                "title": "One",
                "resource_type": "documentation",
                "local_path": "a.md",
                "topic_bucket_ids": ["databricks_basics"],
                "split": "validation",
            }
        ]
    )

    with pytest.raises(ManifestError, match="split"):
        parse_manifest(payload, project_root=tmp_path)


def test_manifest_rejects_missing_enabled_local_file(tmp_path: Path) -> None:
    payload = _minimal_manifest_dict(
        sources=[
            {
                "source_id": "src1",
                "title": "One",
                "resource_type": "documentation",
                "local_path": "missing.md",
                "topic_bucket_ids": ["databricks_basics"],
                "enabled": True,
            }
        ]
    )

    with pytest.raises(ManifestError, match="local_path does not exist"):
        parse_manifest(payload, project_root=tmp_path, require_local_files=True)


def test_manifest_allows_missing_file_when_disabled(tmp_path: Path) -> None:
    payload = _minimal_manifest_dict(
        sources=[
            {
                "source_id": "src1",
                "title": "One",
                "resource_type": "documentation",
                "local_path": "missing.md",
                "topic_bucket_ids": ["databricks_basics"],
                "enabled": False,
            }
        ]
    )

    manifest = parse_manifest(payload, project_root=tmp_path, require_local_files=True)
    assert manifest.enabled_sources() == []


def test_starter_manifest_loads() -> None:
    manifest_path = PROJECT_ROOT / "data" / "manifests" / "databricks_ld_foundations.json"
    manifest = load_manifest(manifest_path)

    assert isinstance(manifest, SourceManifest)
    assert manifest.pack_id == "databricks_ld_foundations"
    assert len(manifest.topic_buckets) == 6
    assert len(manifest.sources) == 15
    # All sources are enabled now that content acquisition has run.
    assert len(manifest.enabled_sources()) == 15
    assert all(s.original_url for s in manifest.sources)


def test_build_source_pack_with_generated_fixtures(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    doc_path = raw_dir / "delta_streaming.md"
    doc_path.write_text(
        "# Delta Streaming\n\n"
        "Use readStream and writeStream with checkpointLocation when building "
        "incremental pipelines over Delta Lake tables in production workloads "
        "that require reliable exactly-once processing semantics.",
        encoding="utf-8",
    )

    payload = _minimal_manifest_dict(
        sources=[
            {
                "source_id": "delta_streaming",
                "title": "Delta Lake table streaming reads and writes",
                "resource_type": "documentation",
                "original_url": "https://docs.databricks.com/aws/en/delta/delta-streaming",
                "local_path": str(doc_path.relative_to(tmp_path)),
                "topic_bucket_ids": ["ingestion_incremental", "lakehouse_delta"],
                "split": "train",
                "priority": 1,
                "enabled": True,
            }
        ]
    )
    manifest_path = _write_manifest(tmp_path, payload)
    output_dir = tmp_path / "out"

    pack_index = build_source_pack(
        manifest_path,
        output_dir,
        project_root=tmp_path,
        require_local_files=True,
    )

    assert pack_index["pack_id"] == "test_pack"
    assert pack_index["ingested_source_count"] == 1
    assert pack_index["total_section_count"] >= 1
    assert pack_index["study_note_task_count"] >= 1
    assert pack_index["total_chunk_count"] >= 1

    pack_file = output_dir / "source_pack.json"
    tasks_file = output_dir / "study_note_tasks.jsonl"
    normalized_manifest = output_dir / "manifest.normalized.json"
    document_file = output_dir / "documents" / "delta_streaming.json"

    assert pack_file.exists()
    assert tasks_file.exists()
    assert normalized_manifest.exists()
    assert document_file.exists()

    loaded_doc = load_document(document_file)
    assert loaded_doc.format_metadata["source_pack"]["source_id"] == "delta_streaming"
    assert loaded_doc.format_metadata["source_pack"]["split"] == "train"

    tasks = [
        json.loads(line)
        for line in tasks_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(tasks) == pack_index["total_chunk_count"]

    task = tasks[0]
    assert task["pack_id"] == "test_pack"
    assert task["source_id"] == "delta_streaming"
    assert task["expected_output_schema"] == STUDY_NOTES_OUTPUT_SCHEMA
    assert task["chunk_id"]
    assert task["chunk_index"] == 0
    assert task["chunk_word_count"] > 0
    assert task["source_section_indexes"]
    assert "readStream" in task["source_content"] or "Delta Streaming" in task["source_content"]
    assert "Output valid JSON only." in task["prompt"]
    assert "<<<PASTE DOCUMENTATION OR TRANSCRIPT HERE>>>" not in task["prompt"]
    assert pack_index["sources"][0]["chunk_count"] == len(tasks)


def test_gold_example_matches_study_notes_schema() -> None:
    """The client-provided gold example must conform to the study-note schema."""
    example_path = (
        PROJECT_ROOT
        / "data"
        / "manifests"
        / "examples"
        / "study_note_example_delta_streaming.json"
    )
    example = json.loads(example_path.read_text(encoding="utf-8"))

    assert set(example.keys()) == set(STUDY_NOTES_OUTPUT_SCHEMA.keys())

    assert isinstance(example["title"], str) and example["title"]
    assert isinstance(example["summary"], str) and example["summary"]

    for field in (
        "key_concepts",
        "important_features_or_tools",
        "practical_workflow",
        "common_mistakes_or_confusions",
    ):
        template_keys = set(STUDY_NOTES_OUTPUT_SCHEMA[field][0].keys())
        assert isinstance(example[field], list) and example[field]
        for item in example[field]:
            assert set(item.keys()) == template_keys, f"{field} item keys mismatch"

    parameter_keys = set(
        STUDY_NOTES_OUTPUT_SCHEMA["important_features_or_tools"][0][
            "important_parameters"
        ][0].keys()
    )
    for feature in example["important_features_or_tools"]:
        for parameter in feature["important_parameters"]:
            assert set(parameter.keys()) == parameter_keys

    assert isinstance(example["project_usage_notes"], list)
    assert all(isinstance(note, str) for note in example["project_usage_notes"])


def test_study_notes_prompt_includes_schema_and_content() -> None:
    prompt = build_study_notes_prompt(content="Example Databricks content.")
    assert "key_concepts" in prompt
    assert "Example Databricks content." in prompt
    assert "Do not invent features" in prompt


def test_build_source_pack_skip_missing_files(tmp_path: Path) -> None:
    payload = _minimal_manifest_dict(
        sources=[
            {
                "source_id": "missing_doc",
                "title": "Missing",
                "resource_type": "documentation",
                "local_path": "does_not_exist.md",
                "topic_bucket_ids": ["databricks_basics"],
                "enabled": True,
            }
        ]
    )
    manifest_path = _write_manifest(tmp_path, payload)
    output_dir = tmp_path / "out"

    pack_index = build_source_pack(
        manifest_path,
        output_dir,
        project_root=tmp_path,
        skip_missing_files=True,
    )

    assert pack_index["skipped_source_count"] == 1
    assert pack_index["study_note_task_count"] == 0
    assert pack_index["sources"][0]["skipped"] is True
    assert pack_index["sources"][0]["document_path"] is None


def test_chunk_document_splits_oversized_section() -> None:
    long_text = " ".join(f"word{i}" for i in range(0, 260))
    document = Document(
        document_id="doc_split",
        source_type="manual_text",
        source_path=None,
        modality="text",
        content_type="plain_text",
        ingestor="test",
        method="manual",
        sections=[Section(index=0, text=long_text)],
    )

    config = ChunkingConfig(target_words=80, max_words=100, min_words=10, overlap_words=15)
    chunks = chunk_document(document, config=config)

    assert len(chunks) >= 3
    assert all(chunk.word_count <= (config.max_words + config.overlap_words) for chunk in chunks)
    assert all(chunk.source_section_indexes == [0] for chunk in chunks)
    assert any(chunk.split_reason == "split_oversized_section" for chunk in chunks)


def test_chunk_document_merges_short_sections() -> None:
    sections = [
        Section(index=0, text="Spark basics intro."),
        Section(index=1, text="Delta Lake quick overview."),
        Section(index=2, text="Unity Catalog access control notes."),
    ]
    document = Document(
        document_id="doc_merge",
        source_type="manual_text",
        source_path=None,
        modality="text",
        content_type="plain_text",
        ingestor="test",
        method="manual",
        sections=sections,
    )

    chunks = chunk_document(
        document,
        config=ChunkingConfig(target_words=40, max_words=120, min_words=5, overlap_words=0),
    )

    assert len(chunks) == 1
    assert chunks[0].source_section_indexes == [0, 1, 2]
    assert chunks[0].split_reason == "merged_small_sections"


def test_chunk_document_preserves_provenance() -> None:
    sections = [
        Section(
            index=0,
            text="Intro transcript segment",
            heading="Intro",
            start_time_s=0.0,
            end_time_s=6.5,
        ),
        Section(
            index=1,
            text="Second transcript segment with key points",
            heading="Key Points",
            start_time_s=6.5,
            end_time_s=15.0,
        ),
    ]
    document = Document(
        document_id="doc_prov",
        source_type="manual_text",
        source_path=None,
        modality="audio",
        content_type="transcript",
        ingestor="test",
        method="manual",
        sections=sections,
    )

    chunks = chunk_document(
        document,
        config=ChunkingConfig(target_words=100, max_words=200, min_words=5, overlap_words=0),
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.source_section_indexes == [0, 1]
    assert chunk.source_headings == ["Intro", "Key Points"]
    assert chunk.source_time_range_s == (0.0, 15.0)


def test_chunk_document_merges_tiny_trailing_chunk() -> None:
    """Undersized trailing chunk merges backward into the previous chunk."""
    body = " ".join(f"word{i}" for i in range(50))
    trailing = "tiny trailing leftover"
    document = Document(
        document_id="doc_trail",
        source_type="manual_text",
        source_path=None,
        modality="text",
        content_type="plain_text",
        ingestor="test",
        method="manual",
        sections=[
            Section(index=0, text=body),
            Section(index=1, text=trailing),
        ],
    )

    # target_words=50 flushes the first section; trailing is below min_words=20.
    chunks = chunk_document(
        document,
        config=ChunkingConfig(
            target_words=50,
            max_words=120,
            min_words=20,
            overlap_words=0,
        ),
    )

    assert len(chunks) == 1
    assert chunks[0].source_section_indexes == [0, 1]
    assert "tiny trailing leftover" in chunks[0].text
    assert chunks[0].word_count >= 20


def test_chunk_document_keeps_sole_undersized_chunk() -> None:
    """A document whose only chunk is below min_words is kept, not dropped."""
    document = Document(
        document_id="doc_tiny",
        source_type="manual_text",
        source_path=None,
        modality="text",
        content_type="plain_text",
        ingestor="test",
        method="manual",
        sections=[Section(index=0, text="only five short words here")],
    )

    chunks = chunk_document(
        document,
        config=ChunkingConfig(
            target_words=40,
            max_words=80,
            min_words=20,
            overlap_words=0,
        ),
    )

    assert len(chunks) == 1
    assert chunks[0].word_count == 5
    assert "only five short words here" in chunks[0].text


def test_chunk_document_enforces_min_words_across_short_sections() -> None:
    """Short sections that would flush early still merge up to min_words."""
    sections = [
        Section(index=0, text=" ".join(f"a{i}" for i in range(30))),
        Section(index=1, text=" ".join(f"b{i}" for i in range(30))),
        Section(index=2, text="short tail words only"),
    ]
    document = Document(
        document_id="doc_min",
        source_type="manual_text",
        source_path=None,
        modality="text",
        content_type="plain_text",
        ingestor="test",
        method="manual",
        sections=sections,
    )

    chunks = chunk_document(
        document,
        config=ChunkingConfig(
            target_words=30,
            max_words=100,
            min_words=25,
            overlap_words=0,
        ),
    )

    assert all(chunk.word_count >= 25 for chunk in chunks)
    assert sum(len(c.source_section_indexes) for c in chunks) >= 3
