from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingestion.folder_manifest import manifest_from_folder, scan_ingestible_files
from ingestion.source_pack import SourcePackError, build_source_pack_from_folder


def test_scan_ingestible_files_finds_nested_and_ignores_junk(tmp_path: Path) -> None:
    (tmp_path / "visible.md").write_text("# Top", encoding="utf-8")
    (tmp_path / "nested" / "deep").mkdir(parents=True)
    (tmp_path / "nested" / "deep" / "note.txt").write_text("Nested text", encoding="utf-8")
    (tmp_path / ".hidden.md").write_text("hidden", encoding="utf-8")
    (tmp_path / "skip.xyz").write_text("unsupported", encoding="utf-8")
    (tmp_path / ".secret").mkdir()
    (tmp_path / ".secret" / "file.md").write_text("nope", encoding="utf-8")
    (tmp_path / "__MACOSX" / "junk.md").mkdir(parents=True)
    (tmp_path / "__MACOSX" / "junk.md" / "mac.md").write_text("mac", encoding="utf-8")
    (tmp_path / ".DS_Store").write_text("store", encoding="utf-8")

    found = scan_ingestible_files(tmp_path, recursive=True)
    names = {path.name for path in found}

    assert names == {"visible.md", "note.txt"}
    assert all(path.is_absolute() for path in found)


def test_scan_ingestible_files_non_recursive(tmp_path: Path) -> None:
    (tmp_path / "top.md").write_text("# Top", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "inner.md").write_text("# Inner", encoding="utf-8")

    found = scan_ingestible_files(tmp_path, recursive=False)

    assert [path.name for path in found] == ["top.md"]


def test_scan_ingestible_files_requires_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "not_a_dir.txt"
    file_path.write_text("x", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        scan_ingestible_files(file_path)


def test_manifest_from_folder_stable_ids_and_defaults(tmp_path: Path) -> None:
    (tmp_path / "Delta Streaming.md").write_text(
        "# Delta\n\n"
        "Delta Lake provides ACID transactions and time travel for data lakes "
        "used in analytics pipelines across production and development workspaces.",
        encoding="utf-8",
    )
    nested = tmp_path / "guides"
    nested.mkdir()
    (nested / "Delta Streaming.md").write_text(
        "# Duplicate name\n\n"
        "A second document with the same filename stem in a nested folder "
        "to exercise source_id collision handling during folder pack builds.",
        encoding="utf-8",
    )

    manifest = manifest_from_folder(
        tmp_path,
        pack_id="my_pack",
        title="My Pack",
        domain="Test domain",
    )

    assert manifest.pack_id == "my_pack"
    assert manifest.title == "My Pack"
    assert manifest.domain == "Test domain"
    assert len(manifest.topic_buckets) == 1
    assert manifest.topic_buckets[0].id == "local_folder"

    source_ids = {source.source_id for source in manifest.sources}
    assert len(source_ids) == 2
    assert "delta_streaming" in source_ids
    assert any(source_id.startswith("delta_streaming_") for source_id in source_ids)

    for source in manifest.sources:
        assert Path(source.local_path).is_absolute()
        assert source.local_path.startswith(str(tmp_path.resolve()))
        assert source.topic_bucket_ids == ["local_folder"]
        assert source.split == "unassigned"
        assert source.enabled is True
        assert source.resource_type == "documentation"


def test_manifest_from_folder_resource_type_inference(tmp_path: Path) -> None:
    (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4\n")
    (tmp_path / "slides.pptx").write_bytes(b"PK\x03\x04")
    (tmp_path / "notes.txt").write_text("text", encoding="utf-8")

    manifest = manifest_from_folder(tmp_path)
    by_id = {source.source_id: source for source in manifest.sources}

    assert by_id["doc"].resource_type == "documentation"
    assert by_id["slides"].resource_type == "tutorial"
    assert by_id["notes"].resource_type == "article"


def test_build_source_pack_from_folder_end_to_end(tmp_path: Path) -> None:
    (tmp_path / "intro.md").write_text(
        "# Intro\n\n"
        "Basics of the lakehouse platform covering workspaces, notebooks, "
        "compute clusters, and common onboarding workflows for new users.",
        encoding="utf-8",
    )
    nested = tmp_path / "more"
    nested.mkdir()
    (nested / "delta.txt").write_text(
        "Delta Lake supports ACID transactions and time travel for analytics "
        "tables used across production data pipelines and streaming jobs.",
        encoding="utf-8",
    )

    output_dir = tmp_path / "out"
    pack_index = build_source_pack_from_folder(
        tmp_path,
        output_dir,
        pack_id="folder_test_pack",
    )

    assert pack_index["pack_id"] == "folder_test_pack"
    assert pack_index["ingested_source_count"] == 2
    assert pack_index["study_note_task_count"] >= 2
    assert pack_index["total_chunk_count"] >= 2

    tasks = [
        json.loads(line)
        for line in (output_dir / "study_note_tasks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source_ids = {task["source_id"] for task in tasks}
    assert "intro" in source_ids
    assert "delta" in source_ids
    assert all(len([t for t in tasks if t["source_id"] == sid]) >= 1 for sid in source_ids)

    assert (output_dir / "source_pack.json").exists()
    assert (output_dir / "manifest.normalized.json").exists()
    assert (output_dir / "documents" / "intro.json").exists()
    assert (output_dir / "documents" / "delta.json").exists()

    normalized = json.loads(
        (output_dir / "manifest.normalized.json").read_text(encoding="utf-8")
    )
    assert normalized["pack_id"] == "folder_test_pack"
    assert len(normalized["sources"]) == 2


def test_build_source_pack_from_folder_empty_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(SourcePackError, match="No ingestible files found"):
        build_source_pack_from_folder(empty, tmp_path / "out")
