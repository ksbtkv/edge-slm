from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.dispatch import ingest, register, supported_extensions
from ingestion.schema import Document

from tests.ingestion_test_utils import (
    assert_document_basics,
    assert_roundtrip,
    sample_document,
)


def test_schema_document_roundtrips() -> None:
    assert_roundtrip(sample_document(), "schema")


def test_dispatch_registers_core_text_extensions() -> None:
    supported = set(supported_extensions())

    expected = {
        ".pdf",
        ".pptx",
        ".txt",
        ".text",
        ".md",
        ".markdown",
        ".mp3",
        ".wav",
        ".m4a",
        ".flac",
        ".aac",
        ".ogg",
        ".opus",
        ".webm",
        ".mp4",
        ".mov",
        ".mkv",
        ".avi",
    }

    assert expected.issubset(supported)


def test_dispatch_rejects_registration_collision() -> None:
    def dummy_ingestor(path: str | Path, **kwargs) -> Document:
        return sample_document()

    with pytest.raises(ValueError, match="already registered"):
        register(".txt", dummy_ingestor)


def test_dispatch_unknown_extension_lists_supported(tmp_path: Path) -> None:
    unsupported = tmp_path / "sample.xyz"
    unsupported.write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError, match="Supported:"):
        ingest(unsupported)


def test_text_ingestion_uses_generated_fixture(tmp_path: Path) -> None:
    text_path = tmp_path / "sample.txt"
    text_path.write_text(
        "Sample Title\n\nFirst paragraph here.\n\nSecond paragraph here.",
        encoding="utf-8",
    )

    document = ingest(text_path)

    assert_document_basics(document, content_type="plain_text", modality="text")
    assert document.ingestor == "text_ingestor"
    assert document.section_count == 3
    assert_roundtrip(document, "text")


def test_markdown_ingestion_uses_generated_fixture(tmp_path: Path) -> None:
    markdown_path = tmp_path / "sample.md"
    markdown_path.write_text(
        "# Main Title\n\n"
        "Intro text covering lakehouse basics and workspace setup for new users "
        "who are learning Databricks for the first time on this project.\n\n"
        "## Section Two\n\n"
        "More content about Delta Lake tables, ACID transactions, and time travel "
        "features used across analytics and streaming pipelines in production.",
        encoding="utf-8",
    )

    document = ingest(markdown_path)

    assert_document_basics(document, content_type="markdown_text", modality="text")
    assert document.ingestor == "markdown_ingestor"
    assert document.section_count == 2
    assert document.sections[0].heading == "Main Title"
    assert document.sections[1].heading == "Section Two"
    # Body text must not include the markdown heading line.
    assert not document.sections[0].text.startswith("#")
    assert "Intro text covering" in document.sections[0].text
    assert_roundtrip(document, "markdown")


def test_markdown_skips_header_only_sections(tmp_path: Path) -> None:
    markdown_path = tmp_path / "headers_only.md"
    markdown_path.write_text("# Title only\n\n## Also empty\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="No text content found"):
        ingest(markdown_path)


def test_markdown_carries_orphan_heading_to_next_body(tmp_path: Path) -> None:
    markdown_path = tmp_path / "orphan.md"
    markdown_path.write_text(
        "## A\n\n"
        "## B\n\n"
        "Body text with enough words to pass the ingest body-word threshold "
        "for markdown section quality filtering in the pipeline.",
        encoding="utf-8",
    )

    document = ingest(markdown_path)

    assert document.section_count == 1
    assert document.sections[0].heading == "B"
    assert "Body text with enough words" in document.sections[0].text
    assert "##" not in document.sections[0].text
