from __future__ import annotations

import tempfile
from pathlib import Path

from ingestion.schema import SCHEMA_VERSION, Document, Section
from ingestion.serialize import load_document, save_document


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_ROOT = PROJECT_ROOT / "pipeline"
RAW_DATA = PROJECT_ROOT / "data" / "raw"


def assert_roundtrip(document: Document, label: str = "document") -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / f"{label}.json"
        save_document(document, output_path)
        loaded = load_document(output_path)

    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.document_id == document.document_id
    assert loaded.section_count == document.section_count
    assert loaded.total_word_count == document.total_word_count


def assert_section_indexes(document: Document) -> None:
    assert [section.index for section in document.sections] == list(
        range(document.section_count)
    )


def assert_document_basics(
    document: Document,
    *,
    content_type: str,
    modality: str | None = None,
) -> None:
    assert document.schema_version == SCHEMA_VERSION
    assert document.content_type == content_type
    if modality is not None:
        assert document.modality == modality
    assert document.document_id
    assert document.ingestor
    assert document.method
    assert document.section_count > 0
    assert all(section.text.strip() for section in document.sections)
    assert_section_indexes(document)


def sample_document() -> Document:
    return Document(
        document_id="test_roundtrip",
        source_type="manual_text",
        source_path=None,
        modality="text",
        content_type="plain_text",
        ingestor="smoke_test",
        method="manual",
        sections=[
            Section(index=0, text="Hello world.", extraction_method="manual"),
        ],
        title="Roundtrip test",
    )


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Expected test file missing: {path}")
    return path
