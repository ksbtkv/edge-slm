"""
Smoke test for the ingestion pipeline.

Exercises schema round-trip, dispatch routing, and all format ingestors
against files in data/raw/. Run from the project root:

    PYTHONPATH=pipeline python -m tests.smoke_test_ingestion
"""

from __future__ import annotations

import importlib
import sys
import tempfile
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_ROOT = PROJECT_ROOT / "pipeline"
RAW_DATA = PROJECT_ROOT / "data" / "raw"

if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from ingestion.dispatch import ingest, supported_extensions
from ingestion.schema import SCHEMA_VERSION, Document, Section
from ingestion.serialize import load_document, save_document


def show(label: str, doc: Document) -> None:
    print(f"\n=== {label} ===")
    print(f"  schema_version : {doc.schema_version}")
    print(f"  document_id    : {doc.document_id}")
    print(f"  modality       : {doc.modality}")
    print(f"  content_type   : {doc.content_type}")
    print(f"  ingestor       : {doc.ingestor}")
    print(f"  method         : {doc.method}")
    print(f"  section_count  : {doc.section_count}")
    print(f"  total_words    : {doc.total_word_count}")
    if doc.sections:
        first = doc.sections[0]
        preview = first.text[:120].replace("\n", " ")
        print(f"  first section  : {preview!r}")


def assert_roundtrip(doc: Document, label: str) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / f"{label}.json"
        save_document(doc, output_path)
        loaded = load_document(output_path)
    assert loaded.section_count == doc.section_count
    assert loaded.schema_version == SCHEMA_VERSION
    print(f"  ✓ serialize round-trip OK ({label})")


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Expected test file missing: {path}")
    return path


def skip_if_import_error(module: str, label: str) -> bool:
    try:
        importlib.import_module(module)
        return False
    except ImportError as error:
        print(f"\n=== {label} ===")
        print(f"  ~ skipped ({error})")
        return True


def test_schema_roundtrip() -> None:
    doc = Document(
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

    assert_roundtrip(doc, "schema")
    print("\n=== schema round-trip ===")
    print("  ✓ save_document / load_document OK")


def test_text() -> None:
    txt_path = require_file(RAW_DATA / "text" / "SampleText.txt")
    doc = ingest(txt_path)
    show("text — SampleText.txt", doc)
    assert doc.content_type == "plain_text"
    assert doc.section_count > 0
    assert_roundtrip(doc, "text")
    print("\n=== text ingest ===")
    print("  ✓ OK")


def test_markdown() -> None:
    md_path = require_file(RAW_DATA / "markdown" / "audio_video_ingestion.md")
    doc = ingest(md_path)
    show("markdown — audio_video_ingestion.md", doc)
    assert doc.content_type == "markdown_text"
    assert doc.section_count > 0
    assert_roundtrip(doc, "markdown")
    print("\n=== markdown ingest ===")
    print("  ✓ OK")


def test_pdf() -> None:
    if skip_if_import_error("pymupdf4llm", "pdf ingest"):
        return

    clean_pdf = require_file(RAW_DATA / "pdf" / "EDGE SLM PROJECT.pdf")
    sample_pdf = require_file(RAW_DATA / "pdf" / "Sample.pdf")
    slide_pdf = RAW_DATA / "pdf" / "STAT3401_Lecture_11_Seasonality.pdf"
    ocr_pdf = RAW_DATA / "pdf" / "OCR.pdf"

    doc = ingest(clean_pdf)
    show("pdf — EDGE SLM PROJECT.pdf", doc)
    assert doc.content_type == "pdf_text"
    assert doc.section_count > 0

    sample_doc = ingest(sample_pdf)
    show("pdf — Sample.pdf", sample_doc)
    assert sample_doc.section_count > 0

    if slide_pdf.exists():
        print("\n=== pdf — slide deck (expect empty-page warning) ===")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            slide_doc = ingest(slide_pdf)
            if caught:
                print(f"  ✓ warning fired: {caught[0].message}")
            else:
                print("  ~ no empty-page warning")
        show("pdf — STAT3401_Lecture_11_Seasonality.pdf", slide_doc)
        assert slide_doc.section_count > 0

    if ocr_pdf.exists():
        try:
            ocr_doc = ingest(ocr_pdf)
            show("pdf — OCR.pdf", ocr_doc)
            print("  ~ OCR.pdf ingested (may have empty pages without OCR enabled)")
        except RuntimeError as error:
            print(f"\n=== pdf — OCR.pdf ===")
            print(f"  ~ expected for image-only PDF without OCR: {error}")

    assert_roundtrip(doc, "pdf")

    not_a_pdf = PIPELINE_ROOT / "ingestion" / "schema.py"
    print("\n=== pdf failure guard ===")
    try:
        ingest(not_a_pdf)
        print("  ✗ FAILED: no exception for non-PDF input")
    except ValueError as error:
        print(f"  ✓ correctly raised ValueError: {error}")

    print("\n=== pdf ingest ===")
    print("  ✓ OK")


def test_pptx() -> None:
    if skip_if_import_error("pptx", "pptx ingest"):
        return

    pptx_path = require_file(RAW_DATA / "pptx" / "SamplePPT.pptx")
    doc = ingest(pptx_path)
    show("pptx — SamplePPT.pptx", doc)
    assert doc.content_type == "slide_text"
    assert doc.section_count > 0
    assert_roundtrip(doc, "pptx")
    print("\n=== pptx ingest ===")
    print("  ✓ OK")


def test_audio() -> None:
    if skip_if_import_error("faster_whisper", "audio ingest"):
        return

    for audio_path in sorted((RAW_DATA / "audio").glob("*")):
        if audio_path.suffix.lower() not in {".mp3", ".wav", ".m4a", ".flac"}:
            continue

        doc = ingest(audio_path)
        show(f"audio — {audio_path.name}", doc)
        assert doc.content_type == "transcript"
        assert doc.modality == "audio"
        assert doc.section_count > 0
        assert_roundtrip(doc, audio_path.stem)

    print("\n=== audio ingest ===")
    print("  ✓ OK")


def test_video() -> None:
    if skip_if_import_error("faster_whisper", "video ingest"):
        return

    import shutil

    if shutil.which("ffmpeg") is None:
        print("\n=== video ingest ===")
        print("  ~ skipped (ffmpeg not found in PATH)")
        return

    video_dir = RAW_DATA / "video"
    video_files = [
        path for path in sorted(video_dir.glob("*"))
        if path.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    ]

    if not video_files:
        print("\n=== video ingest ===")
        print("  ~ skipped (no video files in data/raw/video)")
        return

    for video_path in video_files:
        doc = ingest(video_path)
        show(f"video — {video_path.name}", doc)
        assert doc.content_type == "transcript"
        assert doc.modality == "video"
        assert doc.section_count > 0
        assert_roundtrip(doc, video_path.stem)

    print("\n=== video ingest ===")
    print("  ✓ OK")


def main() -> None:
    print("Raw data root:", RAW_DATA)
    print("Supported extensions:", supported_extensions())

    test_schema_roundtrip()
    test_text()
    test_markdown()
    test_pdf()
    test_pptx()
    test_audio()
    test_video()

    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()
