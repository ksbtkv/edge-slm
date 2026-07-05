"""
Scanned-PDF OCR fallback tests for pdf_ingestor.

Fixtures are generated programmatically with pymupdf (render a text page to
a pixmap, re-insert it as an image), so no files under data/raw are needed.
OCR tests skip when Tesseract language data is not installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pymupdf = pytest.importorskip("pymupdf")

from ingestion import pdf_ingestor
from ingestion.pdf_ingestor import ingest_pdf, is_garbage_text

from tests.ingestion_test_utils import (
    assert_document_basics,
    assert_roundtrip,
)


TESSDATA_AVAILABLE = pdf_ingestor.find_tessdata() is not None

requires_tesseract = pytest.mark.skipif(
    not TESSDATA_AVAILABLE,
    reason="Tesseract language data not installed (brew install tesseract)",
)


SAMPLE_TEXT = (
    "Delta Lake provides ACID transactions for data lakes. "
    "Structured Streaming reads Delta tables incrementally."
)


def make_image_page_pdf(
    output_path: Path,
    *,
    native_pages: list[str] = (),
    image_pages: list[str] = (),
) -> Path:
    """
    Build a PDF with the given native-text pages followed by image-only
    pages (text rendered to a pixmap and inserted as a picture).
    """
    doc = pymupdf.open()

    for text in native_pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 500), text, fontsize=14)

    for text in image_pages:
        source = pymupdf.open()
        source_page = source.new_page()
        source_page.insert_textbox(
            pymupdf.Rect(50, 50, 550, 500), text, fontsize=14
        )
        pixmap = source_page.get_pixmap(dpi=200)
        source.close()

        page = doc.new_page()
        page.insert_image(page.rect, pixmap=pixmap)

    doc.save(output_path)
    doc.close()
    return output_path


@pytest.mark.optional_dependency
@requires_tesseract
def test_fully_scanned_pdf_is_ocred(tmp_path: Path) -> None:
    pdf_path = make_image_page_pdf(
        tmp_path / "scanned.pdf", image_pages=[SAMPLE_TEXT]
    )

    document = ingest_pdf(pdf_path)

    assert_document_basics(document, content_type="pdf_text", modality="document")
    assert document.method == "pymupdf4llm+pymupdf_ocr"
    assert document.section_count == 1

    section = document.sections[0]
    assert section.extraction_method == "pymupdf_ocr"
    assert section.page_number == 1
    assert section.raw_text is not None
    assert "Delta Lake" in section.text

    pdf_meta = document.format_metadata["pdf"]
    assert pdf_meta["ocr_available"] is True
    assert pdf_meta["pages_ocred"] == [1]
    assert pdf_meta["ocr_failed_pages"] == []
    assert pdf_meta["empty_pages"] == []

    assert_roundtrip(document, "scanned_pdf")


@pytest.mark.optional_dependency
@requires_tesseract
def test_mixed_pdf_ocrs_only_image_pages(tmp_path: Path) -> None:
    pdf_path = make_image_page_pdf(
        tmp_path / "mixed.pdf",
        native_pages=["Native text on the first page."],
        image_pages=[SAMPLE_TEXT],
    )

    document = ingest_pdf(pdf_path)

    assert document.section_count == 2
    assert document.method == "pymupdf4llm+pymupdf_ocr"

    native, ocred = document.sections
    assert (native.page_number, native.extraction_method) == (1, "pymupdf4llm")
    assert (ocred.page_number, ocred.extraction_method) == (2, "pymupdf_ocr")
    assert [section.index for section in document.sections] == [0, 1]

    assert document.format_metadata["pdf"]["pages_ocred"] == [2]


@pytest.mark.optional_dependency
def test_scanned_pdf_without_tesseract_raises_with_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = make_image_page_pdf(
        tmp_path / "scanned.pdf", image_pages=[SAMPLE_TEXT]
    )
    monkeypatch.setattr(pdf_ingestor, "find_tessdata", lambda: None)

    with pytest.raises(RuntimeError, match="No extractable text"):
        ingest_pdf(pdf_path)

    with pytest.raises(RuntimeError, match="Install Tesseract"):
        ingest_pdf(pdf_path)


@pytest.mark.optional_dependency
def test_ocr_fallback_opt_out(tmp_path: Path) -> None:
    pdf_path = make_image_page_pdf(
        tmp_path / "scanned.pdf", image_pages=[SAMPLE_TEXT]
    )

    with pytest.raises(RuntimeError, match="No extractable text"):
        ingest_pdf(pdf_path, ocr_fallback=False)


def test_is_garbage_text_guard() -> None:
    assert is_garbage_text("")
    assert is_garbage_text("   \n  ")
    assert is_garbage_text("\ufffd\ufffd\ufffd some text")
    assert is_garbage_text("|| .. -- ~~ ^^ ((")
    assert not is_garbage_text(SAMPLE_TEXT)
    assert not is_garbage_text("Short heading, page 3.")
