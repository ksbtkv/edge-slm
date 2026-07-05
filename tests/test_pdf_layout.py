"""Layout-retry tests for browser print-to-PDF exports."""

from __future__ import annotations

from pathlib import Path

import pytest

pymupdf = pytest.importorskip("pymupdf")

from ingestion.pdf_ingestor import (
    ingest_pdf,
    page_needs_layout_retry,
)
from tests.ingestion_test_utils import PROJECT_ROOT, assert_document_basics


LEARN_PRINT_PDF = (
    PROJECT_ROOT.parent
    / "meetings"
    / "Delta Lake table streaming reads and writes - Azure Databricks _ Microsoft Learn.pdf"
)


def test_page_needs_layout_retry_detects_chrome_only_extraction() -> None:
    chrome = (
        "6/29/26, 11:39 AM Delta Lake table streaming reads and writes\n"
        "https://learn.microsoft.com/en-us/azure/databricks/structured-streaming/delta-lake 1/12"
    )
    native = (
        "Delta Lake table streaming reads and writes\n"
        "This page describes how to use Delta Lake tables as sources and sinks "
        "for Spark Structured Streaming with readStream and writeStream. "
        "Delta Lake solves common performance and reliability problems for "
        "streaming systems and files. The benefits include coalescing small "
        "files, exactly-once processing, and efficient file discovery."
    )
    assert page_needs_layout_retry(extracted_text=chrome, native_text=native)


def test_page_needs_layout_retry_ignores_matching_extraction() -> None:
    text = "Delta Lake streaming reads and writes with readStream and writeStream."
    assert not page_needs_layout_retry(extracted_text=text, native_text=text)


@pytest.mark.optional_dependency
@pytest.mark.skipif(not LEARN_PRINT_PDF.exists(), reason="client print PDF fixture missing")
def test_learn_print_pdf_uses_layout_retry() -> None:
    document = ingest_pdf(LEARN_PRINT_PDF)

    assert_document_basics(document, content_type="pdf_text", modality="document")
    assert document.total_word_count > 1000
    assert document.format_metadata["pdf"]["layout_retry_used"] is True
    assert document.method == "pymupdf4llm_layout"

    combined = "\n".join(section.text for section in document.sections)
    for term in ("readStream", "writeStream", "checkpointLocation", "foreachBatch"):
        assert term in combined
