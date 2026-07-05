"""
PDF ingestion for the Edge SLM pipeline.

Converts a PDF file into a section-based Document (one Section per page).
Extraction uses pymupdf4llm for Markdown-structured text. When classic
extraction is suspiciously thin compared to native PyMuPDF text (common on
browser print-to-PDF exports), a layout-aware pymupdf4llm pass is retried.
Pages with no extractable text (scanned/image-only pages) fall back to OCR via PyMuPDF's
embedded Tesseract engine when Tesseract language data is installed
(`brew install tesseract` on macOS). Chunking is deferred to the source pack — this
module does not split pages into word windows.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pymupdf
import pymupdf4llm

from ingestion.schema import (
    SOURCE_TYPE_LOCAL_FILE,
    Document,
    Section,
    new_document_id,
)

# Deterministic classic extractor for the primary pass — no layout-AI drift.
# Layout-aware extraction is a selective fallback when classic output is
# suspiciously thin compared to native PyMuPDF text (e.g. browser print PDFs).
pymupdf4llm.use_layout(False)

# A page whose classic extraction has fewer words than this AND much less than
# native PyMuPDF text is likely a layout-heavy export (print-to-PDF web pages).
THIN_PAGE_WORD_THRESHOLD = 40
NATIVE_TEXT_DOMINANCE_RATIO = 2.0


OCR_DPI = 300
OCR_INSTALL_HINT = (
    "Install Tesseract language data to enable OCR for scanned pages "
    "(macOS: `brew install tesseract`; Windows: "
    "`winget install UB-Mannheim.TesseractOCR`)."
)

# ISO language -> Tesseract traineddata code. Unknown languages fall back to
# English rather than failing the whole ingest.
_TESSERACT_LANGUAGES = {
    "en": "eng",
    "de": "deu",
    "fr": "fra",
    "es": "spa",
    "it": "ita",
    "pt": "por",
    "nl": "nld",
    "ru": "rus",
    "zh": "chi_sim",
    "ja": "jpn",
    "ko": "kor",
}


def find_tessdata() -> str | None:
    """
    Locate the Tesseract language-data directory, or None if unavailable.

    PyMuPDF's OCR uses MuPDF's embedded Tesseract engine, so only the
    traineddata files are needed — not the tesseract binary. Discovery
    honours TESSDATA_PREFIX and probes well-known install locations.
    """
    try:
        return pymupdf.get_tessdata() or None
    except Exception:
        return None


def tesseract_language(language: str | None) -> str:
    if not language:
        return "eng"
    return _TESSERACT_LANGUAGES.get(language.lower().split("-")[0], "eng")


def _word_count(text: str) -> int:
    return len(text.split())


def page_needs_layout_retry(
    *,
    extracted_text: str,
    native_text: str,
) -> bool:
    """
    True when classic pymupdf4llm output is much thinner than native text.

    Typical on browser print-to-PDF exports where headers/footers extract but
    body copy does not under use_layout(False).
    """
    extracted_words = _word_count(extracted_text.strip())
    native_words = _word_count(native_text.strip())
    if native_words < THIN_PAGE_WORD_THRESHOLD:
        return False
    if extracted_words >= native_words / NATIVE_TEXT_DOMINANCE_RATIO:
        return False
    return extracted_words < THIN_PAGE_WORD_THRESHOLD


def needs_layout_retry(path: Path, page_chunks: list[dict]) -> bool:
    """True if any page looks like a layout-heavy export."""
    with pymupdf.open(path) as doc:
        for page_index, chunk in enumerate(page_chunks):
            page_number = chunk.get("metadata", {}).get("page_number", page_index + 1)
            if page_number < 1 or page_number > len(doc):
                continue
            extracted_text = chunk.get("text") or ""
            native_text = doc[page_number - 1].get_text()
            if page_needs_layout_retry(
                extracted_text=extracted_text,
                native_text=native_text,
            ):
                return True
    return False


def extract_page_chunks(path: Path) -> tuple[list[dict], bool]:
    """
    Extract page chunks with classic layout first, then retry with layout-aware
    extraction when native text is present but classic output is suspiciously thin.
    """
    pymupdf4llm.use_layout(False)
    page_chunks = pymupdf4llm.to_markdown(path, page_chunks=True)

    if not needs_layout_retry(path, page_chunks):
        return page_chunks, False

    pymupdf4llm.use_layout(True)
    try:
        page_chunks = pymupdf4llm.to_markdown(path, page_chunks=True)
    finally:
        pymupdf4llm.use_layout(False)

    return page_chunks, True


def is_garbage_text(text: str) -> bool:
    """
    Heuristic guard against OCR noise: blank output, unprintable characters,
    replacement characters, or output with almost no alphanumeric content
    (e.g. OCR of a decorative page rendered as stray punctuation).
    """
    stripped = text.strip()
    if not stripped:
        return True

    length = len(stripped)
    printable_ratio = (
        sum(ch.isprintable() or ch.isspace() for ch in stripped) / length
    )
    replacement_ratio = stripped.count("\ufffd") / length

    non_space = [ch for ch in stripped if not ch.isspace()]
    alnum_ratio = sum(ch.isalnum() for ch in non_space) / max(1, len(non_space))

    return printable_ratio < 0.90 or replacement_ratio > 0.05 or alnum_ratio < 0.50


def ocr_pdf_pages(
    path: Path,
    page_numbers: list[int],
    *,
    language: str | None,
    tessdata: str,
) -> tuple[dict[int, str], list[int]]:
    """
    OCR the given 1-based pages of a PDF.

    Returns (page_number -> raw OCR text) for pages that produced usable text,
    plus the list of pages where OCR failed or produced only garbage.
    """
    ocr_texts: dict[int, str] = {}
    failed_pages: list[int] = []
    lang = tesseract_language(language)

    with pymupdf.open(path) as doc:
        for page_number in page_numbers:
            try:
                page = doc[page_number - 1]
                textpage = page.get_textpage_ocr(
                    full=True,
                    dpi=OCR_DPI,
                    language=lang,
                    tessdata=tessdata,
                )
                raw_text = page.get_text(textpage=textpage)
            except Exception as error:
                warnings.warn(f"OCR failed on page {page_number} of {path}: {error}")
                failed_pages.append(page_number)
                continue

            if is_garbage_text(raw_text):
                failed_pages.append(page_number)
                continue

            ocr_texts[page_number] = raw_text

    return ocr_texts, failed_pages


def ingest_pdf(
    source_path: str | Path,
    *,
    title: str | None = None,
    language: str | None = "en",
    ocr_fallback: bool = True,
) -> Document:
    """
    Ingest a PDF into the standardised Document schema.

    One non-empty page becomes one Section. Pages with no extractable text
    are retried with OCR (when `ocr_fallback` is True and Tesseract language
    data is installed); pages still empty after that are skipped with a
    warning. A PDF with no extractable text at all raises RuntimeError.
    """
    path = Path(source_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if not path.is_file():
        raise ValueError(f"Expected a file, got a directory: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a .pdf file, got suffix '{path.suffix}': {path}"
        )

    try:
        page_chunks, layout_retry_used = extract_page_chunks(path)
    except Exception as error:
        raise RuntimeError(f"Failed to parse PDF: {path}") from error

    page_count = len(page_chunks)
    extraction_method = "pymupdf4llm_layout" if layout_retry_used else "pymupdf4llm"

    # (page_number, text, raw_text, extraction_method) for every page that
    # yielded content; Sections are built once, in page order, at the end.
    page_entries: list[tuple[int, str, str, str]] = []
    pages_without_native_text: list[int] = []

    for page_index, chunk in enumerate(page_chunks):
        raw_text = chunk.get("text") or ""
        text = raw_text.strip()
        page_number = chunk.get("metadata", {}).get("page_number", page_index + 1)

        if not text:
            pages_without_native_text.append(page_number)
            continue

        page_entries.append((page_number, text, raw_text, extraction_method))

    tessdata = find_tessdata() if ocr_fallback else None
    ocr_attempted = bool(pages_without_native_text) and tessdata is not None
    ocr_failed_pages: list[int] = []
    empty_pages: list[int] = list(pages_without_native_text)

    if ocr_attempted:
        ocr_texts, ocr_failed_pages = ocr_pdf_pages(
            path,
            pages_without_native_text,
            language=language,
            tessdata=tessdata,
        )
        for page_number, raw_text in ocr_texts.items():
            page_entries.append(
                (page_number, raw_text.strip(), raw_text, "pymupdf_ocr")
            )
        empty_pages = ocr_failed_pages

    ocr_hint = "" if tessdata else f" {OCR_INSTALL_HINT}"

    if not page_entries:
        raise RuntimeError(
            f"No extractable text found in any of {page_count} pages: {path}. "
            f"The PDF may be scanned (image-only), encrypted, or corrupt."
            f"{ocr_hint}"
        )

    if empty_pages:
        warnings.warn(
            f"{len(empty_pages)} of {page_count} pages had no text and were "
            f"skipped in {path}: pages {sorted(empty_pages)}.{ocr_hint}"
        )

    page_entries.sort(key=lambda entry: entry[0])
    sections = [
        Section(
            index=index,
            text=text,
            raw_text=raw_text if raw_text else None,
            page_number=page_number,
            extraction_method=extraction_method,
        )
        for index, (page_number, text, raw_text, extraction_method) in enumerate(
            page_entries
        )
    ]

    pages_ocred = sorted(
        entry[0] for entry in page_entries if entry[3] == "pymupdf_ocr"
    )
    method_parts = [extraction_method]
    if pages_ocred:
        method_parts.append("pymupdf_ocr")
    method = "+".join(method_parts)

    return Document(
        document_id=new_document_id("pdf"),
        source_type=SOURCE_TYPE_LOCAL_FILE,
        source_path=str(path),
        modality="document",
        content_type="pdf_text",
        ingestor="pdf_ingestor",
        method=method,
        sections=sections,
        title=title or path.stem,
        language=language,
        format_metadata={
            "pdf": {
                "page_count": page_count,
                "pages_with_text": len(sections),
                "pages_without_text": len(empty_pages),
                "empty_pages": sorted(empty_pages),
                "ocr_available": tessdata is not None,
                "pages_ocred": pages_ocred,
                "ocr_failed_pages": sorted(ocr_failed_pages),
                "layout_retry_used": layout_retry_used,
            }
        },
    )
