"""
Plain text ingestion for the Edge SLM pipeline.

Converts .txt files into a section-based Document. Paragraphs separated by
blank lines become individual Sections. Chunking is deferred to the source pack.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from ingestion.schema import (
    SOURCE_TYPE_LOCAL_FILE,
    Document,
    Section,
    new_document_id,
)


logger = logging.getLogger(__name__)


SUPPORTED_TEXT_EXTENSIONS = {
    ".txt",
    ".text",
}


def read_text_file(input_path: Path) -> tuple[str, str]:
    """Read a text file, trying common encodings."""
    encodings_to_try = ["utf-8", "utf-8-sig", "latin-1"]
    last_error = None

    for encoding in encodings_to_try:
        try:
            return input_path.read_text(encoding=encoding), encoding
        except UnicodeDecodeError as error:
            last_error = error

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"Could not decode text file: {input_path}. Last error: {last_error}",
    )


def clean_text(text: str) -> str:
    """Lightly clean plain text while preserving paragraph structure."""
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_lines: list[str] = []
    previous_line_was_blank = False

    for line in text.split("\n"):
        clean_line = re.sub(r"[ \t]+", " ", line).strip()

        if not clean_line:
            if not previous_line_was_blank:
                cleaned_lines.append("")
            previous_line_was_blank = True
            continue

        cleaned_lines.append(clean_line)
        previous_line_was_blank = False

    return "\n".join(cleaned_lines).strip()


def split_into_paragraphs(text: str) -> list[str]:
    """Split text on blank lines into paragraph blocks."""
    return [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]


def detect_title_from_text(text: str, fallback_title: str) -> str:
    """Use the first short non-empty line as a title when possible."""
    for line in text.split("\n"):
        clean_line = line.strip()
        if not clean_line:
            continue
        if len(clean_line.split()) <= 15:
            return clean_line
        break
    return fallback_title


def ingest_text(
    input_path: str | Path,
    *,
    title: str | None = None,
    language: str | None = "en",
) -> Document:
    """Ingest a plain text file into the standardised Document schema."""
    text_path = Path(input_path)

    if not text_path.exists():
        raise FileNotFoundError(f"Text file not found: {text_path}")

    if not text_path.is_file():
        raise ValueError(f"Input path is not a file: {text_path}")

    if text_path.suffix.lower() not in SUPPORTED_TEXT_EXTENSIONS:
        raise ValueError(
            f"Unsupported text file type: {text_path.suffix}. "
            f"Supported: {sorted(SUPPORTED_TEXT_EXTENSIONS)}"
        )

    raw_text, encoding_used = read_text_file(text_path)
    cleaned_text = clean_text(raw_text)
    raw_paragraphs = split_into_paragraphs(raw_text)
    cleaned_paragraphs = split_into_paragraphs(cleaned_text)

    sections: list[Section] = []

    for index, paragraph in enumerate(cleaned_paragraphs):
        raw_paragraph = raw_paragraphs[index] if index < len(raw_paragraphs) else None
        sections.append(
            Section(
                index=index,
                text=paragraph,
                raw_text=raw_paragraph,
                extraction_method="plain_text_file",
            )
        )

    if not sections:
        raise RuntimeError(f"No text content found in file: {text_path}")

    final_title = title or detect_title_from_text(
        cleaned_text,
        fallback_title=text_path.stem,
    )

    return Document(
        document_id=new_document_id("txt"),
        source_type=SOURCE_TYPE_LOCAL_FILE,
        source_path=str(text_path),
        modality="text",
        content_type="plain_text",
        ingestor="text_ingestor",
        method="plain_text_file",
        sections=sections,
        title=final_title,
        language=language,
        format_metadata={
            "text": {
                "encoding_used": encoding_used,
                "original_character_count": len(raw_text),
                "cleaned_character_count": len(cleaned_text),
                "line_count": len(cleaned_text.splitlines()),
                "paragraph_count": len(sections),
            }
        },
    )
