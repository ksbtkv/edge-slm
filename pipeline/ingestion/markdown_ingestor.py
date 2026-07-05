"""
Markdown ingestion for the Edge SLM pipeline.

Converts Markdown files into a section-based Document. Heading-delimited blocks
become Sections with the heading stored on each Section. Chunking is deferred
to the source pack.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from ingestion.schema import (
    SOURCE_TYPE_LOCAL_FILE,
    Document,
    Section,
    new_document_id,
)


logger = logging.getLogger(__name__)


SUPPORTED_MARKDOWN_EXTENSIONS = {
    ".md",
    ".markdown",
}


def read_markdown_file(input_path: Path) -> tuple[str, str]:
    """Read a Markdown file, trying common encodings."""
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
        f"Could not decode Markdown file: {input_path}. Last error: {last_error}",
    )


def clean_markdown_text(text: str) -> str:
    """Normalise line endings and collapse runs of blank lines."""
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_lines: list[str] = []
    blank_line_count = 0

    for line in text.split("\n"):
        clean_line = line.rstrip()

        if not clean_line.strip():
            blank_line_count += 1
            if blank_line_count <= 1:
                cleaned_lines.append("")
            continue

        cleaned_lines.append(clean_line)
        blank_line_count = 0

    return "\n".join(cleaned_lines).strip()


def get_markdown_heading(line: str) -> tuple[int, str] | None:
    """Detect a Markdown heading line, returning (level, text) or None."""
    match = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line.strip())
    if not match:
        return None
    return len(match.group(1)), match.group(2).strip()


def detect_title_from_markdown(text: str, fallback_title: str) -> str:
    """Detect a document title from Markdown content."""
    first_heading = None

    for line in text.split("\n"):
        heading = get_markdown_heading(line)
        if not heading:
            continue
        heading_level, heading_text = heading
        if heading_level == 1:
            return heading_text
        if first_heading is None:
            first_heading = heading_text

    if first_heading:
        return first_heading

    for line in text.split("\n"):
        clean_line = line.strip()
        if not clean_line:
            continue
        if len(clean_line.split()) <= 15:
            return clean_line
        break

    return fallback_title


def count_markdown_headings(text: str) -> int:
    return sum(1 for line in text.split("\n") if get_markdown_heading(line))


def count_code_blocks(text: str) -> int:
    count = 0
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            count += 1
    return count // 2


def split_markdown_into_sections(text: str) -> list[dict[str, Any]]:
    """Split Markdown into heading-delimited sections."""
    sections: list[dict[str, Any]] = []
    current_lines: list[str] = []
    current_heading: str | None = None

    for line in text.split("\n"):
        heading = get_markdown_heading(line)

        if heading:
            if current_lines:
                sections.append(
                    {
                        "section_heading": current_heading,
                        "text": "\n".join(current_lines).strip(),
                    }
                )
            current_heading = heading[1]
            current_lines = [line]
            continue

        current_lines.append(line)

    if current_lines:
        sections.append(
            {
                "section_heading": current_heading,
                "text": "\n".join(current_lines).strip(),
            }
        )

    return sections


def ingest_markdown(
    input_path: str | Path,
    *,
    title: str | None = None,
    language: str | None = "en",
) -> Document:
    """Ingest a Markdown file into the standardised Document schema."""
    markdown_path = Path(input_path)

    if not markdown_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {markdown_path}")

    if not markdown_path.is_file():
        raise ValueError(f"Input path is not a file: {markdown_path}")

    if markdown_path.suffix.lower() not in SUPPORTED_MARKDOWN_EXTENSIONS:
        raise ValueError(
            f"Unsupported Markdown file type: {markdown_path.suffix}. "
            f"Supported: {sorted(SUPPORTED_MARKDOWN_EXTENSIONS)}"
        )

    raw_text, encoding_used = read_markdown_file(markdown_path)
    cleaned_text = clean_markdown_text(raw_text)
    raw_sections = split_markdown_into_sections(raw_text)
    cleaned_sections = split_markdown_into_sections(cleaned_text)

    sections: list[Section] = []

    for index, section in enumerate(cleaned_sections):
        section_text = section["text"]
        if not section_text.strip():
            continue

        raw_section_text = (
            raw_sections[index]["text"] if index < len(raw_sections) else None
        )

        sections.append(
            Section(
                index=len(sections),
                text=section_text,
                raw_text=raw_section_text,
                heading=section["section_heading"],
                extraction_method="markdown_file",
            )
        )

    if not sections:
        raise RuntimeError(f"No text content found in file: {markdown_path}")

    final_title = title or detect_title_from_markdown(
        cleaned_text,
        fallback_title=markdown_path.stem,
    )

    return Document(
        document_id=new_document_id("md"),
        source_type=SOURCE_TYPE_LOCAL_FILE,
        source_path=str(markdown_path),
        modality="text",
        content_type="markdown_text",
        ingestor="markdown_ingestor",
        method="markdown_file",
        sections=sections,
        title=final_title,
        language=language,
        format_metadata={
            "markdown": {
                "encoding_used": encoding_used,
                "original_character_count": len(raw_text),
                "cleaned_character_count": len(cleaned_text),
                "line_count": len(cleaned_text.splitlines()),
                "heading_count": count_markdown_headings(cleaned_text),
                "code_block_count": count_code_blocks(cleaned_text),
                "section_count": len(sections),
            }
        },
    )
