"""
Typed schema contract for the Edge SLM ingestion pipeline.

Every ingestor returns a `Document`. Every downstream stage reads a `Document`.
This module defines that contract and nothing else: no chunking, no file I/O,
no extraction. Serialization lives in `serialize.py`; routing lives in
`dispatch.py`.

Design notes
------------
1. The atomic unit is a `Section`, not a chunk. A Section is a faithful
   structural unit of the source (a page, a slide, a heading-delimited block,
   a transcript segment). Ingestion does NOT impose a fixed-size window — that
   is a source-pack / chunking concern. The deliberate absence of a `Chunk` type is the
   physical encoding of "defer chunking to the source pack".

2. Location/provenance fields on a Section are flat and optional. A PDF section
   populates `page_number`; a slide populates `slide_number`; a transcript
   segment populates `start_time_s`/`end_time_s`. Any format leaves the rest as
   None. This keeps the schema granularity-agnostic: whether a PDF maps to one
   Section per page or one per heading is an *ingestor* decision, not a schema
   decision.

3. The core schema is typed and locked. Format-specific extras (OCR stats,
   slide counts, video metadata) go in `Document.format_metadata`, an open bag.
   This is what keeps the contract thin: adding a format never edits the core
   types.

4. `raw_text` is retained alongside cleaned `text` so cleaning is re-runnable
   without re-ingesting. CONTRACT: `raw_text` is byte-faithful — it is the
   *unmodified* output of the extraction backend, with no whitespace collapse,
   newline normalization, or case change applied by our cleaning code. This is
   not a nicety. Deferred, downstream cleaning that reads original textual cues
   (de-hyphenation across line breaks, ASR caption-overlap dedup, anything that
   re-derives from the original) is only possible because `raw_text` is faithful.
   A half-cleaned `raw_text` is worse than `None`: it looks recoverable and is
   not. An ingestor that has no faithful original to offer sets `raw_text=None`
   rather than storing an already-processed copy.

5. The schema holds JSON-native data only: str, int, float, bool, None, list,
   and dict of those. This is why `dataclasses.asdict(doc)` round-trips through
   `json.dumps` with no custom encoder. Anything richer (e.g. a `datetime`) must
   be stringified at the boundary — `created_at` already is. Do not add a field
   whose type is not JSON-native, or `serialize` breaks silently on load.

6. An empty `sections` list is structurally valid here. "This source produced
   no text" is a real error, but it is an *ingestor* policy (raise loudly at the
   ingestor), not a schema invariant.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# Bumped from Sahaj's "0.1". This is a deliberately incompatible, section-based
# contract. `serialize.load_document` checks this on load, so old chunk-based
# "0.1" files are rejected loudly — that rejection is the intended behaviour.
SCHEMA_VERSION = "1.0"


# Source types — where the bytes came from. Constants avoid spelling drift.
SOURCE_TYPE_LOCAL_FILE = "local_file"
SOURCE_TYPE_YOUTUBE_URL = "youtube_url"
SOURCE_TYPE_URL = "url"
SOURCE_TYPE_MANUAL_TEXT = "manual_text"


def utc_now_iso() -> str:
    """Current UTC time as a clean ISO-8601 string, e.g. 2026-06-29T04:30:00Z."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def new_document_id(prefix: str = "doc") -> str:
    """
    Short unique document id, e.g. 'pdf_a1b2c3d4'.

    The prefix is for human debugging only; uniqueness comes from the uuid4
    fragment. Ingestors should pass their own prefix ("pdf", "pptx", ...).
    """
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass
class Section:
    """
    One faithful structural unit of a source document.

    Required:
        index: ordinal position within the document (0-based).
        text:  cleaned text — the canonical content downstream stages read.

    Optional location/provenance (populated per format; the rest stay None):
        raw_text:          byte-faithful original — the unmodified extraction
                           backend output, no whitespace/newline/case changes.
                           Retained so deferred cleaning can re-derive from the
                           original. Set to None if no faithful original exists;
                           never store a half-cleaned copy here.
        heading:           section / slide heading if the format has one.
        page_number:       1-based page (PDF).
        slide_number:      1-based slide (PPTX).
        start_time_s:      transcript segment start (audio / video).
        end_time_s:        transcript segment end (audio / video).
        speaker:           transcript speaker label, if known.
        confidence:        extraction confidence (e.g. ASR), if known.
        extraction_method: how this text was produced, e.g. "pymupdf4llm",
                           "pymupdf_ocr", "faster-whisper". This is where the
                           OCR-vs-text distinction lives — no separate flag.
    """

    index: int
    text: str
    raw_text: str | None = None
    heading: str | None = None
    page_number: int | None = None
    slide_number: int | None = None
    start_time_s: float | None = None
    end_time_s: float | None = None
    speaker: str | None = None
    confidence: float | None = None
    extraction_method: str | None = None

    def __post_init__(self) -> None:
        # Cheap, loud guards — catch real wiring bugs at construction, ordered
        # cheapest-first.
        if not isinstance(self.index, int) or self.index < 0:
            raise ValueError(
                f"Section.index must be a non-negative int, got {self.index!r}"
            )
        if not isinstance(self.text, str):
            raise TypeError(
                f"Section.text must be str, got {type(self.text).__name__}"
            )

    @property
    def word_count(self) -> int:
        """Whitespace word count of the cleaned text. Derived, never stored."""
        return len(self.text.split())


@dataclass
class Document:
    """
    The single contract between ingestion and the rest of the pipeline.

    Required (every ingestor supplies these explicitly):
        document_id, source_type, source_path, modality, content_type,
        ingestor, method, sections.

    Stamped / defaulted:
        title, language, format_metadata, schema_version, created_at.

    `format_metadata` is an intentionally open bag for format-specific extras
    (OCR stats, author, slide counts, video info). The core fields above are
    typed and locked; the bag absorbs everything format-specific so that adding
    a new format never edits this class.

    Field semantics:
        modality:     original source kind — "document", "slides", "audio",
                      "video", "text".
        content_type: extracted content kind — "pdf_text", "slide_text",
                      "transcript", "markdown_text", "plain_text".
        method:       the extraction method label, e.g. "pymupdf4llm".
    """

    document_id: str
    source_type: str
    source_path: str | None
    modality: str
    content_type: str
    ingestor: str
    method: str
    sections: list[Section]
    title: str | None = None
    language: str | None = "en"
    format_metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    created_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        if not self.document_id:
            raise ValueError("Document.document_id must be a non-empty string")
        if not isinstance(self.sections, list):
            raise TypeError(
                f"Document.sections must be a list, got "
                f"{type(self.sections).__name__}"
            )
        if not all(isinstance(section, Section) for section in self.sections):
            raise TypeError("Document.sections must contain only Section instances")

    @property
    def section_count(self) -> int:
        return len(self.sections)

    @property
    def total_word_count(self) -> int:
        return sum(section.word_count for section in self.sections)