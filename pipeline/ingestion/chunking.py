"""
Model-sized chunking built on top of section-based ingestion documents.

This module intentionally uses dependency-free word counts as the sizing unit.
The API is kept explicit so a tokenizer-backed counter can replace the internal
word counting later without changing source-pack call sites.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from ingestion.schema import Document, Section

logger = logging.getLogger(__name__)


def _word_count(text: str) -> int:
    return len(text.split())


def _split_paragraphs(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    return parts or [text.strip()]


def _split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    return parts or [text.strip()]


@dataclass(frozen=True)
class ChunkingConfig:
    target_words: int = 450
    max_words: int = 700
    min_words: int = 120
    overlap_words: int = 60
    preserve_section_boundaries: bool = False

    def __post_init__(self) -> None:
        if self.target_words <= 0:
            raise ValueError("target_words must be > 0")
        if self.max_words < self.target_words:
            raise ValueError("max_words must be >= target_words")
        if self.min_words < 0:
            raise ValueError("min_words must be >= 0")
        if self.overlap_words < 0:
            raise ValueError("overlap_words must be >= 0")


@dataclass
class TextChunk:
    chunk_id: str
    chunk_index: int
    text: str
    word_count: int
    source_section_indexes: list[int]
    source_headings: list[str] = field(default_factory=list)
    source_pages: list[int] = field(default_factory=list)
    source_slides: list[int] = field(default_factory=list)
    source_time_range_s: tuple[float, float] | None = None
    split_reason: str = "single_section"


@dataclass
class _ChunkAccumulator:
    text_parts: list[str] = field(default_factory=list)
    section_indexes: list[int] = field(default_factory=list)
    headings: set[str] = field(default_factory=set)
    pages: set[int] = field(default_factory=set)
    slides: set[int] = field(default_factory=set)
    start_time_s: float | None = None
    end_time_s: float | None = None
    split_reason: str = "single_section"

    def add_section_text(self, section: Section, text: str) -> None:
        normalized = text.strip()
        if not normalized:
            return
        self.text_parts.append(normalized)
        if section.index not in self.section_indexes:
            self.section_indexes.append(section.index)
        if section.heading:
            self.headings.add(section.heading)
        if section.page_number is not None:
            self.pages.add(section.page_number)
        if section.slide_number is not None:
            self.slides.add(section.slide_number)
        if section.start_time_s is not None:
            self.start_time_s = (
                section.start_time_s
                if self.start_time_s is None
                else min(self.start_time_s, section.start_time_s)
            )
        if section.end_time_s is not None:
            self.end_time_s = (
                section.end_time_s
                if self.end_time_s is None
                else max(self.end_time_s, section.end_time_s)
            )

    @property
    def text(self) -> str:
        return "\n\n".join(self.text_parts).strip()

    @property
    def word_count(self) -> int:
        return _word_count(self.text)

    def merge_from(self, other: _ChunkAccumulator) -> None:
        """Append another accumulator's content and provenance."""
        for part in other.text_parts:
            if part.strip():
                self.text_parts.append(part.strip())
        for index in other.section_indexes:
            if index not in self.section_indexes:
                self.section_indexes.append(index)
        self.headings.update(other.headings)
        self.pages.update(other.pages)
        self.slides.update(other.slides)
        if other.start_time_s is not None:
            self.start_time_s = (
                other.start_time_s
                if self.start_time_s is None
                else min(self.start_time_s, other.start_time_s)
            )
        if other.end_time_s is not None:
            self.end_time_s = (
                other.end_time_s
                if self.end_time_s is None
                else max(self.end_time_s, other.end_time_s)
            )
        if len(self.section_indexes) > 1:
            self.split_reason = "merged_small_sections"
        elif other.split_reason == "split_oversized_section":
            self.split_reason = "split_oversized_section"

    def to_text_chunk(self, *, document_id: str, chunk_index: int) -> TextChunk:
        time_range: tuple[float, float] | None = None
        if self.start_time_s is not None and self.end_time_s is not None:
            time_range = (self.start_time_s, self.end_time_s)

        return TextChunk(
            chunk_id=f"{document_id}__c{chunk_index:04d}",
            chunk_index=chunk_index,
            text=self.text,
            word_count=self.word_count,
            source_section_indexes=sorted(self.section_indexes),
            source_headings=sorted(self.headings),
            source_pages=sorted(self.pages),
            source_slides=sorted(self.slides),
            source_time_range_s=time_range,
            split_reason=self.split_reason,
        )


def chunk_document(document: Document, config: ChunkingConfig | None = None) -> list[TextChunk]:
    config = config or ChunkingConfig()
    chunks: list[_ChunkAccumulator] = []
    current = _ChunkAccumulator()

    for section in document.sections:
        section_parts = _split_section(section, config=config)
        for part_index, part_text in enumerate(section_parts):
            part_words = _word_count(part_text)
            if part_words == 0:
                continue

            if current.word_count > 0 and (
                current.word_count + part_words > config.max_words
                or config.preserve_section_boundaries
                and part_index == 0
                and current.section_indexes
                and section.index != current.section_indexes[-1]
            ):
                chunks.append(current)
                current = _ChunkAccumulator()

            current.add_section_text(section, part_text)

            if len(section_parts) > 1:
                current.split_reason = "split_oversized_section"
            elif len(current.section_indexes) > 1:
                current.split_reason = "merged_small_sections"
            else:
                current.split_reason = "single_section"

            if current.word_count >= config.target_words:
                chunks.append(current)
                current = _ChunkAccumulator()

    if current.word_count > 0:
        chunks.append(current)

    chunks = _enforce_min_words(chunks, config=config, document_id=document.document_id)
    chunks = _apply_overlap(chunks, config=config)
    return [
        chunk.to_text_chunk(document_id=document.document_id, chunk_index=index)
        for index, chunk in enumerate(chunks)
    ]


def _enforce_min_words(
    chunks: list[_ChunkAccumulator],
    *,
    config: ChunkingConfig,
    document_id: str,
) -> list[_ChunkAccumulator]:
    """
    Merge or keep chunks that fall below ``config.min_words``.

    Preference order for undersized chunks:
    1. Merge forward into the next chunk when combined size <= max_words
    2. Else merge backward into the previous chunk when combined size <= max_words
    3. Else keep (single-chunk documents) or drop with a log when a neighbor
       exists but merging would exceed max_words
    """
    if config.min_words <= 0 or not chunks:
        return chunks

    if len(chunks) == 1:
        if chunks[0].word_count < config.min_words:
            logger.warning(
                "Document %s: sole chunk has %d words (< min_words=%d); keeping it",
                document_id,
                chunks[0].word_count,
                config.min_words,
            )
        return chunks

    result: list[_ChunkAccumulator] = []
    index = 0
    while index < len(chunks):
        chunk = chunks[index]
        if chunk.word_count >= config.min_words:
            result.append(chunk)
            index += 1
            continue

        # Prefer merging forward into the next chunk.
        if index + 1 < len(chunks):
            nxt = chunks[index + 1]
            if chunk.word_count + nxt.word_count <= config.max_words:
                nxt_merged = _ChunkAccumulator(
                    text_parts=list(nxt.text_parts),
                    section_indexes=list(nxt.section_indexes),
                    headings=set(nxt.headings),
                    pages=set(nxt.pages),
                    slides=set(nxt.slides),
                    start_time_s=nxt.start_time_s,
                    end_time_s=nxt.end_time_s,
                    split_reason=nxt.split_reason,
                )
                # Prepend undersized chunk into next.
                prepended = _ChunkAccumulator(
                    text_parts=list(chunk.text_parts),
                    section_indexes=list(chunk.section_indexes),
                    headings=set(chunk.headings),
                    pages=set(chunk.pages),
                    slides=set(chunk.slides),
                    start_time_s=chunk.start_time_s,
                    end_time_s=chunk.end_time_s,
                    split_reason=chunk.split_reason,
                )
                prepended.merge_from(nxt_merged)
                chunks[index + 1] = prepended
                index += 1
                continue

        # Else merge backward into the previous result chunk.
        if result and result[-1].word_count + chunk.word_count <= config.max_words:
            result[-1].merge_from(chunk)
            index += 1
            continue

        # Cannot merge without exceeding max_words — drop undersized chunk.
        logger.info(
            "Document %s: dropping undersized chunk (%d words < min_words=%d) "
            "that cannot merge within max_words=%d",
            document_id,
            chunk.word_count,
            config.min_words,
            config.max_words,
        )
        index += 1

    if not result and chunks:
        # All chunks were undersized and dropped; keep the largest one.
        largest = max(chunks, key=lambda c: c.word_count)
        logger.warning(
            "Document %s: all chunks below min_words; keeping largest (%d words)",
            document_id,
            largest.word_count,
        )
        return [largest]

    return result


def _split_section(section: Section, *, config: ChunkingConfig) -> list[str]:
    if section.word_count <= config.max_words:
        return [section.text]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_words = 0

    for paragraph in _split_paragraphs(section.text):
        paragraph_words = _word_count(paragraph)
        if paragraph_words > config.max_words:
            pieces = _split_sentences(paragraph)
        else:
            pieces = [paragraph]

        for piece in pieces:
            piece_words = _word_count(piece)
            if piece_words > config.max_words:
                words = piece.split()
                for start in range(0, len(words), config.max_words):
                    slice_words = words[start : start + config.max_words]
                    slice_text = " ".join(slice_words).strip()
                    if not slice_text:
                        continue
                    if current_words > 0:
                        chunks.append("\n\n".join(current_parts))
                        current_parts = []
                        current_words = 0
                    chunks.append(slice_text)
                continue

            if current_words > 0 and current_words + piece_words > config.max_words:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_words = 0

            current_parts.append(piece)
            current_words += piece_words

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return [chunk for chunk in chunks if chunk.strip()]


def _apply_overlap(
    chunks: list[_ChunkAccumulator], *, config: ChunkingConfig
) -> list[_ChunkAccumulator]:
    if config.overlap_words <= 0:
        return chunks

    overlapped: list[_ChunkAccumulator] = []
    previous_tail = ""

    for chunk in chunks:
        chunk_text = chunk.text
        if previous_tail and chunk.split_reason == "split_oversized_section":
            chunk_text = f"{previous_tail}\n\n{chunk_text}".strip()

        out = _ChunkAccumulator(
            text_parts=[chunk_text],
            section_indexes=list(chunk.section_indexes),
            headings=set(chunk.headings),
            pages=set(chunk.pages),
            slides=set(chunk.slides),
            start_time_s=chunk.start_time_s,
            end_time_s=chunk.end_time_s,
            split_reason=chunk.split_reason,
        )
        overlapped.append(out)

        if chunk.split_reason == "split_oversized_section":
            words = chunk.text.split()
            previous_tail = " ".join(words[-config.overlap_words :]) if words else ""
        else:
            previous_tail = ""

    return overlapped
