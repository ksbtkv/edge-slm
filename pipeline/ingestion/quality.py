"""
Shared content-quality helpers for ingestion and chunking.

Ingestors use these to skip empty or header-only sections before they reach
the chunker. Chunking enforces its own ``min_words`` threshold separately.
"""

from __future__ import annotations

# Minimum body words for a section to be kept at ingest time.
MIN_BODY_WORDS = 15


def section_word_count(text: str) -> int:
    """Whitespace word count of section body text."""
    return len(text.split()) if text else 0


def is_effectively_empty(text: str) -> bool:
    """True when text is blank or whitespace-only."""
    return not text or not text.strip()


def is_below_body_word_threshold(
    text: str,
    *,
    min_words: int = MIN_BODY_WORDS,
) -> bool:
    """True when body text has fewer than ``min_words`` words."""
    return section_word_count(text) < min_words
