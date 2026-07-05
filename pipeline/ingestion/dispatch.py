"""
Extension -> ingestor routing for the Edge SLM ingestion pipeline.

`ingest(path)` is the single public entry point for Stage 1: hand it a file,
get back a `Document`, regardless of format. It owns one job — pick the right
ingestor by file extension — and nothing else.

Two deliberate design choices:

1. Ingestors are pure functions, `(path, **kwargs) -> Document`, with NO
   registration side effects of their own. The wiring (which extension maps to
   which function) lives here, centrally, in `_wire_default_ingestors`. This is
   chosen over a decorator-registry on purpose: a decorator registers as an
   import-time side effect, so a format silently goes missing if its module is
   never imported. Central wiring is greppable, explicit, and side-effect-free —
   and it keeps each ingestor independently importable (the GUI can call
   `ingest_pdf` directly) and independently testable.

   Adding a format = write `<fmt>_ingestor.py` + add one import and one
   `register(...)` line below. Two visible touches, no magic.

2. Unknown extension and missing file both fail loudly, with a message that
   lists what *is* supported — never a silent skip or a None return.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Callable

from ingestion.schema import Document


logger = logging.getLogger(__name__)


# An ingestor is any callable that turns a path into a Document. Extra keyword
# arguments (e.g. title, language) are passed straight through by `ingest`.
Ingestor = Callable[..., Document]


_REGISTRY: dict[str, Ingestor] = {}


def register(extension: str, ingestor: Ingestor) -> None:
    """
    Map a file extension to an ingestor. Loud on a real collision.

    `extension` is matched case-insensitively and must include the leading dot
    (".pdf"). Re-registering the *same* callable is a no-op (safe re-import);
    registering a *different* callable for an already-claimed extension raises,
    so two ingestors can never silently fight over one format.
    """
    if not extension.startswith("."):
        raise ValueError(f"extension must start with '.', got {extension!r}")

    ext = extension.lower()
    existing = _REGISTRY.get(ext)

    if existing is not None and existing is not ingestor:
        raise ValueError(
            f"extension {ext!r} is already registered to "
            f"{existing.__name__!r}; refusing to overwrite with "
            f"{getattr(ingestor, '__name__', ingestor)!r}."
        )

    _REGISTRY[ext] = ingestor


def supported_extensions() -> list[str]:
    """Sorted list of currently routable extensions."""
    return sorted(_REGISTRY)


def ingest(path: str | Path, **kwargs) -> Document:
    """
    Route a file to its ingestor by extension and return a Document.

    Guards run cheapest-first. Keyword arguments are forwarded to the selected
    ingestor unchanged.

    Raises:
        FileNotFoundError: path does not exist.
        ValueError:        path is not a file, has no extension, or the
                           extension has no registered ingestor.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    ext = file_path.suffix.lower()
    if not ext:
        raise ValueError(f"File has no extension, cannot route: {file_path}")

    ingestor = _REGISTRY.get(ext)
    if ingestor is None:
        raise ValueError(
            f"No ingestor registered for {ext!r} ({file_path.name}). "
            f"Supported: {supported_extensions() or '[none registered]'}."
        )

    return ingestor(file_path, **kwargs)


def _lazy_ingestor(module: str, func_name: str) -> Ingestor:
    """Create a tiny proxy so optional dependencies load only on use."""

    def _ingest(path: str | Path, **kwargs) -> Document:
        try:
            ingestor = getattr(importlib.import_module(module), func_name)
        except ImportError as error:
            raise ImportError(
                f"Cannot ingest with {module}.{func_name}; install the "
                f"optional dependencies for this format. Original error: {error}"
            ) from error

        return ingestor(path, **kwargs)

    _ingest.__name__ = func_name
    return _ingest


def _register_ingestor(extension: str, module: str, func_name: str) -> None:
    """Register one ingestor without importing its optional dependencies."""
    register(extension, _lazy_ingestor(module, func_name))


def _wire_default_ingestors() -> None:
    """
    Central wiring: register the real ingestors.

    Each ingestor is registered through a lazy proxy so importing dispatch is
    always cheap. Optional dependencies are imported only when a matching file
    is actually ingested.
    """
    _register_ingestor(".pdf", "ingestion.pdf_ingestor", "ingest_pdf")
    _register_ingestor(".pptx", "ingestion.pptx_ingestor", "ingest_pptx")
    _register_ingestor(".txt", "ingestion.text_ingestor", "ingest_text")
    _register_ingestor(".text", "ingestion.text_ingestor", "ingest_text")
    _register_ingestor(".md", "ingestion.markdown_ingestor", "ingest_markdown")
    _register_ingestor(".markdown", "ingestion.markdown_ingestor", "ingest_markdown")

    for extension in (
        ".mp3",
        ".wav",
        ".m4a",
        ".flac",
        ".aac",
        ".ogg",
        ".opus",
        ".webm",
        ".mp4",
        ".mov",
        ".mkv",
        ".avi",
    ):
        _register_ingestor(extension, "ingestion.audio_video_ingestor", "ingest_media")


_wire_default_ingestors()