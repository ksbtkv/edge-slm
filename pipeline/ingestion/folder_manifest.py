"""
Build a source-pack manifest from a local directory tree.

Scans a folder for ingestible files, assigns stable source_ids and metadata,
and returns a `SourceManifest` ready for `build_source_pack_from_manifest`.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from ingestion.dispatch import supported_extensions
from ingestion.source_manifest import (
    MANIFEST_VERSION,
    SourceManifest,
    SourceRecord,
    TopicBucket,
)

logger = logging.getLogger(__name__)

_JUNK_NAMES = frozenset({"Thumbs.db", ".DS_Store"})
_AUDIO_VIDEO_EXTENSIONS = frozenset(
    {
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
    }
)
_DEFAULT_TOPIC_BUCKET = TopicBucket(
    id="local_folder",
    label="Local folder",
    description="Files discovered from a local directory batch scan.",
)


def scan_ingestible_files(
    folder: str | Path,
    *,
    recursive: bool = True,
) -> list[Path]:
    """
    Return sorted paths to ingestible files under ``folder``.

    Raises ``FileNotFoundError`` if ``folder`` does not exist and
    ``NotADirectoryError`` if it is not a directory.
    """
    root = Path(folder)
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root}")

    supported = {ext.lower() for ext in supported_extensions()}
    discovered: list[Path] = []

    if recursive:
        candidates = root.rglob("*")
    else:
        candidates = root.iterdir()

    for path in candidates:
        if not path.is_file():
            continue
        if _is_skippable_path(path, root):
            continue
        if path.suffix.lower() not in supported:
            continue
        discovered.append(path.resolve())

    return sorted(discovered)


def count_unsupported_files(
    folder: str | Path,
    *,
    recursive: bool = True,
) -> int:
    """Count non-ingestible files that are not hidden or junk."""
    root = Path(folder).resolve()
    if not root.exists() or not root.is_dir():
        return 0

    supported = {ext.lower() for ext in supported_extensions()}
    skipped = 0

    if recursive:
        candidates = root.rglob("*")
    else:
        candidates = root.iterdir()

    for path in candidates:
        if not path.is_file():
            continue
        if _is_skippable_path(path, root):
            continue
        if path.suffix.lower() not in supported:
            skipped += 1

    return skipped


def manifest_from_folder(
    folder: str | Path,
    *,
    pack_id: str | None = None,
    title: str | None = None,
    domain: str | None = None,
    topic_bucket_ids: list[str] | None = None,
    recursive: bool = True,
) -> SourceManifest:
    """
    Build a ``SourceManifest`` from discovered files under ``folder``.

    Each file becomes one enabled ``SourceRecord`` with an absolute
    ``local_path``, stable ``source_id``, and inferred ``resource_type``.
    """
    root = Path(folder).resolve()
    files = scan_ingestible_files(root, recursive=recursive)

    bucket_ids = topic_bucket_ids or ["local_folder"]
    topic_buckets = [_DEFAULT_TOPIC_BUCKET]

    resolved_pack_id = pack_id or _slugify_stem(root.name) or "local_folder_pack"
    resolved_title = title or root.name
    resolved_domain = domain or "Local documents"

    used_ids: dict[str, int] = {}
    sources: list[SourceRecord] = []

    for index, file_path in enumerate(files):
        relative = file_path.relative_to(root)
        base_id = _slugify_stem(file_path.stem)
        source_id = _unique_source_id(base_id, relative, used_ids)
        sources.append(
            SourceRecord(
                source_id=source_id,
                title=_humanize_stem(file_path.stem),
                resource_type=_infer_resource_type(file_path.suffix),
                local_path=str(file_path),
                topic_bucket_ids=list(bucket_ids),
                priority=index + 1,
                split="unassigned",
                enabled=True,
            )
        )

    return SourceManifest(
        manifest_version=MANIFEST_VERSION,
        pack_id=resolved_pack_id,
        title=resolved_title,
        description=f"Auto-generated manifest from folder: {root}",
        domain=resolved_domain,
        topic_buckets=topic_buckets,
        sources=sources,
    )


def _is_skippable_path(path: Path, folder: Path) -> bool:
    if path.name in _JUNK_NAMES:
        return True
    try:
        rel_parts = path.relative_to(folder).parts
    except ValueError:
        return True
    for part in rel_parts:
        if part.startswith("."):
            return True
        if part == "__MACOSX":
            return True
    return False


def _slugify_stem(stem: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")
    return slug or "file"


def _humanize_stem(stem: str) -> str:
    text = re.sub(r"[_\-]+", " ", stem).strip()
    return text or stem


def _unique_source_id(
    base_id: str,
    relative_path: Path,
    used_ids: dict[str, int],
) -> str:
    if base_id not in used_ids:
        used_ids[base_id] = 1
        return base_id

    digest = hashlib.sha256(str(relative_path).encode("utf-8")).hexdigest()[:4]
    candidate = f"{base_id}_{digest}"
    while candidate in used_ids:
        digest = hashlib.sha256(f"{relative_path}:{digest}".encode("utf-8")).hexdigest()[
            :4
        ]
        candidate = f"{base_id}_{digest}"
    used_ids[candidate] = 1
    return candidate


def _infer_resource_type(suffix: str) -> str:
    ext = suffix.lower()
    if ext in (".pdf", ".md", ".markdown"):
        return "documentation"
    if ext == ".pptx":
        return "tutorial"
    if ext in (".txt", ".text"):
        return "article"
    if ext in _AUDIO_VIDEO_EXTENSIONS:
        return "video_transcript"
    return "documentation"
