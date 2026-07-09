"""
One-time acquisition of video transcripts for a source-pack manifest.

For each `video_transcript` source in the manifest:
1. Download audio only from `original_url` with yt-dlp into
   data/raw/databricks/videos/audio/ (gitignored).
2. Transcribe with faster-whisper (reusing the pipeline's ingestion logic).
3. Write blank-line-separated paragraphs to the manifest's `local_path`, so the
   plain-text ingestor produces one Section per paragraph at pack-build time.

For each `playlist` source, write an index Markdown file (video titles + URLs)
to `local_path` without downloading the videos themselves.

With `--expand-playlists`, each playlist video becomes its own `video_transcript`
manifest entry (downloaded + transcribed), and the parent playlist index source
is disabled to avoid thin duplicate chunks.

Usage:
    python scripts/fetch_transcripts.py [--manifest PATH] [--model base.en]
                                        [--only SOURCE_ID ...] [--force]
                                        [--skip-download] [--expand-playlists]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "pipeline"))

logger = logging.getLogger("fetch_transcripts")

DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "databricks_ld_foundations.json"
AUDIO_DIR = PROJECT_ROOT / "data" / "raw" / "databricks" / "videos" / "audio"
PLAYLIST_TRANSCRIPT_DIR = "data/raw/databricks/videos/playlist"


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_manifest_sources(manifest_path: Path) -> tuple[list[dict], list[dict]]:
    manifest = load_manifest(manifest_path)
    sources = manifest["sources"]
    videos = [s for s in sources if s["resource_type"] == "video_transcript"]
    playlists = [s for s in sources if s["resource_type"] == "playlist"]
    return videos, playlists


def _completed_audio_files(source_id: str) -> list[Path]:
    """Downloaded audio files for a source, ignoring partial downloads."""
    return [
        path
        for path in sorted(AUDIO_DIR.glob(f"{source_id}.*"))
        if not path.name.endswith((".part", ".ytdl"))
    ]


def download_audio(url: str, source_id: str) -> Path:
    """Download best audio for a video and return the local audio file path."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    existing = _completed_audio_files(source_id)
    if existing:
        logger.info("Audio already downloaded for %s: %s", source_id, existing[0])
        return existing[0]

    output_template = str(AUDIO_DIR / f"{source_id}.%(ext)s")
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "-f",
        "bestaudio",
        "--extract-audio",
        "--audio-format",
        "m4a",
        "-o",
        output_template,
        url,
    ]
    logger.info("Downloading audio for %s from %s", source_id, url)
    subprocess.run(command, check=True)

    downloaded = _completed_audio_files(source_id)
    if not downloaded:
        raise RuntimeError(f"yt-dlp reported success but no audio file found for {source_id}")
    return downloaded[0]


def transcribe_to_text(audio_path: Path, *, model, title: str) -> str:
    """Transcribe audio and return blank-line-separated paragraph text."""
    from ingestion.audio_video_ingestor import transcribe_audio_file

    document = transcribe_audio_file(
        audio_path,
        model=model,
        title=title,
        merge_sections=True,
        target_section_words=120,
        max_section_duration_seconds=90.0,
        max_gap_seconds=8.0,
    )
    return "\n\n".join(section.text for section in document.sections) + "\n"


def fetch_playlist_entries(playlist_url: str) -> dict[str, Any]:
    """Fetch flat playlist metadata via yt-dlp JSON."""
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--flat-playlist",
        "-J",
        playlist_url,
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def write_playlist_index(source: dict, playlist: dict[str, Any] | None = None) -> None:
    """Write a Markdown index of playlist entries to the source's local_path."""
    local_path = PROJECT_ROOT / source["local_path"]
    if playlist is None:
        logger.info("Fetching playlist index for %s", source["source_id"])
        playlist = fetch_playlist_entries(source["original_url"])

    lines = [
        f"# {playlist.get('title') or source['title']}",
        "",
        f"Playlist: {source['original_url']}",
        "",
    ]
    for position, entry in enumerate(playlist.get("entries") or [], start=1):
        video_id = entry.get("id")
        video_url = entry.get("url") or f"https://www.youtube.com/watch?v={video_id}"
        title = entry.get("title") or video_id
        duration = entry.get("duration")
        duration_note = f" ({int(duration) // 60} min)" if duration else ""
        lines.append(f"{position}. [{title}]({video_url}){duration_note}")

    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote playlist index: %s", local_path)


def playlist_prefix(source_id: str) -> str:
    """Map playlist source_id to a short prefix for child source_ids."""
    mapping = {
        "video_associate_playlist": "pl_associate",
        "video_spark_de_playlist": "pl_spark_de",
    }
    if source_id in mapping:
        return mapping[source_id]
    # Fallback: strip common prefixes
    slug = source_id
    for prefix in ("video_", "playlist_"):
        if slug.startswith(prefix):
            slug = slug[len(prefix) :]
    if slug.endswith("_playlist"):
        slug = slug[: -len("_playlist")]
    return f"pl_{slug}"


def slugify_title(title: str, *, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_").lower()
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("_")
    return slug or "video"


def child_source_id(playlist_source_id: str, position: int, title: str, video_id: str) -> str:
    prefix = playlist_prefix(playlist_source_id)
    slug = slugify_title(title) if title else (video_id or "video")
    return f"{prefix}_{position:02d}_{slug}"


def build_playlist_child_source(
    playlist_source: dict[str, Any],
    *,
    position: int,
    entry: dict[str, Any],
) -> dict[str, Any]:
    """Create a video_transcript manifest entry for one playlist video."""
    video_id = entry.get("id") or f"pos{position}"
    title = entry.get("title") or f"Playlist video {position}"
    video_url = entry.get("url") or f"https://www.youtube.com/watch?v={video_id}"
    source_id = child_source_id(playlist_source["source_id"], position, title, str(video_id))
    local_path = f"{PLAYLIST_TRANSCRIPT_DIR}/{source_id}.txt"
    return {
        "source_id": source_id,
        "title": title,
        "resource_type": "video_transcript",
        "original_url": video_url,
        "local_path": local_path,
        "topic_bucket_ids": list(playlist_source.get("topic_bucket_ids") or []),
        "description": (
            f"Expanded from playlist {playlist_source['source_id']} "
            f"(position {position})."
        ),
        "priority": int(playlist_source.get("priority") or 50) + position,
        "split": playlist_source.get("split") or "unassigned",
        "enabled": True,
        "notes": f"Parent playlist: {playlist_source['source_id']}",
        "parent_playlist_id": playlist_source["source_id"],
    }


def expand_playlist_sources(
    manifest: dict[str, Any],
    *,
    playlist_ids: set[str] | None = None,
    fetch_entries=fetch_playlist_entries,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Add per-video transcript sources for playlist entries and disable parents.

    Returns (updated_manifest, newly_added_child_sources).
    """
    updated = json.loads(json.dumps(manifest))
    existing_ids = {s["source_id"] for s in updated["sources"]}
    existing_urls = {
        s.get("original_url") for s in updated["sources"] if s.get("original_url")
    }
    added: list[dict[str, Any]] = []

    for source in list(updated["sources"]):
        if source.get("resource_type") != "playlist":
            continue
        if playlist_ids is not None and source["source_id"] not in playlist_ids:
            continue

        logger.info("Expanding playlist %s", source["source_id"])
        playlist = fetch_entries(source["original_url"])
        write_playlist_index(source, playlist)

        for position, entry in enumerate(playlist.get("entries") or [], start=1):
            if not entry:
                continue
            child = build_playlist_child_source(source, position=position, entry=entry)
            if child["source_id"] in existing_ids:
                logger.info("Skipping existing child %s", child["source_id"])
                continue
            if child["original_url"] in existing_urls:
                logger.info(
                    "Skipping duplicate URL for playlist position %d: %s",
                    position,
                    child["original_url"],
                )
                continue
            updated["sources"].append(child)
            existing_ids.add(child["source_id"])
            existing_urls.add(child["original_url"])
            added.append(child)

        # Disable thin index-only parent once children exist
        for src in updated["sources"]:
            if src["source_id"] == source["source_id"]:
                src["enabled"] = False
                notes = src.get("notes") or ""
                disable_note = "Disabled after --expand-playlists (children hold transcripts)."
                if disable_note not in notes:
                    src["notes"] = f"{notes} {disable_note}".strip()
                break

    return updated, added


def acquire_video_transcript(
    source: dict[str, Any],
    *,
    whisper_model,
    force: bool,
    skip_download: bool,
) -> None:
    """Download (unless skipped) and transcribe one video_transcript source."""
    source_id = source["source_id"]
    local_path = PROJECT_ROOT / source["local_path"]
    if local_path.exists() and not force:
        logger.info("Skipping %s (transcript exists)", source_id)
        return

    if skip_download:
        candidates = _completed_audio_files(source_id)
        if not candidates:
            raise FileNotFoundError(f"no downloaded audio for {source_id}")
        audio_path = candidates[0]
    else:
        audio_path = download_audio(source["original_url"], source_id)

    logger.info("Transcribing %s (%s)", source_id, audio_path.name)
    text = transcribe_to_text(audio_path, model=whisper_model, title=source["title"])

    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(text, encoding="utf-8")
    logger.info("Wrote transcript: %s (%d words)", local_path, len(text.split()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--model",
        default="base.en",
        help="faster-whisper model name (tiny.en, base.en, small.en, ...)",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Restrict to these source_ids (videos and/or playlists)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-create transcripts/indexes even if local_path already exists",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only transcribe audio already present in the audio directory",
    )
    parser.add_argument(
        "--expand-playlists",
        action="store_true",
        help=(
            "Expand playlist sources into per-video transcript entries, "
            "update the manifest, disable parent indexes, then transcribe"
        ),
    )
    parser.add_argument(
        "--expand-only",
        action="store_true",
        help="With --expand-playlists: only update the manifest, do not download/transcribe",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    failures: list[str] = []

    if args.expand_playlists:
        manifest = load_manifest(args.manifest)
        playlist_ids = set(args.only) if args.only else None
        # If --only lists video ids only, still expand all playlists when flag set
        # unless any of the --only ids are playlist sources.
        if playlist_ids is not None:
            playlist_source_ids = {
                s["source_id"]
                for s in manifest["sources"]
                if s.get("resource_type") == "playlist"
            }
            overlap = playlist_ids & playlist_source_ids
            playlist_ids = overlap if overlap else None

        try:
            updated, added = expand_playlist_sources(
                manifest, playlist_ids=playlist_ids
            )
            args.manifest.write_text(
                json.dumps(updated, indent=2) + "\n", encoding="utf-8"
            )
            logger.info(
                "Expanded playlists: added %d child sources to %s",
                len(added),
                args.manifest,
            )
        except Exception:
            logger.exception("Playlist expansion failed")
            return 1

        if args.expand_only:
            return 0

    videos, playlists = load_manifest_sources(args.manifest)
    if args.only:
        only = set(args.only)
        videos = [s for s in videos if s["source_id"] in only]
        playlists = [s for s in playlists if s["source_id"] in only]

    # Skip index writes for disabled playlists (already expanded)
    for source in playlists:
        if source.get("enabled") is False:
            logger.info("Skipping disabled playlist index %s", source["source_id"])
            continue
        local_path = PROJECT_ROOT / source["local_path"]
        if local_path.exists() and not args.force:
            logger.info("Skipping %s (index exists)", source["source_id"])
            continue
        try:
            write_playlist_index(source)
        except Exception:
            logger.exception("Playlist index failed for %s", source["source_id"])
            failures.append(source["source_id"])

    whisper_model = None
    for source in videos:
        if source.get("enabled") is False:
            continue
        source_id = source["source_id"]
        try:
            if whisper_model is None:
                from ingestion.audio_video_ingestor import load_whisper_model

                whisper_model = load_whisper_model(model_name=args.model)
            acquire_video_transcript(
                source,
                whisper_model=whisper_model,
                force=args.force,
                skip_download=args.skip_download,
            )
        except Exception:
            logger.exception("Transcript failed for %s", source_id)
            failures.append(source_id)

    if failures:
        logger.error("Failed sources: %s", ", ".join(failures))
        return 1
    logger.info("All requested sources processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
