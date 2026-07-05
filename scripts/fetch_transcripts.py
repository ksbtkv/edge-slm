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

Usage:
    python scripts/fetch_transcripts.py [--manifest PATH] [--model base.en]
                                        [--only SOURCE_ID ...] [--force]
                                        [--skip-download]
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "pipeline"))

logger = logging.getLogger("fetch_transcripts")

DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "databricks_ld_foundations.json"
AUDIO_DIR = PROJECT_ROOT / "data" / "raw" / "databricks" / "videos" / "audio"


def load_manifest_sources(manifest_path: Path) -> tuple[list[dict], list[dict]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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


def write_playlist_index(source: dict) -> None:
    """Write a Markdown index of playlist entries to the source's local_path."""
    local_path = PROJECT_ROOT / source["local_path"]
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--flat-playlist",
        "-J",
        source["original_url"],
    ]
    logger.info("Fetching playlist index for %s", source["source_id"])
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    playlist = json.loads(result.stdout)

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
        help="Restrict to these source_ids",
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    videos, playlists = load_manifest_sources(args.manifest)
    if args.only:
        videos = [s for s in videos if s["source_id"] in args.only]
        playlists = [s for s in playlists if s["source_id"] in args.only]

    failures: list[str] = []

    for source in playlists:
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
        source_id = source["source_id"]
        local_path = PROJECT_ROOT / source["local_path"]
        if local_path.exists() and not args.force:
            logger.info("Skipping %s (transcript exists)", source_id)
            continue

        try:
            if args.skip_download:
                candidates = _completed_audio_files(source_id)
                if not candidates:
                    raise FileNotFoundError(f"no downloaded audio for {source_id}")
                audio_path = candidates[0]
            else:
                audio_path = download_audio(source["original_url"], source_id)

            if whisper_model is None:
                from ingestion.audio_video_ingestor import load_whisper_model

                whisper_model = load_whisper_model(model_name=args.model)

            logger.info("Transcribing %s (%s)", source_id, audio_path.name)
            text = transcribe_to_text(audio_path, model=whisper_model, title=source["title"])

            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text(text, encoding="utf-8")
            logger.info(
                "Wrote transcript: %s (%d words)", local_path, len(text.split())
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
