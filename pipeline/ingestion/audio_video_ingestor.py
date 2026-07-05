"""
Audio and video ingestion for the Edge SLM pipeline.

Turns media sources into section-based Documents. Local audio/video files are
transcribed with faster-whisper; YouTube sources use captions with optional
ASR fallback. Chunking is deferred to the source pack.
"""

from __future__ import annotations

import html
import logging
import re
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, TYPE_CHECKING
from urllib.parse import urlparse

from ingestion.schema import (
    SOURCE_TYPE_LOCAL_FILE,
    SOURCE_TYPE_YOUTUBE_URL,
    Document,
    Section,
    new_document_id,
)

if TYPE_CHECKING:
    from faster_whisper import WhisperModel


logger = logging.getLogger(__name__)


SUPPORTED_AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".aac",
    ".ogg",
    ".opus",
    ".webm",
}

SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
}

TIMESTAMP_LINE_PATTERN = re.compile(
    r"(?P<start>(?:\d{1,2}:)?\d{2}:\d{2}[\.,]\d{3})\s+-->\s+"
    r"(?P<end>(?:\d{1,2}:)?\d{2}:\d{2}[\.,]\d{3})"
)


def is_supported_audio_file(input_path: Path) -> bool:
    return input_path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS


def is_supported_video_file(input_path: Path) -> bool:
    return input_path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS


def is_youtube_url(source: str) -> bool:
    parsed = urlparse(source)
    if parsed.scheme not in {"http", "https"}:
        return False
    host = parsed.netloc.lower()
    return "youtube.com" in host or "youtu.be" in host


def ensure_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg was not found. Please install FFmpeg and make sure it is "
            "available in your terminal PATH."
        )


def extract_audio_from_video(
    video_path: str | Path,
    output_audio_path: str | Path,
) -> Path:
    ensure_ffmpeg_available()

    video_file = Path(video_path)
    output_file = Path(output_audio_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting audio from video: %s", video_file)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_file),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_file),
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg failed while extracting audio.\n"
            f"Command: {' '.join(command)}\n"
            f"Error:\n{result.stderr}"
        )

    return output_file


def normalize_word_for_overlap(word: str) -> str:
    return re.sub(r"^\W+|\W+$", "", word.lower())


def find_word_overlap(
    previous_text: str,
    current_text: str,
    max_overlap_words: int = 30,
) -> int:
    previous_normalized = [
        normalize_word_for_overlap(word) for word in previous_text.split()
    ]
    current_normalized = [
        normalize_word_for_overlap(word) for word in current_text.split()
    ]

    max_possible_overlap = min(
        len(previous_normalized),
        len(current_normalized),
        max_overlap_words,
    )

    for overlap_size in range(max_possible_overlap, 0, -1):
        if previous_normalized[-overlap_size:] == current_normalized[:overlap_size]:
            return overlap_size

    return 0


def append_without_repeating_overlap(previous_text: str, current_text: str) -> str:
    previous_text = previous_text.strip()
    current_text = current_text.strip()

    if not previous_text:
        return current_text
    if not current_text:
        return previous_text

    previous_normalized = re.sub(r"\s+", " ", previous_text.lower()).strip()
    current_normalized = re.sub(r"\s+", " ", current_text.lower()).strip()

    if current_normalized in previous_normalized:
        return previous_text

    overlap_size = find_word_overlap(
        previous_text=previous_text,
        current_text=current_text,
    )

    if overlap_size == 0:
        return f"{previous_text} {current_text}".strip()

    remaining_words = current_text.split()[overlap_size:]
    if not remaining_words:
        return previous_text

    return f"{previous_text} {' '.join(remaining_words)}".strip()


def merge_transcript_sections(
    raw_sections: list[Section],
    *,
    target_section_words: int = 100,
    max_section_duration_seconds: float = 60.0,
    max_gap_seconds: float = 8.0,
) -> list[Section]:
    """Merge small transcript sections into fewer larger sections."""
    merged: list[Section] = []
    current_text = ""
    current_start: float | None = None
    current_end: float | None = None

    def flush() -> None:
        nonlocal current_text, current_start, current_end
        if not current_text.strip():
            return
        merged.append(
            Section(
                index=len(merged),
                text=current_text.strip(),
                raw_text=current_text.strip(),
                start_time_s=current_start,
                end_time_s=current_end,
                extraction_method="transcript_merge",
            )
        )
        current_text = ""
        current_start = None
        current_end = None

    for section in raw_sections:
        text = section.text.strip()
        if not text:
            continue

        start_time = section.start_time_s
        end_time = section.end_time_s

        if current_start is None:
            current_text = text
            current_start = start_time
            current_end = end_time
            continue

        gap_seconds = 0.0
        if current_end is not None and start_time is not None:
            gap_seconds = start_time - current_end

        candidate_text = append_without_repeating_overlap(current_text, text)
        candidate_word_count = len(candidate_text.split())
        candidate_duration = 0.0
        if current_start is not None and end_time is not None:
            candidate_duration = end_time - current_start

        should_start_new = (
            gap_seconds > max_gap_seconds
            or candidate_word_count > target_section_words
            or candidate_duration > max_section_duration_seconds
        )

        if should_start_new:
            flush()
            current_text = text
            current_start = start_time
            current_end = end_time
        else:
            current_text = candidate_text
            current_end = end_time

    flush()
    return merged


def load_whisper_model(
    model_name: str = "tiny.en",
    device: str = "cpu",
    compute_type: str = "int8",
) -> WhisperModel:
    from faster_whisper import WhisperModel
    logger.info(
        "Loading faster-whisper model: model=%s, device=%s, compute_type=%s",
        model_name,
        device,
        compute_type,
    )
    return WhisperModel(model_name, device=device, compute_type=compute_type)


def _build_transcript_document(
    *,
    document_id: str,
    source_type: str,
    source_path: str,
    modality: str,
    title: str,
    language: str | None,
    method: str,
    sections: list[Section],
    format_metadata: dict[str, Any] | None = None,
) -> Document:
    if not sections:
        raise RuntimeError("No transcript sections were produced.")

    return Document(
        document_id=document_id,
        source_type=source_type,
        source_path=source_path,
        modality=modality,
        content_type="transcript",
        ingestor="audio_video_ingestor",
        method=method,
        sections=sections,
        title=title,
        language=language,
        format_metadata=format_metadata or {},
    )


def transcribe_audio_file(
    input_path: str | Path,
    *,
    model: WhisperModel | None = None,
    model_name: str = "tiny.en",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = "en",
    document_id_prefix: str = "audio",
    modality: str = "audio",
    title: str | None = None,
    source_type: str = SOURCE_TYPE_LOCAL_FILE,
    source_path_for_json: str | Path | None = None,
    ingestion_method: str = "faster-whisper",
    format_metadata: dict[str, Any] | None = None,
    merge_sections: bool = False,
    target_section_words: int = 100,
    max_section_duration_seconds: float = 60.0,
    max_gap_seconds: float = 8.0,
) -> Document:
    audio_path = Path(input_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if not audio_path.is_file():
        raise ValueError(f"Input path is not a file: {audio_path}")

    if not is_supported_audio_file(audio_path):
        raise ValueError(
            f"Unsupported audio file type: {audio_path.suffix}. "
            f"Supported: {sorted(SUPPORTED_AUDIO_EXTENSIONS)}"
        )

    document_id = new_document_id(document_id_prefix)

    if model is None:
        model = load_whisper_model(
            model_name=model_name,
            device=device,
            compute_type=compute_type,
        )

    logger.info("Starting transcription for: %s", audio_path)

    segments, info = model.transcribe(str(audio_path), language=language, beam_size=5)
    detected_language = getattr(info, "language", language)
    duration_seconds = getattr(info, "duration", None)

    sections: list[Section] = []

    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue

        sections.append(
            Section(
                index=len(sections),
                text=text,
                raw_text=text,
                start_time_s=round(float(segment.start), 2),
                end_time_s=round(float(segment.end), 2),
                extraction_method="faster-whisper",
            )
        )

    if merge_sections:
        sections = merge_transcript_sections(
            sections,
            target_section_words=target_section_words,
            max_section_duration_seconds=max_section_duration_seconds,
            max_gap_seconds=max_gap_seconds,
        )

    if duration_seconds is None and sections:
        duration_seconds = sections[-1].end_time_s

    source_path = (
        str(source_path_for_json)
        if source_path_for_json is not None
        else str(audio_path)
    )

    metadata = dict(format_metadata or {})
    metadata.setdefault("transcript", {})
    metadata["transcript"].update(
        {
            "model": model_name,
            "duration_seconds": duration_seconds,
            "section_count": len(sections),
            "sections_merged": merge_sections,
        }
    )

    return _build_transcript_document(
        document_id=document_id,
        source_type=source_type,
        source_path=source_path,
        modality=modality,
        title=title or audio_path.stem,
        language=detected_language,
        method=ingestion_method,
        sections=sections,
        format_metadata=metadata,
    )


def transcribe_video_file(
    input_path: str | Path,
    *,
    model: WhisperModel | None = None,
    model_name: str = "tiny.en",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = "en",
) -> Document:
    video_path = Path(input_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not video_path.is_file():
        raise ValueError(f"Input path is not a file: {video_path}")

    if not is_supported_video_file(video_path):
        raise ValueError(
            f"Unsupported video file type: {video_path.suffix}. "
            f"Supported: {sorted(SUPPORTED_VIDEO_EXTENSIONS)}"
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        extracted_audio_path = Path(temp_dir) / f"{video_path.stem}_audio.wav"
        extract_audio_from_video(
            video_path=video_path,
            output_audio_path=extracted_audio_path,
        )

        return transcribe_audio_file(
            extracted_audio_path,
            model=model,
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            language=language,
            document_id_prefix="video",
            modality="video",
            title=video_path.stem,
            source_type=SOURCE_TYPE_LOCAL_FILE,
            source_path_for_json=video_path,
            ingestion_method="ffmpeg_extract_audio_then_faster-whisper",
        )


def fetch_youtube_info(youtube_url: str) -> dict[str, Any]:
    import yt_dlp

    ydl_options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_options) as ydl:
        return ydl.extract_info(youtube_url, download=False)


def find_caption_language_key(
    captions: dict[str, Any],
    preferred_language: str,
) -> str | None:
    if not captions:
        return None

    if preferred_language in captions:
        return preferred_language

    preferred_lower = preferred_language.lower()

    for language_key in captions:
        if language_key.lower() == preferred_lower:
            return language_key

    for language_key in captions:
        if language_key.lower().startswith(preferred_lower + "-"):
            return language_key

    return None


def choose_vtt_caption_track(
    info: dict[str, Any],
    preferred_language: str = "en",
    prefer_manual_captions: bool = True,
) -> tuple[dict[str, Any], str, str]:
    manual_captions = info.get("subtitles") or {}
    automatic_captions = info.get("automatic_captions") or {}

    if prefer_manual_captions:
        caption_sources = [("manual", manual_captions), ("automatic", automatic_captions)]
    else:
        caption_sources = [("automatic", automatic_captions), ("manual", manual_captions)]

    for source_name, captions in caption_sources:
        language_key = find_caption_language_key(captions, preferred_language)
        if language_key is None:
            continue

        for track in captions.get(language_key, []):
            if track.get("ext") == "vtt" and track.get("url"):
                return track, language_key, source_name

    raise ValueError(f"No VTT captions found for language '{preferred_language}'.")


def download_text_from_url(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_vtt_timestamp(timestamp: str) -> float:
    normalized = timestamp.replace(",", ".")
    parts = normalized.split(":")

    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])

    raise ValueError(f"Invalid VTT timestamp: {timestamp}")


def clean_vtt_text(text: str) -> str:
    text = re.sub(r"<(?:\d{1,2}:)?\d{2}:\d{2}[\.,]\d{3}>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def parse_webvtt_to_sections(
    vtt_text: str,
    *,
    merge_small_sections: bool = True,
    target_section_words: int = 100,
    max_section_duration_seconds: float = 60.0,
    max_gap_seconds: float = 8.0,
) -> list[Section]:
    raw_sections: list[Section] = []

    for block in re.split(r"\n\s*\n", vtt_text.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        first_line = lines[0].upper()
        if first_line.startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue

        timestamp_line_index = None
        timestamp_match = None

        for index, line in enumerate(lines):
            match = TIMESTAMP_LINE_PATTERN.search(line)
            if match:
                timestamp_line_index = index
                timestamp_match = match
                break

        if timestamp_line_index is None or timestamp_match is None:
            continue

        start_seconds = parse_vtt_timestamp(timestamp_match.group("start"))
        end_seconds = parse_vtt_timestamp(timestamp_match.group("end"))
        caption_text = clean_vtt_text(" ".join(lines[timestamp_line_index + 1 :]))

        if not caption_text:
            continue

        raw_sections.append(
            Section(
                index=len(raw_sections),
                text=caption_text,
                raw_text=caption_text,
                start_time_s=round(start_seconds, 2),
                end_time_s=round(end_seconds, 2),
                extraction_method="youtube_vtt",
            )
        )

    if not merge_small_sections:
        return raw_sections

    return merge_transcript_sections(
        raw_sections,
        target_section_words=target_section_words,
        max_section_duration_seconds=max_section_duration_seconds,
        max_gap_seconds=max_gap_seconds,
    )


def ingest_youtube_captions(
    youtube_url: str,
    preferred_language: str = "en",
    prefer_manual_captions: bool = True,
    merge_caption_sections_enabled: bool = True,
    target_section_words: int = 100,
    max_section_duration_seconds: float = 60.0,
    max_gap_seconds: float = 8.0,
) -> Document:
    info = fetch_youtube_info(youtube_url)
    document_id = new_document_id("youtube")

    caption_track, selected_language, caption_source = choose_vtt_caption_track(
        info=info,
        preferred_language=preferred_language,
        prefer_manual_captions=prefer_manual_captions,
    )

    sections = parse_webvtt_to_sections(
        download_text_from_url(caption_track["url"]),
        merge_small_sections=merge_caption_sections_enabled,
        target_section_words=target_section_words,
        max_section_duration_seconds=max_section_duration_seconds,
        max_gap_seconds=max_gap_seconds,
    )

    video_title = info.get("title") or "YouTube video"

    return _build_transcript_document(
        document_id=document_id,
        source_type=SOURCE_TYPE_YOUTUBE_URL,
        source_path=youtube_url,
        modality="video",
        title=video_title,
        language=selected_language,
        method=f"youtube_{caption_source}_captions",
        sections=sections,
        format_metadata={
            "youtube": {
                "video_id": info.get("id"),
                "title": video_title,
                "webpage_url": info.get("webpage_url") or youtube_url,
                "channel": info.get("channel"),
                "uploader": info.get("uploader"),
                "duration_seconds": info.get("duration"),
                "caption_language": selected_language,
                "caption_source": caption_source,
                "caption_format": caption_track.get("ext"),
                "asr_fallback_used": False,
                "caption_sections_merged": merge_caption_sections_enabled,
            }
        },
    )


def download_youtube_audio(
    youtube_url: str,
    output_dir: str | Path,
) -> tuple[Path, dict[str, Any]]:
    import yt_dlp

    ensure_ffmpeg_available()

    output_folder = Path(output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_template = output_folder / "youtube_audio.%(ext)s"

    ydl_options = {
        "format": "bestaudio/best",
        "outtmpl": str(output_template),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_options) as ydl:
        info = ydl.extract_info(youtube_url, download=True)

    for downloaded_item in info.get("requested_downloads") or []:
        possible_path = (
            downloaded_item.get("filepath")
            or downloaded_item.get("_filename")
            or downloaded_item.get("filename")
        )
        if possible_path:
            candidate = Path(possible_path)
            if candidate.exists() and candidate.is_file():
                return candidate, info

    for extension in [".webm", ".opus", ".m4a", ".mp3", ".wav", ".aac", ".ogg"]:
        candidate = output_folder / f"youtube_audio{extension}"
        if candidate.exists() and candidate.is_file():
            return candidate, info

    raise RuntimeError("yt-dlp finished, but no expected audio file was found.")


def ingest_youtube_with_asr(
    youtube_url: str,
    *,
    model: WhisperModel | None = None,
    model_name: str = "tiny.en",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = "en",
    merge_sections: bool = True,
    target_section_words: int = 100,
    max_section_duration_seconds: float = 60.0,
    max_gap_seconds: float = 8.0,
) -> Document:
    with tempfile.TemporaryDirectory() as temp_dir:
        downloaded_audio_path, info = download_youtube_audio(
            youtube_url=youtube_url,
            output_dir=temp_dir,
        )

        video_title = info.get("title") or "YouTube video"

        return transcribe_audio_file(
            downloaded_audio_path,
            model=model,
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            language=language,
            document_id_prefix="youtube",
            modality="video",
            title=video_title,
            source_type=SOURCE_TYPE_YOUTUBE_URL,
            source_path_for_json=youtube_url,
            ingestion_method="youtube_audio_faster-whisper",
            format_metadata={
                "youtube": {
                    "video_id": info.get("id"),
                    "title": video_title,
                    "webpage_url": info.get("webpage_url") or youtube_url,
                    "channel": info.get("channel"),
                    "uploader": info.get("uploader"),
                    "duration_seconds": info.get("duration"),
                    "caption_source": None,
                    "asr_fallback_used": True,
                    "asr_sections_merged": merge_sections,
                }
            },
            merge_sections=merge_sections,
            target_section_words=target_section_words,
            max_section_duration_seconds=max_section_duration_seconds,
            max_gap_seconds=max_gap_seconds,
        )


def ingest_youtube_url(
    youtube_url: str,
    *,
    preferred_language: str = "en",
    prefer_manual_captions: bool = True,
    use_captions: bool = True,
    allow_asr_fallback: bool = True,
    model: WhisperModel | None = None,
    model_name: str = "tiny.en",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = "en",
    merge_caption_sections_enabled: bool = True,
    target_section_words: int = 100,
    max_section_duration_seconds: float = 60.0,
    max_gap_seconds: float = 8.0,
) -> Document:
    if use_captions:
        try:
            return ingest_youtube_captions(
                youtube_url=youtube_url,
                preferred_language=preferred_language,
                prefer_manual_captions=prefer_manual_captions,
                merge_caption_sections_enabled=merge_caption_sections_enabled,
                target_section_words=target_section_words,
                max_section_duration_seconds=max_section_duration_seconds,
                max_gap_seconds=max_gap_seconds,
            )
        except Exception as error:
            if not allow_asr_fallback:
                raise
            logger.warning(
                "YouTube captions failed. Falling back to local ASR. Error: %s",
                error,
            )

    return ingest_youtube_with_asr(
        youtube_url=youtube_url,
        model=model,
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        language=language,
        merge_sections=merge_caption_sections_enabled,
        target_section_words=target_section_words,
        max_section_duration_seconds=max_section_duration_seconds,
        max_gap_seconds=max_gap_seconds,
    )


def ingest_media(
    input_path: str | Path,
    *,
    model: WhisperModel | None = None,
    model_name: str = "tiny.en",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = "en",
) -> Document:
    """Ingest a local audio or video file."""
    media_path = Path(input_path)

    if is_supported_audio_file(media_path):
        return transcribe_audio_file(
            media_path,
            model=model,
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            language=language,
        )

    if is_supported_video_file(media_path):
        return transcribe_video_file(
            media_path,
            model=model,
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            language=language,
        )

    raise ValueError(
        f"Unsupported media file type: {media_path.suffix.lower()}. "
        f"Supported audio: {sorted(SUPPORTED_AUDIO_EXTENSIONS)}. "
        f"Supported video: {sorted(SUPPORTED_VIDEO_EXTENSIONS)}."
    )


def ingest_media_source(
    input_source: str | Path,
    *,
    model: WhisperModel | None = None,
    model_name: str = "tiny.en",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = "en",
    preferred_youtube_caption_language: str = "en",
    prefer_manual_youtube_captions: bool = True,
    use_youtube_captions: bool = True,
    allow_youtube_asr_fallback: bool = True,
    merge_youtube_caption_sections: bool = True,
    youtube_target_section_words: int = 100,
    youtube_max_section_duration_seconds: float = 60.0,
    youtube_max_gap_seconds: float = 8.0,
) -> Document:
    """Ingest either a local media file or a YouTube URL."""
    source_text = str(input_source)

    if is_youtube_url(source_text):
        return ingest_youtube_url(
            youtube_url=source_text,
            preferred_language=preferred_youtube_caption_language,
            prefer_manual_captions=prefer_manual_youtube_captions,
            use_captions=use_youtube_captions,
            allow_asr_fallback=allow_youtube_asr_fallback,
            model=model,
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            language=language,
            merge_caption_sections_enabled=merge_youtube_caption_sections,
            target_section_words=youtube_target_section_words,
            max_section_duration_seconds=youtube_max_section_duration_seconds,
            max_gap_seconds=youtube_max_gap_seconds,
        )

    return ingest_media(
        input_path=input_source,
        model=model,
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        language=language,
    )
