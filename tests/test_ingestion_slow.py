from __future__ import annotations

import os
import shutil
import warnings

import pytest

from ingestion.dispatch import ingest

from tests.ingestion_test_utils import (
    RAW_DATA,
    assert_document_basics,
    assert_roundtrip,
    require_file,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_SLOW_INGESTION") != "1",
    reason="set RUN_SLOW_INGESTION=1 to run sample-file ingestion tests",
)


@pytest.mark.slow
@pytest.mark.optional_dependency
def test_pdf_sample_files() -> None:
    pytest.importorskip("pymupdf4llm")

    clean_pdf = require_file(RAW_DATA / "pdf" / "EDGE SLM PROJECT.pdf")
    sample_pdf = require_file(RAW_DATA / "pdf" / "Sample.pdf")
    slide_pdf = require_file(RAW_DATA / "pdf" / "STAT3401_Lecture_11_Seasonality.pdf")
    ocr_pdf = RAW_DATA / "pdf" / "OCR.pdf"

    clean_doc = ingest(clean_pdf)
    assert_document_basics(clean_doc, content_type="pdf_text", modality="document")
    assert clean_doc.ingestor == "pdf_ingestor"
    assert_roundtrip(clean_doc, "pdf")

    sample_doc = ingest(sample_pdf)
    assert_document_basics(sample_doc, content_type="pdf_text", modality="document")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        slide_doc = ingest(slide_pdf)

    assert_document_basics(slide_doc, content_type="pdf_text", modality="document")
    assert caught, "expected empty-page warning for slide deck fixture"

    if ocr_pdf.exists():
        from ingestion.pdf_ingestor import find_tessdata

        if find_tessdata() is None:
            with pytest.raises(RuntimeError, match="No extractable text"):
                ingest(ocr_pdf)
        else:
            ocr_doc = ingest(ocr_pdf)
            assert_document_basics(ocr_doc, content_type="pdf_text", modality="document")
            assert ocr_doc.format_metadata["pdf"]["pages_ocred"]


@pytest.mark.slow
@pytest.mark.optional_dependency
def test_pptx_sample_file() -> None:
    pytest.importorskip("pptx")

    pptx_path = require_file(RAW_DATA / "pptx" / "SamplePPT.pptx")
    document = ingest(pptx_path)

    assert_document_basics(document, content_type="slide_text", modality="slides")
    assert document.ingestor == "pptx_ingestor"
    assert_roundtrip(document, "pptx")


@pytest.mark.slow
@pytest.mark.media
@pytest.mark.optional_dependency
def test_audio_sample_files() -> None:
    pytest.importorskip("faster_whisper")

    audio_dir = RAW_DATA / "audio"
    audio_files = [
        path
        for path in sorted(audio_dir.glob("*"))
        if path.suffix.lower() in {".mp3", ".wav", ".m4a", ".flac"}
    ]
    assert audio_files, f"no audio files found in {audio_dir}"

    for audio_path in audio_files:
        document = ingest(audio_path)
        assert_document_basics(document, content_type="transcript", modality="audio")
        assert document.ingestor == "audio_video_ingestor"
        assert_roundtrip(document, audio_path.stem)


@pytest.mark.slow
@pytest.mark.media
@pytest.mark.optional_dependency
def test_video_sample_files() -> None:
    pytest.importorskip("faster_whisper")

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not found in PATH")

    video_dir = RAW_DATA / "video"
    video_files = [
        path
        for path in sorted(video_dir.glob("*"))
        if path.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    ]
    assert video_files, f"no video files found in {video_dir}"

    for video_path in video_files:
        document = ingest(video_path)
        assert_document_basics(document, content_type="transcript", modality="video")
        assert document.ingestor == "audio_video_ingestor"
        assert_roundtrip(document, video_path.stem)
