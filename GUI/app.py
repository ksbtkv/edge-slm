"""
Edge SLM Pipeline Runner — Streamlit GUI

Main GUI for the Edge SLM ingestion pipeline.

Supports:
- PDF
- PPTX
- plain text
- Markdown
- local audio/video
- YouTube URLs

Run from the project root:

    streamlit run GUI/app.py
"""

from __future__ import annotations

import html
import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import streamlit as st

# need this to be the first Streamlit command, otherwise Streamlit complains.
st.set_page_config(
    page_title="Edge SLM Pipeline Runner",
    page_icon="▶",
    layout="wide",
)

# find the project root from this file location so the app works even if my
# terminal is inside the main project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = PROJECT_ROOT / "pipeline"

# add pipeline/ to Python's import path so I can import my teammate's
# ingestion package without needing PYTHONPATH every single time.
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from ingestion.dispatch import ingest, supported_extensions  # noqa: E402
from ingestion.serialize import save_document  # noqa: E402


UPLOAD_DIR = PROJECT_ROOT / "data" / "gui_uploads"
OUTPUT_DIR = PROJECT_ROOT / "data" / "gui_outputs"

MODELS = [
    "Llama 3.2 3B",
    "Qwen2.5 3B",
    "Phi-3.5 Mini 3.8B",
    "Gemma 3 4B",
]

LORA_RANKS = [
    "8 — faster, smaller adapter",
    "16 — balanced",
    "32 — higher capacity",
    "64 — maximum capacity",
]

QUANTIZATION_TARGETS = [
    "Q4_K_M — recommended",
    "Q2_K — minimum size",
    "Q5_K_M — better quality",
    "Q8_0 — maximum quality",
]

TURNAROUND_OPTIONS = [
    "Standard (24–48 hrs)",
    "Priority queue",
]

ALL_EXTENSIONS = [
    "pdf", "pptx",
    "txt", "text", "md", "markdown",
    "mp3", "wav", "m4a", "flac", "aac", "ogg", "opus", "webm",
    "mp4", "mov", "mkv", "avi",
]


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #F8F9FB;
}

header[data-testid="stHeader"] {
    background: #FFFFFF;
    border-bottom: 1px solid #E8EAED;
}

[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E8EAED;
}

h1 {
    font-weight: 700 !important;
    color: #0F172A !important;
    letter-spacing: -0.5px !important;
}

h2 {
    font-weight: 600 !important;
    color: #0F172A !important;
    font-size: 1.1rem !important;
    margin-top: 1.5rem !important;
    letter-spacing: -0.2px !important;
}

h3 {
    font-weight: 600 !important;
    color: #0F172A !important;
    font-size: 1rem !important;
}

.section-eyebrow {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #5B4FE9;
    margin-bottom: 4px;
    margin-top: 8px;
}

.metric-card {
    background: #FFFFFF;
    border: 1px solid #E8EAED;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}

.metric-card .metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #0F172A;
    line-height: 1.1;
}

.metric-card .metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 4px;
}

.stage-card {
    background: #FFFFFF;
    border: 1px solid #E8EAED;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 10px;
    display: flex;
    align-items: flex-start;
    gap: 14px;
}

.stage-card.done { border-left: 4px solid #16A34A; }
.stage-card.running { border-left: 4px solid #D97706; }
.stage-card.error { border-left: 4px solid #DC2626; }
.stage-card.waiting { border-left: 4px solid #E8EAED; }

.stage-number {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    flex-shrink: 0;
    margin-top: 2px;
}

.stage-number.done { background: #DCFCE7; color: #16A34A; }
.stage-number.running { background: #FEF3C7; color: #D97706; }
.stage-number.error { background: #FEE2E2; color: #DC2626; }
.stage-number.waiting { background: #F1F5F9; color: #94A3B8; }

.stage-content { flex: 1; }

.stage-title {
    font-size: 0.88rem;
    font-weight: 600;
    color: #0F172A;
    margin: 0 0 2px 0;
}

.stage-description {
    font-size: 0.77rem;
    color: #64748B;
    margin: 0 0 4px 0;
}

.stage-detail {
    font-size: 0.74rem;
    font-weight: 500;
    margin: 0;
}

.stage-detail.done { color: #16A34A; }
.stage-detail.running { color: #D97706; }
.stage-detail.error { color: #DC2626; }
.stage-detail.waiting { color: #94A3B8; }

.status-badge {
    font-size: 0.68rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 999px;
    display: inline-block;
    letter-spacing: 0.3px;
    vertical-align: middle;
}

.status-badge.done { background: #DCFCE7; color: #16A34A; }
.status-badge.running { background: #FEF3C7; color: #D97706; }
.status-badge.error { background: #FEE2E2; color: #DC2626; }
.status-badge.waiting { background: #F1F5F9; color: #94A3B8; }

.info-card {
    background: #FFFFFF;
    border: 1px solid #E8EAED;
    border-radius: 12px;
    padding: 4px 20px;
    margin: 12px 0;
}

.info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #F1F5F9;
    font-size: 0.82rem;
    gap: 12px;
}

.info-row:last-child { border-bottom: none; }
.info-label { color: #64748B; font-weight: 500; flex-shrink: 0; }
.info-value { color: #0F172A; font-weight: 600; text-align: right; word-break: break-all; }

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #5B4FE9 0%, #7C6FF7 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.2rem !important;
    letter-spacing: 0.2px !important;
}

.stButton > button[kind="primary"]:hover { opacity: 0.9 !important; }
.stButton > button[kind="primary"]:disabled { opacity: 0.4 !important; }

.stSelectbox > div > div {
    border-radius: 8px !important;
    background: #FFFFFF !important;
}

.stTextInput > div > div > input {
    border-radius: 8px !important;
    background: #FFFFFF !important;
}

hr { border-color: #E8EAED !important; }

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E8EAED;
    border-radius: 10px;
    padding: 10px 14px !important;
}

[data-testid="stMetricValue"] {
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    color: #0F172A !important;
}

[data-testid="stMetricLabel"] {
    font-size: 0.7rem !important;
    color: #64748B !important;
    font-weight: 500 !important;
}

.brand-strip {
    background: linear-gradient(135deg, #5B4FE9 0%, #7C6FF7 100%);
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 16px;
}

.brand-strip .brand-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: white;
    margin: 0;
}

.brand-strip .brand-sub {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.75);
    margin: 2px 0 0 0;
}

.last-run-card {
    background: #F8F9FB;
    border: 1px solid #E8EAED;
    border-radius: 10px;
    padding: 12px 14px;
    margin-top: 8px;
}

.last-run-card .run-status {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 4px;
}

.last-run-card .run-name {
    font-size: 0.82rem;
    font-weight: 500;
    color: #0F172A;
    word-break: break-all;
}

.preview-card {
    background: #F8F9FB;
    border: 1px solid #E8EAED;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 12px;
}

.preview-header {
    font-size: 0.82rem;
    font-weight: 600;
    color: #0F172A;
    margin-bottom: 4px;
}

.preview-text {
    font-size: 0.82rem;
    color: #334155;
    line-height: 1.6;
    white-space: pre-wrap;
}
</style>
"""


def initialise_session_state() -> None:
    # keep all run outputs in session state so Streamlit can re-render the
    # page without losing the last successful ingestion result.
    defaults = {
        "document": None,
        "output_path": None,
        "config_path": None,
        "data_ingestion_done": False,
        "ingestion_error": None,
        "last_filename": None,
        "downloaded_youtube_path": None,
        "original_youtube_url": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_pipeline_state() -> None:
    # clear the old result before starting a new run so the GUI does not show
    # stale files or stale errors from a previous attempt.
    st.session_state["document"] = None
    st.session_state["output_path"] = None
    st.session_state["config_path"] = None
    st.session_state["data_ingestion_done"] = False
    st.session_state["ingestion_error"] = None
    st.session_state["last_filename"] = None
    st.session_state["downloaded_youtube_path"] = None
    st.session_state["original_youtube_url"] = None


def save_uploaded_file(uploaded_file) -> Path:
    # Streamlit keeps uploaded files in memory, so I save the file to disk first.
    # The ingestion dispatcher expects a normal local file path.
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    output_path = UPLOAD_DIR / uploaded_file.name

    with output_path.open("wb") as file:
        file.write(uploaded_file.getbuffer())

    return output_path


def build_output_path(stem: str) -> Path:
    # clean the filename so the output JSON path is safe and predictable.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = stem.replace(" ", "_").lower()
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", safe_name).strip("_")

    if not safe_name:
        safe_name = "source"

    return OUTPUT_DIR / f"{safe_name}.ingested.json"


def utc_now_iso() -> str:
    # I store timestamps in UTC so every run config has a consistent time format.
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def parse_lora_rank(lora_rank_label: str) -> int:
    # The UI label is friendly, but the downstream training script needs a number.
    return int(lora_rank_label.split("—")[0].strip())


def parse_quantization_target(quantization_label: str) -> str:
    # The UI label is friendly, but quantization scripts usually need just Q4_K_M etc.
    return quantization_label.split("—")[0].strip()


def save_run_config(
    *,
    document,
    output_path: Path,
    input_source: str,
    base_model: str,
    lora_rank: str,
    quantization_target: str,
    hardware_target: str,
    turnaround: str,
    notification_email: str,
) -> Path:
    """
    I save this run config as the handoff file for the next stages.

    The ingestion JSON contains the extracted document.
    This config JSON tells teammates which model, LoRA rank, quantization target,
    and input file belong to this run.
    """
    run_id = f"run_{uuid.uuid4().hex[:8]}"

    config = {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "input_source": input_source,
        "ingested_document": str(output_path),
        "document_id": document.document_id,
        "base_model": base_model,
        "lora_rank": parse_lora_rank(lora_rank),
        "quantization_target": parse_quantization_target(quantization_target),
        "hardware_target": hardware_target,
        "turnaround": turnaround,
        "notification_email": notification_email,
        "pipeline_status": {
            "ingestion": "done",
            "instruction_pairs": "pending",
            "finetuning": "pending",
            "quantization": "pending",
            "evaluation": "pending",
        },
    }

    config_path = OUTPUT_DIR / f"{run_id}_config.json"

    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)

    return config_path


def looks_like_youtube_url(value: str) -> bool:
    value = value.strip().lower()

    return (
        value.startswith("https://www.youtube.com/")
        or value.startswith("https://youtube.com/")
        or value.startswith("https://youtu.be/")
        or value.startswith("http://www.youtube.com/")
        or value.startswith("http://youtube.com/")
        or value.startswith("http://youtu.be/")
    )


def get_youtube_video_id(youtube_url: str) -> str:
    # I use the YouTube video ID to make a clean downloaded filename.
    parsed_url = urlparse(youtube_url)

    if "youtu.be" in parsed_url.netloc:
        video_id = parsed_url.path.strip("/")
    else:
        query_values = parse_qs(parsed_url.query)
        video_id = query_values.get("v", ["youtube_source"])[0]

    if not video_id:
        video_id = "youtube_source"

    return re.sub(r"[^a-zA-Z0-9_-]", "_", video_id)


def download_youtube_audio(youtube_url: str) -> Path:
    # dispatch.ingest() works with local files, not raw YouTube URLs.
    # download the YouTube audio first, then send the local audio file to ingest().
    from yt_dlp import YoutubeDL

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    video_id = get_youtube_video_id(youtube_url)
    output_template = str(UPLOAD_DIR / f"youtube_{video_id}.%(ext)s")

    ydl_options = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": output_template,
        "quiet": True,
        "noplaylist": True,
        "restrictfilenames": True,
    }

    with YoutubeDL(ydl_options) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        downloaded_path = Path(ydl.prepare_filename(info))

    if downloaded_path.exists():
        return downloaded_path

    possible_files = sorted(UPLOAD_DIR.glob(f"youtube_{video_id}.*"))

    if possible_files:
        return possible_files[0]

    raise FileNotFoundError(
        f"YouTube audio download completed, but no file found for {video_id}."
    )


def run_ingestion(
    uploaded_file,
    *,
    base_model: str,
    lora_rank: str,
    quantization_target: str,
    hardware_target: str,
    turnaround: str,
    notification_email: str,
) -> None:
    # This is the normal local-file path:
    # upload -> save to disk -> ingest -> save document JSON -> save run config.
    reset_pipeline_state()

    try:
        with st.spinner(f"Ingesting {uploaded_file.name}..."):
            input_path = save_uploaded_file(uploaded_file)
            document = ingest(input_path)

            output_path = build_output_path(input_path.stem)
            save_document(document, output_path)

            config_path = save_run_config(
                document=document,
                output_path=output_path,
                input_source=str(input_path),
                base_model=base_model,
                lora_rank=lora_rank,
                quantization_target=quantization_target,
                hardware_target=hardware_target,
                turnaround=turnaround,
                notification_email=notification_email,
            )

        st.session_state["document"] = document
        st.session_state["output_path"] = output_path
        st.session_state["config_path"] = config_path
        st.session_state["data_ingestion_done"] = True
        st.session_state["ingestion_error"] = None
        st.session_state["last_filename"] = uploaded_file.name

    except Exception as error:
        st.session_state["ingestion_error"] = error
        st.session_state["data_ingestion_done"] = False


def run_ingestion_from_youtube(
    youtube_url: str,
    *,
    base_model: str,
    lora_rank: str,
    quantization_target: str,
    hardware_target: str,
    turnaround: str,
    notification_email: str,
) -> None:
    # This is the YouTube path:
    # YouTube URL -> download audio -> ingest audio file -> save document JSON -> save run config.
    reset_pipeline_state()

    try:
        with st.spinner("Downloading and transcribing YouTube audio..."):
            downloaded_audio_path = download_youtube_audio(youtube_url)
            document = ingest(downloaded_audio_path)

            # I add the original YouTube URL as metadata so we don't lose where it came from.
            if hasattr(document, "format_metadata") and isinstance(
                document.format_metadata, dict
            ):
                document.format_metadata["youtube_url"] = youtube_url
                document.format_metadata["downloaded_audio_path"] = str(
                    downloaded_audio_path
                )

            video_id = get_youtube_video_id(youtube_url)
            output_path = build_output_path(f"youtube_{video_id}")
            save_document(document, output_path)

            config_path = save_run_config(
                document=document,
                output_path=output_path,
                input_source=youtube_url,
                base_model=base_model,
                lora_rank=lora_rank,
                quantization_target=quantization_target,
                hardware_target=hardware_target,
                turnaround=turnaround,
                notification_email=notification_email,
            )

        st.session_state["document"] = document
        st.session_state["output_path"] = output_path
        st.session_state["config_path"] = config_path
        st.session_state["data_ingestion_done"] = True
        st.session_state["ingestion_error"] = None
        st.session_state["last_filename"] = document.title or youtube_url
        st.session_state["downloaded_youtube_path"] = downloaded_audio_path
        st.session_state["original_youtube_url"] = youtube_url

    except Exception as error:
        st.session_state["ingestion_error"] = error
        st.session_state["data_ingestion_done"] = False


def render_stage_card(
    number: int,
    title: str,
    description: str,
    status: str,
    detail: str = "",
) -> None:
    icon_map = {
        "done": "✓",
        "running": "⟳",
        "error": "✕",
        "waiting": str(number),
    }

    detail_html = (
        f'<p class="stage-detail {status}">{html.escape(detail)}</p>'
        if detail
        else ""
    )

    st.markdown(
        f"""
        <div class="stage-card {status}">
            <div class="stage-number {status}">{icon_map.get(status, number)}</div>
            <div class="stage-content">
                <p class="stage-title">
                    {html.escape(title)}
                    <span class="status-badge {status}" style="margin-left:8px;">
                        {status}
                    </span>
                </p>
                <p class="stage-description">{html.escape(description)}</p>
                {detail_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_stages() -> None:
    st.markdown(
        '<p class="section-eyebrow">Pipeline stages</p>',
        unsafe_allow_html=True,
    )

    ingestion_done = st.session_state["data_ingestion_done"]
    ingestion_error = st.session_state["ingestion_error"]
    document = st.session_state.get("document")

    if ingestion_error is not None:
        status, detail = "error", "see error below"
    elif ingestion_done:
        status = "done"
        detail = (
            f"{document.section_count} sections · {document.total_word_count:,} words"
            if document
            else "complete"
        )
    else:
        status, detail = "waiting", ""

    render_stage_card(
        1,
        "Data ingestion",
        "Convert source files to structured JSON",
        status,
        detail,
    )
    render_stage_card(
        2,
        "Instruction pairs",
        "Generate Q&A pairs via LLM synthesis",
        "waiting",
        "awaiting data ingestion",
    )
    render_stage_card(
        3,
        "Fine-tuning",
        "LoRA training on Pawsey A100",
        "waiting",
        "awaiting instruction pairs",
    )
    render_stage_card(
        4,
        "Quantization",
        "Convert checkpoint to GGUF format",
        "waiting",
        "awaiting fine-tuning",
    )
    render_stage_card(
        5,
        "Evaluation",
        "Benchmark against dataset",
        "waiting",
        "awaiting quantization",
    )


def get_section_location(section) -> str:
    parts = []

    if section.page_number is not None:
        parts.append(f"Page {section.page_number}")

    if section.slide_number is not None:
        parts.append(f"Slide {section.slide_number}")

    if section.start_time_s is not None and section.end_time_s is not None:
        parts.append(f"{section.start_time_s:.1f}s → {section.end_time_s:.1f}s")

    if section.heading:
        parts.append(f'"{section.heading}"')

    return " · ".join(parts) if parts else "No location metadata"


def render_document_summary() -> None:
    document = st.session_state.get("document")
    output_path = st.session_state.get("output_path")
    config_path = st.session_state.get("config_path")

    if document is None:
        return

    st.divider()
    st.markdown(
        '<p class="section-eyebrow">Ingestion result</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{document.section_count}</div>
                <div class="metric-label">Sections</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{document.total_word_count:,}</div>
                <div class="metric-label">Words</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{html.escape(str(document.schema_version))}</div>
                <div class="metric-label">Schema</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    original_url = st.session_state.get("original_youtube_url")
    source_display = original_url if original_url else (document.source_path or "—")

    def info_row(label: str, value) -> str:
        safe_label = html.escape(str(label))
        safe_value = html.escape(str(value))
        return (
            f'<div class="info-row">'
            f'<span class="info-label">{safe_label}</span>'
            f'<span class="info-value">{safe_value}</span>'
            f'</div>'
        )

    st.markdown(
        f"""
        <div class="info-card">
            {info_row("Title", document.title or "Not available")}
            {info_row("Document ID", document.document_id)}
            {info_row("Source", source_display)}
            {info_row("Modality", document.modality)}
            {info_row("Content type", document.content_type)}
            {info_row("Method", document.method)}
            {info_row("Output file", output_path.name if output_path else "—")}
            {info_row("Run config", config_path.name if config_path else "—")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Preview first 5 sections"):
        if not document.sections:
            st.warning("No sections were extracted.")
            return

        for section in document.sections[:5]:
            preview = html.escape(section.text[:600])
            truncated = len(section.text) > 600

            st.markdown(
                f"""
                <div class="preview-card">
                    <div class="preview-header">
                        Section {section.index}
                        <span style="color:#64748B; font-weight:400; font-size:0.73rem; margin-left:8px;">
                            {section.word_count} words · {html.escape(get_section_location(section))}
                        </span>
                    </div>
                    <div class="preview-text">
                        {preview}{'… (truncated)' if truncated else ''}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_error() -> None:
    error = st.session_state.get("ingestion_error")

    if error is not None:
        st.error(f"Ingestion failed: {error}")

        with st.expander("Show full traceback"):
            st.exception(error)


def render_sidebar() -> str:
    st.sidebar.markdown(
        """
        <div class="brand-strip">
            <p class="brand-title">Edge SLM</p>
            <p class="brand-sub">UWA AI Club × Visagio</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mode = st.sidebar.radio(
        "Mode",
        ["Pipeline runner", "Evaluation"],
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.markdown(
        '<p class="section-eyebrow" style="padding-left:4px;">Last run</p>',
        unsafe_allow_html=True,
    )

    if st.session_state["data_ingestion_done"]:
        name = html.escape(st.session_state.get("last_filename") or "Untitled")

        st.sidebar.markdown(
            f"""
            <div class="last-run-card">
                <div class="run-status" style="color:#16A34A;">✓ Complete</div>
                <div class="run-name">{name}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            """
            <div class="last-run-card">
                <div class="run-status" style="color:#94A3B8;">No completed run yet</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.sidebar.divider()
    st.sidebar.markdown(
        '<p class="section-eyebrow" style="padding-left:4px;">Supported formats</p>',
        unsafe_allow_html=True,
    )
    st.sidebar.caption(", ".join(supported_extensions()))

    return mode


def main() -> None:
    # I inject the CSS here so the app keeps the clean dashboard look.
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    initialise_session_state()

    mode = render_sidebar()

    if mode == "Evaluation":
        st.title("Evaluation")
        st.caption("Benchmark the fine-tuned model against a dataset.")
        st.info(
            "Evaluation will be connected once fine-tuning and "
            "quantization stages are complete."
        )
        return

    st.title("Pipeline runner")
    st.caption("Configure and prepare a fine-tuning run")

    left_col, right_col = st.columns([1.15, 1], gap="large")

    with left_col:
        st.markdown(
            '<p class="section-eyebrow">1 — Data source</p>',
            unsafe_allow_html=True,
        )

        st.selectbox(
            "Input format",
            [
                "Auto detect",
                "PDF documents",
                "PPTX slide decks",
                "Plain text / Markdown",
                "Audio / video files",
            ],
        )

        uploaded_file = st.file_uploader(
            "Drop files or click to browse",
            type=ALL_EXTENSIONS,
            help="Supports PDF, PPTX, TXT, MD, MP3, WAV, MP4, and more.",
        )

        st.markdown(
            """
            <p style="font-size:0.82rem; font-weight:500; color:#64748B; margin:8px 0 4px 0;">
                Or paste a YouTube URL
            </p>
            """,
            unsafe_allow_html=True,
        )

        youtube_url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown(
            '<p class="section-eyebrow">2 — Model configuration</p>',
            unsafe_allow_html=True,
        )

        base_model = st.selectbox("Base model", MODELS)

        lora_rank = st.selectbox(
            "LoRA rank",
            LORA_RANKS,
            index=1,
        )

        quantization_target = st.selectbox(
            "Quantization target",
            QUANTIZATION_TARGETS,
            index=0,
        )

        st.divider()
        st.markdown(
            '<p class="section-eyebrow">3 — Hardware target</p>',
            unsafe_allow_html=True,
        )

        hw1, hw2, hw3 = st.columns(3)
        hw1.metric("RAM", "16 GB")
        hw2.metric("VRAM", "2–8 GB")
        hw3.metric("Mode", "multitask")

        # I keep this fixed for now because the current project target is Pawsey.
        # Later, this can become a real selector if the team supports local runs too.
        hardware_target = "Pawsey / HPC"

        st.divider()
        st.markdown(
            '<p class="section-eyebrow">4 — Pawsey job</p>',
            unsafe_allow_html=True,
        )

        turnaround = st.selectbox("Expected turnaround", TURNAROUND_OPTIONS)

        notification_email = st.text_input(
            "Notification email",
            placeholder="you@visagio.com",
        )

        st.divider()

        clean_url = youtube_url.strip()
        has_file = uploaded_file is not None
        has_url = bool(clean_url) and looks_like_youtube_url(clean_url)

        if clean_url and not looks_like_youtube_url(clean_url):
            st.warning("This does not look like a valid YouTube URL.")

        run_clicked = st.button(
            "▶ Prepare pipeline run",
            type="primary",
            use_container_width=True,
            disabled=not (has_file or has_url),
        )

        if run_clicked:
            # I give YouTube priority if both a file and URL are provided,
            # because the user most likely pasted the URL intentionally.
            if has_url:
                run_ingestion_from_youtube(
                    clean_url,
                    base_model=base_model,
                    lora_rank=lora_rank,
                    quantization_target=quantization_target,
                    hardware_target=hardware_target,
                    turnaround=turnaround,
                    notification_email=notification_email,
                )
                st.rerun()

            elif has_file:
                run_ingestion(
                    uploaded_file,
                    base_model=base_model,
                    lora_rank=lora_rank,
                    quantization_target=quantization_target,
                    hardware_target=hardware_target,
                    turnaround=turnaround,
                    notification_email=notification_email,
                )
                st.rerun()

    with right_col:
        render_pipeline_stages()
        render_error()
        render_document_summary()

    st.divider()
    st.caption(
        "Stage 1 — Data ingestion — is connected to the live pipeline. "
        "The run config JSON is saved for stages 2–5."
    )


if __name__ == "__main__":
    main()