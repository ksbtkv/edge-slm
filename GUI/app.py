from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import streamlit as st

st.set_page_config(
    page_title="Edge SLM Pipeline Runner",
    page_icon="▶",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = PROJECT_ROOT / "pipeline"
RUNS_DIR = PROJECT_ROOT / "data" / "gui_runs"

if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from ingestion.dispatch import supported_extensions  # noqa: E402


ALL_EXTENSIONS = [
    "pdf", "pptx",
    "txt", "text", "md", "markdown",
    "mp3", "wav", "m4a", "flac", "aac", "ogg", "opus", "webm",
    "mp4", "mov", "mkv", "avi",
]

MODEL_PRESETS = [
    "Qwen3 4B Instruct 4-bit",
    "Qwen2.5 3B",
    "Llama 3.2 3B",
    "Phi-3.5 Mini 3.8B",
    "Gemma 3 4B",
]

LORA_RANKS = [
    "8 — faster",
    "16 — balanced",
    "32 — stronger",
    "64 — larger",
]

QUANTIZATION_TARGETS = [
    "Q4_K_M — recommended",
    "Q5_K_M — better quality",
    "Q8_0 — maximum quality",
]

HARDWARE_TARGETS = [
    "Local Mac / MLX",
    "Pawsey / HPC",
]

RUN_TYPES = [
    "Local smoke test",
    "Standard training run",
    "Priority run",
]

DEFAULT_MLX_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"


CUSTOM_CSS = """
<style>
:root {
    --bg: #F7F8FC;
    --surface: #FFFFFF;
    --surface-soft: #F8FAFC;
    --border: #E5E7EB;
    --border-strong: #CBD5E1;
    --text: #0F172A;
    --muted: #64748B;
    --primary: #5B4FE9;
    --primary-dark: #4338CA;
    --primary-soft: #EEF2FF;
    --green: #16A34A;
    --green-soft: #DCFCE7;
    --red: #DC2626;
    --red-soft: #FEE2E2;
    --amber: #D97706;
    --amber-soft: #FEF3C7;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(91, 79, 233, 0.075), transparent 28rem),
        linear-gradient(180deg, #FFFFFF 0%, var(--bg) 48%, var(--bg) 100%);
    color: var(--text);
}

.block-container {
    padding-top: 1.7rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

header[data-testid="stHeader"] {
    background: rgba(255, 255, 255, 0.82);
    backdrop-filter: blur(18px);
    border-bottom: 1px solid rgba(226, 232, 240, 0.85);
}

[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: var(--text);
}

h1 {
    color: var(--text) !important;
    font-weight: 850 !important;
    letter-spacing: -0.045em !important;
    margin-bottom: 0.15rem !important;
}

h2, h3 {
    color: var(--text) !important;
}

.app-caption {
    color: var(--muted);
    font-size: 0.98rem;
    line-height: 1.55;
    margin-bottom: 1.5rem;
}

.sidebar-brand {
    background: linear-gradient(135deg, #5B4FE9 0%, #7C6FF7 100%);
    border-radius: 20px;
    padding: 18px;
    margin-bottom: 18px;
    box-shadow: 0 16px 36px rgba(91, 79, 233, 0.24);
}

.sidebar-title {
    color: #FFFFFF !important;
    font-size: 1.28rem;
    font-weight: 850;
    margin-bottom: 3px;
}

.sidebar-sub {
    color: rgba(255, 255, 255, 0.78) !important;
    font-size: 0.78rem;
    font-weight: 650;
}

.run-pill {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    color: #166534 !important;
    border-radius: 15px;
    padding: 12px 14px;
    font-size: 0.84rem;
    font-weight: 750;
    line-height: 1.45;
}

.empty-pill {
    background: #F8FAFC;
    border: 1px solid var(--border);
    color: var(--muted) !important;
    border-radius: 15px;
    padding: 12px 14px;
    font-size: 0.84rem;
    font-weight: 700;
}

.section-kicker {
    color: var(--primary);
    font-size: 0.74rem;
    font-weight: 850;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 0.45rem;
}

.section-note {
    color: var(--muted);
    font-size: 0.86rem;
    line-height: 1.5;
    margin-top: -0.15rem;
    margin-bottom: 0.9rem;
}

.panel {
    background: rgba(255,255,255,0.92);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 20px;
    box-shadow: 0 18px 44px rgba(15, 23, 42, 0.045);
}

.input-note {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    color: #475569;
    border-radius: 14px;
    padding: 12px 14px;
    font-size: 0.83rem;
    line-height: 1.5;
    margin-top: 1rem;
}

.status-chip {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 999px;
    padding: 5px 10px;
    font-size: 0.66rem;
    font-weight: 850;
    letter-spacing: 0.45px;
    text-transform: uppercase;
}

.status-ready {
    background: var(--primary-soft);
    color: var(--primary-dark);
}

.status-waiting {
    background: #F1F5F9;
    color: #64748B;
}

.status-done {
    background: var(--green-soft);
    color: var(--green);
}

.status-error {
    background: var(--red-soft);
    color: var(--red);
}

.step-title {
    color: var(--text);
    font-size: 1rem;
    font-weight: 850;
    margin-bottom: 0.18rem;
}

.step-description {
    color: var(--muted);
    font-size: 0.84rem;
    line-height: 1.48;
}

.step-detail {
    color: var(--text);
    font-size: 0.78rem;
    font-weight: 800;
    margin-top: 0.55rem;
}

.step-technical {
    color: #94A3B8;
    font-size: 0.72rem;
    line-height: 1.4;
    margin-top: 0.35rem;
}

.step-number {
    width: 34px;
    height: 34px;
    border-radius: 999px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.88rem;
    font-weight: 850;
    background: #F1F5F9;
    color: #64748B;
}

.step-number-ready {
    background: var(--primary-soft);
    color: var(--primary-dark);
}

.step-number-done {
    background: var(--green-soft);
    color: var(--green);
}

.step-number-error {
    background: var(--red-soft);
    color: var(--red);
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 22px !important;
    border: 1px solid #E5E7EB !important;
    box-shadow: 0 12px 32px rgba(15, 23, 42, 0.04) !important;
    background: rgba(255,255,255,0.92) !important;
}

.path-box {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 10px 12px;
    color: #334155;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.78rem;
    word-break: break-all;
    margin-bottom: 0.75rem;
}

.success-note {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    color: #166534;
    border-radius: 14px;
    padding: 12px 14px;
    font-size: 0.83rem;
    line-height: 1.5;
    margin-top: 1rem;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #5B4FE9 0%, #7C6FF7 100%) !important;
    border: none !important;
    border-radius: 14px !important;
    color: white !important;
    font-weight: 800 !important;
    min-height: 42px !important;
    box-shadow: 0 12px 24px rgba(91, 79, 233, 0.22);
}

.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 16px 28px rgba(91, 79, 233, 0.28);
}

.stButton > button[kind="primary"]:disabled {
    opacity: 0.45 !important;
    box-shadow: none !important;
}

div[data-baseweb="select"] > div,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 13px !important;
    color: #0F172A !important;
}

div[data-baseweb="select"] * {
    color: #0F172A !important;
}

[data-testid="stTextInput"] input::placeholder {
    color: #94A3B8 !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: #FFFFFF !important;
    border: 1px dashed #CBD5E1 !important;
    border-radius: 18px !important;
    overflow: hidden !important;
}

[data-testid="stFileUploaderDropzone"] * {
    color: #334155 !important;
}

[data-testid="stFileUploaderFile"] {
    background: #F8FAFC !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
}

[data-testid="stFileUploaderFile"] * {
    color: #0F172A !important;
}

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 17px;
    padding: 14px 16px !important;
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.035);
}

[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 0.78rem !important;
}

[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-weight: 850 !important;
}

hr {
    border-color: #E5E7EB !important;
}

code {
    border-radius: 10px !important;
}
</style>
"""


def initialise_session_state() -> None:
    defaults = {
        "run_summary": None,
        "enrichment_summary": None,
        "training_summary": None,
        "finetuning_summary": None,
        "final_check_summary": None,
        "run_error": None,
        "enrichment_error": None,
        "training_error": None,
        "finetuning_error": None,
        "final_check_error": None,
        "source_pack_logs": None,
        "enrichment_logs": None,
        "training_logs": None,
        "finetuning_logs": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def parse_lora_rank(label: str) -> int:
    return int(label.split("—")[0].strip())


def parse_quantization_target(label: str) -> str:
    return label.split("—")[0].strip()


def safe_filename(name: str) -> str:
    cleaned = name.replace(" ", "_")
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", cleaned)
    return cleaned.strip("_") or "uploaded_file"


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
    parsed_url = urlparse(youtube_url)

    if "youtu.be" in parsed_url.netloc:
        video_id = parsed_url.path.strip("/")
    else:
        query_values = parse_qs(parsed_url.query)
        video_id = query_values.get("v", ["youtube_source"])[0]

    if not video_id:
        video_id = "youtube_source"

    return re.sub(r"[^a-zA-Z0-9_-]", "_", video_id)


def download_youtube_audio(youtube_url: str, input_dir: Path) -> Path:
    from yt_dlp import YoutubeDL

    input_dir.mkdir(parents=True, exist_ok=True)

    video_id = get_youtube_video_id(youtube_url)
    output_template = str(input_dir / f"youtube_{video_id}.%(ext)s")

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

    possible_files = sorted(input_dir.glob(f"youtube_{video_id}.*"))

    if possible_files:
        return possible_files[0]

    raise FileNotFoundError(
        f"YouTube audio download completed, but no file was found for {video_id}."
    )


def save_uploaded_files(uploaded_files, input_dir: Path) -> list[Path]:
    input_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for uploaded_file in uploaded_files:
        output_name = safe_filename(uploaded_file.name)
        output_path = input_dir / output_name

        counter = 1
        while output_path.exists():
            output_path = input_dir / f"{output_path.stem}_{counter}{output_path.suffix}"
            counter += 1

        output_path.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(output_path)

    return saved_paths


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0

    with path.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def read_preview(path: Path, max_chars: int = 3000) -> str:
    if not path.exists():
        return ""

    text = path.read_text(encoding="utf-8")
    return text[:max_chars]


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def update_run_config(config_path: Path, updates: dict) -> None:
    config = read_json(config_path)
    config.update(updates)
    write_json(config_path, config)


def update_pipeline_status(config_path: Path, stage: str, status: str) -> None:
    config = read_json(config_path)
    pipeline_status = config.get("pipeline_status", {})
    pipeline_status[stage] = status
    config["pipeline_status"] = pipeline_status
    write_json(config_path, config)


def run_command(
    command: list[str],
    *,
    env_extra: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PIPELINE_ROOT)

    if env_extra:
        env.update({key: value for key, value in env_extra.items() if value})

    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def parse_key_value_stats(output: str, keys: list[str]) -> dict:
    stats = {}

    for key in keys:
        match = re.search(rf"{key}=([0-9]+)", output)
        if match:
            stats[key] = int(match.group(1))

    return stats


def parse_finetuning_stats(output: str) -> dict:
    stats = {}

    peak_match = re.search(r"Peak mem\s+([0-9.]+)\s+GB", output)
    if peak_match:
        stats["peak_mem_gb"] = float(peak_match.group(1))

    train_matches = re.findall(r"Iter\s+([0-9]+): Train loss\s+([0-9.]+)", output)
    if train_matches:
        last_iter, last_loss = train_matches[-1]
        stats["last_train_iter"] = int(last_iter)
        stats["last_train_loss"] = float(last_loss)

    val_matches = re.findall(r"Iter\s+([0-9]+): Val loss\s+([0-9.]+)", output)
    if val_matches:
        last_iter, last_loss = val_matches[-1]
        stats["last_val_iter"] = int(last_iter)
        stats["last_val_loss"] = float(last_loss)

    return stats


def force_unassigned_tasks_to_train(tasks_path: Path) -> dict:
    lines = tasks_path.read_text(encoding="utf-8").splitlines()
    updated_lines = []
    split_counts = {}
    changed = 0

    for line in lines:
        if not line.strip():
            continue

        task = json.loads(line)
        original_split = task.get("split")

        if original_split in (None, "", "unassigned"):
            task["split"] = "train"
            changed += 1

        split = task.get("split", "missing")
        split_counts[split] = split_counts.get(split, 0) + 1
        updated_lines.append(json.dumps(task, ensure_ascii=False))

    tasks_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")

    return {
        "changed_to_train": changed,
        "split_counts": split_counts,
    }


def prepare_mlx_smoke_dataset(training_dir: Path) -> dict:
    train_path = training_dir / "train.jsonl"
    valid_path = training_dir / "valid.jsonl"
    test_path = training_dir / "test.jsonl"

    train_count = count_jsonl_lines(train_path)

    if train_count == 0:
        raise ValueError("train.jsonl is empty. Create the training dataset before testing local training.")

    copied_to_valid = False
    copied_to_test = False

    if count_jsonl_lines(valid_path) == 0:
        shutil.copyfile(train_path, valid_path)
        copied_to_valid = True

    if count_jsonl_lines(test_path) == 0:
        shutil.copyfile(train_path, test_path)
        copied_to_test = True

    return {
        "train_count": count_jsonl_lines(train_path),
        "valid_count": count_jsonl_lines(valid_path),
        "test_count": count_jsonl_lines(test_path),
        "copied_train_to_valid": copied_to_valid,
        "copied_train_to_test": copied_to_test,
    }


def build_source_pack_run(
    *,
    uploaded_files,
    youtube_url: str,
    ui_config: dict,
) -> None:
    st.session_state["run_summary"] = None
    st.session_state["enrichment_summary"] = None
    st.session_state["training_summary"] = None
    st.session_state["finetuning_summary"] = None
    st.session_state["final_check_summary"] = None
    st.session_state["run_error"] = None
    st.session_state["enrichment_error"] = None
    st.session_state["training_error"] = None
    st.session_state["finetuning_error"] = None
    st.session_state["final_check_error"] = None
    st.session_state["source_pack_logs"] = None
    st.session_state["enrichment_logs"] = None
    st.session_state["training_logs"] = None
    st.session_state["finetuning_logs"] = None

    run_id = f"run_{uuid.uuid4().hex[:8]}"
    run_dir = RUNS_DIR / run_id
    input_dir = run_dir / "inputs"
    source_pack_dir = run_dir / "source_pack"
    enrichment_dir = run_dir / "enrichment"
    training_dir = run_dir / "training"
    adapters_dir = run_dir / "adapters"
    evaluation_dir = run_dir / "evaluation"
    config_path = run_dir / "run_config.json"

    for directory in [
        run_dir,
        input_dir,
        source_pack_dir,
        enrichment_dir,
        training_dir,
        adapters_dir,
        evaluation_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    saved_files = []

    if uploaded_files:
        saved_files.extend(save_uploaded_files(uploaded_files, input_dir))

    clean_url = youtube_url.strip()

    if clean_url:
        downloaded_audio = download_youtube_audio(clean_url, input_dir)
        saved_files.append(downloaded_audio)

    if not saved_files:
        raise ValueError("Upload at least one file or provide a YouTube URL.")

    saved_config = dict(ui_config)
    saved_config.pop("anthropic_api_key", None)

    initial_config = {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "run_dir": str(run_dir),
        "input_dir": str(input_dir),
        "source_pack_dir": str(source_pack_dir),
        "enrichment_dir": str(enrichment_dir),
        "training_dir": str(training_dir),
        "adapters_dir": str(adapters_dir),
        "evaluation_dir": str(evaluation_dir),
        "tasks_jsonl": str(source_pack_dir / "study_note_tasks.jsonl"),
        "input_files": [str(path) for path in saved_files],
        "youtube_url": clean_url or None,
        "ui_settings": saved_config,
        "pipeline_status": {
            "source_pack": "running",
            "teacher_enrichment": "pending",
            "training_pairs": "pending",
            "fine_tuning": "pending",
            "evaluation": "pending",
        },
    }

    write_json(config_path, initial_config)

    command = [
        sys.executable,
        "scripts/build_folder_pack.py",
        str(input_dir),
        "-o",
        str(source_pack_dir),
        "--pack-id",
        run_id,
        "--domain",
        "GUI upload",
    ]

    result = run_command(command)

    st.session_state["source_pack_logs"] = {
        "command": " ".join(command),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }

    if result.returncode != 0:
        update_pipeline_status(config_path, "source_pack", "error")
        update_run_config(config_path, {"error": result.stderr})
        raise RuntimeError(result.stderr or "File preparation failed.")

    tasks_path = source_pack_dir / "study_note_tasks.jsonl"
    source_pack_path = source_pack_dir / "source_pack.json"
    manifest_path = source_pack_dir / "manifest.normalized.json"

    split_update = force_unassigned_tasks_to_train(tasks_path)
    task_count = count_jsonl_lines(tasks_path)

    completed_config = {
        **initial_config,
        "source_pack_json": str(source_pack_path),
        "manifest_json": str(manifest_path),
        "study_note_task_count": task_count,
        "split_update": split_update,
        "pipeline_status": {
            "source_pack": "done",
            "teacher_enrichment": "pending",
            "training_pairs": "pending",
            "fine_tuning": "pending",
            "evaluation": "pending",
        },
    }

    write_json(config_path, completed_config)

    st.session_state["run_summary"] = {
        "run_id": run_id,
        "run_dir": run_dir,
        "input_dir": input_dir,
        "source_pack_dir": source_pack_dir,
        "enrichment_dir": enrichment_dir,
        "training_dir": training_dir,
        "adapters_dir": adapters_dir,
        "evaluation_dir": evaluation_dir,
        "tasks_path": tasks_path,
        "source_pack_path": source_pack_path,
        "manifest_path": manifest_path,
        "config_path": config_path,
        "input_files": saved_files,
        "task_count": task_count,
        "split_update": split_update,
        "ui_settings": saved_config,
    }


def run_teacher_enrichment_sample(
    *,
    sample_size: int,
    anthropic_api_key: str,
) -> None:
    st.session_state["enrichment_summary"] = None
    st.session_state["training_summary"] = None
    st.session_state["finetuning_summary"] = None
    st.session_state["final_check_summary"] = None
    st.session_state["enrichment_error"] = None
    st.session_state["training_error"] = None
    st.session_state["finetuning_error"] = None
    st.session_state["final_check_error"] = None
    st.session_state["enrichment_logs"] = None
    st.session_state["training_logs"] = None
    st.session_state["finetuning_logs"] = None

    summary = st.session_state.get("run_summary")

    if not summary:
        raise ValueError("Prepare files before generating learning examples.")

    resolved_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    if not resolved_key:
        raise ValueError("Claude API key is missing.")

    tasks_path = Path(summary["tasks_path"])
    enrichment_dir = Path(summary["enrichment_dir"])
    config_path = Path(summary["config_path"])

    enrichment_dir.mkdir(parents=True, exist_ok=True)
    update_pipeline_status(config_path, "teacher_enrichment", "running")

    command = [
        sys.executable,
        "scripts/enrich_tasks.py",
        str(tasks_path),
        str(enrichment_dir),
        "--sample",
        str(sample_size),
        "--no-batch",
    ]

    result = run_command(
        command,
        env_extra={"ANTHROPIC_API_KEY": resolved_key},
    )

    combined_output = "\n".join(
        part for part in [result.stdout, result.stderr] if part.strip()
    )

    st.session_state["enrichment_logs"] = {
        "command": " ".join(command),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }

    if result.returncode != 0:
        update_pipeline_status(config_path, "teacher_enrichment", "error")
        update_run_config(config_path, {"teacher_enrichment_error": combined_output})
        raise RuntimeError(combined_output or "Learning example generation failed.")

    stats = parse_key_value_stats(
        combined_output,
        [
            "tasks",
            "already_done",
            "new",
            "retried",
            "rejected",
            "tokens_in",
            "tokens_out",
        ],
    )

    output_files = sorted(path for path in enrichment_dir.glob("**/*") if path.is_file())

    update_run_config(
        config_path,
        {
            "teacher_enrichment": {
                "status": "done",
                "sample_size": sample_size,
                "enrichment_dir": str(enrichment_dir),
                "stats": stats,
                "completed_at": utc_now_iso(),
            },
            "pipeline_status": {
                "source_pack": "done",
                "teacher_enrichment": "done",
                "training_pairs": "pending",
                "fine_tuning": "pending",
                "evaluation": "pending",
            },
        },
    )

    st.session_state["enrichment_summary"] = {
        "enrichment_dir": enrichment_dir,
        "sample_size": sample_size,
        "stats": stats,
        "output_files": output_files,
    }


def export_training_pairs(val_fraction: float = 0.0) -> None:
    st.session_state["training_summary"] = None
    st.session_state["finetuning_summary"] = None
    st.session_state["final_check_summary"] = None
    st.session_state["training_error"] = None
    st.session_state["finetuning_error"] = None
    st.session_state["final_check_error"] = None
    st.session_state["training_logs"] = None
    st.session_state["finetuning_logs"] = None

    summary = st.session_state.get("run_summary")
    enrichment_summary = st.session_state.get("enrichment_summary")

    if not summary:
        raise ValueError("Prepare files before creating the training dataset.")

    if not enrichment_summary:
        raise ValueError("Generate learning examples before creating the training dataset.")

    tasks_path = Path(summary["tasks_path"])
    enrichment_dir = Path(summary["enrichment_dir"])
    training_dir = Path(summary["training_dir"])
    config_path = Path(summary["config_path"])

    training_dir.mkdir(parents=True, exist_ok=True)
    force_unassigned_tasks_to_train(tasks_path)
    update_pipeline_status(config_path, "training_pairs", "running")

    command = [
        sys.executable,
        "scripts/export_training_pairs.py",
        str(tasks_path),
        str(enrichment_dir),
        str(training_dir),
        "--val-fraction",
        str(val_fraction),
    ]

    result = run_command(command)

    combined_output = "\n".join(
        part for part in [result.stdout, result.stderr] if part.strip()
    )

    st.session_state["training_logs"] = {
        "command": " ".join(command),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }

    if result.returncode != 0:
        update_pipeline_status(config_path, "training_pairs", "error")
        update_run_config(config_path, {"training_pairs_error": combined_output})
        raise RuntimeError(combined_output or "Training dataset creation failed.")

    stats = parse_key_value_stats(
        combined_output,
        [
            "train",
            "valid",
            "eval",
            "holdout",
            "skipped_unenriched",
            "skipped_other_split",
        ],
    )

    output_files = {
        "train": training_dir / "train.jsonl",
        "valid": training_dir / "valid.jsonl",
        "eval_references": training_dir / "eval_references.jsonl",
        "holdout_references": training_dir / "holdout_references.jsonl",
        "export_summary": training_dir / "export_summary.json",
    }

    update_run_config(
        config_path,
        {
            "training_pairs": {
                "status": "done",
                "training_dir": str(training_dir),
                "val_fraction": val_fraction,
                "stats": stats,
                "output_files": {name: str(path) for name, path in output_files.items()},
                "completed_at": utc_now_iso(),
            },
            "pipeline_status": {
                "source_pack": "done",
                "teacher_enrichment": "done",
                "training_pairs": "done",
                "fine_tuning": "pending",
                "evaluation": "pending",
            },
        },
    )

    st.session_state["training_summary"] = {
        "training_dir": training_dir,
        "val_fraction": val_fraction,
        "stats": stats,
        "output_files": output_files,
    }


def run_finetuning_smoke_test(
    *,
    iters: int,
    save_every: int,
    steps_per_eval: int,
    active_mlx_model: str,
) -> None:
    st.session_state["finetuning_summary"] = None
    st.session_state["final_check_summary"] = None
    st.session_state["finetuning_error"] = None
    st.session_state["final_check_error"] = None
    st.session_state["finetuning_logs"] = None

    summary = st.session_state.get("run_summary")
    training_summary = st.session_state.get("training_summary")

    if not summary:
        raise ValueError("Prepare files before testing local training.")

    if not training_summary:
        raise ValueError("Create the training dataset before testing local training.")

    training_dir = Path(summary["training_dir"])
    adapters_dir = Path(summary["adapters_dir"])
    config_path = Path(summary["config_path"])

    adapters_dir.mkdir(parents=True, exist_ok=True)
    mlx_dataset_status = prepare_mlx_smoke_dataset(training_dir)

    update_pipeline_status(config_path, "fine_tuning", "running")

    command = [
        "bash",
        "scripts/train_local_mlx.sh",
        str(training_dir),
        str(adapters_dir),
        "--iters",
        str(iters),
        "--save-every",
        str(save_every),
        "--steps-per-eval",
        str(steps_per_eval),
    ]

    result = run_command(
        command,
        env_extra={"EDGE_SLM_STUDENT_MLX": active_mlx_model},
    )

    combined_output = "\n".join(
        part for part in [result.stdout, result.stderr] if part.strip()
    )

    st.session_state["finetuning_logs"] = {
        "command": " ".join(command),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }

    if result.returncode != 0:
        update_pipeline_status(config_path, "fine_tuning", "error")
        update_run_config(config_path, {"fine_tuning_error": combined_output})
        raise RuntimeError(combined_output or "Local training test failed.")

    stats = parse_finetuning_stats(combined_output)
    adapter_files = sorted(path for path in adapters_dir.glob("*") if path.is_file())

    update_run_config(
        config_path,
        {
            "fine_tuning": {
                "status": "done",
                "mode": "local_mlx_smoke_test",
                "active_mlx_model": active_mlx_model,
                "iters": iters,
                "save_every": save_every,
                "steps_per_eval": steps_per_eval,
                "training_dir": str(training_dir),
                "adapters_dir": str(adapters_dir),
                "mlx_dataset_status": mlx_dataset_status,
                "stats": stats,
                "adapter_files": [str(path) for path in adapter_files],
                "completed_at": utc_now_iso(),
            },
            "pipeline_status": {
                "source_pack": "done",
                "teacher_enrichment": "done",
                "training_pairs": "done",
                "fine_tuning": "done",
                "evaluation": "pending",
            },
        },
    )

    st.session_state["finetuning_summary"] = {
        "training_dir": training_dir,
        "adapters_dir": adapters_dir,
        "active_mlx_model": active_mlx_model,
        "iters": iters,
        "save_every": save_every,
        "steps_per_eval": steps_per_eval,
        "mlx_dataset_status": mlx_dataset_status,
        "stats": stats,
        "adapter_files": adapter_files,
    }


def run_final_pipeline_check() -> None:
    st.session_state["final_check_summary"] = None
    st.session_state["final_check_error"] = None

    summary = st.session_state.get("run_summary")
    finetuning_summary = st.session_state.get("finetuning_summary")

    if not summary:
        raise ValueError("Prepare files before creating the final report.")

    if not finetuning_summary:
        raise ValueError("Run the local training test before creating the final report.")

    source_pack_dir = Path(summary["source_pack_dir"])
    enrichment_dir = Path(summary["enrichment_dir"])
    training_dir = Path(summary["training_dir"])
    adapters_dir = Path(summary["adapters_dir"])
    evaluation_dir = Path(summary["evaluation_dir"])
    config_path = Path(summary["config_path"])

    evaluation_dir.mkdir(parents=True, exist_ok=True)
    report_path = evaluation_dir / "pipeline_report.json"

    checks = {
        "run_config": config_path,
        "source_pack_json": source_pack_dir / "source_pack.json",
        "manifest_json": source_pack_dir / "manifest.normalized.json",
        "study_note_tasks": source_pack_dir / "study_note_tasks.jsonl",
        "enrichment_summary": enrichment_dir / "run_summary.json",
        "train_jsonl": training_dir / "train.jsonl",
        "valid_jsonl": training_dir / "valid.jsonl",
        "test_jsonl": training_dir / "test.jsonl",
        "export_summary": training_dir / "export_summary.json",
        "adapter_weights": adapters_dir / "adapters.safetensors",
        "adapter_config": adapters_dir / "adapter_config.json",
    }

    file_checks = {}

    for name, path in checks.items():
        file_checks[name] = {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "line_count": count_jsonl_lines(path) if path.suffix == ".jsonl" else None,
        }

    missing = [
        name
        for name, details in file_checks.items()
        if not details["exists"] or details["size_bytes"] == 0
    ]

    status = "passed" if not missing else "warning"

    report = {
        "run_id": summary["run_id"],
        "created_at": utc_now_iso(),
        "status": status,
        "message": (
            "All required pipeline artifacts are present."
            if not missing
            else "Some expected pipeline artifacts are missing or empty."
        ),
        "missing_or_empty": missing,
        "counts": {
            "input_files": len(summary["input_files"]),
            "study_note_tasks": count_jsonl_lines(source_pack_dir / "study_note_tasks.jsonl"),
            "train_rows": count_jsonl_lines(training_dir / "train.jsonl"),
            "valid_rows": count_jsonl_lines(training_dir / "valid.jsonl"),
            "test_rows": count_jsonl_lines(training_dir / "test.jsonl"),
            "adapter_files": len([path for path in adapters_dir.glob("*") if path.is_file()]),
        },
        "file_checks": file_checks,
        "note": (
            "This is a final pipeline integrity check. A true model-quality "
            "evaluation script is not present in this branch yet."
        ),
    }

    write_json(report_path, report)

    update_run_config(
        config_path,
        {
            "final_check": {
                "status": status,
                "report_path": str(report_path),
                "missing_or_empty": missing,
                "completed_at": utc_now_iso(),
            },
            "pipeline_status": {
                "source_pack": "done",
                "teacher_enrichment": "done",
                "training_pairs": "done",
                "fine_tuning": "done",
                "evaluation": "done" if status == "passed" else "warning",
            },
        },
    )

    st.session_state["final_check_summary"] = {
        "status": status,
        "report_path": report_path,
        "missing_or_empty": missing,
        "file_checks": file_checks,
        "counts": report["counts"],
    }


def section_title(title: str, note: str | None = None) -> None:
    st.markdown(
        f'<div class="section-kicker">{html.escape(title)}</div>',
        unsafe_allow_html=True,
    )

    if note:
        st.markdown(
            f'<div class="section-note">{html.escape(note)}</div>',
            unsafe_allow_html=True,
        )


def status_for_step(step: str) -> tuple[str, str]:
    run_summary = st.session_state.get("run_summary")
    enrichment_summary = st.session_state.get("enrichment_summary")
    training_summary = st.session_state.get("training_summary")
    finetuning_summary = st.session_state.get("finetuning_summary")
    final_check_summary = st.session_state.get("final_check_summary")

    errors = {
        "prepare": st.session_state.get("run_error"),
        "examples": st.session_state.get("enrichment_error"),
        "dataset": st.session_state.get("training_error"),
        "training": st.session_state.get("finetuning_error"),
        "report": st.session_state.get("final_check_error"),
    }

    if errors.get(step):
        return "error", "Needs attention"

    if step == "prepare":
        if run_summary:
            return "done", f"{run_summary['task_count']} task(s) prepared"
        return "ready", "Ready to start"

    if step == "examples":
        if enrichment_summary:
            stats = enrichment_summary.get("stats", {})
            return "done", f"{stats.get('new', 0)} example(s) generated"
        if run_summary:
            return "ready", "Ready"
        return "waiting", "Waiting for files"

    if step == "dataset":
        if training_summary:
            stats = training_summary.get("stats", {})
            return "done", f"{stats.get('train', 0)} training row(s)"
        if enrichment_summary:
            return "ready", "Ready"
        return "waiting", "Waiting for examples"

    if step == "training":
        if finetuning_summary:
            stats = finetuning_summary.get("stats", {})
            peak = stats.get("peak_mem_gb")
            if peak:
                return "done", f"Passed · {peak:.2f} GB peak"
            return "done", "Passed"
        if training_summary:
            return "ready", "Ready"
        return "waiting", "Waiting for dataset"

    if step == "report":
        if final_check_summary:
            missing = len(final_check_summary.get("missing_or_empty", []))
            if missing == 0:
                return "done", "All files present"
            return "error", f"{missing} missing item(s)"
        if finetuning_summary:
            return "ready", "Ready"
        return "waiting", "Waiting for training test"

    return "waiting", "Waiting"


def render_step_card(
    *,
    number: int,
    title: str,
    description: str,
    technical: str,
    status: str,
    detail: str,
    button_key: str,
    disabled: bool,
    action,
) -> None:
    status_class = {
        "ready": "status-ready",
        "waiting": "status-waiting",
        "done": "status-done",
        "error": "status-error",
    }.get(status, "status-waiting")

    number_class = {
        "ready": "step-number-ready",
        "done": "step-number-done",
        "error": "step-number-error",
    }.get(status, "")

    number_text = "✓" if status == "done" else "!" if status == "error" else str(number)

    with st.container(border=True):
        left, right = st.columns([0.76, 0.24])

        with left:
            icon_col, text_col = st.columns([0.11, 0.89])

            with icon_col:
                st.markdown(
                    f'<div class="step-number {number_class}">{html.escape(number_text)}</div>',
                    unsafe_allow_html=True,
                )

            with text_col:
                st.markdown(
                    f"""
<div class="step-title">{html.escape(title)}</div>
<div class="step-description">{html.escape(description)}</div>
<div class="step-technical">{html.escape(technical)}</div>
<div class="step-detail">{html.escape(detail)}</div>
                    """,
                    unsafe_allow_html=True,
                )

        with right:
            st.markdown(
                f'<div style="text-align:right; margin-bottom:10px;"><span class="status-chip {status_class}">{html.escape(status)}</span></div>',
                unsafe_allow_html=True,
            )
            clicked = st.button(
                "Run",
                type="primary",
                use_container_width=True,
                disabled=disabled,
                key=button_key,
            )

        if clicked:
            action()


def render_sidebar() -> str:
    st.sidebar.markdown(
        """
<div class="sidebar-brand">
    <div class="sidebar-title">Edge SLM</div>
    <div class="sidebar-sub">UWA AI Club × Visagio</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    mode = st.sidebar.radio(
        "View",
        ["Pipeline runner", "Final report"],
    )

    st.sidebar.divider()

    st.sidebar.markdown("### Latest run")
    summary = st.session_state.get("run_summary")
    final_check_summary = st.session_state.get("final_check_summary")
    finetuning_summary = st.session_state.get("finetuning_summary")
    training_summary = st.session_state.get("training_summary")
    enrichment_summary = st.session_state.get("enrichment_summary")

    if final_check_summary and summary:
        label = "final report created"
    elif finetuning_summary and summary:
        label = "training test passed"
    elif training_summary and summary:
        label = "dataset created"
    elif enrichment_summary and summary:
        label = "examples generated"
    elif summary:
        label = "files prepared"
    else:
        label = None

    if summary and label:
        st.sidebar.markdown(
            f"""
<div class="run-pill">
    ✓ {html.escape(summary["run_id"])}<br>
    {html.escape(label)}<br>
    {summary["task_count"]} prepared task(s)
</div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            '<div class="empty-pill">No run completed yet</div>',
            unsafe_allow_html=True,
        )

    st.sidebar.divider()
    st.sidebar.markdown("### Supported inputs")
    st.sidebar.caption(", ".join(supported_extensions()))

    return mode


def render_input_and_settings() -> tuple[list, str, dict]:
    section_title(
        "Input",
        "Add content and choose the run settings.",
    )

    uploaded_files = st.file_uploader(
        "Upload files",
        type=ALL_EXTENSIONS,
        accept_multiple_files=True,
        help="Supported: PDF, PPTX, TXT, Markdown, audio, and video files.",
    )

    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="Optional",
    )

    st.divider()

    section_title(
        "Model and run settings",
        "These settings are saved with the run. The active MLX model path is used during the local training test.",
    )

    model_preset = st.selectbox(
        "Model family",
        MODEL_PRESETS,
    )

    active_mlx_model = st.text_input(
        "Active MLX model path",
        value=DEFAULT_MLX_MODEL,
        help="This is passed to Stage 4 using EDGE_SLM_STUDENT_MLX.",
    )

    col1, col2 = st.columns(2)

    with col1:
        lora_rank = st.selectbox("LoRA rank", LORA_RANKS, index=1)
        hardware_target = st.selectbox("Hardware target", HARDWARE_TARGETS)

    with col2:
        quantization_target = st.selectbox("Quantization target", QUANTIZATION_TARGETS)
        run_type = st.selectbox("Run type", RUN_TYPES)

    st.divider()

    section_title(
        "Teacher model access",
        "Used only for generating examples. The key is not saved to run_config.json.",
    )

    anthropic_api_key = st.text_input(
        "Claude API key",
        type="password",
        placeholder="Paste key here, or use ANTHROPIC_API_KEY in terminal",
    )

    sample_size = st.number_input(
        "Example sample size",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
    )

    ui_config = {
        "model_preset": model_preset,
        "active_mlx_model": active_mlx_model,
        "lora_rank": parse_lora_rank(lora_rank),
        "quantization_target": parse_quantization_target(quantization_target),
        "hardware_target": hardware_target,
        "run_type": run_type,
        "sample_size": int(sample_size),
        "anthropic_api_key": anthropic_api_key,
        "note": (
            "The active MLX model path is used during the local MLX smoke test. "
            "Other settings are saved for handover and future automation."
        ),
    }

    return uploaded_files, youtube_url, ui_config


def render_workflow(uploaded_files, youtube_url: str, ui_config: dict) -> None:
    section_title(
        "Workflow",
        "Run each step from top to bottom.",
    )

    clean_url = youtube_url.strip()
    has_files = bool(uploaded_files)
    has_valid_url = bool(clean_url) and looks_like_youtube_url(clean_url)
    has_api_key = bool(ui_config.get("anthropic_api_key")) or bool(os.environ.get("ANTHROPIC_API_KEY"))

    prepare_status, prepare_detail = status_for_step("prepare")
    examples_status, examples_detail = status_for_step("examples")
    dataset_status, dataset_detail = status_for_step("dataset")
    training_status, training_detail = status_for_step("training")
    report_status, report_detail = status_for_step("report")

    def handle_prepare_files() -> None:
        try:
            with st.spinner("Preparing files..."):
                build_source_pack_run(
                    uploaded_files=uploaded_files,
                    youtube_url=clean_url if has_valid_url else "",
                    ui_config=ui_config,
                )
            st.rerun()
        except Exception as error:
            st.session_state["run_error"] = error
            st.rerun()

    def handle_generate_examples() -> None:
        try:
            with st.spinner("Generating examples..."):
                run_teacher_enrichment_sample(
                    sample_size=int(ui_config.get("sample_size", 1)),
                    anthropic_api_key=ui_config.get("anthropic_api_key", ""),
                )
            st.rerun()
        except Exception as error:
            st.session_state["enrichment_error"] = error
            st.rerun()

    def handle_create_dataset() -> None:
        try:
            with st.spinner("Creating training dataset..."):
                export_training_pairs(val_fraction=0.0)
            st.rerun()
        except Exception as error:
            st.session_state["training_error"] = error
            st.rerun()

    def handle_test_training() -> None:
        try:
            with st.spinner("Running local training test..."):
                run_finetuning_smoke_test(
                    iters=10,
                    save_every=10,
                    steps_per_eval=10,
                    active_mlx_model=ui_config.get("active_mlx_model", DEFAULT_MLX_MODEL),
                )
            st.rerun()
        except Exception as error:
            st.session_state["finetuning_error"] = error
            st.rerun()

    def handle_final_report() -> None:
        try:
            with st.spinner("Creating final report..."):
                run_final_pipeline_check()
            st.rerun()
        except Exception as error:
            st.session_state["final_check_error"] = error
            st.rerun()

    render_step_card(
        number=1,
        title="Prepare files",
        description="Create a clean run folder and convert your content into structured tasks.",
        technical="Uses scripts/build_folder_pack.py",
        status=prepare_status,
        detail=prepare_detail,
        button_key="prepare_files",
        disabled=not (has_files or has_valid_url),
        action=handle_prepare_files,
    )

    if clean_url and not looks_like_youtube_url(clean_url):
        st.caption("The YouTube URL does not look valid yet.")

    render_step_card(
        number=2,
        title="Generate examples",
        description="Use the teacher model to create learning examples from the prepared content.",
        technical="Uses scripts/enrich_tasks.py",
        status=examples_status,
        detail=examples_detail if has_api_key else "Enter Claude API key to continue",
        button_key="generate_examples",
        disabled=not st.session_state.get("run_summary") or not has_api_key,
        action=handle_generate_examples,
    )

    render_step_card(
        number=3,
        title="Create dataset",
        description="Turn the generated examples into train.jsonl and related training files.",
        technical="Uses scripts/export_training_pairs.py",
        status=dataset_status,
        detail=dataset_detail,
        button_key="create_dataset",
        disabled=not st.session_state.get("enrichment_summary"),
        action=handle_create_dataset,
    )

    render_step_card(
        number=4,
        title="Test training",
        description="Run a small local MLX training test and create adapter weights.",
        technical="Uses scripts/train_local_mlx.sh with 10 iterations",
        status=training_status,
        detail=training_detail,
        button_key="test_training",
        disabled=not st.session_state.get("training_summary"),
        action=handle_test_training,
    )

    render_step_card(
        number=5,
        title="Create report",
        description="Check all required outputs and save a final pipeline report.",
        technical="Creates evaluation/pipeline_report.json",
        status=report_status,
        detail=report_detail,
        button_key="create_report",
        disabled=not st.session_state.get("finetuning_summary"),
        action=handle_final_report,
    )


def render_simple_summary() -> None:
    summary = st.session_state.get("run_summary")
    enrichment_summary = st.session_state.get("enrichment_summary")
    training_summary = st.session_state.get("training_summary")
    finetuning_summary = st.session_state.get("finetuning_summary")
    final_check_summary = st.session_state.get("final_check_summary")

    if not summary:
        return

    st.divider()
    section_title(
        "Run summary",
        "A simple view of what the current run has completed.",
    )

    with st.container(border=True):
        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("Tasks", summary.get("task_count", 0))

        if enrichment_summary:
            c2.metric("Examples", enrichment_summary.get("stats", {}).get("new", 0))
        else:
            c2.metric("Examples", 0)

        if training_summary:
            c3.metric("Train rows", training_summary.get("stats", {}).get("train", 0))
        else:
            c3.metric("Train rows", 0)

        if finetuning_summary:
            c4.metric("Adapters", len(finetuning_summary.get("adapter_files", [])))
        else:
            c4.metric("Adapters", 0)

        if final_check_summary:
            c5.metric("Report", final_check_summary.get("status", "—"))
        else:
            c5.metric("Report", "Pending")

        if final_check_summary and not final_check_summary.get("missing_or_empty"):
            st.markdown(
                '<div class="success-note">All required pipeline artifacts are present.</div>',
                unsafe_allow_html=True,
            )


def render_generated_files() -> None:
    summary = st.session_state.get("run_summary")
    enrichment_summary = st.session_state.get("enrichment_summary")
    training_summary = st.session_state.get("training_summary")
    finetuning_summary = st.session_state.get("finetuning_summary")
    final_check_summary = st.session_state.get("final_check_summary")

    if not summary:
        return

    with st.expander("Generated files"):
        file_rows = [
            ("Run folder", summary.get("run_dir")),
            ("Run config", summary.get("config_path")),
            ("Prepared tasks", summary.get("tasks_path")),
            ("Source pack", summary.get("source_pack_dir")),
        ]

        if enrichment_summary:
            file_rows.append(("Generated examples", enrichment_summary.get("enrichment_dir")))

        if training_summary:
            file_rows.append(("Training dataset", training_summary.get("training_dir")))

        if finetuning_summary:
            file_rows.append(("Adapter folder", finetuning_summary.get("adapters_dir")))

        if final_check_summary:
            file_rows.append(("Final report", final_check_summary.get("report_path")))

        for label, path in file_rows:
            if path:
                st.markdown(f"**{label}**")
                st.markdown(
                    f'<div class="path-box">{html.escape(str(path))}</div>',
                    unsafe_allow_html=True,
                )


def render_previews() -> None:
    summary = st.session_state.get("run_summary")
    training_summary = st.session_state.get("training_summary")
    final_check_summary = st.session_state.get("final_check_summary")

    if not summary:
        return

    with st.expander("JSON previews"):
        tasks_path = Path(summary["tasks_path"])

        st.markdown("#### Prepared tasks")
        st.code(read_preview(tasks_path), language="json")

        if training_summary:
            output_files = training_summary.get("output_files", {})
            train_path = output_files.get("train")
            export_summary_path = output_files.get("export_summary")

            if train_path and Path(train_path).exists():
                st.markdown("#### train.jsonl")
                st.code(read_preview(Path(train_path)), language="json")

            if export_summary_path and Path(export_summary_path).exists():
                st.markdown("#### export_summary.json")
                st.code(read_preview(Path(export_summary_path)), language="json")

        if final_check_summary:
            report_path = final_check_summary.get("report_path")

            if report_path and Path(report_path).exists():
                st.markdown("#### pipeline_report.json")
                st.code(read_preview(Path(report_path)), language="json")


def render_logs() -> None:
    logs = [
        ("File preparation logs", st.session_state.get("source_pack_logs")),
        ("Example generation logs", st.session_state.get("enrichment_logs")),
        ("Dataset creation logs", st.session_state.get("training_logs")),
        ("Local training logs", st.session_state.get("finetuning_logs")),
    ]

    if not any(log for _, log in logs):
        return

    with st.expander("Technical logs"):
        for title, log in logs:
            if not log:
                continue

            st.markdown(f"#### {title}")

            st.write("Command:")
            st.code(log.get("command", ""))

            if log.get("stdout"):
                st.write("stdout:")
                st.code(log["stdout"])

            if log.get("stderr"):
                st.write("stderr:")
                st.code(log["stderr"])


def render_errors() -> None:
    errors = [
        ("File preparation failed.", st.session_state.get("run_error")),
        ("Example generation failed.", st.session_state.get("enrichment_error")),
        ("Dataset creation failed.", st.session_state.get("training_error")),
        ("Local training test failed.", st.session_state.get("finetuning_error")),
        ("Final report creation failed.", st.session_state.get("final_check_error")),
    ]

    active_errors = [(title, error) for title, error in errors if error]

    if not active_errors:
        return

    st.divider()
    section_title("Errors")

    for title, error in active_errors:
        st.error(title)

        with st.expander("Show details"):
            st.code(str(error))


def render_final_report_page() -> None:
    st.title("Final report")
    st.markdown(
        """
<div class="app-caption">
    This page shows the latest final pipeline report. It checks whether the expected
    files were created successfully. It is not a model-quality benchmark.
</div>
        """,
        unsafe_allow_html=True,
    )

    final_check_summary = st.session_state.get("final_check_summary")

    if not final_check_summary:
        st.info("No final report has been created yet. Complete the workflow in Pipeline runner.")
        return

    counts = final_check_summary.get("counts", {})
    missing = final_check_summary.get("missing_or_empty", [])

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Status", final_check_summary.get("status", "unknown"))
        c2.metric("Missing/empty", len(missing))
        c3.metric("Train rows", counts.get("train_rows", 0))

        c4, c5, c6 = st.columns(3)
        c4.metric("Prepared tasks", counts.get("study_note_tasks", 0))
        c5.metric("Test rows", counts.get("test_rows", 0))
        c6.metric("Adapter files", counts.get("adapter_files", 0))

        st.markdown("#### Report file")
        st.markdown(
            f'<div class="path-box">{html.escape(str(final_check_summary.get("report_path")))}</div>',
            unsafe_allow_html=True,
        )

        if missing:
            st.warning("Some expected artifacts are missing or empty.")
            for item in missing:
                st.write(f"- {item}")
        else:
            st.success("All required pipeline artifacts are present.")

    render_previews()


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    initialise_session_state()

    mode = render_sidebar()

    if mode == "Final report":
        render_final_report_page()
        return

    st.title("Pipeline runner")
    st.markdown(
        """
<div class="app-caption">
    Upload content, choose model settings, generate examples, create a training dataset,
    run a small local training test, and create a final handover report.
</div>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([0.98, 1.22], gap="large")

    with left_col:
        uploaded_files, youtube_url, ui_config = render_input_and_settings()

    with right_col:
        render_workflow(uploaded_files, youtube_url, ui_config)

    render_errors()
    render_simple_summary()
    render_generated_files()
    render_previews()
    render_logs()

    st.caption(
        "Stage 5 creates a pipeline integrity report because this branch does not yet include a model-quality evaluation script."
    )


if __name__ == "__main__":
    main()
