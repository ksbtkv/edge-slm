# Edge SLM

Offline edge small-language-model pipeline for turning heterogeneous learning
sources into reproducible **study-note task** datasets.

The pipeline ends at `study_note_tasks.jsonl`. LLM enrichment, instruction-pair
export, and LoRA training run in a **separate HPC pipeline** on Pawsey.

## Current status

The committed codebase covers **Stages 0–1.5**:

| Stage | Status | What it does |
|---|---|---|
| Ingestion | Done | Convert local files into a shared `Document` / `Section` schema |
| Source pack | Done | Curate sources via manifest, ingest a pack, write dataset artifacts |
| Chunking | Done | Turn sections into model-sized chunks for study-note tasks |
| Acquisition | Done | One-time scripts to download transcripts and export doc pages |

```text
source file  ->  Document (sections)  ->  TextChunk  ->  study_note_tasks.jsonl
                                                              |
                                                              v
                                              (HPC pipeline — separate repo)
                                    LLM enrichment -> training pairs -> LoRA
```

## Project layout

```text
edge_slm/
  pipeline/ingestion/     # ingestion, chunking, source-pack code
  scripts/                # one-time acquisition (transcripts, doc exports)
  tests/                  # pytest + smoke scripts
  data/manifests/         # source-pack manifests (tracked)
  data/raw/               # local sample/source files (gitignored)
  data/processed/         # generated pack outputs (gitignored)
```

Key modules:

- `schema.py` / `serialize.py` — `Document` contract and JSON round-trip
- `dispatch.py` — `ingest(path)` routes by file extension
- `*_ingestor.py` — format-specific extractors (PDF, PPTX, text, Markdown, audio/video)
- `quality.py` — shared ingest filters (e.g. `MIN_BODY_WORDS` for thin markdown sections)
- `chunking.py` — model-sized chunking over `Document.sections`
- `source_manifest.py` / `source_pack.py` — manifest validation and pack builder
- `study_notes_schema.py` — structured study-note prompt and output schema for tasks

## Supported formats

`ingest(path)` currently routes:

| Extension | Ingestor |
|---|---|
| `.pdf` | `pdf_ingestor` (pymupdf4llm + OCR fallback for scanned pages) |
| `.pptx` | `pptx_ingestor` (python-pptx) |
| `.txt`, `.text` | `text_ingestor` |
| `.md`, `.markdown` | `markdown_ingestor` |
| `.mp3`, `.wav`, `.m4a`, `.flac`, `.aac`, `.ogg`, `.opus` | `audio_video_ingestor` |
| `.mp4`, `.mov`, `.mkv`, `.avi`, `.webm` | `audio_video_ingestor` (ffmpeg + faster-whisper) |

Optional dependencies load lazily — importing `dispatch` does not require every
format backend to be installed.

Scanned (image-only) PDF pages are OCRed automatically via PyMuPDF's embedded
Tesseract engine when Tesseract language data is installed (macOS:
`brew install tesseract`; Windows: see [Windows setup](#windows-setup)).
Without it, scanned pages are skipped with a warning (and a fully scanned PDF
fails with an install hint). OCR sections carry
`extraction_method="pymupdf_ocr"`, and per-page OCR stats land in
`format_metadata["pdf"]`. Pass `ocr_fallback=False` to `ingest_pdf` to opt
out.

## Install

From the project root, using your `edge-slm` environment:

```bash
pip install -r requirements-dev.txt
```

Core text and Markdown ingestion use only the Python standard library:

```bash
pip install -r requirements.txt
```

Optional format dependencies are split by surface:

```bash
pip install -r requirements-ingestion-pdf.txt
pip install -r requirements-ingestion-pptx.txt
pip install -r requirements-ingestion-media.txt
```

For all current ingestors:

```bash
pip install -r requirements-ingestion-all.txt
```

Media ingestion also needs the system `ffmpeg` binary for local video extraction:

```bash
brew install ffmpeg
```

## Windows setup

The pipeline runs on Windows without code changes; only the shell syntax and
system binaries differ.

Environment variables: the `VAR=value command` prefix used in the examples
below is bash syntax. On Windows set the variable first:

```powershell
# PowerShell
$env:PYTHONPATH = "pipeline"; python -c "from ingestion.source_pack import build_source_pack; ..."
```

```bat
:: Command Prompt
set PYTHONPATH=pipeline && python -c "from ingestion.source_pack import build_source_pack; ..."
```

The same applies to `RUN_SLOW_INGESTION=1`.

System binaries:

```powershell
# ffmpeg (media ingestion)
winget install ffmpeg

# Tesseract language data (scanned-PDF OCR)
winget install UB-Mannheim.TesseractOCR
```

PyMuPDF finds the Tesseract language data automatically in the default
install location (`C:\Program Files\Tesseract-OCR\tessdata`); for a custom
location, set the `TESSDATA_PREFIX` environment variable to the `tessdata`
directory.

## Quick start: ingest one file

```bash
PYTHONPATH=pipeline python -c "
from ingestion.dispatch import ingest
from ingestion.serialize import save_document

doc = ingest('data/raw/pdf/EDGE SLM PROJECT.pdf')
print(doc.section_count, doc.total_word_count)
save_document(doc, 'output/example.json')
"
```

## Sections vs chunks

Ingestion produces faithful **sections** (page, slide, heading block, or
transcript segment). Chunking then builds **model-sized inputs** from one or
more sections:

- `Document.sections` preserve source structure and provenance
- `TextChunk` records are sizing-aware task inputs with merged/split provenance
- Each study-note task references chunk metadata (`chunk_id`,
  `source_section_indexes`, headings, page/slide/time range)
- Markdown ingestion skips header-only / thin bodies via `quality.py`
  (`MIN_BODY_WORDS`); chunking then enforces `min_words` before tasks are written

Inspect chunks for a file:

```bash
PYTHONPATH=pipeline python -c "
from ingestion.dispatch import ingest
from ingestion.chunking import chunk_document

doc = ingest('data/raw/pdf/EDGE SLM PROJECT.pdf')
for c in chunk_document(doc):
    print(c.chunk_id, c.word_count, c.source_section_indexes, c.split_reason)
"
```

Default chunk sizing (`ChunkingConfig`):

- `target_words=450`
- `max_words=700`
- `min_words=120` — enforced minimum chunk size (undersized chunks merge with neighbours or are dropped; a document's sole short chunk is kept)
- `overlap_words=60`

## Databricks source pack

The Databricks L&D source-pack workflow builds reproducible dataset artifacts
from a curated manifest of local documentation, tutorials, and transcripts.

Starter manifest:

- `data/manifests/databricks_ld_foundations.json` — source list mirroring the
  client's dataset source pack document; each `original_url` matches the
  hyperlink targets in that document
- `data/manifests/examples/study_note_example_delta_streaming.json` — the
  client-provided gold example output (Delta Lake streaming study notes), used
  as a schema-compliance reference for the HPC enrichment pipeline

Place curated source files under `data/raw/databricks/`, then enable matching
manifest entries with `"enabled": true`. All entries in the starter manifest
ship disabled until local files are added.

Build the pack:

```bash
PYTHONPATH=pipeline python -c "
from ingestion.source_pack import build_source_pack
build_source_pack(
    'data/manifests/databricks_ld_foundations.json',
    'data/processed/source_packs/databricks_ld_foundations',
    skip_missing_files=True,
)
"
```

Outputs land in `data/processed/source_packs/databricks_ld_foundations/`:

- `manifest.normalized.json` — validated manifest
- `documents/<source_id>.json` — one ingested `Document` per source
- `source_pack.json` — pack index with provenance and counts
- `study_note_tasks.jsonl` — **final pipeline output** — one task per chunk

Audit undersized tasks in an existing pack:

```bash
PYTHONPATH=pipeline python scripts/audit_chunk_quality.py \
  data/processed/source_packs/databricks_ld_foundations --threshold 120
```

Tune chunking when building a pack:

```bash
PYTHONPATH=pipeline python -c "
from ingestion.chunking import ChunkingConfig
from ingestion.source_pack import build_source_pack

build_source_pack(
    'data/manifests/databricks_ld_foundations.json',
    'data/processed/source_packs/databricks_ld_foundations',
    skip_missing_files=True,
    chunking_config=ChunkingConfig(
        target_words=450,
        max_words=700,
        min_words=120,
        overlap_words=60,
        preserve_section_boundaries=False,
    ),
)
"
```

### Acquiring the source content

One-time acquisition scripts populate `data/raw/databricks/` from the manifest's
`original_url`s (`pip install -r requirements-acquisition.txt` first).

**Recommended expansion workflow** (official docs + video gaps → 500+ chunks):

```bash
# 1. Discover in-scope docs.databricks.com pages (llms.txt + optional sitemap)
#    Review data/manifests/generated/databricks_docs_candidates.json, then apply.
python scripts/discover_databricks_docs.py --no-sitemap          # candidates only
python scripts/discover_databricks_docs.py --apply --limit 250   # merge into manifest

# 2. Export documentation/article pages -> Markdown via trafilatura
python scripts/fetch_docs.py

# 3. Expand playlist sources into per-video transcript entries (updates manifest,
#    disables thin parent indexes). Then download + transcribe videos.
PYTHONPATH=pipeline python scripts/fetch_transcripts.py --expand-playlists --expand-only
PYTHONPATH=pipeline python scripts/fetch_transcripts.py

# 4. Rebuild pack and audit undersized chunks
PYTHONPATH=pipeline python -c "
from ingestion.source_pack import build_source_pack
build_source_pack(
    'data/manifests/databricks_ld_foundations.json',
    'data/processed/source_packs/databricks_ld_foundations',
    skip_missing_files=True,
)
"
PYTHONPATH=pipeline python scripts/audit_chunk_quality.py \
    data/processed/source_packs/databricks_ld_foundations
```

Shorter one-shot acquisition (existing manifest URLs only):

```bash
# Video transcripts: yt-dlp audio download + faster-whisper transcription.
# Without --expand-playlists, playlist sources get an index .md only.
PYTHONPATH=pipeline python scripts/fetch_transcripts.py

# Documentation/article pages -> Markdown via trafilatura
python scripts/fetch_docs.py
```

Scripts skip sources whose `local_path` already exists (`--force` to re-fetch,
`--only SOURCE_ID` to restrict, `--skip-download` to transcribe existing audio).
The anchor certification video is ~7.5 hours, so its download + transcription
takes a while.

## Folder batch (no manifest)

For ad-hoc local folders, skip hand-authoring a manifest and build a pack directly
from a directory tree. The scanner discovers supported files recursively, assigns
stable `source_id`s, and writes the same artifacts as the curated manifest
workflow (`manifest.normalized.json` is included so you can inspect or edit and
re-run manually later).

Use the **curated manifest workflow** (Databricks L&D pack above) when you need
`original_url` provenance, topic-bucket curation, train/eval splits, or
production dataset labels. Folder batch is best for one-off local document dumps.

Python API:

```bash
PYTHONPATH=pipeline python -c "
from ingestion.source_pack import build_source_pack_from_folder

build_source_pack_from_folder(
    '/path/to/folder',
    'data/processed/source_packs/my_folder_pack',
    pack_id='my_folder_pack',
    domain='Local documents',
)
"
```

CLI:

```bash
PYTHONPATH=pipeline python scripts/build_folder_pack.py /path/to/folder \\
    -o data/processed/source_packs/my_folder_pack \\
    --pack-id my_folder_pack \\
    --domain "Local documents"
```

Flags: `--no-recursive`, `--pack-id`, `--title`, `--domain`, repeatable
`--topic-bucket-id`, and optional chunking overrides (`--target-words`,
`--max-words`, `--min-words`, `--overlap-words`).

Hidden files, `__MACOSX`, `.DS_Store`, and unsupported extensions are skipped.
An empty folder (no ingestible files) raises an error listing supported
extensions.

Non-Databricks files work with the same ingestion and chunking layers. Only the
checked-in Databricks manifest and study-note prompt are domain-specific.

## HPC handoff

This repo's deliverable is `study_note_tasks.jsonl`. Each line includes:

| Field | Purpose |
|---|---|
| `source_content` | Chunk text (~450 words) for the LLM to summarise |
| `prompt` | Full instructions + embedded content + output rules |
| `expected_output_schema` | Target JSON shape for structured study notes |
| `split`, `topic_bucket_ids`, `source_id`, `original_url` | Dataset labels and provenance |

The gold example at `data/manifests/examples/study_note_example_delta_streaming.json`
shows the expected output quality and schema.

Downstream (separate HPC pipeline): LLM enrichment → validated study-note JSON
→ instruction pairs → LoRA fine-tuning on Pawsey.

### Packaging a dated handoff snapshot

After a successful pack build, copy the deliverables into a dated folder under
`data/processed/handoffs/` (gitignored with the rest of `data/processed/`):

```bash
DATE=$(date +%Y%m%d)
HANDOFF=data/processed/handoffs/databricks_ld_foundations_$DATE
mkdir -p "$HANDOFF"
cp data/processed/source_packs/databricks_ld_foundations/study_note_tasks.jsonl "$HANDOFF/"
cp data/processed/source_packs/databricks_ld_foundations/source_pack.json "$HANDOFF/"
cp data/processed/source_packs/databricks_ld_foundations/manifest.normalized.json "$HANDOFF/"
# Add HANDOFF.md with task counts, splits, and the gold-example path, then
# copy the folder to the Pawsey HPC workspace.
```

Current local snapshot (example): `data/processed/handoffs/databricks_ld_foundations_20260709/`.

## Test

Fast tests avoid heavy optional dependencies and local fixture files:

```bash
PYTHONPATH=pipeline pytest
```

Full sample-file ingestion (PDF, PPTX, audio, video) is opt-in:

```bash
RUN_SLOW_INGESTION=1 PYTHONPATH=pipeline pytest
```

Manual smoke test with a human-readable report (requires files under
`data/raw/`):

```bash
PYTHONPATH=pipeline python -m tests.smoke_test_ingestion
```

Expected local fixture layout for smoke/slow tests:

```text
data/raw/
  text/SampleText.txt
  markdown/audio_video_ingestion.md
  pdf/EDGE SLM PROJECT.pdf
  pdf/Sample.pdf
  pptx/SamplePPT.pptx
  audio/
  video/
```
