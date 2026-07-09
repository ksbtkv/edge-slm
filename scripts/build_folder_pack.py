"""
Build a source pack from a local folder (no curated manifest required).

Scans the input directory for ingestible files, auto-generates a manifest,
and writes the same artifacts as the manifest workflow:

- manifest.normalized.json
- documents/<source_id>.json
- source_pack.json
- study_note_tasks.jsonl

Usage:
    PYTHONPATH=pipeline python scripts/build_folder_pack.py /path/to/folder \\
        -o data/processed/source_packs/my_folder_pack \\
        --pack-id my_folder_pack \\
        --domain "Local documents"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "pipeline"))

from ingestion.chunking import ChunkingConfig
from ingestion.source_pack import SourcePackError, build_source_pack_from_folder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Local folder to scan for ingestible files",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for generated pack artifacts",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan files directly under input_dir (not subfolders)",
    )
    parser.add_argument(
        "--pack-id",
        default=None,
        help="Pack identifier (default: slug of folder name)",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Human-readable pack title (default: folder name)",
    )
    parser.add_argument(
        "--domain",
        default=None,
        help='Domain label (default: "Local documents")',
    )
    parser.add_argument(
        "--topic-bucket-id",
        action="append",
        dest="topic_bucket_ids",
        default=None,
        help="Topic bucket id for all sources (repeatable; default: local_folder)",
    )
    parser.add_argument(
        "--target-words",
        type=int,
        default=None,
        help="Chunking target_words override",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=None,
        help="Chunking max_words override",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=None,
        help="Chunking min_words override",
    )
    parser.add_argument(
        "--overlap-words",
        type=int,
        default=None,
        help="Chunking overlap_words override",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    chunking_kwargs = {
        key: value
        for key, value in (
            ("target_words", args.target_words),
            ("max_words", args.max_words),
            ("min_words", args.min_words),
            ("overlap_words", args.overlap_words),
        )
        if value is not None
    }
    chunking_config = ChunkingConfig(**chunking_kwargs) if chunking_kwargs else None

    try:
        pack_index = build_source_pack_from_folder(
            args.input_dir,
            args.output_dir,
            recursive=not args.no_recursive,
            pack_id=args.pack_id,
            title=args.title,
            domain=args.domain,
            topic_bucket_ids=args.topic_bucket_ids,
            chunking_config=chunking_config,
        )
    except (SourcePackError, FileNotFoundError, NotADirectoryError) as error:
        logging.error("%s", error)
        return 1

    logging.info(
        "Pack %r: %d sources, %d tasks -> %s",
        pack_index["pack_id"],
        pack_index["ingested_source_count"],
        pack_index["study_note_task_count"],
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
