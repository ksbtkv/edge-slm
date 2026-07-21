"""
One-time export of documentation/article sources for a source-pack manifest.

For every non-video source in the manifest (documentation, tutorial, article,
training portal, certification page, course outline), fetch `original_url`,
extract the main content with trafilatura, and write Markdown to the
manifest's `local_path`.

Pages that fail to fetch or yield too little text are reported at the end and
left for manual export.

Usage:
    python scripts/fetch_docs.py [--manifest PATH] [--only SOURCE_ID ...] [--force]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger("fetch_docs")

DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "databricks_ld_foundations.json"
NON_DOC_RESOURCE_TYPES = {"video_transcript", "playlist"}
MIN_CONTENT_WORDS = 80


def fetch_markdown(url: str) -> str | None:
    """Fetch a URL and return extracted Markdown, or None on failure."""
    import trafilatura

    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None
    return trafilatura.extract(
        downloaded,
        output_format="markdown",
        include_formatting=True,
        include_tables=True,
        include_links=False,
        favor_recall=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch even if local_path already exists",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    sources = [
        s
        for s in manifest["sources"]
        if s["resource_type"] not in NON_DOC_RESOURCE_TYPES
    ]
    if args.only:
        sources = [s for s in sources if s["source_id"] in args.only]

    failures: list[tuple[str, str]] = []

    for source in sources:
        source_id = source["source_id"]
        local_path = PROJECT_ROOT / source["local_path"]
        url = source["original_url"]

        if local_path.exists() and not args.force:
            logger.info("Skipping %s (file exists)", source_id)
            continue

        logger.info("Fetching %s: %s", source_id, url)
        try:
            markdown = fetch_markdown(url)
        except Exception as error:
            failures.append((source_id, f"fetch error: {error}"))
            logger.exception("Fetch failed for %s", source_id)
            continue

        if not markdown or len(markdown.split()) < MIN_CONTENT_WORDS:
            word_count = len(markdown.split()) if markdown else 0
            failures.append(
                (source_id, f"extracted only {word_count} words; export manually")
            )
            logger.warning("Content too thin for %s (%d words)", source_id, word_count)
            continue

        content = f"# {source['title']}\n\nSource: {url}\n\n{markdown.strip()}\n"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(content, encoding="utf-8")
        logger.info("Wrote %s (%d words)", local_path, len(markdown.split()))

    if failures:
        logger.error("Sources needing manual export:")
        for source_id, reason in failures:
            logger.error("  %s: %s", source_id, reason)
        return 1
    logger.info("All requested doc sources exported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
