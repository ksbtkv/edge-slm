"""
Discover official Databricks documentation URLs and seed the source-pack manifest.

Primary seed: https://docs.databricks.com/llms.txt (topic-grouped markdown index).
Optional fallback: sitemap XML URLs (soft-fail if unavailable).

Usage:
    python scripts/discover_databricks_docs.py [--apply] [--limit N]
    python scripts/discover_databricks_docs.py --llms-txt PATH --apply
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger("discover_databricks_docs")

DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "databricks_ld_foundations.json"
DEFAULT_CANDIDATES_DIR = PROJECT_ROOT / "data" / "manifests" / "generated"
DEFAULT_LLMS_TXT_URL = "https://docs.databricks.com/llms.txt"
DEFAULT_SITEMAP_URLS = (
    "https://docs.databricks.com/en/doc-sitemap.xml",
    "https://docs.databricks.com/sitemap.xml",
)

# Path prefixes (matched against URL path after stripping cloud provider segments)
# mapped to existing topic buckets in databricks_ld_foundations.
TOPIC_PREFIXES: dict[str, tuple[str, ...]] = {
    "databricks_basics": (
        "/getting-started/",
        "/workspace/",
        "/compute/",
        "/notebooks/",
        "/introduction/",
        "/lakehouse/",
        "/files/",
        "/libraries/",
        "/spark/",
    ),
    "lakehouse_delta": (
        "/delta/",
        "/tables/",
        "/transactions/",
        "/iceberg/",
        "/optimizations/",
    ),
    "spark_sql_pyspark": (
        "/spark/",
        "/sql/",
        "/pyspark/",
        "/getting-started/dataframes",
        "/languages/python",
        "/udf/",
    ),
    "ingestion_incremental": (
        "/structured-streaming/",
        "/ingestion/",
        "/auto-loader/",
        "/connect/streaming/",
    ),
    "production_pipelines": (
        "/ldp/",
        "/jobs/",
        "/designer/",
        "/data-engineering/",
        "/dev-tools/ci-cd/",
        "/repos/",
    ),
    "governance_security": (
        "/data-governance/",
        "/unity-catalog/",
        "/catalogs/",
        "/volumes/",
        "/schemas/",
        "/security/",
        "/views/",
        "/catalog-explorer/",
    ),
}

EXCLUDE_PREFIXES: tuple[str, ...] = (
    "/machine-learning/",
    "/agents/",
    "/mlflow/",
    "/mlflow3/",
    "/genie/",
    "/genie-one/",
    "/genie-code/",
    "/large-language-models/",
    "/release-notes/",
    "/partner-connect/",
    "/partners/",
    "/marketplace/",
    "/admin/",
    "/api/",
    "/reference/",
    "/resources/",
    "/error-messages/",
    "/migration/",
    "/integrations/",
    "/ai-search/",
    "/ai-gateway/",
    "/ai-bi/",
    "/omnigent/",
    "/databricks-ai/",
    "/agent-skills/",
    "/oltp/",
    "/clean-rooms/",
    "/data-sharing/",
    "/opensharing/",
    "/business-semantics/",
    "/dashboards/",
    "/visualizations/",
    "/external-access/",
    "/query/formats/opensharing",
    # Partner-specific Lakeflow Connect connector deep-dives (keep overview pages via include)
    "/ingestion/lakeflow-connect/google-",
    "/ingestion/lakeflow-connect/hubspot",
    "/ingestion/lakeflow-connect/salesforce",
    "/ingestion/lakeflow-connect/service-now",
    "/ingestion/lakeflow-connect/workday",
    "/ingestion/lakeflow-connect/netsuite",
    "/ingestion/lakeflow-connect/zendesk",
    "/ingestion/lakeflow-connect/jira",
    "/ingestion/lakeflow-connect/sharepoint",
    "/ingestion/lakeflow-connect/dynamics",
    "/ingestion/lakeflow-connect/oracle",
    "/ingestion/lakeflow-connect/sql-server",
    "/ingestion/lakeflow-connect/postgresql",
    "/ingestion/lakeflow-connect/mysql",
    "/ingestion/lakeflow-connect/db2",
    "/ingestion/lakeflow-connect/teradata",
    "/ingestion/lakeflow-connect/ga4",
    "/ingestion/lakeflow-connect/google_analytics",
    "/ingestion/lakeflow-connect/google-analytics",
)

CLOUD_SEGMENTS = frozenset({"aws", "azure", "gcp"})
LINK_RE = re.compile(r"^\s*-\s*\[([^\]]+)\]\((https?://[^)\s]+)\)(?:\s*-\s*(.*))?$", re.M)
PRIORITY_START = 100
DOCS_LOCAL_DIR = "data/raw/databricks/docs/official"


def fetch_text(url: str, *, timeout: float = 60.0) -> str:
    """Fetch a URL and return decoded text."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "edge-slm-doc-discovery/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def normalize_docs_url(url: str) -> str | None:
    """
    Normalize a docs.databricks.com URL to a cloud-agnostic form.

    Strips trailing slashes (except root), fragments, and query strings.
    Collapses /aws|azure|gcp/en/... to /... when present.
    Returns None for non-docs hosts.
    """
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"}:
        return None
    host = (parsed.netloc or "").lower()
    if host not in {"docs.databricks.com", "www.docs.databricks.com"}:
        return None

    parts = [p for p in parsed.path.split("/") if p]
    # /aws/en/delta/tutorial -> /delta/tutorial
    if len(parts) >= 2 and parts[0] in CLOUD_SEGMENTS and parts[1] == "en":
        parts = parts[2:]
    # /en/delta/tutorial -> /delta/tutorial
    elif parts and parts[0] == "en":
        parts = parts[1:]

    path = "/" + "/".join(parts) if parts else "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return urlunparse(("https", "docs.databricks.com", path, "", "", ""))


def path_of(url: str) -> str:
    return urlparse(url).path or "/"


def is_excluded(path: str) -> bool:
    lowered = path.lower()
    return any(lowered == prefix.rstrip("/") or lowered.startswith(prefix) for prefix in EXCLUDE_PREFIXES)


def match_topic_buckets(path: str) -> list[str]:
    """Return topic bucket ids whose include prefixes match the URL path."""
    lowered = path.lower()
    matched: list[str] = []
    for bucket_id, prefixes in TOPIC_PREFIXES.items():
        for prefix in prefixes:
            prefix_l = prefix.lower()
            if lowered == prefix_l.rstrip("/") or lowered.startswith(prefix_l):
                matched.append(bucket_id)
                break
    return matched


def slug_from_url(url: str, *, max_len: int = 60) -> str:
    """Build a stable slug from a normalized docs URL path."""
    path = path_of(url).strip("/")
    if not path:
        return "index"
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", path).strip("_").lower()
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("_")
    return slug or "index"


def source_id_for_url(url: str) -> str:
    return f"doc_{slug_from_url(url)}"


def local_path_for_url(url: str) -> str:
    return f"{DOCS_LOCAL_DIR}/{slug_from_url(url)}.md"


def parse_llms_txt(text: str) -> list[dict[str, str]]:
    """Parse llms.txt markdown into {title, url, description} records."""
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for match in LINK_RE.finditer(text):
        title, url, description = match.group(1), match.group(2), (match.group(3) or "").strip()
        normalized = normalize_docs_url(url)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        records.append(
            {
                "title": title.strip(),
                "url": normalized,
                "description": description,
            }
        )
    return records


def parse_sitemap_xml(text: str) -> list[str]:
    """Extract <loc> URLs from a sitemap (handles default XML namespaces)."""
    root = ET.fromstring(text)
    urls: list[str] = []
    for elem in root.iter():
        if elem.tag.endswith("loc") and elem.text:
            urls.append(elem.text.strip())
    return urls


def discover_from_llms_txt(text: str) -> list[dict[str, Any]]:
    """Filter llms.txt entries to in-scope documentation candidates."""
    candidates: list[dict[str, Any]] = []
    for record in parse_llms_txt(text):
        path = path_of(record["url"])
        if is_excluded(path):
            continue
        buckets = match_topic_buckets(path)
        if not buckets:
            continue
        candidates.append(
            {
                "title": record["title"],
                "original_url": record["url"],
                "description": record["description"]
                or f"Official Databricks documentation: {record['title']}",
                "topic_bucket_ids": buckets,
                "source_id": source_id_for_url(record["url"]),
                "local_path": local_path_for_url(record["url"]),
                "resource_type": "documentation",
                "discovery_source": "llms.txt",
            }
        )
    return candidates


def discover_from_sitemap_urls(urls: list[str]) -> list[dict[str, Any]]:
    """Filter raw sitemap URLs to in-scope documentation candidates."""
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_url in urls:
        normalized = normalize_docs_url(raw_url)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        path = path_of(normalized)
        if is_excluded(path):
            continue
        buckets = match_topic_buckets(path)
        if not buckets:
            continue
        title = path.strip("/").replace("/", " / ") or "Databricks documentation"
        candidates.append(
            {
                "title": title,
                "original_url": normalized,
                "description": f"Official Databricks documentation page: {path}",
                "topic_bucket_ids": buckets,
                "source_id": source_id_for_url(normalized),
                "local_path": local_path_for_url(normalized),
                "resource_type": "documentation",
                "discovery_source": "sitemap",
            }
        )
    return candidates


def merge_candidates(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate candidates by original_url, preferring earlier groups."""
    merged: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    seen_ids: set[str] = set()
    for group in groups:
        for candidate in group:
            url = candidate["original_url"]
            source_id = candidate["source_id"]
            if url in seen_urls or source_id in seen_ids:
                continue
            seen_urls.add(url)
            seen_ids.add(source_id)
            merged.append(candidate)
    return merged


def existing_manifest_urls(manifest: dict[str, Any]) -> set[str]:
    urls: set[str] = set()
    for source in manifest.get("sources", []):
        raw = source.get("original_url")
        if not raw:
            continue
        normalized = normalize_docs_url(raw) or raw.rstrip("/")
        urls.add(normalized)
    return urls


def existing_manifest_ids(manifest: dict[str, Any]) -> set[str]:
    return {s["source_id"] for s in manifest.get("sources", []) if "source_id" in s}


def filter_new_candidates(
    candidates: list[dict[str, Any]],
    *,
    existing_urls: set[str],
    existing_ids: set[str],
) -> list[dict[str, Any]]:
    """Drop candidates already present in the manifest (by URL or source_id)."""
    return [
        candidate
        for candidate in candidates
        if candidate["original_url"] not in existing_urls
        and candidate["source_id"] not in existing_ids
    ]


def candidate_to_manifest_source(candidate: dict[str, Any], *, priority: int) -> dict[str, Any]:
    return {
        "source_id": candidate["source_id"],
        "title": candidate["title"],
        "resource_type": candidate.get("resource_type", "documentation"),
        "original_url": candidate["original_url"],
        "local_path": candidate["local_path"],
        "topic_bucket_ids": list(candidate["topic_bucket_ids"]),
        "description": candidate["description"],
        "priority": priority,
        "split": "train",
        "enabled": True,
        "notes": f"Auto-discovered from {candidate.get('discovery_source', 'docs index')}.",
    }


def apply_candidates_to_manifest(
    manifest: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    priority_start: int = PRIORITY_START,
) -> tuple[dict[str, Any], int]:
    """
    Append new documentation sources to a manifest copy.

    Returns (updated_manifest, number_added).
    """
    updated = json.loads(json.dumps(manifest))  # deep copy via JSON
    existing_urls = existing_manifest_urls(updated)
    existing_ids = existing_manifest_ids(updated)
    to_add = filter_new_candidates(
        candidates, existing_urls=existing_urls, existing_ids=existing_ids
    )
    for index, candidate in enumerate(to_add):
        # Ensure unique source_id if collision after filter edge cases
        source_id = candidate["source_id"]
        if source_id in existing_ids:
            suffix = 2
            while f"{source_id}_{suffix}" in existing_ids:
                suffix += 1
            candidate = {**candidate, "source_id": f"{source_id}_{suffix}"}
        source = candidate_to_manifest_source(
            candidate, priority=priority_start + index
        )
        updated["sources"].append(source)
        existing_ids.add(source["source_id"])
        existing_urls.add(source["original_url"])
    return updated, len(to_add)


def load_discovery_sources(
    *,
    llms_txt: str | None,
    llms_txt_url: str,
    sitemap_urls: tuple[str, ...],
    try_sitemap: bool,
) -> list[dict[str, Any]]:
    """Load and merge candidates from llms.txt and optional sitemaps."""
    if llms_txt is not None:
        llms_body = llms_txt
    else:
        logger.info("Fetching llms.txt from %s", llms_txt_url)
        llms_body = fetch_text(llms_txt_url)

    llms_candidates = discover_from_llms_txt(llms_body)
    logger.info("llms.txt yielded %d in-scope candidates", len(llms_candidates))

    sitemap_candidates: list[dict[str, Any]] = []
    if try_sitemap:
        for sitemap_url in sitemap_urls:
            try:
                logger.info("Trying sitemap %s", sitemap_url)
                body = fetch_text(sitemap_url)
                raw_urls = parse_sitemap_xml(body)
                batch = discover_from_sitemap_urls(raw_urls)
                logger.info("Sitemap %s yielded %d in-scope URLs", sitemap_url, len(batch))
                sitemap_candidates.extend(batch)
            except (urllib.error.URLError, urllib.error.HTTPError, ET.ParseError, TimeoutError) as error:
                logger.warning("Sitemap unavailable (%s): %s", sitemap_url, error)
            except Exception as error:  # noqa: BLE001 — soft-fail discovery
                logger.warning("Sitemap failed (%s): %s", sitemap_url, error)

    return merge_candidates(llms_candidates, sitemap_candidates)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Candidate list JSON path (default: data/manifests/generated/databricks_docs_candidates.json)",
    )
    parser.add_argument(
        "--llms-txt",
        type=Path,
        default=None,
        help="Use a local llms.txt file instead of fetching",
    )
    parser.add_argument("--llms-txt-url", default=DEFAULT_LLMS_TXT_URL)
    parser.add_argument(
        "--no-sitemap",
        action="store_true",
        help="Skip sitemap fallback discovery",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Merge new candidates into the manifest",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap number of new candidates written/applied",
    )
    parser.add_argument(
        "--priority-start",
        type=int,
        default=PRIORITY_START,
        help="Starting priority for applied sources",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    llms_text = args.llms_txt.read_text(encoding="utf-8") if args.llms_txt else None
    candidates = load_discovery_sources(
        llms_txt=llms_text,
        llms_txt_url=args.llms_txt_url,
        sitemap_urls=DEFAULT_SITEMAP_URLS,
        try_sitemap=not args.no_sitemap,
    )

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    existing_urls = existing_manifest_urls(manifest)
    existing_ids = existing_manifest_ids(manifest)
    new_candidates = filter_new_candidates(
        candidates, existing_urls=existing_urls, existing_ids=existing_ids
    )
    if args.limit is not None:
        new_candidates = new_candidates[: max(0, args.limit)]

    output_path = args.output or (
        DEFAULT_CANDIDATES_DIR / "databricks_docs_candidates.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": "llms.txt (+ optional sitemap)",
        "total_discovered": len(candidates),
        "new_candidates": len(new_candidates),
        "candidates": new_candidates,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %d new candidates to %s", len(new_candidates), output_path)

    if args.apply:
        updated, added = apply_candidates_to_manifest(
            manifest,
            new_candidates,
            priority_start=args.priority_start,
        )
        args.manifest.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
        logger.info("Applied %d sources to %s (now %d total)", added, args.manifest, len(updated["sources"]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
