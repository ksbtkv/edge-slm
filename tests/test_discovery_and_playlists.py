"""Tests for docs discovery and playlist expansion helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import discover_databricks_docs as discover  # noqa: E402
import fetch_transcripts as transcripts  # noqa: E402


SAMPLE_LLMS = """# Databricks Documentation

## Overview and getting started
- [What is Databricks?](https://docs.databricks.com/introduction/) - Learn what is Databricks.
- [Build an ETL pipeline](https://docs.databricks.com/getting-started/data-pipeline-get-started) - Lakeflow pipelines.

## Data engineering
- [Structured streaming](https://docs.databricks.com/structured-streaming/concepts) - Incremental workloads.
- [Auto Loader](https://docs.databricks.com/ingestion/cloud-object-storage/auto-loader/) - Ingest from cloud storage.
- [Lakeflow Jobs](https://docs.databricks.com/jobs/) - Orchestrate workflows.

## Machine learning and AI
- [AI and machine learning overview](https://docs.databricks.com/machine-learning/) - Out of scope.
- [Build AI agents](https://docs.databricks.com/agents/) - Out of scope.

## Data governance and security
- [Unity Catalog overview](https://docs.databricks.com/data-governance/unity-catalog/) - Governance.
- [Security overview](https://docs.databricks.com/security/) - Security.

## Additional resources
- [Release notes](https://docs.databricks.com/release-notes/) - Excluded.
- [Pricing](https://www.databricks.com/product/pricing) - Non-docs host.
"""


def test_normalize_docs_url_strips_cloud_and_trailing_slash() -> None:
    assert (
        discover.normalize_docs_url("https://docs.databricks.com/aws/en/delta/tutorial/")
        == "https://docs.databricks.com/delta/tutorial"
    )
    assert (
        discover.normalize_docs_url("https://docs.databricks.com/en/jobs/")
        == "https://docs.databricks.com/jobs"
    )
    assert discover.normalize_docs_url("https://www.databricks.com/learn") is None


def test_match_and_exclude_prefixes() -> None:
    assert "lakehouse_delta" in discover.match_topic_buckets("/delta/tutorial")
    assert "ingestion_incremental" in discover.match_topic_buckets(
        "/structured-streaming/concepts"
    )
    assert discover.match_topic_buckets("/machine-learning/automl") == []
    assert discover.is_excluded("/machine-learning/automl")
    assert discover.is_excluded("/release-notes/runtime")
    assert not discover.is_excluded("/delta/tutorial")


def test_source_id_slug_stability() -> None:
    url = "https://docs.databricks.com/delta/tutorial"
    assert discover.source_id_for_url(url) == "doc_delta_tutorial"
    assert discover.local_path_for_url(url).endswith("delta_tutorial.md")


def test_discover_from_llms_txt_filters_scope() -> None:
    candidates = discover.discover_from_llms_txt(SAMPLE_LLMS)
    urls = {c["original_url"] for c in candidates}
    assert "https://docs.databricks.com/introduction" in urls
    assert "https://docs.databricks.com/jobs" in urls
    assert "https://docs.databricks.com/data-governance/unity-catalog" in urls
    assert "https://docs.databricks.com/machine-learning" not in urls
    assert "https://docs.databricks.com/agents" not in urls
    assert "https://docs.databricks.com/release-notes" not in urls
    assert all(c["topic_bucket_ids"] for c in candidates)


def test_filter_and_apply_dedupes_existing_manifest() -> None:
    candidates = discover.discover_from_llms_txt(SAMPLE_LLMS)
    manifest = {
        "manifest_version": "1.0",
        "pack_id": "test",
        "title": "Test",
        "description": "Test",
        "domain": "Test",
        "topic_buckets": [{"id": "databricks_basics", "label": "Basics"}],
        "sources": [
            {
                "source_id": "doc_aws_legacy",
                "title": "Legacy AWS overview",
                "resource_type": "documentation",
                "original_url": "https://docs.databricks.com/aws/en/introduction/",
                "local_path": "data/raw/x.md",
                "topic_bucket_ids": ["databricks_basics"],
                "priority": 1,
                "split": "train",
                "enabled": True,
            }
        ],
    }
    existing_urls = discover.existing_manifest_urls(manifest)
    existing_ids = discover.existing_manifest_ids(manifest)
    # Cloud URL normalizes to /introduction — should dedupe that candidate
    assert "https://docs.databricks.com/introduction" in existing_urls

    new = discover.filter_new_candidates(
        candidates, existing_urls=existing_urls, existing_ids=existing_ids
    )
    assert all(c["original_url"] != "https://docs.databricks.com/introduction" for c in new)

    updated, added = discover.apply_candidates_to_manifest(manifest, new, priority_start=100)
    assert added == len(new)
    assert len(updated["sources"]) == 1 + added
    assert updated["sources"][-1]["priority"] >= 100
    assert updated["sources"][-1]["enabled"] is True


def test_playlist_child_source_id_and_build() -> None:
    playlist = {
        "source_id": "video_associate_playlist",
        "title": "Associate playlist",
        "topic_bucket_ids": ["databricks_basics", "lakehouse_delta"],
        "priority": 15,
        "split": "unassigned",
    }
    entry = {
        "id": "abc123",
        "title": "Lakehouse Platform Overview!",
        "url": "https://www.youtube.com/watch?v=abc123",
        "duration": 600,
    }
    child = transcripts.build_playlist_child_source(playlist, position=1, entry=entry)
    assert child["source_id"] == "pl_associate_01_lakehouse_platform_overview"
    assert child["resource_type"] == "video_transcript"
    assert child["topic_bucket_ids"] == ["databricks_basics", "lakehouse_delta"]
    assert child["parent_playlist_id"] == "video_associate_playlist"
    assert child["local_path"].endswith(".txt")


def test_expand_playlist_sources_adds_children_and_disables_parent(tmp_path: Path) -> None:
    playlist_source = {
        "source_id": "video_spark_de_playlist",
        "title": "Spark DE playlist",
        "resource_type": "playlist",
        "original_url": "https://www.youtube.com/playlist?list=TEST",
        "local_path": "data/raw/databricks/videos/spark_index.md",
        "topic_bucket_ids": ["spark_sql_pyspark"],
        "priority": 20,
        "split": "train",
        "enabled": True,
    }
    # write_playlist_index writes under PROJECT_ROOT; redirect via local_path in tmp
    # by monkeypatching PROJECT_ROOT on the module for this test.
    index_rel = "videos/spark_index.md"
    playlist_source["local_path"] = index_rel

    manifest = {
        "manifest_version": "1.0",
        "pack_id": "test",
        "title": "Test",
        "description": "Test",
        "domain": "Test",
        "topic_buckets": [{"id": "spark_sql_pyspark", "label": "Spark"}],
        "sources": [playlist_source],
    }

    fake_playlist = {
        "title": "Fake playlist",
        "entries": [
            {
                "id": "vid1",
                "title": "Intro to Spark",
                "url": "https://www.youtube.com/watch?v=vid1",
            },
            {
                "id": "vid2",
                "title": "DataFrames",
                "url": "https://www.youtube.com/watch?v=vid2",
            },
        ],
    }

    original_root = transcripts.PROJECT_ROOT
    transcripts.PROJECT_ROOT = tmp_path
    try:
        updated, added = transcripts.expand_playlist_sources(
            manifest,
            fetch_entries=lambda _url: fake_playlist,
        )
    finally:
        transcripts.PROJECT_ROOT = original_root

    assert len(added) == 2
    assert len(updated["sources"]) == 3
    parent = next(s for s in updated["sources"] if s["source_id"] == "video_spark_de_playlist")
    assert parent["enabled"] is False
    children = [s for s in updated["sources"] if s["resource_type"] == "video_transcript"]
    assert len(children) == 2
    assert children[0]["source_id"].startswith("pl_spark_de_01_")
    assert (tmp_path / index_rel).exists()
