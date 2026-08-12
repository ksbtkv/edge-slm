from __future__ import annotations

import pytest

from wizard.model_catalog import CATALOG, get_entry


def test_catalog_is_non_empty():
    assert len(CATALOG) > 0


def test_catalog_names_are_unique():
    names = [entry.name for entry in CATALOG]
    assert len(names) == len(set(names))


def test_catalog_entries_have_required_fields():
    for entry in CATALOG:
        assert entry.name
        assert entry.display_name
        assert entry.ollama_ref
        assert entry.description
        assert entry.approx_download_gb > 0
        assert entry.min_ram_gb > 0


def test_get_entry_returns_matching_entry():
    entry = get_entry(CATALOG[0].name)
    assert entry == CATALOG[0]


def test_get_entry_raises_on_unknown_name():
    with pytest.raises(KeyError):
        get_entry("not-a-real-model")
