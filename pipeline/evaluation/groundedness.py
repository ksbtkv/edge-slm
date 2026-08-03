"""
Tier 2 — automated groundedness.

A study note is grounded when the concrete names it mentions (features,
tools, parameters, concepts) actually appear in the source chunk. Names the
model invented won't be found, so the grounded fraction is a cheap,
deterministic hallucination signal.
"""

from __future__ import annotations

import re
from typing import Any


def extract_claimed_names(note: dict[str, Any]) -> list[str]:
    """Concrete names the note claims came from the source."""
    names: list[str] = []
    for feature in note.get("important_features_or_tools", []) or []:
        if isinstance(feature, dict):
            if isinstance(feature.get("name"), str):
                names.append(feature["name"])
            for param in feature.get("important_parameters") or []:
                if isinstance(param, dict) and isinstance(
                    param.get("parameter"), str
                ):
                    names.append(param["parameter"])
    for concept in note.get("key_concepts", []) or []:
        if isinstance(concept, dict) and isinstance(concept.get("concept"), str):
            names.append(concept["concept"])
    return [name for name in (n.strip() for n in names) if name]


def _normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[`'\"()\[\]{}*_]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def name_is_grounded(name: str, source: str) -> bool:
    """
    True if the name (or all of its significant words) occurs in the source.

    Whole-name substring match first; otherwise fall back to word-level
    matching so paraphrased titles like "Delta Lake time travel" count when
    both "delta lake" and "time travel" appear separately.
    """
    norm_name = _normalise(name)
    norm_source = _normalise(source)
    if not norm_name:
        return True
    if norm_name in norm_source:
        return True
    words = [w for w in norm_name.split() if len(w) > 2]
    if not words:
        return False
    return all(word in norm_source for word in words)


def groundedness_score(note: dict[str, Any], source: str) -> dict[str, Any]:
    """Fraction of claimed names found in the source, with the misses."""
    names = extract_claimed_names(note)
    if not names:
        return {"score": 1.0, "claimed": 0, "ungrounded": []}
    ungrounded = [name for name in names if not name_is_grounded(name, source)]
    return {
        "score": (len(names) - len(ungrounded)) / len(names),
        "claimed": len(names),
        "ungrounded": ungrounded,
    }
