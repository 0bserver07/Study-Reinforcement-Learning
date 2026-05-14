"""Load and flatten keyword groups from the YAML config."""

from __future__ import annotations

from pathlib import Path

import yaml

from lit_builder import config


def load_keywords(path: Path | None = None) -> dict[str, list[str]]:
    """Read the keyword YAML and return ``{group_name: [terms, ...]}``.

    Defaults to ``config.KEYWORDS_PATH``. The YAML is expected to have a
    top-level ``groups`` mapping; each value is a list of strings.
    """
    target = Path(path) if path is not None else config.KEYWORDS_PATH
    with target.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    groups = data.get("groups", {}) or {}
    out: dict[str, list[str]] = {}
    for group_name, terms in groups.items():
        if not terms:
            out[group_name] = []
            continue
        # Coerce to str and strip; keep order from the file.
        out[group_name] = [str(t).strip() for t in terms if str(t).strip()]
    return out


def all_terms(groups: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Flatten groups to ``[(group, term), ...]`` sorted by term length desc.

    Longer terms come first so that substring/word-boundary matches against
    longer phrases are recorded before any shorter sub-phrase steals the slot
    in callers that early-exit. Stable order is otherwise preserved.
    """
    flat: list[tuple[str, str]] = []
    for group_name, terms in groups.items():
        for term in terms:
            flat.append((group_name, term))
    flat.sort(key=lambda gt: len(gt[1]), reverse=True)
    return flat
