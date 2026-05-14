"""Match keyword terms against a paper's searchable text."""

from __future__ import annotations

import re
from functools import lru_cache

from lit_builder.models import Paper


def searchable_text(paper: Paper) -> str:
    """Return the lowercased, newline-joined haystack for ``paper``.

    Includes title, tldr, abstract, keywords, and primary_area. Missing
    fields are coerced to the empty string.
    """
    parts = [
        paper.title or "",
        paper.tldr or "",
        paper.abstract or "",
        paper.keywords or "",
        paper.primary_area or "",
    ]
    return "\n".join(p.lower() for p in parts)


@lru_cache(maxsize=4096)
def _compile_term(term: str) -> re.Pattern[str]:
    """Compile a word-boundary regex for ``term``.

    A term matches if it appears bounded on each side by either a
    non-alphanumeric character OR the start/end of the haystack. We hand-roll
    the boundary instead of using ``\\b`` because ``\\b`` does not fire next
    to non-word characters that some terms contain (e.g. the ``-`` in
    ``low-precision`` or the ``/`` in ``i/o complexity``).
    """
    escaped = re.escape(term.lower())
    pattern = rf"(?:^|(?<=[^a-z0-9])){escaped}(?:(?=[^a-z0-9])|$)"
    return re.compile(pattern, re.IGNORECASE)


def find_hits(text: str, terms: list[tuple[str, str]]) -> list[str]:
    """Return matched term strings, deduped, in the order ``terms`` lists them.

    ``text`` is expected to already be lowercased (see ``searchable_text``).
    Match is case-insensitive substring with non-alphanumeric word boundaries,
    so e.g. ``moe`` matches ``MoE layer`` but not ``moebius``.
    """
    seen: set[str] = set()
    hits: list[str] = []
    for _group, term in terms:
        if term in seen:
            continue
        if _compile_term(term).search(text):
            hits.append(term)
            seen.add(term)
    return hits


def filter_paper(paper: Paper, terms: list[tuple[str, str]]) -> list[str]:
    """Convenience: build the haystack for ``paper`` and return its hits."""
    return find_hits(searchable_text(paper), terms)
