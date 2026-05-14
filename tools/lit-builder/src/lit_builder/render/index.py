"""Index pages: per-venue paper index and per-topic pages.

Pure functions. Take in already-loaded rows, produce markdown strings.
"""

from __future__ import annotations

from collections import defaultdict

from ..models import DigestResult, Paper, ScoreResult
from .markdown import slugify


def _bucket_for(score: ScoreResult) -> int:
    """Score 3 / 2 / 1 / 0 (None coerces to 0)."""
    if score.llm_score is None:
        return 0
    return int(score.llm_score)


def _line_for(paper: Paper, score: ScoreResult) -> str:
    bucket = _bucket_for(score)
    snippet = (score.llm_reason or "").strip().replace("\n", " ")
    if len(snippet) > 100:
        snippet = snippet[:100]
    return f"- **{bucket}** — [{paper.title}]({paper.id}.md) — {snippet}"


def render_index(papers_with_scores: list[tuple[Paper, ScoreResult]]) -> str:
    """Render the per-venue index, grouped by score bucket (3, 2, 1, 0)."""
    buckets: dict[int, list[tuple[Paper, ScoreResult]]] = defaultdict(list)
    for paper, score in papers_with_scores:
        buckets[_bucket_for(score)].append((paper, score))

    # Stable order within a bucket: keep insertion order from caller (assumed
    # already ordered, e.g. by get_scored DESC + id ASC).
    lines: list[str] = ["# Papers", ""]

    headers = {3: "Score 3", 2: "Score 2", 1: "Score 1", 0: "Score 0 / unscored"}

    for bucket in (3, 2, 1, 0):
        entries = buckets.get(bucket, [])
        if bucket == 0 and not entries:
            continue
        lines.append(f"## {headers[bucket]}")
        lines.append("")
        if not entries:
            lines.append("_(none)_")
        else:
            for paper, score in entries:
                lines.append(_line_for(paper, score))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_topic_pages(
    papers_with_scores: list[tuple[Paper, ScoreResult]],
    digests_by_id: dict[str, DigestResult],
) -> dict[str, str]:
    """Group papers by `digest.method_category`. Return {filename: markdown}.

    Only papers that have a digest are included. Filename is `{slug}.md`.
    """
    by_category: dict[str, list[tuple[Paper, ScoreResult, DigestResult]]] = defaultdict(list)
    for paper, score in papers_with_scores:
        digest = digests_by_id.get(paper.id)
        if digest is None:
            continue
        category = digest.method_category or "uncategorized"
        by_category[category].append((paper, score, digest))

    out: dict[str, str] = {}
    for category, items in by_category.items():
        slug = slugify(category) or "uncategorized"
        filename = f"{slug}.md"
        lines = [f"# {category}", ""]
        lines.append(f"_{len(items)} paper(s)_")
        lines.append("")
        for paper, score, digest in items:
            summary = (digest.summary or score.llm_reason or "").strip().replace("\n", " ")
            if len(summary) > 200:
                summary = summary[:200]
            bucket = _bucket_for(score)
            lines.append(
                f"- **{bucket}** — [{paper.title}](../papers/{paper.venue}/{paper.id}.md)"
                f" — {summary}"
            )
        out[filename] = "\n".join(lines).rstrip() + "\n"

    return out
