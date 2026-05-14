"""Single-paper markdown rendering.

Pure functions only -- no DB, no filesystem. Inputs are dataclass rows from
`lit_builder.models`. Output is a markdown string.
"""

from __future__ import annotations

import re

from ..models import DigestResult, Paper, ScoreResult


def _join_authors(authors: str) -> str:
    """Authors are stored semicolon-separated; render as comma-separated."""
    if not authors:
        return ""
    parts = [a.strip() for a in authors.split(";") if a.strip()]
    return ", ".join(parts)


def _format_keywords(keywords: str) -> str:
    if not keywords:
        return ""
    parts = [k.strip() for k in keywords.split(";") if k.strip()]
    return ", ".join(parts)


def slugify(text: str) -> str:
    """Lowercase, replace any non-[a-z0-9]+ run with '-', strip leading/trailing '-'."""
    s = text.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def paper_to_markdown(
    paper: Paper,
    score: ScoreResult | None,
    digest: DigestResult | None,
) -> str:
    """Render a single paper as a standalone markdown document."""
    lines: list[str] = []
    lines.append(f"# {paper.title}")
    lines.append("")
    lines.append(f"**Venue:** {paper.venue} ({paper.status})")
    lines.append(f"**Authors:** {_join_authors(paper.authors)}")
    if paper.site_url:
        lines.append(f"**OpenReview:** [{paper.site_url}]({paper.site_url})")
    if paper.github_url:
        lines.append(f"**GitHub:** [link]({paper.github_url})")
    if paper.project_url:
        lines.append(f"**Project page:** [link]({paper.project_url})")

    if score is not None:
        lines.append("")
        lines.append("## Relevance")
        lines.append("")
        llm_score_str = "—" if score.llm_score is None else str(score.llm_score)
        reason = score.llm_reason or "(no reason given)"
        lines.append(f"**LLM score:** {llm_score_str}/3 — {reason}")
        hits = ", ".join(score.keyword_hits) if score.keyword_hits else ""
        lines.append(f"**Keyword hits:** `{hits}`")

    lines.append("")
    lines.append("## TLDR")
    lines.append(paper.tldr if paper.tldr else "(none provided)")

    lines.append("")
    lines.append("## Abstract")
    lines.append(paper.abstract if paper.abstract else "(none provided)")

    lines.append("")
    lines.append("## Keywords")
    lines.append(_format_keywords(paper.keywords))

    if digest is not None:
        lines.append("")
        lines.append("## Digest")
        lines.append("")
        lines.append(f"- **Method category:** {digest.method_category or '—'}")
        lines.append(f"- **Claimed speedup:** {digest.claimed_speedup or '—'}")
        lines.append(f"- **Hardware target:** {digest.hardware_target or '—'}")
        tags = ", ".join(digest.sutro_relevance_tags) if digest.sutro_relevance_tags else ""
        lines.append(f"- **Sutro tags:** {tags}")
        lines.append("")
        lines.append(digest.summary or "")

    # Trailing newline for clean diff appearance.
    return "\n".join(lines) + "\n"
