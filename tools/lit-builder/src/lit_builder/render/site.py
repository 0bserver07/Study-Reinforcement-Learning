"""Site-level orchestration: read DB, write venue pages, mkdocs config, exports."""

from __future__ import annotations

from pathlib import Path

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from .. import config, data
from .index import render_index, render_topic_pages
from .markdown import paper_to_markdown


_MKDOCS_TEMPLATE = """site_name: lit-builder
site_description: ICLR/NeurIPS literature builder
theme:
  name: material
  features:
    - navigation.tabs
    - navigation.sections
    - content.code.copy
nav:
  - Home: index.md
  - Papers: papers/
  - Topics: topics/
markdown_extensions:
  - admonition
  - toc:
      permalink: true
  - pymdownx.superfences
"""


def render_venue(
    venue: str,
    *,
    db_path: Path | None = None,
    output_dir: Path | None = None,
    min_score: int = 0,
) -> dict[str, int]:
    """Materialize all markdown for a venue.

    Writes:
      - {output_dir}/{venue}/{paper_id}.md for every scored paper.
      - {output_dir}/{venue}/index.md
      - {DOCS_TOPICS}/{venue}/{slug}.md per method category.

    Returns counts: {"papers_written": N, "topics_written": M}.
    """
    out_papers = output_dir or config.DOCS_PAPERS
    venue_dir = out_papers / venue
    venue_dir.mkdir(parents=True, exist_ok=True)

    topics_dir = config.DOCS_TOPICS / venue
    topics_dir.mkdir(parents=True, exist_ok=True)

    conn = data.connect(db_path)
    try:
        scored = data.get_scored(conn, venue, min_score=min_score)

        digests_by_id: dict[str, object] = {}
        papers_written = 0

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(f"render {venue}", total=len(scored))
            for paper, score in scored:
                digest = data.get_digest(conn, venue, paper.id)
                if digest is not None:
                    digests_by_id[paper.id] = digest
                md = paper_to_markdown(paper, score, digest)
                (venue_dir / f"{paper.id}.md").write_text(md, encoding="utf-8")
                papers_written += 1
                progress.advance(task)

        index_md = render_index(scored)
        (venue_dir / "index.md").write_text(index_md, encoding="utf-8")

        topic_pages = render_topic_pages(scored, digests_by_id)  # type: ignore[arg-type]
        for filename, md in topic_pages.items():
            (topics_dir / filename).write_text(md, encoding="utf-8")
    finally:
        conn.close()

    return {"papers_written": papers_written, "topics_written": len(topic_pages)}


def write_mkdocs_config(
    *,
    output_path: Path | None = None,
    force: bool = False,
) -> Path:
    """Write a minimal mkdocs.yml at project root if missing.

    Idempotent: if the file exists, leave it alone unless `force=True`.
    """
    path = output_path or (config.ROOT / "mkdocs.yml")
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_MKDOCS_TEMPLATE, encoding="utf-8")
    return path


def discoveries_export(
    venue: str,
    *,
    db_path: Path | None = None,
    min_score: int = 3,
) -> str:
    """Produce a markdown blob for pasting into a SutroYaro DISCOVERIES.md.

    Each line: `- [{title}]({site_url}) — {one-line digest summary or llm_reason}`.
    Score 3 only by default. Caller decides where to write the result.
    """
    conn = data.connect(db_path)
    try:
        scored = data.get_scored(conn, venue, min_score=min_score)
        lines: list[str] = []
        for paper, score in scored:
            digest = data.get_digest(conn, venue, paper.id)
            if digest is not None and digest.summary:
                blurb = digest.summary
            else:
                blurb = score.llm_reason or ""
            blurb = blurb.strip().replace("\n", " ")
            # Keep the line tight -- one-liner.
            if len(blurb) > 240:
                blurb = blurb[:240]
            url = paper.site_url or ""
            lines.append(f"- [{paper.title}]({url}) — {blurb}")
    finally:
        conn.close()

    if not lines:
        return ""
    return "\n".join(lines) + "\n"
