"""Stage-1 keyword filter pass over a venue's papers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from lit_builder.data import connect, get_papers, upsert_score
from lit_builder.filter.keywords import all_terms, load_keywords
from lit_builder.filter.match import filter_paper
from lit_builder.models import ScoreResult


def filter_venue(
    venue: str,
    *,
    db_path: Path | None = None,
    status_like: str = "Accept%",
) -> dict[str, int]:
    """Run the keyword pre-filter over every paper in ``venue``.

    Loads keywords, scans each paper, and writes a ``ScoreResult`` row for
    every paper that has at least one keyword hit. ``llm_score`` stays
    ``None`` -- the LLM stage fills it in later.

    Returns counts: ``total_scanned``, ``passed``, ``skipped``.
    """
    groups = load_keywords()
    terms = all_terms(groups)

    conn = connect(db_path)
    try:
        papers = get_papers(conn, venue, status_like=status_like)
        total = len(papers)
        passed = 0
        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("hits={task.fields[hits]}"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(
                f"filter {venue}", total=total, hits=0
            )
            for paper in papers:
                hits = filter_paper(paper, terms)
                if hits:
                    score = ScoreResult(
                        paper_id=paper.id,
                        venue=venue,
                        keyword_hits=hits,
                        llm_score=None,
                        llm_reason="",
                        scored_at=now_iso,
                    )
                    upsert_score(conn, score)
                    passed += 1
                progress.update(task, advance=1, hits=passed)
    finally:
        try:
            conn.close()
        except Exception:
            pass

    return {
        "total_scanned": total,
        "passed": passed,
        "skipped": total - passed,
    }
