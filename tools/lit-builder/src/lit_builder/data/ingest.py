"""Ingest: read raw papercopilot JSON, normalize, upsert into SQLite."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from .. import config
from ..models import Paper, PapercopilotRecord
from . import db


def _coerce_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return str(v)


def _coerce_rating(v: Any) -> float | None:
    """`rating_avg` can be [mean, std], a scalar, a string, or missing."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        if not v:
            return None
        try:
            return float(v[0])
        except (TypeError, ValueError):
            return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _coerce_citation(v: Any) -> int | None:
    """`gs_citation` may be int, str, or missing/garbage."""
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        try:
            return int(v)
        except (ValueError, OverflowError):
            return None
    if isinstance(v, str):
        s = v.strip().replace(",", "")
        if not s:
            return None
        try:
            return int(s)
        except ValueError:
            try:
                return int(float(s))
            except ValueError:
                return None
    return None


def normalize(record: PapercopilotRecord, venue: str) -> Paper:
    """Map a raw papercopilot record to our Paper dataclass.

    Pure function. Missing keys default to "" (or None for numerics).
    Stashes the full raw record as JSON in `raw_json` for later forensics.
    """
    return Paper(
        id=_coerce_str(record.get("id")),
        venue=venue,
        title=_coerce_str(record.get("title")),
        abstract=_coerce_str(record.get("abstract")),
        tldr=_coerce_str(record.get("tldr")),
        keywords=_coerce_str(record.get("keywords")),
        primary_area=_coerce_str(record.get("primary_area")),
        authors=_coerce_str(record.get("author")),
        status=_coerce_str(record.get("status")),
        track=_coerce_str(record.get("track")),
        site_url=_coerce_str(record.get("site")),
        github_url=_coerce_str(record.get("github")),
        project_url=_coerce_str(record.get("project")),
        rating_avg=_coerce_rating(record.get("rating_avg")),
        citation_count=_coerce_citation(record.get("gs_citation")),
        raw_json=json.dumps(record, ensure_ascii=False, sort_keys=True),
    )


def _load_raw(venue: str) -> list[PapercopilotRecord]:
    path = config.raw_path(venue)
    if not path.exists():
        raise FileNotFoundError(
            f"raw papercopilot file missing: {path}. "
            f"Run lit_builder.data.fetch.fetch({venue!r}) first."
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"expected a list at {path}, got {type(data).__name__}")
    return data


def ingest(venue: str, *, db_path: Path | None = None) -> dict[str, int]:
    """Load raw JSON for `venue`, normalize, and upsert into the DB.

    Returns: {"total": N, "accepted": M, "by_status": {status: count, ...}}.
    """
    console = Console()
    raw = _load_raw(venue)
    console.print(f"[bold]ingest[/bold] {venue}: {len(raw):,} raw records")

    by_status: Counter[str] = Counter()
    papers: list[Paper] = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as prog:
        task = prog.add_task("normalize", total=len(raw))
        for rec in raw:
            paper = normalize(rec, venue)
            papers.append(paper)
            by_status[paper.status or "(missing)"] += 1
            prog.advance(task)

    conn = db.connect(db_path)
    try:
        # Chunk the upsert so a malformed row can be tracked down quickly.
        BATCH = 2000
        written = 0
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as prog:
            task = prog.add_task("upsert", total=len(papers))
            for i in range(0, len(papers), BATCH):
                chunk = papers[i : i + BATCH]
                written += db.upsert_papers(conn, chunk)
                prog.advance(task, len(chunk))
    finally:
        conn.close()

    accepted = sum(c for s, c in by_status.items() if s.startswith("Accept"))
    console.print(
        f"[green]ingested[/green] {written:,} rows "
        f"({accepted:,} accepted) into {db_path or config.DB_PATH}"
    )

    return {
        "total": len(papers),
        "accepted": accepted,
        "by_status": dict(by_status),
    }
