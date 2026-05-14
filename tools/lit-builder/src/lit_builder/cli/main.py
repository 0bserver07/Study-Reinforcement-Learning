"""CLI for lit-builder.

Designed to be called by humans and by other coding agents (Codex, Claude Code).
Each command takes positional args, prints structured output, and exits non-zero
on error.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from lit_builder import config
from lit_builder.data import (
    connect,
    fetch as data_fetch,
    get_digest,
    get_paper,
    get_score,
    get_scored,
    ingest as data_ingest,
)
from lit_builder.filter import filter_venue
from lit_builder.render import (
    discoveries_export,
    paper_to_markdown,
    render_venue,
    write_mkdocs_config,
)
from lit_builder.score import deepen_paper, score_venue

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    help="ICLR / NeurIPS literature builder. Pipeline: fetch -> ingest -> filter -> score -> render.",
)
console = Console()


def _print_kv(d: dict) -> None:
    """Print a result dict as key=value lines (machine-readable)."""
    for k, v in d.items():
        console.print(f"{k}={v}")


@app.command()
def fetch(
    venue: str = typer.Argument(..., help="e.g. iclr2026, nips2025"),
    force: bool = typer.Option(False, "--force", help="Re-download even if cached"),
) -> None:
    """Download papercopilot JSON for a venue to data/raw/."""
    config.ensure_dirs()
    path = data_fetch(venue, force=force)
    _print_kv({"venue": venue, "path": str(path), "size_bytes": path.stat().st_size})


@app.command()
def ingest(
    venue: str = typer.Argument(...),
) -> None:
    """Ingest the fetched JSON into the SQLite DB."""
    config.ensure_dirs()
    counts = data_ingest(venue)
    _print_kv(
        {
            "venue": venue,
            "total": counts.get("total", 0),
            "accepted": counts.get("accepted", 0),
        }
    )
    console.print("[dim]by_status:[/dim]", json.dumps(counts.get("by_status", {})))


@app.command(name="filter")
def filter_cmd(
    venue: str = typer.Argument(...),
    status_like: str = typer.Option(
        "%", "--status", help="SQL LIKE pattern on the `status` column. Default '%' matches all."
    ),
) -> None:
    """Run the keyword pre-filter. Writes ScoreResult rows with keyword_hits set."""
    config.ensure_dirs()
    counts = filter_venue(venue, status_like=status_like)
    _print_kv({"venue": venue, **counts})


@app.command()
def score(
    venue: str = typer.Argument(...),
    limit: Optional[int] = typer.Option(None, "--limit", help="Max papers to score this run"),
    only_unscored: bool = typer.Option(
        True, "--only-unscored/--rescore", help="Skip rows that already have an llm_score"
    ),
) -> None:
    """LLM triage pass: Haiku scores keyword survivors 0–3."""
    config.ensure_dirs()
    counts = score_venue(venue, limit=limit, only_unscored=only_unscored)
    _print_kv({"venue": venue, **counts})


@app.command()
def deepen(
    venue: str = typer.Argument(...),
    paper_id: str = typer.Argument(...),
) -> None:
    """On-demand deep digest for a single paper. Writes to digests table."""
    config.ensure_dirs()
    digest = deepen_paper(venue, paper_id)
    _print_kv(
        {
            "venue": venue,
            "paper_id": paper_id,
            "method_category": digest.method_category,
            "claimed_speedup": digest.claimed_speedup,
            "hardware_target": digest.hardware_target,
            "tags": ",".join(digest.sutro_relevance_tags),
        }
    )
    console.print()
    console.print(digest.summary)


@app.command(name="list")
def list_cmd(
    venue: str = typer.Argument(...),
    min_score: int = typer.Option(2, "--min-score", help="Show papers with llm_score >= this"),
    limit: int = typer.Option(50, "--limit"),
) -> None:
    """List scored papers, ordered by relevance score."""
    conn = connect()
    try:
        rows = get_scored(conn, venue, min_score=min_score, limit=limit)
    finally:
        conn.close()

    if not rows:
        console.print(f"[yellow]No papers scored >= {min_score} for {venue}[/yellow]")
        raise typer.Exit(code=0)

    table = Table(title=f"{venue} — score >= {min_score}")
    table.add_column("score", justify="right")
    table.add_column("id")
    table.add_column("title", overflow="fold")
    table.add_column("reason", overflow="fold")

    for paper, sr in rows:
        table.add_row(
            str(sr.llm_score) if sr.llm_score is not None else "-",
            paper.id,
            paper.title,
            (sr.llm_reason or "")[:120],
        )
    console.print(table)
    console.print(f"[dim]rows={len(rows)}[/dim]")


@app.command()
def show(
    venue: str = typer.Argument(...),
    paper_id: str = typer.Argument(...),
) -> None:
    """Show full markdown for a paper (paper + score + digest if present)."""
    conn = connect()
    try:
        paper = get_paper(conn, venue, paper_id)
        sr = get_score(conn, venue, paper_id)
        dg = get_digest(conn, venue, paper_id)
    finally:
        conn.close()

    if paper is None:
        console.print(f"[red]not found:[/red] venue={venue} id={paper_id}")
        raise typer.Exit(code=1)

    md = paper_to_markdown(paper, sr, dg)
    sys.stdout.write(md)
    sys.stdout.write("\n")


@app.command()
def render(
    venue: str = typer.Argument(...),
    min_score: int = typer.Option(0, "--min-score"),
    write_config: bool = typer.Option(True, "--mkdocs/--no-mkdocs", help="Write mkdocs.yml if absent"),
) -> None:
    """Write per-paper markdown + venue index + topic pages to docs/."""
    config.ensure_dirs()
    counts = render_venue(venue, min_score=min_score)
    _print_kv({"venue": venue, **counts})
    if write_config:
        path = write_mkdocs_config()
        console.print(f"[dim]mkdocs config: {path}[/dim]")


@app.command()
def export_discoveries(
    venue: str = typer.Argument(...),
    min_score: int = typer.Option(3, "--min-score"),
    out: Optional[Path] = typer.Option(None, "--out", help="Write to file instead of stdout"),
) -> None:
    """Emit a DISCOVERIES.md-style snippet for the most relevant papers."""
    text = discoveries_export(venue, min_score=min_score)
    if out is not None:
        out.write_text(text)
        console.print(f"[green]wrote {len(text)} bytes to {out}[/green]")
    else:
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")


@app.command()
def serve() -> None:
    """Run `mkdocs serve` from the project root."""
    if shutil.which("mkdocs") is None:
        console.print("[red]mkdocs not installed. Try: pip install mkdocs mkdocs-material[/red]")
        raise typer.Exit(code=1)
    write_mkdocs_config()
    subprocess.run(["mkdocs", "serve"], check=False, cwd=str(config.ROOT))


@app.command()
def status(venue: Optional[str] = typer.Argument(None)) -> None:
    """Print DB state. With no arg, summarize all venues; with one, deep-dive."""
    conn = connect()
    try:
        if venue is None:
            cursor = conn.execute(
                "SELECT venue, COUNT(*) FROM papers GROUP BY venue ORDER BY venue"
            )
            for v, n in cursor.fetchall():
                cursor2 = conn.execute(
                    "SELECT COUNT(*), SUM(CASE WHEN llm_score IS NOT NULL THEN 1 ELSE 0 END) "
                    "FROM scores WHERE venue=?",
                    (v,),
                )
                filtered, scored = cursor2.fetchone()
                console.print(f"{v}: papers={n} filtered={filtered or 0} scored={scored or 0}")
        else:
            cursor = conn.execute(
                "SELECT status, COUNT(*) FROM papers WHERE venue=? GROUP BY status ORDER BY 2 DESC",
                (venue,),
            )
            table = Table(title=f"{venue} status breakdown")
            table.add_column("status")
            table.add_column("count", justify="right")
            for s, n in cursor.fetchall():
                table.add_row(s or "(empty)", str(n))
            console.print(table)
            cursor = conn.execute(
                "SELECT llm_score, COUNT(*) FROM scores WHERE venue=? GROUP BY llm_score ORDER BY 1",
                (venue,),
            )
            score_table = Table(title=f"{venue} llm_score distribution")
            score_table.add_column("llm_score")
            score_table.add_column("count", justify="right")
            for s, n in cursor.fetchall():
                score_table.add_row(str(s) if s is not None else "(unscored)", str(n))
            console.print(score_table)
    finally:
        conn.close()


if __name__ == "__main__":
    app()
