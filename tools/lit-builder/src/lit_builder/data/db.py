"""SQLite access layer.

Connection lifecycle is the caller's responsibility. All write helpers use
INSERT OR REPLACE so reruns are idempotent.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from .. import config
from ..models import DigestResult, Paper, ScoreResult, SCHEMA_SQL


# ---------- connection ------------------------------------------------------


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    """Open the SQLite DB, apply the schema, set sane pragmas, return conn.

    Caller is responsible for closing.
    """
    path = db_path or config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


# ---------- writes ----------------------------------------------------------


_PAPER_COLS = (
    "id",
    "venue",
    "title",
    "abstract",
    "tldr",
    "keywords",
    "primary_area",
    "authors",
    "status",
    "track",
    "site_url",
    "github_url",
    "project_url",
    "rating_avg",
    "citation_count",
    "raw_json",
)
_PAPER_INSERT_SQL = (
    f"INSERT OR REPLACE INTO papers ({', '.join(_PAPER_COLS)}) "
    f"VALUES ({', '.join('?' for _ in _PAPER_COLS)})"
)


def _paper_row(p: Paper) -> tuple:
    return (
        p.id,
        p.venue,
        p.title,
        p.abstract,
        p.tldr,
        p.keywords,
        p.primary_area,
        p.authors,
        p.status,
        p.track,
        p.site_url,
        p.github_url,
        p.project_url,
        p.rating_avg,
        p.citation_count,
        p.raw_json,
    )


def upsert_papers(conn: sqlite3.Connection, papers: Iterable[Paper]) -> int:
    """Bulk INSERT OR REPLACE into `papers`. Returns rows written."""
    rows = [_paper_row(p) for p in papers]
    if not rows:
        return 0
    with conn:
        conn.executemany(_PAPER_INSERT_SQL, rows)
    return len(rows)


def upsert_score(conn: sqlite3.Connection, score: ScoreResult) -> None:
    """Insert or replace a single score row."""
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO scores "
            "(paper_id, venue, keyword_hits, llm_score, llm_reason, scored_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                score.paper_id,
                score.venue,
                json.dumps(list(score.keyword_hits)),
                score.llm_score,
                score.llm_reason,
                score.scored_at,
            ),
        )


def upsert_digest(conn: sqlite3.Connection, digest: DigestResult) -> None:
    """Insert or replace a single digest row."""
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO digests "
            "(paper_id, venue, method_category, claimed_speedup, hardware_target, "
            " sutro_relevance_tags, summary, digested_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                digest.paper_id,
                digest.venue,
                digest.method_category,
                digest.claimed_speedup,
                digest.hardware_target,
                json.dumps(list(digest.sutro_relevance_tags)),
                digest.summary,
                digest.digested_at,
            ),
        )


# ---------- reads -----------------------------------------------------------


def _row_to_paper(r: sqlite3.Row) -> Paper:
    return Paper(
        id=r["id"],
        venue=r["venue"],
        title=r["title"],
        abstract=r["abstract"],
        tldr=r["tldr"],
        keywords=r["keywords"],
        primary_area=r["primary_area"],
        authors=r["authors"],
        status=r["status"],
        track=r["track"],
        site_url=r["site_url"],
        github_url=r["github_url"],
        project_url=r["project_url"],
        rating_avg=r["rating_avg"],
        citation_count=r["citation_count"],
        raw_json=r["raw_json"],
    )


def _row_to_score(r: sqlite3.Row) -> ScoreResult:
    raw_hits = r["keyword_hits"] or "[]"
    try:
        hits = json.loads(raw_hits)
    except json.JSONDecodeError:
        hits = []
    return ScoreResult(
        paper_id=r["paper_id"],
        venue=r["venue"],
        keyword_hits=list(hits),
        llm_score=r["llm_score"],
        llm_reason=r["llm_reason"] or "",
        scored_at=r["scored_at"] or "",
    )


def _row_to_digest(r: sqlite3.Row) -> DigestResult:
    raw_tags = r["sutro_relevance_tags"] or "[]"
    try:
        tags = json.loads(raw_tags)
    except json.JSONDecodeError:
        tags = []
    return DigestResult(
        paper_id=r["paper_id"],
        venue=r["venue"],
        method_category=r["method_category"] or "",
        claimed_speedup=r["claimed_speedup"] or "",
        hardware_target=r["hardware_target"] or "",
        sutro_relevance_tags=list(tags),
        summary=r["summary"] or "",
        digested_at=r["digested_at"] or "",
    )


def get_papers(
    conn: sqlite3.Connection,
    venue: str,
    *,
    status_like: str = "Accept%",
    limit: int | None = None,
) -> list[Paper]:
    """Return papers in `venue` whose status matches `status_like` (SQL LIKE)."""
    sql = "SELECT * FROM papers WHERE venue = ? AND status LIKE ?"
    params: list = [venue, status_like]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    cur = conn.execute(sql, params)
    return [_row_to_paper(r) for r in cur.fetchall()]


def get_paper(conn: sqlite3.Connection, venue: str, paper_id: str) -> Paper | None:
    cur = conn.execute(
        "SELECT * FROM papers WHERE venue = ? AND id = ?",
        (venue, paper_id),
    )
    row = cur.fetchone()
    return _row_to_paper(row) if row else None


def get_scored(
    conn: sqlite3.Connection,
    venue: str,
    *,
    min_score: int = 0,
    limit: int | None = None,
) -> list[tuple[Paper, ScoreResult]]:
    """Return (paper, score) pairs for papers with llm_score >= min_score.

    Ordered by llm_score DESC. Papers without a score are excluded.
    """
    sql = (
        "SELECT p.*, "
        "       s.paper_id AS s_paper_id, s.venue AS s_venue, "
        "       s.keyword_hits AS s_keyword_hits, s.llm_score AS s_llm_score, "
        "       s.llm_reason AS s_llm_reason, s.scored_at AS s_scored_at "
        "FROM papers p "
        "JOIN scores s ON s.venue = p.venue AND s.paper_id = p.id "
        "WHERE p.venue = ? AND COALESCE(s.llm_score, 0) >= ? "
        "ORDER BY s.llm_score DESC, p.id ASC"
    )
    params: list = [venue, min_score]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)

    out: list[tuple[Paper, ScoreResult]] = []
    for r in conn.execute(sql, params).fetchall():
        paper = _row_to_paper(r)
        try:
            hits = json.loads(r["s_keyword_hits"] or "[]")
        except json.JSONDecodeError:
            hits = []
        score = ScoreResult(
            paper_id=r["s_paper_id"],
            venue=r["s_venue"],
            keyword_hits=list(hits),
            llm_score=r["s_llm_score"],
            llm_reason=r["s_llm_reason"] or "",
            scored_at=r["s_scored_at"] or "",
        )
        out.append((paper, score))
    return out


def get_score(
    conn: sqlite3.Connection, venue: str, paper_id: str
) -> ScoreResult | None:
    cur = conn.execute(
        "SELECT * FROM scores WHERE venue = ? AND paper_id = ?",
        (venue, paper_id),
    )
    row = cur.fetchone()
    return _row_to_score(row) if row else None


def get_digest(
    conn: sqlite3.Connection, venue: str, paper_id: str
) -> DigestResult | None:
    cur = conn.execute(
        "SELECT * FROM digests WHERE venue = ? AND paper_id = ?",
        (venue, paper_id),
    )
    row = cur.fetchone()
    return _row_to_digest(row) if row else None
