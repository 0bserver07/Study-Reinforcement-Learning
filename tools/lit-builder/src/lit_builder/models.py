"""Shared schema. Every module imports from here.

Two layers:
  1. PapercopilotRecord  -- the raw JSON shape from papercopilot/paperlists.
  2. Paper, ScoreResult, DigestResult -- our normalized DB rows.

SQLite schema is defined in DDL below and applied by data.db.connect().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# ---------- raw papercopilot record ----------------------------------------

# Papercopilot JSON: list[dict]. Keys we use (others ignored):
#   id, title, track, status, tldr, abstract, keywords, primary_area,
#   author, site, github, project, rating_avg, gs_citation
PapercopilotRecord = dict[str, Any]


# ---------- normalized rows ------------------------------------------------


@dataclass
class Paper:
    """Normalized paper row. Mirrors the `papers` table."""

    id: str  # papercopilot id (e.g. "00F7BfXLYJ")
    venue: str  # e.g. "iclr2026"
    title: str
    abstract: str
    tldr: str
    keywords: str  # semicolon-separated as in source
    primary_area: str
    authors: str  # semicolon-separated
    status: str  # "Accept (Oral)" | "Accept (Poster)" | "Reject" | "Withdraw" | ...
    track: str
    site_url: str
    github_url: str
    project_url: str
    rating_avg: Optional[float]
    citation_count: Optional[int]
    raw_json: str = ""  # full source row, JSON-encoded, for forensics


@dataclass
class ScoreResult:
    """Output of the keyword + LLM scoring pass. Mirrors `scores` table."""

    paper_id: str
    venue: str
    keyword_hits: list[str] = field(default_factory=list)
    llm_score: Optional[int] = None  # 0-3 from Haiku
    llm_reason: str = ""
    scored_at: str = ""  # ISO timestamp


@dataclass
class DigestResult:
    """Output of `lit deepen <id>`. Mirrors `digests` table."""

    paper_id: str
    venue: str
    method_category: str  # e.g. "quantization", "sparsity", "optimizer"
    claimed_speedup: str  # free-text, e.g. "2.3x on ImageNet"
    hardware_target: str  # e.g. "GPU", "edge", "TPU", ""
    sutro_relevance_tags: list[str] = field(default_factory=list)
    summary: str = ""  # 3-5 sentences for Sutro audience
    digested_at: str = ""


# ---------- SQLite DDL -----------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS papers (
    id           TEXT NOT NULL,
    venue        TEXT NOT NULL,
    title        TEXT NOT NULL,
    abstract     TEXT NOT NULL DEFAULT '',
    tldr         TEXT NOT NULL DEFAULT '',
    keywords     TEXT NOT NULL DEFAULT '',
    primary_area TEXT NOT NULL DEFAULT '',
    authors      TEXT NOT NULL DEFAULT '',
    status       TEXT NOT NULL DEFAULT '',
    track        TEXT NOT NULL DEFAULT '',
    site_url     TEXT NOT NULL DEFAULT '',
    github_url   TEXT NOT NULL DEFAULT '',
    project_url  TEXT NOT NULL DEFAULT '',
    rating_avg   REAL,
    citation_count INTEGER,
    raw_json     TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (venue, id)
);

CREATE INDEX IF NOT EXISTS idx_papers_status ON papers(status);
CREATE INDEX IF NOT EXISTS idx_papers_venue  ON papers(venue);

CREATE TABLE IF NOT EXISTS scores (
    paper_id      TEXT NOT NULL,
    venue         TEXT NOT NULL,
    keyword_hits  TEXT NOT NULL DEFAULT '[]',  -- JSON array
    llm_score     INTEGER,
    llm_reason    TEXT NOT NULL DEFAULT '',
    scored_at     TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (venue, paper_id),
    FOREIGN KEY (venue, paper_id) REFERENCES papers(venue, id)
);

CREATE INDEX IF NOT EXISTS idx_scores_llm ON scores(llm_score DESC);

CREATE TABLE IF NOT EXISTS digests (
    paper_id              TEXT NOT NULL,
    venue                 TEXT NOT NULL,
    method_category       TEXT NOT NULL DEFAULT '',
    claimed_speedup       TEXT NOT NULL DEFAULT '',
    hardware_target       TEXT NOT NULL DEFAULT '',
    sutro_relevance_tags  TEXT NOT NULL DEFAULT '[]',  -- JSON array
    summary               TEXT NOT NULL DEFAULT '',
    digested_at           TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (venue, paper_id),
    FOREIGN KEY (venue, paper_id) REFERENCES papers(venue, id)
);
"""
