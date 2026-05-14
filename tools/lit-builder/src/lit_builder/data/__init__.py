"""Data layer: fetch papercopilot dumps, ingest to SQLite, query rows."""

from __future__ import annotations

from .db import (
    connect,
    get_digest,
    get_paper,
    get_papers,
    get_score,
    get_scored,
    upsert_digest,
    upsert_papers,
    upsert_score,
)
from .fetch import fetch
from .ingest import ingest, normalize

__all__ = [
    "fetch",
    "ingest",
    "normalize",
    "connect",
    "upsert_papers",
    "upsert_score",
    "upsert_digest",
    "get_papers",
    "get_paper",
    "get_scored",
    "get_score",
    "get_digest",
]
