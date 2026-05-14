"""Project paths and venue registry."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_DB = ROOT / "data" / "db"
DOCS_PAPERS = ROOT / "docs" / "papers"
DOCS_TOPICS = ROOT / "docs" / "topics"
CONFIGS = ROOT / "configs"

DB_PATH = DATA_DB / "lit.sqlite"
KEYWORDS_PATH = CONFIGS / "keywords.yaml"

PAPERCOPILOT_RAW = (
    "https://raw.githubusercontent.com/papercopilot/paperlists/main/{venue_family}/{venue_slug}.json"
)


# venue_slug -> (family, year)
VENUES: dict[str, tuple[str, int]] = {
    "iclr2026": ("iclr", 2026),
    "iclr2025": ("iclr", 2025),
    "iclr2024": ("iclr", 2024),
    "nips2025": ("nips", 2025),
    "nips2024": ("nips", 2024),
    "icml2025": ("icml", 2025),
    "icml2024": ("icml", 2024),
}


def papercopilot_url(venue: str) -> str:
    if venue not in VENUES:
        raise ValueError(f"unknown venue: {venue}. Known: {sorted(VENUES)}")
    family, _year = VENUES[venue]
    return PAPERCOPILOT_RAW.format(venue_family=family, venue_slug=venue)


def raw_path(venue: str) -> Path:
    return DATA_RAW / f"{venue}.json"


def ensure_dirs() -> None:
    for d in (DATA_RAW, DATA_DB, DOCS_PAPERS, DOCS_TOPICS, CONFIGS):
        d.mkdir(parents=True, exist_ok=True)
