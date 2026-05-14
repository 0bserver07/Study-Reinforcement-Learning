"""Tests for the normalize() pure function. No network, no disk writes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make `src/` importable when running directly via `python3 tests/...`.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lit_builder.data.ingest import normalize  # noqa: E402
from lit_builder.models import Paper  # noqa: E402


VENUE = "iclr2026"


def _record_typical() -> dict:
    return {
        "id": "00F7BfXLYJ",
        "title": "Sparse Coding for Energy-Efficient Inference",
        "track": "main",
        "status": "Accept (Oral)",
        "tldr": "we make models cheap",
        "abstract": "A long abstract about sparsity.",
        "keywords": "sparsity;quantization;efficient inference",
        "primary_area": "applications",
        "author": "Alice Zhao;Bob Singh;Carol Diaz",
        "site": "https://openreview.net/forum?id=00F7BfXLYJ",
        "github": "https://github.com/example/sparse",
        "project": "",
        "rating_avg": [7.5, 0.5],
        "gs_citation": 12,
    }


def _record_minimal() -> dict:
    # Almost everything missing. Withdrawn paper.
    return {
        "id": "ZZZminimal",
        "title": "Stub",
        "status": "Withdraw",
    }


def _record_messy() -> dict:
    # rating_avg as scalar, gs_citation as string with comma, missing optional urls.
    return {
        "id": "msg-1",
        "title": "Messy Inputs",
        "track": "main",
        "status": "Accept (Poster)",
        "abstract": "abs",
        "keywords": "rl",
        "primary_area": "rl",
        "author": "Solo Author",
        "rating_avg": 6.0,
        "gs_citation": "1,234",
    }


def test_normalize_typical():
    p = normalize(_record_typical(), VENUE)
    assert isinstance(p, Paper)
    assert p.id == "00F7BfXLYJ"
    assert p.venue == VENUE
    assert p.title == "Sparse Coding for Energy-Efficient Inference"
    assert p.status == "Accept (Oral)"
    assert p.tldr == "we make models cheap"
    assert p.keywords == "sparsity;quantization;efficient inference"
    assert p.authors == "Alice Zhao;Bob Singh;Carol Diaz"
    assert p.site_url.startswith("https://openreview.net/")
    assert p.github_url == "https://github.com/example/sparse"
    assert p.project_url == ""
    assert p.rating_avg == 7.5
    assert p.citation_count == 12
    # raw_json round-trips
    decoded = json.loads(p.raw_json)
    assert decoded["id"] == "00F7BfXLYJ"
    assert decoded["rating_avg"] == [7.5, 0.5]


def test_normalize_minimal():
    p = normalize(_record_minimal(), VENUE)
    assert p.id == "ZZZminimal"
    assert p.venue == VENUE
    assert p.title == "Stub"
    assert p.status == "Withdraw"
    # Missing string fields default to "".
    assert p.abstract == ""
    assert p.tldr == ""
    assert p.keywords == ""
    assert p.authors == ""
    assert p.site_url == ""
    assert p.github_url == ""
    assert p.project_url == ""
    assert p.track == ""
    assert p.primary_area == ""
    # Missing numerics default to None.
    assert p.rating_avg is None
    assert p.citation_count is None


def test_normalize_messy():
    p = normalize(_record_messy(), VENUE)
    assert p.id == "msg-1"
    assert p.status == "Accept (Poster)"
    # Scalar rating coerces to float.
    assert p.rating_avg == 6.0
    # Citation string with comma parses.
    assert p.citation_count == 1234
    # Missing urls default to "".
    assert p.site_url == ""
    assert p.github_url == ""
    assert p.project_url == ""


if __name__ == "__main__":
    test_normalize_typical()
    test_normalize_minimal()
    test_normalize_messy()
    print("ok: all 3 normalize tests passed")
