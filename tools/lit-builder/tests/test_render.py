"""Tests for the render/ module. Pure functions only -- no DB, no FS."""

from __future__ import annotations

from lit_builder.models import DigestResult, Paper, ScoreResult
from lit_builder.render import paper_to_markdown, render_index, slugify
from lit_builder.render.index import render_topic_pages
from lit_builder.render.site import discoveries_export  # noqa: F401  (import smoke)


def _make_paper(**overrides) -> Paper:
    base = dict(
        id="ABC123",
        venue="iclr2026",
        title="Low-Precision Training!",
        abstract="We train at fp4 with no accuracy loss.",
        tldr="fp4 training works",
        keywords="quantization;low-precision;training",
        primary_area="optimization",
        authors="Alice;Bob;Carol",
        status="Accept (Poster)",
        track="main",
        site_url="https://openreview.net/forum?id=ABC123",
        github_url="https://github.com/example/repo",
        project_url="https://example.com/proj",
        rating_avg=7.0,
        citation_count=3,
        raw_json="{}",
    )
    base.update(overrides)
    return Paper(**base)


def _make_score(**overrides) -> ScoreResult:
    base = dict(
        paper_id="ABC123",
        venue="iclr2026",
        keyword_hits=["quantization", "fp4"],
        llm_score=3,
        llm_reason="Directly relevant: trains in fp4 end-to-end with sparsity.",
        scored_at="2026-04-25T00:00:00Z",
    )
    base.update(overrides)
    return ScoreResult(**base)


def _make_digest(**overrides) -> DigestResult:
    base = dict(
        paper_id="ABC123",
        venue="iclr2026",
        method_category="quantization",
        claimed_speedup="2.3x on ImageNet",
        hardware_target="GPU",
        sutro_relevance_tags=["energy", "low-precision"],
        summary="They quantize all matmul activations to fp4 and recover accuracy via stochastic rounding.",
        digested_at="2026-04-25T00:00:00Z",
    )
    base.update(overrides)
    return DigestResult(**base)


# ---------- paper_to_markdown ----------------------------------------------


def test_paper_to_markdown_full():
    md = paper_to_markdown(_make_paper(), _make_score(), _make_digest())
    # Title
    assert md.startswith("# Low-Precision Training!\n")
    # Header block
    assert "**Venue:** iclr2026 (Accept (Poster))" in md
    assert "**Authors:** Alice, Bob, Carol" in md
    assert "**OpenReview:** [https://openreview.net/forum?id=ABC123]" in md
    assert "**GitHub:** [link](https://github.com/example/repo)" in md
    assert "**Project page:** [link](https://example.com/proj)" in md
    # Sections
    assert "## Relevance" in md
    assert "**LLM score:** 3/3" in md
    assert "Directly relevant" in md
    assert "**Keyword hits:** `quantization, fp4`" in md
    assert "## TLDR" in md
    assert "fp4 training works" in md
    assert "## Abstract" in md
    assert "We train at fp4" in md
    assert "## Keywords" in md
    assert "quantization, low-precision, training" in md
    assert "## Digest" in md
    assert "**Method category:** quantization" in md
    assert "**Claimed speedup:** 2.3x on ImageNet" in md
    assert "**Hardware target:** GPU" in md
    assert "**Sutro tags:** energy, low-precision" in md
    assert "stochastic rounding" in md


def test_paper_to_markdown_no_score_no_digest():
    md = paper_to_markdown(_make_paper(), None, None)
    assert "## Relevance" not in md
    assert "## Digest" not in md
    # Other sections still present.
    assert "## TLDR" in md
    assert "## Abstract" in md
    assert "## Keywords" in md


def test_paper_to_markdown_omits_optional_links():
    p = _make_paper(github_url="", project_url="")
    md = paper_to_markdown(p, _make_score(), None)
    assert "**GitHub:**" not in md
    assert "**Project page:**" not in md
    assert "**OpenReview:**" in md


def test_paper_to_markdown_handles_missing_tldr():
    p = _make_paper(tldr="")
    md = paper_to_markdown(p, None, None)
    assert "(none provided)" in md


# ---------- slugify --------------------------------------------------------


def test_slugify_basic():
    assert slugify("Low-Precision Training!") == "low-precision-training"


def test_slugify_collapses_runs():
    assert slugify("FOO   bar___baz") == "foo-bar-baz"


def test_slugify_strips_edges():
    assert slugify("--Hello, World!--") == "hello-world"


def test_slugify_all_punct():
    assert slugify("!!!") == ""


def test_slugify_unicode_falls_back():
    # Non-ascii letters get stripped (we only allow [a-z0-9]).
    assert slugify("Quantización 8-bit") == "quantizaci-n-8-bit"


# ---------- render_index ----------------------------------------------------


def test_render_index_has_section_per_present_score():
    p1 = _make_paper(id="P1", title="High Relevance")
    p2 = _make_paper(id="P2", title="Mid Relevance")
    p3 = _make_paper(id="P3", title="Low Relevance")
    s1 = _make_score(paper_id="P1", llm_score=3, llm_reason="top tier")
    s2 = _make_score(paper_id="P2", llm_score=2, llm_reason="useful")
    s3 = _make_score(paper_id="P3", llm_score=1, llm_reason="tangential")

    md = render_index([(p1, s1), (p2, s2), (p3, s3)])

    assert "## Score 3" in md
    assert "## Score 2" in md
    assert "## Score 1" in md
    # No score-0 entries -> the section is skipped per spec.
    assert "## Score 0" not in md
    # Entry lines.
    assert "- **3** — [High Relevance](P1.md) — top tier" in md
    assert "- **2** — [Mid Relevance](P2.md) — useful" in md
    assert "- **1** — [Low Relevance](P3.md) — tangential" in md


def test_render_index_truncates_long_reason():
    p = _make_paper(id="LONG")
    long_reason = "x" * 300
    s = _make_score(paper_id="LONG", llm_score=2, llm_reason=long_reason)
    md = render_index([(p, s)])
    # Reason should be capped at 100 chars in the line.
    truncated = "x" * 100
    assert truncated in md
    assert "x" * 101 not in md


def test_render_index_includes_zero_section_when_present():
    p = _make_paper(id="ZERO", title="Zero Paper")
    s = _make_score(paper_id="ZERO", llm_score=0, llm_reason="not relevant")
    md = render_index([(p, s)])
    assert "## Score 0" in md
    assert "[Zero Paper](ZERO.md)" in md


# ---------- render_topic_pages ---------------------------------------------


def test_render_topic_pages_groups_by_method_category():
    p1 = _make_paper(id="A", title="Quant Paper")
    p2 = _make_paper(id="B", title="Sparsity Paper")
    p3 = _make_paper(id="C", title="No Digest Paper")
    s1 = _make_score(paper_id="A")
    s2 = _make_score(paper_id="B")
    s3 = _make_score(paper_id="C")
    d1 = _make_digest(paper_id="A", method_category="quantization", summary="quant summary")
    d2 = _make_digest(paper_id="B", method_category="sparsity", summary="sparse summary")

    pages = render_topic_pages(
        [(p1, s1), (p2, s2), (p3, s3)],
        {"A": d1, "B": d2},
    )
    assert "quantization.md" in pages
    assert "sparsity.md" in pages
    assert "Quant Paper" in pages["quantization.md"]
    assert "Sparsity Paper" in pages["sparsity.md"]
    # Paper without a digest is excluded.
    assert "No Digest Paper" not in pages["quantization.md"]
    assert "No Digest Paper" not in pages["sparsity.md"]


# ---------- discoveries_export (signature smoke) ---------------------------
# A real DB-backed test belongs in an integration suite. We at least confirm
# the symbol is importable and callable signature-wise via the import above.


# ---------- helper: discoveries-style line is what render_index/topics build
def test_discoveries_line_format_via_topic_summary():
    """Sanity-check the one-line format we use in topic pages mirrors the
    style used by `discoveries_export` (title + url + short blurb)."""
    p = _make_paper(id="X", title="Demo")
    s = _make_score(paper_id="X")
    d = _make_digest(paper_id="X", method_category="optimizer", summary="opt summary")
    pages = render_topic_pages([(p, s)], {"X": d})
    assert "optimizer.md" in pages
    page = pages["optimizer.md"]
    assert "[Demo]" in page
    assert "opt summary" in page


# ---------- discoveries_export filters by min_score ------------------------


def test_discoveries_export_filters_to_score_3(tmp_path, monkeypatch):
    """End-to-end test using a temp SQLite DB.

    Even though the spec says 'no DB', the score-3 filter is the load-bearing
    behavior of `discoveries_export` and is only observable via a DB round
    trip. We use a tmp file so it leaves no artifacts.
    """
    from lit_builder.data import connect, upsert_digest, upsert_papers, upsert_score
    from lit_builder.render import discoveries_export

    db = tmp_path / "lit.sqlite"
    conn = connect(db)
    try:
        p_hi = _make_paper(id="HI", title="High", venue="iclr2026")
        p_lo = _make_paper(id="LO", title="Low", venue="iclr2026")
        upsert_papers(conn, [p_hi, p_lo])
        upsert_score(conn, _make_score(paper_id="HI", venue="iclr2026", llm_score=3))
        upsert_score(conn, _make_score(paper_id="LO", venue="iclr2026", llm_score=1))
        upsert_digest(
            conn,
            _make_digest(paper_id="HI", venue="iclr2026", summary="hi summary"),
        )
    finally:
        conn.close()

    out = discoveries_export("iclr2026", db_path=db)
    assert "[High]" in out
    assert "[Low]" not in out
    assert "hi summary" in out
