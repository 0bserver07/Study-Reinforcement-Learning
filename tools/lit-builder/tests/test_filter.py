"""Unit tests for the keyword pre-filter. No network, no DB."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running these tests without an editable install.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lit_builder.filter import (  # noqa: E402
    all_terms,
    filter_paper,
    find_hits,
    load_keywords,
    searchable_text,
)
from lit_builder.models import Paper  # noqa: E402


def _terms(*pairs: tuple[str, str]) -> list[tuple[str, str]]:
    return list(pairs)


def test_find_hits_case_insensitive_and_dedup_in_source_order():
    text = "Quantization helps. We also try QUANTIZATION and quantization again. Sparse too.".lower()
    terms = _terms(
        ("precision_and_quantization", "quantization"),
        ("sparsity", "sparse"),
        ("sparsity", "moe"),  # absent
    )
    hits = find_hits(text, terms)
    # Each matched term appears once, in the order it was given in `terms`.
    assert hits == ["quantization", "sparse"]


def test_moe_word_boundary_does_not_match_moebius():
    terms = _terms(("sparsity", "moe"))
    # Should match: word-boundary on space.
    assert find_hits("we propose a MoE layer for routing".lower(), terms) == ["moe"]
    # Should NOT match: 'moe' inside 'moebius'.
    assert find_hits("the moebius strip is fun".lower(), terms) == []
    # Should also NOT match a longer alphanumeric word that merely contains 'moe'.
    assert find_hits("smoethered in adjectives".lower(), terms) == []


def test_low_precision_matches_phrase_and_hyphenated_form():
    terms = _terms(("precision_and_quantization", "low precision"))
    # Multi-word phrase, plain.
    assert find_hits("Low Precision Training works".lower(), terms) == ["low precision"]
    # Hyphenated should match because the hyphen is a non-alphanumeric boundary
    # and the term is matched against its literal form -- BUT the literal term
    # is 'low precision' (with a space). Our boundary regex requires the space
    # to be present, so the hyphenated form should match the SEPARATE term
    # 'low-precision'. Verify both terms behave as expected.
    terms2 = _terms(
        ("precision_and_quantization", "low precision"),
        ("precision_and_quantization", "low-precision"),
    )
    hits = find_hits("we explore low-precision training".lower(), terms2)
    assert "low-precision" in hits
    # The literal "low precision" (with a space) should NOT match the
    # hyphenated text, since the haystack doesn't contain a space there.
    assert "low precision" not in hits


def test_filter_paper_end_to_end():
    paper = Paper(
        id="abc123",
        venue="iclr2026",
        title="Energy-Efficient MoE Training with Low-Precision Optimizers",
        abstract="We study quantization and sparsity. Our method beats Adam.",
        tldr="MoE + int8 + sparsity = win.",
        keywords="moe;quantization;sparsity",
        primary_area="optimization",
        authors="A;B",
        status="Accept (Poster)",
        track="main",
        site_url="",
        github_url="",
        project_url="",
        rating_avg=7.0,
        citation_count=0,
        raw_json="",
    )
    terms = _terms(
        ("precision_and_quantization", "quantization"),
        ("precision_and_quantization", "low-precision"),
        ("precision_and_quantization", "int8"),
        ("sparsity", "sparsity"),
        ("sparsity", "moe"),
        ("optimizers_and_training_dynamics", "adam"),
        ("energy_and_compute", "energy-efficient"),
        ("energy_and_compute", "carbon"),  # absent
    )
    hits = filter_paper(paper, terms)
    # Order matches `terms` order; carbon is filtered out; no duplicates.
    assert hits == [
        "quantization",
        "low-precision",
        "int8",
        "sparsity",
        "moe",
        "adam",
        "energy-efficient",
    ]


def test_searchable_text_lowercases_and_joins_with_newlines():
    paper = Paper(
        id="x",
        venue="v",
        title="TITLE",
        abstract="ABSTRACT",
        tldr="TLDR",
        keywords="K1;K2",
        primary_area="AREA",
        authors="",
        status="",
        track="",
        site_url="",
        github_url="",
        project_url="",
        rating_avg=None,
        citation_count=None,
        raw_json="",
    )
    text = searchable_text(paper)
    assert text == "title\ntldr\nabstract\nk1;k2\narea"


def test_load_keywords_and_all_terms_against_real_config():
    groups = load_keywords()
    assert isinstance(groups, dict)
    assert "sparsity" in groups
    assert "moe" in groups["sparsity"]

    terms = all_terms(groups)
    assert all(isinstance(g, str) and isinstance(t, str) for g, t in terms)
    # Sorted longest-first.
    lengths = [len(t) for _g, t in terms]
    assert lengths == sorted(lengths, reverse=True)
    # Every term from every group is represented once.
    expected = sum(len(v) for v in groups.values())
    assert len(terms) == expected


if __name__ == "__main__":
    # Allow running without pytest: simple driver.
    fns = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e!r}")
    if failed:
        raise SystemExit(1)
    print(f"\nall {len(fns)} tests passed")
