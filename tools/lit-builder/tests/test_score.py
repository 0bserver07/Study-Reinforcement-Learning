"""Tests for the score module. All Anthropic calls are mocked."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Make `src/` importable when pytest is run from the project root.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lit_builder.models import DigestResult, Paper  # noqa: E402
from lit_builder.score import (  # noqa: E402
    build_deepen_prompt,
    build_triage_prompt,
    deepen_one,
    score_one,
)


def _make_paper(**overrides) -> Paper:
    base = dict(
        id="abc123",
        venue="iclr2026",
        title="A Sparse Quantized Optimizer for Energy-Efficient Training",
        abstract=(
            "We propose a 4-bit optimizer that reduces HBM accesses by 3x on a "
            "transformer training workload. Our method exploits weight sparsity "
            "during the gradient update step and avoids unnecessary cache traffic."
        ),
        tldr="4-bit optimizer with sparsity-aware updates.",
        keywords="quantization;sparsity;optimizer;energy",
        primary_area="optimization",
        authors="Alice;Bob",
        status="Accept (Poster)",
        track="main",
        site_url="",
        github_url="",
        project_url="",
        rating_avg=7.0,
        citation_count=0,
    )
    base.update(overrides)
    return Paper(**base)


def _mock_response(text: str) -> MagicMock:
    """Build a minimal mock that mirrors the Anthropic Messages response shape."""
    block = MagicMock()
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


# ---------- build_triage_prompt -------------------------------------------


def test_build_triage_prompt_includes_rubric_and_abstract() -> None:
    paper = _make_paper()
    prompt = build_triage_prompt(paper)

    assert "0 = Not relevant" in prompt
    assert "3 = Highly relevant" in prompt
    # Abstract content should appear verbatim.
    assert "4-bit optimizer that reduces HBM accesses" in prompt
    assert paper.title in prompt
    assert paper.tldr in prompt


def test_build_triage_prompt_truncates_long_abstract() -> None:
    paper = _make_paper(abstract="A" * 5000)
    prompt = build_triage_prompt(paper)
    # Truncated to 1500 chars; the prompt itself adds boilerplate around it.
    assert "A" * 1500 in prompt
    assert "A" * 1501 not in prompt


# ---------- score_one ------------------------------------------------------


def test_score_one_parses_well_formed_json() -> None:
    client = MagicMock()
    client.messages.create.return_value = _mock_response(
        '{"score": 2, "reason": "Quantized optimizer with cache focus."}'
    )

    score, reason = score_one(client, _make_paper())

    assert score == 2
    assert reason == "Quantized optimizer with cache focus."
    # Sanity: model + max_tokens were passed through.
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["max_tokens"] == 200
    assert kwargs["model"].startswith("claude-haiku")


def test_score_one_parses_json_wrapped_in_prose() -> None:
    client = MagicMock()
    client.messages.create.return_value = _mock_response(
        'Sure, here is the score:\n{"score": 3, "reason": "Direct hit."}\nThanks.'
    )
    score, reason = score_one(client, _make_paper())
    assert score == 3
    assert reason == "Direct hit."


def test_score_one_handles_malformed_response() -> None:
    client = MagicMock()
    client.messages.create.return_value = _mock_response("nope, not JSON at all")

    score, reason = score_one(client, _make_paper())

    assert score == 0
    assert reason.startswith("PARSE_ERROR:")
    assert "nope" in reason
    assert len(reason) <= len("PARSE_ERROR: ") + 100


def test_score_one_rejects_out_of_range_score() -> None:
    client = MagicMock()
    client.messages.create.return_value = _mock_response(
        '{"score": 9, "reason": "Way too high."}'
    )
    score, reason = score_one(client, _make_paper())
    assert score == 0
    assert reason.startswith("PARSE_ERROR:")


# ---------- deepen_one -----------------------------------------------------


def test_deepen_one_parses_full_digest() -> None:
    payload = (
        '{"method_category": "quantization", '
        '"claimed_speedup": "3x reduction in HBM accesses", '
        '"hardware_target": "GPU", '
        '"sutro_relevance_tags": ["data-movement", "low-precision", "optimizer"], '
        '"summary": "Sentence one. Sentence two. Sentence three."}'
    )
    client = MagicMock()
    client.messages.create.return_value = _mock_response(payload)

    paper = _make_paper()
    digest = deepen_one(client, paper, keyword_hits=["quantization", "energy"])

    assert isinstance(digest, DigestResult)
    assert digest.paper_id == paper.id
    assert digest.venue == paper.venue
    assert digest.method_category == "quantization"
    assert digest.claimed_speedup == "3x reduction in HBM accesses"
    assert digest.hardware_target == "GPU"
    assert digest.sutro_relevance_tags == [
        "data-movement",
        "low-precision",
        "optimizer",
    ]
    assert "Sentence one." in digest.summary
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["max_tokens"] == 600


def test_deepen_one_handles_malformed_response() -> None:
    client = MagicMock()
    client.messages.create.return_value = _mock_response("not json")

    digest = deepen_one(client, _make_paper(), keyword_hits=[])

    assert digest.summary.startswith("PARSE_ERROR:")
    assert digest.method_category == ""
    assert digest.sutro_relevance_tags == []


def test_build_deepen_prompt_includes_keyword_hits() -> None:
    paper = _make_paper()
    prompt = build_deepen_prompt(paper, keyword_hits=["quantization", "sparsity"])
    assert "quantization, sparsity" in prompt
    assert paper.title in prompt
    assert "method_category" in prompt
