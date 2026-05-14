"""Pass A: cheap relevance triage. One Haiku call per paper, returns 0-3 + reason."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from .. import data
from ..models import Paper, ScoreResult
from .client import get_model

_ABSTRACT_LIMIT = 1500

_RUBRIC = """0 = Not relevant. Topic unrelated to training efficiency, energy, data movement, sparsity, low precision, or local learning.
1 = Tangential. Mentions an efficiency angle but not the paper's main contribution.
2 = Relevant. Efficiency / data-movement / sparsity / quantization / local-learning IS a main contribution.
3 = Highly relevant. Directly advances energy-efficient training or one of the Sutro Group's named priorities (data movement, cache-aware ML, sparse parity, biologically-plausible learning, hardware-aware training)."""

_SYSTEM_PROMPT = (
    "You are a scoring assistant for the Sutro Group, a research collective focused on "
    "energy-efficient AI training: data movement, cache-aware ML, sparsity, low-precision / "
    "quantization, biologically-plausible / local learning, hardware-aware kernels, "
    "optimizers, scaling laws, distillation, sparse parity / grokking benchmarks. "
    "Score papers strictly. Respond with JSON only, no prose."
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def build_triage_prompt(paper: Paper) -> str:
    """Build the user message for triage scoring of a single paper."""
    abstract = (paper.abstract or "")[:_ABSTRACT_LIMIT]
    return (
        "Score the following paper for relevance to the Sutro Group lens.\n\n"
        f"Title: {paper.title}\n"
        f"TLDR: {paper.tldr or '(none)'}\n"
        f"Keywords: {paper.keywords or '(none)'}\n"
        f"Abstract: {abstract or '(none)'}\n\n"
        "Rubric:\n"
        f"{_RUBRIC}\n\n"
        'Respond with JSON only, exactly: {"score": <int 0-3>, "reason": "<one sentence>"}'
    )


def _extract_text(response: Any) -> str:
    """Pull the first text block out of an Anthropic Messages response."""
    content = getattr(response, "content", None)
    if not content:
        return ""
    block = content[0]
    text = getattr(block, "text", None)
    if text is None and isinstance(block, dict):
        text = block.get("text", "")
    return text or ""


def _parse_score_json(text: str) -> tuple[int, str] | None:
    """Find a JSON object in the response and coerce it to (score, reason)."""
    match = _JSON_RE.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "score" not in obj:
        return None
    try:
        score = int(obj["score"])
    except (TypeError, ValueError):
        return None
    if score < 0 or score > 3:
        return None
    reason = str(obj.get("reason", "")).strip()
    return score, reason


def score_one(client: Any, paper: Paper) -> tuple[int, str]:
    """Call Haiku once to score `paper`. Returns (score, reason).

    On parse failure returns (0, "PARSE_ERROR: <first 100 chars of response>").
    """
    prompt = build_triage_prompt(paper)
    response = client.messages.create(
        model=get_model(),
        # deepseek/reasoning models consume thinking tokens against this budget, so 200
        # (enough for Haiku) truncates them. 600 covers both providers.
        max_tokens=600,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text = _extract_text(response)
    parsed = _parse_score_json(text)
    if parsed is None:
        snippet = (text or "").strip().replace("\n", " ")[:100]
        return 0, f"PARSE_ERROR: {snippet}"
    return parsed


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def score_venue(
    venue: str,
    *,
    db_path: Path | None = None,
    limit: int | None = None,
    only_unscored: bool = True,
) -> dict[str, int]:
    """Score every keyword-prefiltered paper in `venue` via Haiku.

    Walks the `scores` table for rows whose `keyword_hits` is non-empty (and
    optionally whose `llm_score` is still NULL), loads the corresponding
    paper, calls Haiku, and writes the score back via `upsert_score`.

    Returns a counts dict: `{"scored": N, "errors": M, "skipped": K}`.
    """
    conn = data.connect(db_path)
    try:
        sql = (
            "SELECT paper_id FROM scores "
            "WHERE venue = ? AND keyword_hits != '[]' AND keyword_hits != ''"
        )
        params: list[Any] = [venue]
        if only_unscored:
            sql += " AND llm_score IS NULL"
        sql += " ORDER BY paper_id ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        paper_ids = [row["paper_id"] for row in conn.execute(sql, params).fetchall()]

        counts = {"scored": 0, "errors": 0, "skipped": 0}
        if not paper_ids:
            return counts

        client = _lazy_client()

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(f"scoring {venue}", total=len(paper_ids))
            for paper_id in paper_ids:
                paper = data.get_paper(conn, venue, paper_id)
                existing = data.get_score(conn, venue, paper_id)
                if paper is None or existing is None:
                    counts["skipped"] += 1
                    progress.advance(task)
                    continue
                try:
                    score, reason = score_one(client, paper)
                except Exception as exc:  # noqa: BLE001 - log + continue
                    counts["errors"] += 1
                    data.upsert_score(
                        conn,
                        ScoreResult(
                            paper_id=existing.paper_id,
                            venue=existing.venue,
                            keyword_hits=existing.keyword_hits,
                            llm_score=0,
                            llm_reason=f"API_ERROR: {type(exc).__name__}: {str(exc)[:120]}",
                            scored_at=_now_iso(),
                        ),
                    )
                    progress.advance(task)
                    continue

                if reason.startswith("PARSE_ERROR"):
                    counts["errors"] += 1
                else:
                    counts["scored"] += 1
                data.upsert_score(
                    conn,
                    ScoreResult(
                        paper_id=existing.paper_id,
                        venue=existing.venue,
                        keyword_hits=existing.keyword_hits,
                        llm_score=score,
                        llm_reason=reason,
                        scored_at=_now_iso(),
                    ),
                )
                progress.advance(task)
        return counts
    finally:
        conn.close()


def _lazy_client() -> Any:
    """Indirection so tests can monkeypatch get_client without importing here."""
    from .client import get_client

    return get_client()
