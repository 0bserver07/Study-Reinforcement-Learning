"""Pass B: structured digest for a single paper, on demand."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import data
from ..models import DigestResult, Paper
from .client import get_model
from .triage import _extract_text  # reuse the response-shape helper

_ABSTRACT_LIMIT = 1500
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_SYSTEM_PROMPT = (
    "You are a research analyst writing for the Sutro Group, a collective focused on "
    "energy-efficient AI training (data movement, cache-aware ML, sparsity, low precision, "
    "biologically-plausible learning, hardware-aware kernels). "
    "Produce concise, technical digests. Respond with JSON only, no prose outside it."
)


def build_deepen_prompt(paper: Paper, keyword_hits: list[str] | None = None) -> str:
    """Build the deepen-pass user message for one paper."""
    abstract = (paper.abstract or "")[:_ABSTRACT_LIMIT]
    hits_line = ", ".join(keyword_hits) if keyword_hits else "(none)"
    return (
        "Write a structured digest of the following paper for the Sutro Group.\n\n"
        f"Title: {paper.title}\n"
        f"TLDR: {paper.tldr or '(none)'}\n"
        f"Keywords: {paper.keywords or '(none)'}\n"
        f"Prior keyword hits: {hits_line}\n"
        f"Abstract: {abstract or '(none)'}\n\n"
        "Required JSON shape:\n"
        "{\n"
        '  "method_category": "string (e.g. quantization, sparsity, optimizer, distillation)",\n'
        '  "claimed_speedup": "string (verbatim claim from abstract, or empty string if none)",\n'
        '  "hardware_target": "string (GPU / edge / TPU / CPU / empty if not stated)",\n'
        '  "sutro_relevance_tags": ["tag1", "tag2", ... 2 to 5 short tags],\n'
        '  "summary": "3-5 sentences for the Sutro audience"\n'
        "}\n"
        "Respond with JSON only."
    )


def _coerce_str(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    return str(val).strip()


def _coerce_tags(val: Any) -> list[str]:
    if isinstance(val, list):
        return [str(t).strip() for t in val if str(t).strip()]
    if isinstance(val, str) and val.strip():
        # Tolerate the model handing back a single string.
        return [val.strip()]
    return []


def _parse_digest_json(text: str) -> dict[str, Any] | None:
    match = _JSON_RE.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def deepen_one(client: Any, paper: Paper, keyword_hits: list[str]) -> DigestResult:
    """Call Haiku to produce a structured digest of `paper`.

    On parse failure returns a DigestResult whose `summary` starts with
    "PARSE_ERROR: ..." and whose other fields are empty.
    """
    prompt = build_deepen_prompt(paper, keyword_hits=keyword_hits)
    response = client.messages.create(
        model=get_model(),
        # See triage.py for why this is generous: deepseek reasoning tokens.
        max_tokens=1500,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text = _extract_text(response)
    obj = _parse_digest_json(text)
    if obj is None:
        snippet = (text or "").strip().replace("\n", " ")[:200]
        return DigestResult(
            paper_id=paper.id,
            venue=paper.venue,
            method_category="",
            claimed_speedup="",
            hardware_target="",
            sutro_relevance_tags=[],
            summary=f"PARSE_ERROR: {snippet}",
        )
    return DigestResult(
        paper_id=paper.id,
        venue=paper.venue,
        method_category=_coerce_str(obj.get("method_category")),
        claimed_speedup=_coerce_str(obj.get("claimed_speedup")),
        hardware_target=_coerce_str(obj.get("hardware_target")),
        sutro_relevance_tags=_coerce_tags(obj.get("sutro_relevance_tags")),
        summary=_coerce_str(obj.get("summary")),
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def deepen_paper(
    venue: str,
    paper_id: str,
    *,
    db_path: Path | None = None,
) -> DigestResult:
    """Produce + persist a digest for `(venue, paper_id)`. Returns the digest."""
    conn = data.connect(db_path)
    try:
        paper = data.get_paper(conn, venue, paper_id)
        if paper is None:
            raise ValueError(f"paper not found: venue={venue!r} id={paper_id!r}")
        existing_score = data.get_score(conn, venue, paper_id)
        keyword_hits = list(existing_score.keyword_hits) if existing_score else []

        from .client import get_client  # lazy so tests can monkeypatch

        client = get_client()
        digest = deepen_one(client, paper, keyword_hits)
        digest.digested_at = _now_iso()
        data.upsert_digest(conn, digest)
        return digest
    finally:
        conn.close()
