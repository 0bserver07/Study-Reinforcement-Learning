"""LLM scoring stage. Pass A = triage (cheap), Pass B = deepen (on demand)."""

from __future__ import annotations

from .client import (
    HAIKU_MODEL,
    OLLAMA_DEFAULT_MODEL,
    OllamaClient,
    get_client,
    get_model,
    get_provider,
)
from .deepen import build_deepen_prompt, deepen_one, deepen_paper
from .triage import build_triage_prompt, score_one, score_venue

__all__ = [
    "HAIKU_MODEL",
    "OLLAMA_DEFAULT_MODEL",
    "OllamaClient",
    "get_client",
    "get_model",
    "get_provider",
    "build_triage_prompt",
    "score_one",
    "score_venue",
    "build_deepen_prompt",
    "deepen_one",
    "deepen_paper",
]
