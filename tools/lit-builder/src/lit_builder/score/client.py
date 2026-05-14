"""LLM provider client. Supports Anthropic (default) and Ollama (cloud or local).

Selection is via the `LIT_PROVIDER` env var:
    LIT_PROVIDER=anthropic   -> uses ANTHROPIC_API_KEY (default)
    LIT_PROVIDER=ollama      -> uses OLLAMA_API_KEY against https://ollama.com,
                                or local Ollama if OLLAMA_HOST is set.

The model is resolved by `get_model()`:
    LIT_MODEL=<anything>     -> override (highest priority)
    else for anthropic       -> HAIKU_MODEL
    else for ollama          -> OLLAMA_DEFAULT_MODEL ("deepseek-v4-pro:cloud")

Both providers expose a `.messages.create(model, max_tokens, system, messages)`
method that returns an object with `.content[0].text`. This means the rest of
the score module (triage.py, deepen.py) and existing tests do NOT need to know
which provider is active.
"""

from __future__ import annotations

import os
from typing import Any

# ---- model defaults -------------------------------------------------------

# Latest Haiku per project's CLAUDE.md (2026-04). Bump when a newer Haiku ships.
HAIKU_MODEL = "claude-haiku-4-5-20251001"

# Default Ollama Cloud model. Override with LIT_MODEL.
OLLAMA_DEFAULT_MODEL = "deepseek-v4-pro:cloud"

OLLAMA_DEFAULT_HOST = "https://ollama.com"


def get_provider() -> str:
    """Resolve the active provider name. Defaults to anthropic."""
    raw = os.environ.get("LIT_PROVIDER", "").strip().lower()
    if raw in {"ollama", "anthropic"}:
        return raw
    # Auto-detect: prefer ollama if its key is set and anthropic is not.
    if os.environ.get("OLLAMA_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        return "ollama"
    return "anthropic"


def get_model() -> str:
    """Resolve the model name for the active provider."""
    override = os.environ.get("LIT_MODEL", "").strip()
    if override:
        return override
    return HAIKU_MODEL if get_provider() == "anthropic" else OLLAMA_DEFAULT_MODEL


# ---- adapters -------------------------------------------------------------


class _TextBlock:
    """Minimal stand-in for an Anthropic text content block."""

    def __init__(self, text: str) -> None:
        self.text = text


class _AnthropicShapeResponse:
    """Minimal stand-in for an Anthropic Messages response."""

    def __init__(self, text: str) -> None:
        self.content = [_TextBlock(text)]


class _OllamaMessages:
    """Anthropic-shaped `.messages.create()` for the Ollama Python client."""

    def __init__(self, ollama_client: Any) -> None:
        self._client = ollama_client

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict[str, Any]],
    ) -> _AnthropicShapeResponse:
        chat_messages: list[dict[str, Any]] = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend(messages)
        response = self._client.chat(
            model=model,
            messages=chat_messages,
            options={"num_predict": max_tokens},
            stream=False,
        )
        # Newer ollama python lib returns a Pydantic ChatResponse with
        # .message.content; older / dict-style returns response['message']['content'].
        text = ""
        msg = getattr(response, "message", None)
        if msg is not None:
            text = getattr(msg, "content", "") or ""
        elif isinstance(response, dict):
            text = (response.get("message") or {}).get("content", "") or ""
        return _AnthropicShapeResponse(text)


class OllamaClient:
    """Anthropic-shaped facade over the Ollama Python client."""

    def __init__(self, host: str, api_key: str | None) -> None:
        from ollama import Client as _OllamaSDKClient

        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = _OllamaSDKClient(host=host, headers=headers or None)
        self.messages = _OllamaMessages(self._client)


# ---- public factory -------------------------------------------------------


def get_client() -> Any:
    """Return an LLM client. Provider chosen via LIT_PROVIDER env var.

    Raises:
        RuntimeError: if the required API key for the active provider is missing.
    """
    provider = get_provider()
    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set (LIT_PROVIDER=anthropic)")
        import anthropic

        return anthropic.Anthropic(api_key=api_key)

    if provider == "ollama":
        host = os.environ.get("OLLAMA_HOST", OLLAMA_DEFAULT_HOST)
        api_key = os.environ.get("OLLAMA_API_KEY")
        # Cloud needs a key; a local host (e.g. http://localhost:11434) does not.
        if host == OLLAMA_DEFAULT_HOST and not api_key:
            raise RuntimeError(
                "OLLAMA_API_KEY not set and OLLAMA_HOST is the cloud endpoint. "
                "Either set OLLAMA_API_KEY or point OLLAMA_HOST at a local server."
            )
        return OllamaClient(host=host, api_key=api_key)

    raise RuntimeError(f"unknown LIT_PROVIDER: {provider!r}")
