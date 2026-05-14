"""Fetch raw papercopilot JSON to disk."""

from __future__ import annotations

from pathlib import Path

import httpx
from rich.console import Console

from .. import config


def fetch(venue: str, *, force: bool = False) -> Path:
    """Download the papercopilot JSON for `venue` to `raw_path(venue)`.

    Streams to disk (the file is ~100MB). Skips work if the destination
    already exists and `force` is False. Returns the local path.
    """
    console = Console()
    config.ensure_dirs()

    dest = config.raw_path(venue)
    if dest.exists() and not force:
        console.print(f"[dim]already cached:[/dim] {dest}")
        return dest

    url = config.papercopilot_url(venue)
    console.print(f"[bold]fetching[/bold] {url}")
    console.print(f"[dim]  -> {dest}[/dim]")

    tmp = dest.with_suffix(dest.suffix + ".part")
    bytes_written = 0
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in resp.iter_bytes(chunk_size=64 * 1024):
                    if chunk:
                        f.write(chunk)
                        bytes_written += len(chunk)
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise

    console.print(f"[green]done[/green]: {bytes_written / (1024 * 1024):.1f} MB")
    return dest
