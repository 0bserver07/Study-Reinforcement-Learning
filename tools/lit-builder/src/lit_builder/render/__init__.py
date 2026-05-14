"""Rendering layer: turn DB rows into markdown + an MkDocs site."""

from __future__ import annotations

from .index import render_index, render_topic_pages
from .markdown import paper_to_markdown, slugify
from .site import discoveries_export, render_venue, write_mkdocs_config

__all__ = [
    "paper_to_markdown",
    "render_index",
    "render_topic_pages",
    "render_venue",
    "write_mkdocs_config",
    "discoveries_export",
    "slugify",
]
