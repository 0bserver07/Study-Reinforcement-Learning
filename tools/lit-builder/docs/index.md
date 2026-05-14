# lit-builder

**ICLR / NeurIPS literature builder with keyword + LLM relevance scoring.**

Pulls ML conference paper lists, filters them through a research lens, and produces an annotated, browsable literature base. Built for the [Sutro Group](https://github.com/cybertronai/SutroYaro) energy-efficient training agenda — generalizes to any lens via `configs/keywords.yaml`.

## What's published

| Venue | Total papers | Keyword-passed | LLM-scored | **Score 3** | Score 2 | Score 1 |
|-------|-------------:|---------------:|-----------:|------------:|--------:|--------:|
| [ICLR 2026](papers/iclr2026/) | 19,813 | 4,842 | 4,842 | **40** | 25 | 134 |

**Score 3** = directly advances Sutro Group priorities (energy-efficient training, sparsity, low-precision, data movement, communication efficiency, training alternatives).

[**→ Browse all 40 score-3 papers**](papers/iclr2026/)

## How it works

```
fetch  →  ingest  →  filter  →  score  →  deepen  →  render
   │          │          │         │          │           │
papercopilot  SQLite     keyword   LLM       on-demand   markdown
paperlists                pre-filter (0–3 + reason)       + mkdocs
```

Each stage is a CLI subcommand. Output lives in `data/db/lit.sqlite`. The site is generated from the DB.

## LLM providers

Two providers, swap with `LIT_PROVIDER`:

### Anthropic
- Default model: `claude-haiku-4-5-20251001`
- Set `ANTHROPIC_API_KEY`
- Override with `LIT_MODEL`

### Ollama (cloud or local)
- **Cloud**: set `OLLAMA_API_KEY` against `https://ollama.com`
- **Local**: set `OLLAMA_HOST=http://localhost:11434`
- Default model: `deepseek-v4-pro:cloud`

**Solid Ollama Cloud models** (any swap-in via `LIT_MODEL`):

| Model | Notes |
|-------|-------|
| `deepseek-v4-pro:cloud` | Default. Reasoning model; ~5s per paper at 200-token limit. |
| `deepseek-v4-flash:cloud` | Faster, lower latency, slightly less robust on edge cases. |
| `gpt-oss:120b` | Strong general scorer. Slightly heavier than deepseek-v4-pro. |
| `qwen3:235b-cloud` | Largest. Best for the deepen stage on borderline papers. |
| `llama3.1:70b` | Solid baseline; available locally too. |

Local examples:
```bash
LIT_PROVIDER=ollama OLLAMA_HOST=http://localhost:11434 LIT_MODEL=llama3.1:8b
LIT_PROVIDER=ollama OLLAMA_HOST=http://localhost:11434 LIT_MODEL=qwen3:14b
```

## Quickstart

```bash
pip install -e .
# or with uv: uv sync && uv run lit ...

lit fetch  iclr2026
lit ingest iclr2026
lit filter iclr2026                    # keyword pre-filter
lit score  iclr2026 --limit 200        # LLM triage on survivors
lit list   iclr2026 --min-score 2      # browse high-relevance
lit deepen iclr2026 <paper_id>         # structured digest on demand
lit render iclr2026                    # write markdown + mkdocs nav
lit serve                              # local mkdocs preview
```

See the [README on GitHub](https://github.com/0bserver07/iclr-lit-builder) for full setup.

## CLI as a tool

The CLI is designed to be called by other coding agents (Codex, Claude Code, Gemini CLI). Every command takes positional args, exits non-zero on error, and prints structured `key=value` output.

```bash
lit --help
lit score --help
```

## Source

[github.com/0bserver07/iclr-lit-builder](https://github.com/0bserver07/iclr-lit-builder)
