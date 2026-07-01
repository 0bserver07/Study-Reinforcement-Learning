> **Note**: this is a copy of `iclr-lit-builder` (see `../../../iclr-lit-builder`), retuned for this repo.
> The original tracks an energy-efficiency lens; here `configs/keywords.yaml` is replaced with an
> RL / RLHF / reasoning / agentic keyword set. The pipeline and CLI docs below are accurate; the
> "Latest run" numbers below are from the original and don't apply. Working data goes in `data/`
> (gitignored). Run: `.venv/bin/lit fetch <venue>` etc., or `pip install -e .` first to get `lit` on PATH.

# lit-builder

Pull ML conference paper lists, filter for the [Sutro Group](https://github.com/cybertronai/SutroYaro) lens (energy-efficient training + broader training-efficiency), and produce an annotated, browsable literature base.

**Live site:** https://0bserver07.github.io/iclr-lit-builder/

Source: [papercopilot/paperlists](https://github.com/papercopilot/paperlists). Starts with ICLR 2026 (~20K records); generalizes to NeurIPS, ICML, etc.

## Latest run: ICLR 2026

| | Count |
|--|------:|
| Papers ingested | **19,813** |
| Keyword-filtered (sent to LLM) | 4,842 |
| **Score 3** (directly Sutro-relevant) | **40** |
| Score 2 (relevant) | 25 |
| Score 1 (tangential) | 134 |
| Score 0 (rejected) | 4,643 |
| Markdown rendered | 4,196 files |

[**→ Browse the score-3 list**](https://0bserver07.github.io/iclr-lit-builder/papers/iclr2026/)

## Pipeline

```
fetch  →  ingest  →  filter (keywords)  →  score (LLM)  →  deepen (on demand)  →  render
```

Each stage is a CLI subcommand and writes to a SQLite database (`data/db/lit.sqlite`). The markdown / MkDocs site is generated from the DB.

## LLM provider: pick one

The scoring stage uses an LLM. Two providers are supported, controlled by `LIT_PROVIDER`. Override the model at any time with `LIT_MODEL=<name>`.

```bash
# Anthropic (default)
export LIT_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
# default model: claude-haiku-4-5-20251001
# override:  export LIT_MODEL=claude-sonnet-4-6

# Ollama Cloud
export LIT_PROVIDER=ollama
export OLLAMA_API_KEY=...
# default model: deepseek-v4-pro:cloud

# Ollama local (no key needed)
export LIT_PROVIDER=ollama
export OLLAMA_HOST=http://localhost:11434
export LIT_MODEL=llama3.1:8b
```

### Solid Ollama Cloud models

Verified to work with the scoring prompt (`lit score`) and the deepen prompt (`lit deepen`). Swap with `LIT_MODEL=<name>`.

| Model | Notes |
|-------|-------|
| `deepseek-v4-pro:cloud` | **Default.** Reasoning model; ~5s per paper at 200-token limit. Best price/quality. |
| `deepseek-v4-flash:cloud` | Faster, lower latency, slightly less robust on edge cases. |
| `gpt-oss:120b` | Strong general scorer. Slightly heavier than deepseek-v4-pro. |
| `qwen3:235b-cloud` | Largest. Best for the deepen stage on borderline papers. |
| `llama3.1:70b` | Solid baseline; available locally too. |

### Solid local models (Ollama, no API key)

```bash
LIT_MODEL=llama3.1:8b      # 4.7 GB, fast, decent
LIT_MODEL=qwen3:14b        # 8 GB, better reasoning
LIT_MODEL=deepseek-v4:7b   # 4 GB, distilled reasoning model
```

## Quickstart

```bash
pip install -e .
# or with uv: uv sync && uv run lit ...

lit fetch  iclr2026
lit ingest iclr2026
lit filter iclr2026                       # keyword pre-filter
lit score  iclr2026 --limit 200           # LLM triage on survivors (0–3 + reason)
lit list   iclr2026 --min-score 2         # browse high-relevance
lit deepen iclr2026 <paper_id>            # structured digest on demand
lit render iclr2026                       # write markdown + mkdocs nav
lit serve                                 # local mkdocs preview
```

## Real example output

Scoring 5 ICLR 2026 candidates on `deepseek-v4-pro:cloud` (Ollama Cloud), ~5s per paper:

| score | title | reason |
|---|---|---|
| 2 | PersonalQ: Select, Quantize, and Serve Personalized Diffusion | Quantization technique for personalized diffusion models that reduces inference memory, aligning with low-precision research. |
| 2 | Reassessing Layer Pruning in LLMs | Layer pruning to reduce computation, directly addressing efficiency and model compression. |
| 1 | Toward Unifying Group Fairness Evaluation from a Sparsity Perspective | References sparsity but only as a lens for fairness evaluation, not as a contribution to training efficiency. |
| 1 | Early Layer Readouts for Robust Knowledge Distillation | Domain generalization via adaptive distillation, only tangential efficiency link. |
| 0 | Concept Alignment for Autonomous Distillation | Robustness and bias mitigation, not energy-efficient training. |

## CLI as a tool

The CLI is designed to be called by other coding agents (Codex, Claude Code). Every command takes positional args, exits non-zero on error, and prints structured key=value output. See `lit --help`.

## Layout

```
src/lit_builder/
  models.py        # shared dataclasses + SQLite DDL
  config.py        # paths, venue registry
  data/            # papercopilot fetch + SQLite ingest
  filter/          # keyword matcher
  score/           # LLM scorer + deepener (Anthropic / Ollama)
  render/          # markdown + mkdocs export
  cli/             # typer commands
configs/keywords.yaml   # editable keyword groups
```

## Status

| Stage | iclr2026 |
|---|---|
| fetch | done: 19,813 raw records (93 MB) |
| ingest | done: 19,813 in DB |
| filter | done: 4,842 keyword candidates |
| score | done: 4,842 / 4,842 LLM-scored via `deepseek-v4-pro:cloud` (40 at score 3, 25 at 2, 134 at 1, 4,643 at 0) |
| deepen | implemented; on-demand per paper |
| render | done: 4,196 markdown pages + index |
| publish | live at https://0bserver07.github.io/iclr-lit-builder/ |

## Tests

```bash
pip install pytest
PYTHONPATH=src python3 -m pytest tests -q       # 33 tests, all mocked LLM
```

---

## Running this in this repo

Installed in an isolated venv at `tools/lit-builder/.venv` (so `lit` here is `tools/lit-builder/.venv/bin/lit`). Working data goes in `tools/lit-builder/data/` (gitignored). Already done: `fetch` + `ingest` + `filter` on ICLR 2026: 19,813 papers ingested, 4,329 matched the RL keyword set in `configs/keywords.yaml`.

What's left needs an LLM credential. Pick one:

```bash
cd tools/lit-builder
LIT=.venv/bin/lit

# Option A, Claude Haiku (cheap, fast):
export LIT_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-...
# Option B, Ollama Cloud (uses the :cloud models already pulled):
export LIT_PROVIDER=ollama OLLAMA_API_KEY=... LIT_MODEL=deepseek-v4-pro:cloud
# Option C, a local model: `ollama pull llama3.1:8b` then LIT_PROVIDER=ollama LIT_MODEL=llama3.1:8b

$LIT score  iclr2026 --limit 4329          # triage all survivors 0-3 (or a smaller --limit to test)
$LIT list   iclr2026 --min-score 2          # the curated, relevance-ranked list
$LIT deepen iclr2026 <paper_id>             # structured digest for the ones that matter
# then: fold the score>=2 list + digests into ../../reference/papers/<topic>/README.md

# more venues are one command each:
$LIT fetch nips2025 && $LIT ingest nips2025 && $LIT filter nips2025
$LIT fetch icml2025 && $LIT ingest icml2025 && $LIT filter icml2025
```

`lit list iclr2026 --min-score -1` shows the keyword survivors before scoring.
