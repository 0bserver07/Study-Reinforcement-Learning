# content-pipeline

Small scripts for drafting write-ups (blog posts, threads) about papers using an LLM API. Auxiliary — not part of the study path. The drafts it produces are starting points; they need editing before they're worth anything.

> Note: an earlier version of this README described a much larger system — templates, workflows, a dozen generator scripts, publishing integrations. Most of that was never built. What's actually here is below. Read the source for the exact command-line flags.

## What's here

```
content-pipeline/
├── config/
│   └── api_keys.env.example     # copy to api_keys.env, fill in keys (api_keys.env is gitignored)
├── generators/
│   ├── llm_client.py            # wrapper around LLM APIs (Anthropic, OpenAI, DeepSeek, Ollama, ...)
│   ├── paper_to_blog.py         # draft a blog post from an arXiv paper
│   ├── paper_to_thread.py       # draft a thread from an arXiv paper
│   └── test_ollama.py           # quick check that a local Ollama model responds
└── outputs/
    └── blogs/                   # generated drafts land here (one example is checked in)
```

## Use

```bash
cp config/api_keys.env.example config/api_keys.env   # then fill in the keys you have
python3 generators/test_ollama.py                    # if you're using a local model
python3 generators/paper_to_blog.py --help           # see the real flags
python3 generators/paper_to_thread.py --help
```

## Rules

- Never commit `config/api_keys.env`. It's in `.gitignore`; keep it that way.
- Anything this produces is unreviewed until a person reads and fixes it. Don't publish raw output. Verify every claim and every citation against the paper.
- If you extend this, keep the README matching reality. Don't describe scripts that don't exist.
