<!-- status: unreviewed | last-reviewed: never -->

# Reference: papers (RL for LLMs, 2022–2025)

Reading lists on RL applied to language models — code generation, program synthesis, RLHF and alignment. The `PAPERS.md` file in each subdirectory is **generated**: the collector in [`../../tools/arxiv-collector/`](../../tools/arxiv-collector/) queries arXiv and dumps titles, authors, dates, and abstracts. Don't hand-edit those — re-run the collector to refresh. The per-topic READMEs are short hand notes on what each area is about.

This is reference material, not the study path. If you're learning RL, start with [`../../CURRICULUM.md`](../../CURRICULUM.md) and the lectures; come here to go deeper on a specific area.

## Subdirectories

- [`RLHF-and-Alignment/`](./RLHF-and-Alignment/) — PPO for LLMs, reward modeling, DPO and its variants, GRPO, reward hacking, multi-objective alignment.
- [`LLM-Code-Generation/`](./LLM-Code-Generation/) — execution feedback as a reward signal, sandboxing, benchmark evaluation (HumanEval, MBPP, APPS, SWE-bench, LiveCodeBench).
- [`LLM-RL-Program-Synthesis/`](./LLM-RL-Program-Synthesis/) — competition-level code generation (AlphaCode, CodeRL), process supervision, reasoning RL (o1-style, DeepSeek-R1).

## Landmark papers, with identifiers you can check

The lecture series covers most of these.

- AlphaCode — *Competition-level code generation with AlphaCode*, Li et al., Science 2022. [arXiv:2203.07814](https://arxiv.org/abs/2203.07814)
- CodeRL — *CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning*, Le et al., NeurIPS 2022. [arXiv:2207.01780](https://arxiv.org/abs/2207.01780)
- InstructGPT — *Training language models to follow instructions with human feedback*, Ouyang et al., 2022. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- DPO — *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, Rafailov et al., 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- GRPO — introduced in *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*, Shao et al., 2024. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- DeepSeek-R1 — *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*, DeepSeek-AI, 2025. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)

Survey papers exist for "RL for LLMs" and "LLMs for code" (2024–2025) — search arXiv rather than trusting a stale citation here; the earlier version of this file listed an ID that didn't match the year it claimed.

## Tooling these papers use

- TRL (Hugging Face) — PPO/DPO/GRPO trainers for transformers. DeepSpeed / FSDP for sharding.
- HumanEval, MBPP, APPS — function- and competition-level code benchmarks. SWE-bench — real GitHub issues. LiveCodeBench — refreshed to limit contamination.
- For executing model-written code safely: containers/gVisor sandboxes, static analysis (Bandit, Semgrep).

## Refreshing the lists

```bash
python3 tools/arxiv-collector/arxiv_paper_collector.py
```

Run it when the `PAPERS.md` files are stale. See [`../../tools/arxiv-collector/README.md`](../../tools/arxiv-collector/README.md).
