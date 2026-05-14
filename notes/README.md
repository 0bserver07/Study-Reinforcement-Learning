<!-- status: unreviewed | last-reviewed: never -->

# Lecture series: deep RL to LLM alignment

A self-study sequence that goes from MDPs and policy gradients up through RLHF, DPO, and the 2024–2025 alignment methods. The lecture bodies **haven't been reviewed yet** — useful as a structured path, but check the math, the code, and the citations against primary sources. `../CURRICULUM.md` is the same path with prerequisites and time estimates; [`../AGENTS.md`](../AGENTS.md) explains the `status:` labels.

Each lecture tries to do four things: give the intuition before the math, show code that runs, point at where the method breaks in practice, and name the papers that introduced it. When a lecture has a matching exercise, it links to [`../exercises/`](../exercises/).

## Lectures and review status

| # | Lecture | Status |
|---|---|---|
| 01 | [MDPs and Bellman equations](./lectures/01-mdps-bellman.md) — exercise: [`01-mdps`](../exercises/01-mdps/) | unreviewed (de-slopped; a fabricated value-function output was removed) |
| 02 | [Policy gradients from scratch](./lectures/02-policy-gradients.md) — exercise: [`02-policy-gradients`](../exercises/02-policy-gradients/) | unreviewed (de-slopped; a broken link and a code bug were fixed) |
| 03 | [Value functions & Q-learning](./lectures/03-value-functions-q-learning.md) — exercise: [`03-q-learning`](../exercises/03-q-learning/) | unreviewed (de-slopped; a dead `Modern-RL-Research/` path and a missing import fixed) |
| 04 | [Actor-critic methods](./lectures/04-actor-critic.md) | unreviewed (de-slopped; a code bug fixed) |
| 05 | [Trust regions and TRPO](./lectures/05-trpo.md) | unreviewed (de-slopped; fabricated training times removed) |
| 06 | [PPO](./lectures/06-ppo.md) | unreviewed (de-slopped; `import gym` → `gymnasium` fixed) |
| 07 | [Off-policy learning: SAC and TD3](./lectures/07-off-policy-rl.md) | unreviewed (de-slopped; an old-API `env.step` call fixed) |
| 08 | [Model-based RL](./lectures/08-model-based-rl.md) | unreviewed (de-slopped; old-API calls + a wrong citation fixed) |
| 09 | [Reward modeling for RLHF](./lectures/09-reward-modeling.md) | unreviewed (de-slopped; citations checked, IDs added) |
| 10 | [PPO for language models](./lectures/10-ppo-for-llms.md) | unreviewed (de-slopped; a broken next-lecture link + unverified compute claims fixed) |
| 11 | [Direct preference optimization](./lectures/11-dpo.md) | unreviewed (de-slopped; a fabricated paper removed) |
| 12 | [Beyond DPO: GRPO, RRHF, IPO](./lectures/12-beyond-dpo.md) | unreviewed (de-slopped; a fabricated benchmark table + a fabricated paper removed) |
| 13 | [RLHF for code generation](./lectures/13-rlhf-code-generation.md) | unreviewed (de-slopped; CodeRL mis-attributed to Meta → fixed to Salesforce; fabricated benchmark numbers removed) |
| 14 | [Constitutional AI, RLAIF, self-improvement](./lectures/14-constitutional-ai-rlaif.md) | unreviewed (new draft) |
| 15 | [RL with verifiable rewards & reasoning models](./lectures/15-rl-verifiable-rewards.md) | unreviewed (new draft) |
| 16 | [Agentic RL: tool use, multi-turn](./lectures/16-agentic-rl.md) | unreviewed (new draft) |
| 17 | [Online & iterative preference optimization](./lectures/17-online-iterative-preference.md) | unreviewed (new draft) |
| 18 | [Distillation of reasoning models](./lectures/18-distillation-reasoning.md) | unreviewed (new draft) |
| 19 | [Offline RL](./lectures/19-offline-rl.md) | unreviewed (new draft) |

Planned: a curated paper layer in [`../reference/papers/`](../reference/papers/), built from `../tools/lit-builder/` once the LLM scoring step has been run (it needs a credential). Optionally: an exploration lecture (intrinsic motivation, count-based methods, RND) — the one remaining foundational gap.

Cheat sheets and diagrams are in [`cheat-sheets/`](./cheat-sheets/) and [`diagrams/`](./diagrams/) — also unreviewed.

## How to use this

Starting from scratch: do 01–05 in order, type out the code yourself, and don't move on from a lecture until you can explain its method without notes. Then 06–08, then 09 onward.

Already know RL, here for the LLM part: skim 01–05 for notation, then go 09 → 10 → 11 → 12. Lecture 13 if you care about code generation specifically.

Here for code generation: 02 (policy-gradient intuition), 10 (PPO for LLMs), 11–13.

## Prerequisites

- Calculus (derivatives, chain rule, gradients), probability (expectations, distributions, KL divergence), basic linear algebra. The math is explained as it comes up.
- Python at an intermediate level; PyTorch basics (the code uses PyTorch); NumPy.
- Budget a few hours per lecture including coding and debugging.

## Study notes that hold up

- Type the code out. Don't paste it.
- Break it on purpose — change a hyperparameter until it fails, then work out why.
- If you can't explain a method simply, you don't have it yet.
- After coding a method, read the original paper. It reads very differently once you've implemented it.
- Print shapes when something's wrong. Most RL bugs are shape or sign errors.

## Supplementary resources

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.)
- Spinning Up in Deep RL (OpenAI) — explanations plus reference implementations
- David Silver's UCL lectures
- Recent papers, by topic, in [`../reference/papers/`](../reference/papers/)

The lectures are meant to stand on their own, but they'll make more sense alongside these.
