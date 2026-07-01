<!-- status: unreviewed | last-reviewed: never -->

# notes: study material

Two layers live in this directory, mixed.

**Trusted, hand-written:**

- **[`cs294-2017/`](./cs294-2017/)**: personal student notes from CS 294 Deep RL (Berkeley, Spring 2017: Levine, Schulman, Finn). 246 lines of working notes from the field being built. Idiosyncratic, kept as written. `status: hand-written`.
- **[`sutton-barto-digest/`](./sutton-barto-digest/)**: short distillation of the four elements of an RL system (policy, reward, value function, model) from Sutton & Barto. `status: hand-written`.

These are old (2017) and informal, but they're a real person's understanding, not AI text. Trusted as starting points.

**AI-drafted, useful as scaffold (`unreviewed`: treat with skepticism):**

- **[`lectures/`](./lectures/)**: a 34-lecture series taking RL from MDPs through RLHF / DPO / GRPO / RLVR / agentic / offline and on into reasoning, systems, and applications. Lectures 01–19 have had an editorial pass: broken links fixed, code bugs caught (`import gym` → `gymnasium`, missing imports, old-API `env.step` calls), citations checked or removed when they didn't resolve, fake-first-person framing stripped. Lectures 20–34 are newer drafts without that pass. **In no case has a person read a lecture end to end and signed off.** Cross-check the math against the cited papers; treat the code as a starting point that needs verification. Index and per-lecture review status below.
- **[`cheat-sheets/`](./cheat-sheets/)**: `RL-Math-Formulas.md` and `RL-Quick-Reference.md`. Audited (caught a wrong KL direction; fixed). Same caveat.
- **[`diagrams/`](./diagrams/)**: `RL-Algorithm-Diagrams.md`. Audited (caught and fixed a wrong DPO loss diagram and a wrong GRPO advantage diagram). Same caveat.

[`../CURRICULUM.md`](../CURRICULUM.md) is the suggested order through everything. [`../AGENTS.md`](../AGENTS.md) explains the `<!-- status: ... -->` convention every doc carries.

## Lecture series: drafts, in order

| # | Lecture | Status |
|---|---|---|
| 01 | [MDPs and Bellman equations](./lectures/01-mdps-bellman.md), exercise: [`01-mdps`](../exercises/01-mdps/) | unreviewed (de-slopped; a fabricated value-function output was removed) |
| 02 | [Policy gradients from scratch](./lectures/02-policy-gradients.md), exercise: [`02-policy-gradients`](../exercises/02-policy-gradients/) | unreviewed (de-slopped; a broken link and a code bug were fixed) |
| 03 | [Value functions & Q-learning](./lectures/03-value-functions-q-learning.md), exercise: [`03-q-learning`](../exercises/03-q-learning/) | unreviewed (de-slopped; a dead `Modern-RL-Research/` path and a missing import fixed) |
| 04 | [Actor-critic methods](./lectures/04-actor-critic.md), exercise: [`04-actor-critic`](../exercises/04-actor-critic/) | unreviewed (de-slopped; a code bug fixed) |
| 05 | [Trust regions and TRPO](./lectures/05-trpo.md) | unreviewed (de-slopped; fabricated training times removed) |
| 06 | [PPO](./lectures/06-ppo.md) | unreviewed (de-slopped; `import gym` → `gymnasium` fixed) |
| 07 | [Off-policy learning: SAC and TD3](./lectures/07-off-policy-rl.md) | unreviewed (de-slopped; an old-API `env.step` call fixed) |
| 08 | [Model-based RL](./lectures/08-model-based-rl.md) | unreviewed (de-slopped; old-API calls + a wrong citation fixed) |
| 09 | [Reward modeling for RLHF](./lectures/09-reward-modeling.md) | unreviewed (de-slopped; citations checked, IDs added) |
| 10 | [PPO for language models](./lectures/10-ppo-for-llms.md) | unreviewed (de-slopped; a broken next-lecture link + unverified compute claims fixed) |
| 11 | [Direct preference optimization](./lectures/11-dpo.md) | unreviewed (de-slopped; a fabricated paper removed) |
| 12 | [Beyond DPO: GRPO, RRHF, IPO](./lectures/12-beyond-dpo.md) | unreviewed (de-slopped; a fabricated benchmark table + a fabricated paper removed) |
| 13 | [RLHF for code generation](./lectures/13-rlhf-code-generation.md), exercise: [`15-grpo-rlvr`](../exercises/15-grpo-rlvr/) (related) | unreviewed (de-slopped; CodeRL mis-attributed to Meta → fixed to Salesforce; fabricated benchmark numbers removed) |
| 14 | [Constitutional AI, RLAIF, self-improvement](./lectures/14-constitutional-ai-rlaif.md) | unreviewed (new draft) |
| 15 | [RL with verifiable rewards & reasoning models](./lectures/15-rl-verifiable-rewards.md), exercise: [`15-grpo-rlvr`](../exercises/15-grpo-rlvr/) | unreviewed (new draft) |
| 16 | [Agentic RL: tool use, multi-turn](./lectures/16-agentic-rl.md) | unreviewed (new draft) |
| 17 | [Online & iterative preference optimization](./lectures/17-online-iterative-preference.md) | unreviewed (new draft) |
| 18 | [Distillation of reasoning models](./lectures/18-distillation-reasoning.md) | unreviewed (new draft) |
| 19 | [Offline RL](./lectures/19-offline-rl.md) | unreviewed (new draft) |
| 20 | [Exploration: from ε-greedy to intrinsic motivation](./lectures/20-exploration.md), exercise: [`20-exploration`](../exercises/20-exploration/) | unreviewed (new draft) |
| 21 | [Multi-agent RL and self-play](./lectures/21-multi-agent-rl.md) | unreviewed (new draft) |
| 22 | [World models](./lectures/22-world-models.md) | unreviewed (new draft) |
| 23 | [Process reward models vs outcome reward models](./lectures/23-process-reward-models.md) | unreviewed (new draft) |
| 24 | [Computer use and browser agents](./lectures/24-computer-use-agents.md) | unreviewed (new draft) |
| 25 | [Long-horizon credit assignment](./lectures/25-long-horizon-credit.md) | unreviewed (new draft) |
| 26 | [RL for mathematical reasoning](./lectures/26-rl-math-reasoning.md) | unreviewed (new draft) |
| 27 | [RLAIF and synthetic preferences at scale](./lectures/27-rlaif.md) | unreviewed (new draft) |
| 28 | [Reward hacking and verifier design](./lectures/28-reward-hacking.md) | unreviewed (new draft) |
| 29 | [Distributed RL systems](./lectures/29-distributed-rl-systems.md) | unreviewed (new draft) |
| 30 | [RL inference infrastructure for LLMs](./lectures/30-rl-inference-infra.md) | unreviewed (new draft) |
| 31 | [Hardware for RL](./lectures/31-hardware-for-rl.md) | unreviewed (new draft) |
| 32 | [Meta-RL and in-context RL](./lectures/32-meta-rl-in-context.md) | unreviewed (new draft) |
| 33 | [Robotics RL](./lectures/33-robotics-rl.md) | unreviewed (new draft) |
| 34 | [Self-distillation and self-improvement loops](./lectures/34-self-distillation.md) | unreviewed (new draft) |

What "unreviewed" means here: nobody has read the lecture end-to-end and signed off on it. The editorial pass (de-slop, fix broken links, catch code bugs, verify citations) has happened for lectures 01–19; that's the parenthetical note next to those rows. Lectures 20–34 are newer drafts that haven't had even that pass yet, so treat them with more caution. The next step for any of them is a person reads it and either flips it to `reviewed` (with today's date in `last-reviewed:`) or notes what's still wrong.

Planned: a curated paper layer in [`../reference/papers/`](../reference/papers/), built from `../tools/lit-builder/` once the LLM scoring step has been run (it needs a credential: see issue #2). Two hand-curated topic READMEs have landed: [`GRPO-RLVR/`](../reference/papers/GRPO-RLVR/) and [`Agentic-RL/`](../reference/papers/Agentic-RL/), but their auto-generated `PAPERS.md` files still need a collector run.

Cheat sheets and diagrams are in [`cheat-sheets/`](./cheat-sheets/) and [`diagrams/`](./diagrams/), also unreviewed. Beyond `RL-Math-Formulas.md` and `RL-Quick-Reference.md`: `RLHF-vs-DPO-vs-GRPO.md` (side-by-side comparison of the alignment methods), `RL-LLM-loops-2026.md` (ASCII data-flow diagrams of every training loop), `KL-control.md` (KL penalties across TRPO/PPO/RLHF/DPO/GRPO), `RL-loss-functions.md` (one block per algorithm with loss, gradient, code, and tradeoff).

## How to use this

Starting from scratch: read the talks/books/courses linked in [`../readme.md`](../readme.md); they're the trusted external material. The hand-written CS294 notes at [`cs294-2017/`](./cs294-2017/) give you one student's path through the same material.

Already know RL, here for the LLM part: lectures 09 → 11 → 12 → 14 → 15 → 17 covers the RLHF → DPO → GRPO → constitutional AI → RLVR → iterative preference optimization arc.

Here for code generation specifically: lecture 02 (policy-gradient intuition), 10 (PPO for LLMs), 13 (RLHF for code), 15 (RLVR: the basis of modern reasoning-RL on code).

## Prerequisites

- Calculus (derivatives, chain rule, gradients), probability (expectations, distributions, KL divergence), basic linear algebra.
- Python at an intermediate level; PyTorch basics; NumPy.
- A few hours per lecture including coding and debugging.
