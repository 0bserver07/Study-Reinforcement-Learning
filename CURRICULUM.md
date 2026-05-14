# Curriculum

The order to go through the material in, with prerequisites, rough time, and a checkpoint for each step — a thing you should be able to do before moving on. The lectures live in [`notes/lectures/`](./notes/lectures/) (index and review status in [`notes/README.md`](./notes/README.md)); exercises in [`exercises/`](./exercises/). The lecture bodies haven't been reviewed yet — treat their claims as unverified and check against the references.

Setup before you start: Python 3.10+, PyTorch, NumPy, Gymnasium (`pip install gymnasium`), and `pytest` for the exercises. A laptop is enough for everything through lecture 08. The LLM lectures (09+) describe full-scale training; you can follow the code without running it at scale.

## Block 1 — foundations (lectures 01–05)

Don't skip this block. The LLM material later leans on all of it.

| Step | Prereq | Time | Checkpoint |
|---|---|---|---|
| 01 — MDPs and Bellman equations | none | 2–3 h | Write down both Bellman equations and explain what value iteration does, without notes. |
| 02 — Policy gradients from scratch | 01, basic calculus | 3–4 h + exercise | Derive ∇J = E[∇ log π · R] and say why the environment dynamics drop out. Then do the exercise: [`exercises/02-policy-gradients/`](./exercises/02-policy-gradients/) — implement REINFORCE, get CartPole solved, tests green. |
| 03 — Value functions & Q-learning | 01, 02 | 2–3 h | Explain when you'd reach for a value-based method instead of a policy-gradient one. |
| 04 — Actor-critic | 02, 03 | 2–3 h | Explain why subtracting a learned baseline reduces variance without biasing the gradient. |
| 05 — Trust regions and TRPO | 02, 04 | 2–4 h | Explain what "trust region" means here and what problem it's solving. |

## Block 2 — modern RL (lectures 06–08)

| Step | Prereq | Time | Checkpoint |
|---|---|---|---|
| 06 — PPO | 02, 04, 05 | 2–3 h | Write the clipped surrogate objective and explain what the clip does and why. |
| 07 — Off-policy learning: SAC, TD3 | 03, 04 | 2–3 h | Explain what a replay buffer buys you and why off-policy methods need extra machinery to stay stable. |
| 08 — Model-based RL | 03 | 2–3 h | Explain when learning a model is worth it and how model error bites you. |

## Block 3 — RL for language models (lectures 09–13)

| Step | Prereq | Time | Checkpoint |
|---|---|---|---|
| 09 — Reward modeling for RLHF | 02 | 2–3 h | Explain how a preference dataset becomes a scalar reward model (Bradley-Terry), and how reward models get gamed. |
| 10 — PPO for language models | 06, 09 | 3–4 h | Walk through the RLHF loop end to end: SFT → reward model → PPO with a KL penalty to the reference policy. |
| 11 — Direct preference optimization | 09, 10 | 2–3 h | Explain how DPO skips the explicit reward model and what it's implicitly optimizing. |
| 12 — Beyond DPO: GRPO, RRHF, IPO | 10, 11 | 2–4 h | Say what GRPO changes relative to PPO and why that matters for reasoning-style training. |
| 13 — RLHF for code generation | 02, 10 | 2–3 h | Explain why unit-test pass/fail is a good reward signal for code and where it falls short. |

## Block 4 — reasoning, agents, and the modern stack (lectures 14–18)

Recent material, and the field moves fast — treat these as a map of the territory, not the last word. Newer drafts, so the `unreviewed` caveat applies maybe more than elsewhere.

| Step | Prereq | Time | Checkpoint |
|---|---|---|---|
| 14 — Constitutional AI, RLAIF, self-improvement | 09, 10, 11 | 2–3 h | Explain the two phases of Constitutional AI; name one bias an LLM judge has and how you'd fight it. |
| 15 — RL with verifiable rewards & reasoning models | 02, 06, 12 | 3–4 h | Write GRPO's group-relative advantage and say why it removes the need for a value network; sketch the R1-Zero loop. Then do the exercise: [`exercises/15-grpo-rlvr/`](./exercises/15-grpo-rlvr/) — implement GRPO on a verifiable toy task. |
| 16 — Agentic RL: tool use, multi-turn | 01, 06, 10, 15 | ~3 h | Explain why long-horizon agentic tasks make credit assignment hard, and why "reward = tests pass" is gameable. |
| 17 — Online & iterative preference optimization | 09, 11, 12 | 2–3 h | Explain why offline DPO underperforms PPO-RLHF, and what iterating the loop fixes. |
| 18 — Distillation of reasoning models | 09, 15 | ~2 h | Explain the R1-distill recipe and why a small model can imitate a big reasoner's chains but couldn't discover them via RL alone. |

## Block 5 — foundational topic that didn't fit earlier (lecture 19)

| Step | Prereq | Time | Checkpoint |
|---|---|---|---|
| 19 — Offline RL (BCQ, CQL, IQL, Decision Transformer) | 03, 04, 07; helpful: 11 | 2–3 h | Explain why naive Q-learning on a fixed dataset produces optimistic Q-values for out-of-distribution actions, and how IQL avoids the problem. |

You can read 19 anywhere after Block 1; it's placed here because the bridge it draws to DPO (lecture 11) — DPO is offline preference learning — is easier to see once you've done the LLM material.

## Planned

- A curated paper layer in [`reference/papers/`](./reference/papers/), built from `tools/lit-builder/` once the LLM scoring step has been run (it needs a credential).
- Optionally: an exploration lecture (intrinsic motivation, count-based methods, RND).

## If you already know RL

Skim 01–05 for notation, then go 09 → 10 → 11 → 12, and 13 if you care about code. The reading lists in [`reference/papers/`](./reference/papers/) are the place to go deeper.

## Working alone vs. with an agent

If you're using an AI tutor (Claude Code, Codex): point it at [`AGENTS.md`](./AGENTS.md). For an exercise, it should have you edit `starter.py`, run `pytest exercises/NN-topic/`, and hand you the next hint from `HINTS.md` when you're stuck — not the solution. An exercise is done when the tests pass and you can explain why your implementation works.
