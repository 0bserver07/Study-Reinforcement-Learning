# Exercises

Small, tested coding exercises that go with the lectures. Each one is a `NN-topic/` directory:

- `README.md` — the task and the acceptance criteria (which tests must pass)
- `starter.py` — a skeleton with `TODO`s; this is the file you edit
- `test_*.py` — `pytest` tests that pass only when your implementation is correct
- `solution/` — a reference implementation, for comparing after you've tried
- `HINTS.md` — graduated hints; hint 1 is a nudge, the last is nearly the answer

## Running them

From the repo root:

```bash
pip install -r exercises/requirements.txt    # torch, numpy, gymnasium, pytest
pytest exercises/02-policy-gradients/         # run one exercise's tests
pytest exercises/                             # run all of them
```

Tests that need PyTorch are skipped cleanly if it isn't installed.

## Working with an AI tutor

If you're using Claude Code or Codex: it should have you edit `starter.py`, run the tests, and when you're stuck give you the *next* hint from `HINTS.md` — not the solution, not two hints ahead. It should only show you `solution/` if you ask for it directly. See [`../AGENTS.md`](../AGENTS.md#how-exercises-work). An exercise is done when its tests pass and you can explain why your implementation works.

## What's here

| Exercise | Goes with | Status |
|---|---|---|
| [`01-mdps/`](./01-mdps/) — value iteration on a gridworld | Lecture 01 | ready |
| [`02-policy-gradients/`](./02-policy-gradients/) — REINFORCE, solve CartPole | Lecture 02 | ready |
| [`03-q-learning/`](./03-q-learning/) — tabular Q-learning on FrozenLake | Lecture 03 | ready |
| [`04-actor-critic/`](./04-actor-critic/) — actor-critic with a learned baseline, on CartPole | Lecture 04 | ready |
| [`15-grpo-rlvr/`](./15-grpo-rlvr/) — GRPO on a verifiable arithmetic toy task | Lecture 15 | ready |

A PPO exercise on a continuous-control env is the next obvious one. Agentic RL doesn't make a fast tested exercise — that's a "go build it" project.
