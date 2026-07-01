<!-- status: unreviewed | last-reviewed: never -->

# homework

Problem sets in the style of a real RL course (the Berkeley CS294 model: HW1 through HW5). Each homework combines theory questions, coding problems, and short reading. Solutions in a separate file, try the problems first.

The coding parts link to [`../exercises/`](../exercises/) (which are tested).

## What's here

| HW | Topic | Lecture(s) | Status |
|---|---|---|---|
| [HW1](./hw01-mdps-and-value-iteration/) | MDPs, Bellman equations, value iteration | L01 | ready |
| HW2 | Policy gradients, REINFORCE | L02 | planned |
| HW3 | Q-learning, DQN | L03 | planned |
| HW4 | Actor-critic, PPO | L04, L06 | planned |
| HW5 | Reward modeling, RLHF, DPO | L09–L11 | planned |
| HW6 | GRPO and RL with verifiable rewards | L12, L15 | planned |
| HW7 | Agentic RL or offline RL (pick one) | L16 or L19 | planned |

## How to use

Each `hwNN/` directory has:

- `README.md`: what the HW covers, what you should know going in, time estimate
- `problems.md`: numbered problems (theory + coding + reading)
- `solutions.md`: full worked solutions; **try every problem first**

Theory problems should take 10–30 minutes each with paper. Coding problems link to the matching `exercises/NN/` (with starter, tests, reference solution, hints). Reading problems point you at specific Sutton & Barto chapters or original papers.

## Why bother with paper-and-pencil theory in 2026

Because if you can't derive the policy gradient theorem from scratch, you don't actually understand what you're optimizing: you're just running someone else's code. Same for the Bellman equations, the contraction proof for value iteration, the variance argument for baselines, the DPO derivation. The theory is short. Doing it once builds the intuition that makes the code make sense.
