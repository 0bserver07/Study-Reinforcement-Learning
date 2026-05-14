# Exercise 01 — value iteration on a gridworld

Goes with [Lecture 01: MDPs and Bellman equations](../../notes/lectures/01-mdps-bellman.md).

You'll implement value iteration — repeated Bellman-optimality backups — and use it to solve a small gridworld exactly. The environment (`GridWorldMDP`) is given; you write the algorithm. No deep learning, no PyTorch — just NumPy and the Bellman equation.

## The task

Fill in the `TODO`s in [`starter.py`](./starter.py). Two functions:

1. `value_iteration(mdp, gamma, theta)` — start `V` at zeros, then sweep: for each non-terminal state, set `V[s] = max_a Σ_{(p, s', r) ∈ transitions(s, a)} p · (r + γ V[s'])`. Keep sweeping until the largest change in a sweep drops below `theta`. Return `V`.
2. `greedy_policy(mdp, V, gamma)` — for each non-terminal state, the action that maximizes `Σ p · (r + γ V[s'])`. Return an int array.

`GridWorldMDP` and `follow_greedy` are given. The MDP interface you use: `mdp.n_states`, `mdp.n_actions`, `mdp.is_terminal(s)`, `mdp.transitions(s, a) -> [(prob, next_state, reward), ...]`. Writing the backup against `transitions` (which returns a list) rather than assuming determinism means the same `value_iteration` works on stochastic MDPs too — handy for the variations below.

## Acceptance criteria

```bash
pytest exercises/01-mdps/
```

passes:

- `value_iteration` matches the hand-computed `V*` on two tiny MDPs (a 2-state one and a 3-state chain).
- `greedy_policy` picks the action that reaches the paying terminal state.
- On the 5×5 gridworld: `V*` is +10 at a goal-adjacent cell, decreases as you move away from the goal, and is 0 at the (terminal) goal; and the greedy policy reaches the goal from every state.

You're done when the tests pass *and* you can explain: why value iteration converges (the Bellman-optimality operator is a γ-contraction in max-norm), and why you skip terminal states in the sweep.

## If you get stuck

[`HINTS.md`](./HINTS.md), one at a time. Reference solution: [`solution/value_iteration.py`](./solution/value_iteration.py) — after you've tried.

## Going further (optional)

- **Stochastic transitions.** Make a `SlipperyGridWorld` where each action does what you asked with probability 0.8 and slips to a perpendicular cell with probability 0.1 each — so `transitions(s, a)` returns three entries. Your `value_iteration` should work unchanged. How does `V*` and the policy change near holes/walls?
- **Add obstacles** with a big negative reward, and watch the policy route around them.
- **Compare to policy iteration** (Lecture 01, Part 5): which does fewer total backups on this gridworld? Which does fewer *sweeps*?
- Sweep γ over `{0.5, 0.9, 0.99, 0.999}` and look at how the value function and the policy change. At low γ, why does the agent stop caring about a far-away goal?
