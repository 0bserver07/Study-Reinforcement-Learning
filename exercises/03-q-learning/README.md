# Exercise 03 — tabular Q-learning on FrozenLake

Goes with [Lecture 03: Value Functions & Q-Learning](../../notes/lectures/03-value-functions-q-learning.md).

You'll implement tabular Q-learning and use it to solve `FrozenLake-v1` with `is_slippery=False` (the deterministic version — Q-learning solves it cleanly). The point is to build the TD update with your own hands: ε-greedy exploration, the bootstrap target `r + γ max_a' Q(s',a')`, and the loop.

## The task

Fill in the `TODO`s in [`starter.py`](./starter.py). Three pieces:

1. `epsilon_greedy(q_row, epsilon, n_actions, rng)` — with probability `epsilon` a uniform random action, otherwise `argmax(q_row)`. Return a Python `int`.
2. `q_update(q_table, state, action, reward, next_state, done, alpha, gamma)` — one Q-learning step, in place: `target = reward` if `done` else `reward + gamma * max_a' Q[next_state, a']`; `Q[state, action] += alpha * (target - Q[state, action])`. Return the TD error.
3. `train(...)` — the loop: pick an ε-greedy action, step the env, do one `q_update`, move on; record per-episode return.

`greedy_policy_return` is given — it runs one episode of the greedy policy. On non-slippery FrozenLake the only reward is 1.0 for reaching the goal, so it returns 1.0 exactly when the greedy policy reaches the goal.

One detail in the given `train`: the Q-table starts at all-ones, not zeros — *optimistic initialization*. With zeros, `argmax` over an all-zero row always returns action 0, so the agent walks into a wall and never explores. Starting Q above the true values makes every untried action look attractive, so the agent tries everything early on. (1.0 happens to be FrozenLake's maximum return.)

## Acceptance criteria

```bash
pytest exercises/03-q-learning/
```

passes:

- `epsilon_greedy` is greedy when `epsilon=0`, and visits every action when `epsilon=1`.
- `q_update` matches hand-computed values on a terminal transition and a bootstrap transition.
- After `train(n_episodes=2000, seed=0)`, the greedy policy reaches the goal (`greedy_policy_return == 1.0`), and a good fraction of late-training episodes succeed.

You're done when the tests pass *and* you can explain: why Q-learning is "off-policy" (you explore with ε-greedy but the `max` in the target learns the *greedy* policy's values), why optimistic initialization is doing real work here, and why the deterministic version is so much easier than slippery FrozenLake.

## If you get stuck

[`HINTS.md`](./HINTS.md), one at a time. Reference solution: [`solution/q_learning.py`](./solution/q_learning.py) — after you've tried.

## Going further (optional)

- Switch to `is_slippery=True` (the default). Now even the optimal policy only wins ~74% of the time, and constant `epsilon=0.1` hurts. Add ε-decay (`epsilon = max(0.01, epsilon * 0.999)` per episode) and bump `n_episodes`; see how high a success rate you can get.
- Try `alpha=1.0` on the deterministic version — it's a deterministic env, so full replacement works and converges fast. On the slippery version, `alpha=1.0` is a disaster — work out why.
- The DQN step (not graded here): take `SimpleDQN` from the lecture, add the replay buffer and target network, train on CartPole.
