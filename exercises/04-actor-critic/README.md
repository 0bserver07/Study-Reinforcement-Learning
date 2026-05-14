# Exercise 04 — Actor-critic on CartPole

Goes with [Lecture 04: Actor-critic methods](../../notes/lectures/04-actor-critic.md).

You'll implement a simple advantage actor-critic and use it to solve `CartPole-v1`. The point is to see concretely what a learned value function buys you: instead of subtracting the episode mean (REINFORCE's crude baseline), you subtract a state-dependent estimate V(s_t). The advantage A_t = G_t - V(s_t) tells the actor whether the return from that state was better or worse than what the critic expected, which cuts variance without biasing the gradient.

## The task

Fill in the `TODO`s in [`starter.py`](./starter.py). Six pieces:

1. `PolicyNet` — same MLP as exercise 02: state vector to action logits.
2. `ValueNet` — a second MLP from the same state vector to a scalar V(s). Output shape for a single state is `[]` (scalar), for a batch it's `[batch]`.
3. `select_action` — identical to exercise 02. Sample from `Categorical(logits=...)`, return `(int, log_prob_with_gradient)`.
4. `compute_returns` — identical to exercise 02. Walk rewards backwards, accumulate `R = r + gamma * R`.
5. `actor_critic_loss(log_probs, returns, values)` — the new piece:
   - `advantages = (returns - values).detach()` — stop the gradient here. The policy gradient must not flow through the critic.
   - `actor_loss = -(stack(log_probs) * advantages).sum()` — policy gradient, negated.
   - `critic_loss = F.mse_loss(values, returns)` — train V(s) to predict G_t.
   - Return `(actor_loss, critic_loss)` separately; the train loop combines them.
6. The update step in `train` — collect `value = critic(state_t)` for each step during rollout; after the episode, compute `loss = actor_loss + value_coef * critic_loss`, then `zero_grad` → `backward` → clip grad norm at 1.0 → `step`.

The train loop scaffolding (episode rollout structure, optimizer setup, `episode_returns` list) is given.

## Acceptance criteria

```bash
pytest exercises/04-actor-critic/
```

passes. That's:

- `compute_returns` matches hand-computed values (`gamma=0.5`, `[1,1,1]` → `[1.75, 1.5, 1.0]`; `gamma=1`, `[1,1,1]` → `[3, 2, 1]`).
- `PolicyNet` and `ValueNet` output the right shapes for single and batched inputs.
- `select_action` returns `(int, scalar tensor that requires grad)`.
- `actor_critic_loss` returns two scalar tensors; the actor loss backprops into `PolicyNet`; the critic loss backprops into `ValueNet`; the actor loss does *not* produce gradients in `ValueNet` (the detach test).
- Integration test: `train(seed=0)` with the default config (600 episodes, `lr=3e-3`) — `max(returns) >= 195` and `mean(returns[-100:]) > 100`.

You're done when the tests pass *and* you can explain: why the advantages are detached before multiplying the log-probs, and what would go wrong if you forgot the `.detach()`.

## If you get stuck

Read [`HINTS.md`](./HINTS.md) — one hint at a time. The reference implementation is in [`solution/actor_critic.py`](./solution/actor_critic.py); look at it after you've made a real attempt.

## Going further (optional)

- Replace the Monte Carlo return target with a 1-step TD target: `r_t + gamma * V(s_{t+1})`. That's the TD(0) critic — faster updates within an episode, but biased. Compare convergence speed.
- Add an entropy bonus: `loss = actor_loss + value_coef * critic_loss - entropy_coef * dist.entropy().sum()`. Watch how it changes early-training exploration and whether it stabilizes the final policy.
- Try `LunarLander-v3` (needs `pip install "gymnasium[box2d]"`). The state space is 8-dimensional and the problem is harder — REINFORCE's variance becomes a real problem there, and actor-critic's state-dependent baseline starts to matter more.
- Share weights between `PolicyNet` and `ValueNet` — one trunk network whose output feeds two heads (a logit head and a value head). This is how A3C and most practical AC implementations work. The gradient interaction between heads is subtle.
