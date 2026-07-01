# Exercise 05 — PPO on CartPole

Goes with [Lecture 06: PPO — Proximal Policy Optimization](../../notes/lectures/06-ppo.md).

You'll implement PPO-Clip from scratch and use it to solve `CartPole-v1`. After REINFORCE (exercise 02) and actor-critic (exercise 04), this is the algorithm that finally gives you stable, sample-efficient learning by (a) re-using each batch of rollouts for several gradient epochs and (b) clipping the probability ratio so a single noisy minibatch can't blow up the policy.

The five pieces you fill in are the same pieces every "real" PPO implementation has — a shared-trunk actor-critic, GAE, the clipped surrogate loss, a rollout collector, and the outer loop. Everything is small enough that the training run finishes in ~15 seconds on a laptop CPU.

## The task

Fill in the TODOs in [`starter.py`](./starter.py). Five pieces:

1. `ActorCriticNet` — small MLP with a shared trunk (`state_dim → 64 → 64`, tanh), a policy head producing logits, and a value head producing a scalar V(s). One network, two heads.
2. `compute_gae` — Generalized Advantage Estimation. Walk the rollout backwards, building up the GAE by mixing one-step TD residuals through `λ`. Return both the advantages and the returns (advantages + values, the critic target).
3. `ppo_clip_loss` — the full PPO loss: clipped policy surrogate + value MSE − entropy bonus. The clipping is what makes this PPO and not just actor-critic with a ratio.
4. `collect_rollouts` — run the env for `n_steps` with the current policy; return tensors of obs, actions, log_probs, rewards, values, dones. Collect everything under `torch.no_grad()` — the loss recomputes log_probs and values from the (post-update) net each minibatch.
5. `train` — the outer loop: rollout → GAE → normalize advantages → `epochs` passes of minibatch SGD on `ppo_clip_loss` → clip grad norm → step. Return per-iteration mean episode reward.

## Setup

```bash
pip install -r exercises/requirements.txt
pytest exercises/05-ppo/
```

## Acceptance criteria

`pytest exercises/05-ppo/` passes. That's:

- `ActorCriticNet` outputs logits of shape `(n_actions,)` and a scalar value for a single state; `(batch, n_actions)` and `(batch,)` for a batch.
- `compute_gae` matches hand-computed values for two test cases (`λ=1`/`γ=1` reduces to Monte Carlo returns-to-go; `λ=0` reduces to one-step TD), and respects the `done` mask at episode boundaries.
- `ppo_clip_loss` returns a scalar that backprops into both the policy log-probs and the value predictions; zero advantages with zero value error and zero entropy give exactly zero loss; the clip engages when the ratio leaves `[1−ε, 1+ε]`; higher entropy lowers the loss (entropy is a bonus).
- `collect_rollouts` returns tensors with the right shapes and `requires_grad=False` log_probs and values (they're the "old" rollout, treated as constants in the loss).
- Integration test: `train(seed=0, total_steps=50_000)` reaches mean episode reward > 150 over the last 10 logged iterations. CartPole's max is 500; random is ~21. The run takes ~15s on a laptop CPU.

You're done when the tests pass *and* you can explain:

- **Why the clip prevents large policy updates.** Specifically, what does the `min(unclipped, clipped)` give you that a one-sided clip wouldn't?
- **What GAE's `λ` controls.** What happens at `λ=0`? At `λ=1`? Why is `λ=0.95` the standard default?
- **Why entropy is added to the loss.** What goes wrong if you set `entropy_coef=0` and the policy gets unlucky early?

## If you get stuck

Read [`HINTS.md`](./HINTS.md) — one hint at a time. The reference implementation is in [`solution/ppo.py`](./solution/ppo.py); look at it after you've made a real attempt.

## Going further (optional)

- Add **value clipping**: when computing the value loss, clip the value prediction to within ε of the rolled-out value (`v_clipped = v_old + clamp(v_new - v_old, -ε, ε)`), and take the max of the unclipped and clipped MSE. This is a standard PPO refinement that prevents the value function from making large jumps. Compare convergence.
- Track the **clip fraction** (the fraction of ratios that hit the clip boundary). If it's > 0.5 your clip is too tight or your `epochs` is too high; if it's < 0.05 your clip is doing nothing. Print it during training and tune.
- Try `LunarLander-v3` (needs `pip install "gymnasium[box2d]"`). Same algorithm, just an 8-dim state space and continuous-ish dynamics; you'll need more `total_steps` (~200k) and possibly a smaller `lr`.
- The full PPO paper uses a separate optimizer and a learning-rate schedule that anneals to zero. Add the schedule and see if it helps.
- Replace the categorical policy with a Gaussian one (`Normal(mean, std)` heads), and try a continuous-control env like `Pendulum-v1`. The clipping, GAE, and outer loop are unchanged — only the policy distribution differs.
