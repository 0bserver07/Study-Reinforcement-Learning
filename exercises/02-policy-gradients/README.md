# Exercise 02: REINFORCE on CartPole

Goes with [Lecture 02: Policy gradients from scratch](../../notes/lectures/02-policy-gradients.md).

You'll implement REINFORCE (Monte Carlo policy gradient) and use it to solve `CartPole-v1`. The point is to build the policy-gradient update with your own hands: sampling actions, computing returns-to-go, turning `∇J = E[∑ ∇log π(a|s) · A]` into a loss you can `.backward()`.

## The task

Fill in the `TODO`s in [`starter.py`](./starter.py). There are five pieces:

1. `PolicyNet`: a small MLP from a state vector to action **logits** (no softmax in the network).
2. `select_action`: sample an action from the policy; return `(action, log_prob)` where `log_prob` is a scalar tensor that still carries gradient.
3. `compute_returns`: discounted return-to-go for each timestep: `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …`.
4. `reinforce_loss`: the REINFORCE objective written as a loss to *minimize*. Normalize the returns (zero mean, unit std) first: that's a baseline plus scaling; it cuts variance and doesn't bias the gradient in expectation.
5. `train`: the loop: roll out an episode, compute the loss, then `zero_grad` → `backward` → clip the gradient (`torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)`) → `step`. The clip matters here: a noisy episode can produce a huge gradient and the policy collapses.

## Acceptance criteria

```bash
pytest exercises/02-policy-gradients/
```

passes. That's:

- `compute_returns` matches hand-computed returns for known reward sequences (including `γ = 1` and length-1 episodes).
- `PolicyNet` outputs the right shape for single and batched inputs.
- `select_action` returns a valid action and a scalar `log_prob` that requires grad.
- `reinforce_loss` returns a scalar that backpropagates into the policy.
- The integration test: with the default config (`train(seed=0)` → 800 episodes, `lr=1e-3`, gradient clipping), training learns CartPole: the best episode reaches at least 195 (a random policy never gets close), and the last-100 average stays well above random. This runs the full loop, so it takes a couple of minutes. REINFORCE is noisy: it can hit 500 and then dip; that's expected, and it's what a value-function baseline (Lecture 04) helps with.

You're done when the tests pass *and* you can explain: why the environment dynamics don't appear in the gradient, and why subtracting the mean return doesn't bias it.

## If you get stuck

Read [`HINTS.md`](./HINTS.md), one hint at a time. The reference implementation is in [`solution/reinforce.py`](./solution/reinforce.py); look at it after you've made a real attempt.

## Going further (optional)

- Replace the normalized-returns baseline with a learned value-function baseline (`b(s) = V(s)`), trained with MSE against the returns. That's the step from REINFORCE toward actor-critic (Lecture 04).
- Add an entropy bonus to the loss and watch how it changes exploration.
- Try `LunarLander-v3` (needs `pip install "gymnasium[box2d]"`): harder, and REINFORCE's variance starts to hurt.
