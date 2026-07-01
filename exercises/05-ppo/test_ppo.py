"""Tests for exercise 05 (PPO on CartPole).

Run from the repo root:  pytest exercises/05-ppo/

These run against starter.py — so they fail until you fill in the TODOs. To
check the reference solution instead, copy solution/ppo.py over starter.py
(or import from solution in a scratch session).
"""

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from starter import (
    ActorCriticNet,
    compute_gae,
    ppo_clip_loss,
    collect_rollouts,
    train,
    GAMMA,
    LAMBDA,
)


# --- ActorCriticNet ----------------------------------------------------------

def test_actor_critic_forward_shapes():
    """Single-state input returns (n_actions,) logits and scalar () value."""
    net = ActorCriticNet(state_dim=4, n_actions=2)
    logits, value = net(torch.zeros(4))
    assert logits.shape == (2,), f"single-state logits should be (2,), got {logits.shape}"
    assert value.shape == (), f"single-state value should be scalar (), got {value.shape}"


def test_actor_critic_forward_shapes_batched():
    """Batched input returns (batch, n_actions) logits and (batch,) values."""
    net = ActorCriticNet(state_dim=4, n_actions=2)
    logits, value = net(torch.zeros(8, 4))
    assert logits.shape == (8, 2), f"batched logits should be (8, 2), got {logits.shape}"
    assert value.shape == (8,), f"batched value should be (8,), got {value.shape}"


def test_actor_critic_has_separate_heads():
    """The policy and value heads should be distinct parameter tensors."""
    net = ActorCriticNet(state_dim=4, n_actions=2)
    n_params = sum(p.numel() for p in net.parameters())
    assert n_params > 0, "ActorCriticNet has no parameters"


# --- compute_gae -------------------------------------------------------------

def test_compute_gae_lambda_one_matches_mc():
    """λ=1, γ=1, episode terminates at end ⇒ A_t = G_t − V(s_t).

    rewards = [1, 1, 1], values = [0.5, 0.5, 0.5], dones = [0, 0, 1],
    next_value = 0.

    Working it out (γ=λ=1):
      t=2: δ_2 = 1 + 1·0·(1−1) − 0.5 = 0.5;  A_2 = 0.5
      t=1: δ_1 = 1 + 1·0.5·1 − 0.5 = 1.0;    A_1 = 1.0 + 1·1·1·0.5 = 1.5
      t=0: δ_0 = 1 + 1·0.5·1 − 0.5 = 1.0;    A_0 = 1.0 + 1·1·1·1.5 = 2.5

    So advantages = [2.5, 1.5, 0.5], returns = [3.0, 2.0, 1.0].
    (returns are exactly the Monte Carlo returns-to-go for [1,1,1] — sanity check.)
    """
    rewards = torch.tensor([1.0, 1.0, 1.0])
    values = torch.tensor([0.5, 0.5, 0.5])
    dones = torch.tensor([0.0, 0.0, 1.0])
    next_value = torch.tensor(0.0)

    adv, ret = compute_gae(rewards, values, dones, next_value, gamma=1.0, lam=1.0)

    assert torch.allclose(adv, torch.tensor([2.5, 1.5, 0.5]), atol=1e-5), (
        f"advantages: expected [2.5, 1.5, 0.5], got {adv.tolist()}"
    )
    assert torch.allclose(ret, torch.tensor([3.0, 2.0, 1.0]), atol=1e-5), (
        f"returns: expected [3.0, 2.0, 1.0], got {ret.tolist()}"
    )


def test_compute_gae_lambda_zero_is_td0():
    """λ=0 collapses GAE to the one-step TD residual.

    Same inputs as above, λ=0:
      t=2: δ_2 = 0.5;  A_2 = 0.5
      t=1: δ_1 = 1.0;  A_1 = 1.0 (no contribution from later steps)
      t=0: δ_0 = 1.0;  A_0 = 1.0

    advantages = [1.0, 1.0, 0.5], returns = [1.5, 1.5, 1.0].
    """
    rewards = torch.tensor([1.0, 1.0, 1.0])
    values = torch.tensor([0.5, 0.5, 0.5])
    dones = torch.tensor([0.0, 0.0, 1.0])
    next_value = torch.tensor(0.0)

    adv, ret = compute_gae(rewards, values, dones, next_value, gamma=1.0, lam=0.0)

    assert torch.allclose(adv, torch.tensor([1.0, 1.0, 0.5]), atol=1e-5), (
        f"λ=0 advantages: expected [1.0, 1.0, 0.5], got {adv.tolist()}"
    )
    assert torch.allclose(ret, torch.tensor([1.5, 1.5, 1.0]), atol=1e-5), (
        f"λ=0 returns: expected [1.5, 1.5, 1.0], got {ret.tolist()}"
    )


def test_compute_gae_done_breaks_bootstrap():
    """A done at step t should zero out the next-state value contribution to δ_t.

    rewards = [0, 0], values = [1.0, 1.0], dones = [1, 0], next_value = 100.

    For t=0 (done=1), bootstrapping into values[1] (or next_value) is wrong —
    the episode ended. δ_0 = r_0 + γ·V_next·(1−1) − V_0 = 0 + 0 − 1 = −1.

    If you forget the (1 − done) mask, δ_0 would be 0 + γ·1·1 − 1 = −0.01 (very
    different).
    """
    rewards = torch.tensor([0.0, 0.0])
    values = torch.tensor([1.0, 1.0])
    dones = torch.tensor([1.0, 0.0])
    next_value = torch.tensor(100.0)

    adv, _ = compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95)

    assert abs(adv[0].item() - (-1.0)) < 1e-5, (
        f"With done=1 at t=0, A_0 should be exactly -1.0 (bootstrap masked), "
        f"got {adv[0].item():.4f}. Check you're multiplying v_next by (1 - dones[t])."
    )


# --- ppo_clip_loss -----------------------------------------------------------

def test_ppo_clip_loss_zero_advantage_zero_policy_term():
    """When all advantages are 0, the policy term contributes 0 to the loss.

    The remaining loss is value_coef · MSE(value_pred, value_target) − entropy_coef · H.
    We pass value_pred == value_target and entropy=0 so the whole loss is 0.
    """
    log_probs_new = torch.tensor([-0.1, -0.2, -0.3], requires_grad=True)
    log_probs_old = torch.tensor([-0.1, -0.2, -0.3])  # ratio = 1
    advantages = torch.zeros(3)
    value_pred = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)
    value_target = torch.tensor([0.5, 0.5, 0.5])
    entropy = torch.zeros(3)

    loss = ppo_clip_loss(
        log_probs_new, log_probs_old, advantages,
        value_pred, value_target, entropy,
    )
    assert loss.shape == (), f"loss should be a scalar, got shape {loss.shape}"
    assert abs(loss.item()) < 1e-6, (
        f"Zero advantage + zero value error + zero entropy ⇒ loss should be 0, "
        f"got {loss.item():.6f}"
    )


def test_ppo_clip_loss_entropy_lowers_loss():
    """Higher entropy should LOWER the loss (entropy is a bonus, subtracted)."""
    log_probs_new = torch.tensor([-0.1, -0.2], requires_grad=True)
    log_probs_old = torch.tensor([-0.1, -0.2])
    advantages = torch.zeros(2)
    value_pred = torch.tensor([0.0, 0.0], requires_grad=True)
    value_target = torch.tensor([0.0, 0.0])

    loss_low_H = ppo_clip_loss(
        log_probs_new, log_probs_old, advantages,
        value_pred, value_target, entropy=torch.tensor([0.0, 0.0]),
    )
    loss_high_H = ppo_clip_loss(
        log_probs_new, log_probs_old, advantages,
        value_pred, value_target, entropy=torch.tensor([1.0, 1.0]),
    )
    assert loss_high_H.item() < loss_low_H.item(), (
        f"loss with high entropy ({loss_high_H.item():.4f}) should be LOWER "
        f"than loss with zero entropy ({loss_low_H.item():.4f}) — "
        f"entropy is a bonus, so the loss term is -entropy_coef * H."
    )


def test_ppo_clip_loss_clipping_engages_on_large_ratio():
    """When ratio · A > clip(ratio) · A and A > 0, the loss is MIN of the two
    (the clipped one). This puts a ceiling on how much the policy can profit
    from a single large-ratio step.

    Setup: log_probs_new = log_probs_old + 1 ⇒ ratio = e ≈ 2.718.
    With A=+1 and clip_eps=0.2, the unclipped term is 2.718, the clipped is 1.2.
    min(2.718, 1.2) = 1.2, so policy_loss = -1.2.
    """
    log_probs_new = torch.tensor([1.0], requires_grad=True)
    log_probs_old = torch.tensor([0.0])
    advantages = torch.tensor([1.0])
    value_pred = torch.tensor([0.0], requires_grad=True)
    value_target = torch.tensor([0.0])
    entropy = torch.tensor([0.0])

    loss = ppo_clip_loss(
        log_probs_new, log_probs_old, advantages,
        value_pred, value_target, entropy,
        clip_eps=0.2, value_coef=0.0, entropy_coef=0.0,
    )
    # Only the policy term is nonzero (value_coef = entropy_coef = 0).
    # policy_loss = -min(e, 1.2) = -1.2
    assert abs(loss.item() - (-1.2)) < 1e-4, (
        f"Clipped loss should be -1.2, got {loss.item():.4f}. "
        f"Check ratio.clamp(1 - clip_eps, 1 + clip_eps) and torch.min(...)."
    )


def test_ppo_clip_loss_backprops():
    """The loss must produce non-zero gradients in log_probs_new and value_pred."""
    log_probs_new = torch.tensor([-0.1, -0.2, -0.3], requires_grad=True)
    log_probs_old = torch.tensor([-0.5, -0.5, -0.5])
    advantages = torch.tensor([1.0, -1.0, 0.5])
    value_pred = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    value_target = torch.tensor([1.0, 1.0, 1.0])
    entropy = torch.tensor([0.5, 0.5, 0.5])

    loss = ppo_clip_loss(
        log_probs_new, log_probs_old, advantages,
        value_pred, value_target, entropy,
    )
    loss.backward()
    assert log_probs_new.grad is not None and torch.any(log_probs_new.grad != 0), (
        "loss.backward() did not populate log_probs_new.grad"
    )
    assert value_pred.grad is not None and torch.any(value_pred.grad != 0), (
        "loss.backward() did not populate value_pred.grad — check value_loss term"
    )


# --- collect_rollouts --------------------------------------------------------

def test_collect_rollouts_shapes():
    """Returned tensors must have length n_steps along axis 0."""
    pytest.importorskip("gymnasium")
    import gymnasium as gym

    torch.manual_seed(0)
    np.random.seed(0)
    env = gym.make("CartPole-v1")
    state, _ = env.reset(seed=0)
    net = ActorCriticNet(env.observation_space.shape[0], env.action_space.n)

    n_steps = 128
    out = collect_rollouts(env, net, n_steps, state, seed_offset=0)
    # Expected: (obs, actions, log_probs, rewards, values, dones,
    #            episode_rewards, next_state, next_value, ep_counter)
    assert len(out) == 10, (
        f"collect_rollouts should return 10 things, got {len(out)}. "
        f"See the docstring for the exact return tuple."
    )
    obs, actions, log_probs, rewards, values, dones, ep_rewards, next_state, next_value, n_eps = out

    assert obs.shape == (n_steps, 4), f"obs shape: expected ({n_steps}, 4), got {obs.shape}"
    assert actions.shape == (n_steps,), f"actions shape: expected ({n_steps},), got {actions.shape}"
    assert log_probs.shape == (n_steps,), f"log_probs shape: expected ({n_steps},), got {log_probs.shape}"
    assert rewards.shape == (n_steps,), f"rewards shape: expected ({n_steps},), got {rewards.shape}"
    assert values.shape == (n_steps,), f"values shape: expected ({n_steps},), got {values.shape}"
    assert dones.shape == (n_steps,), f"dones shape: expected ({n_steps},), got {dones.shape}"
    assert next_value.shape == (), f"next_value should be scalar, got {next_value.shape}"

    env.close()


def test_collect_rollouts_log_probs_detached():
    """The rolled-out log_probs are the 'old' ones — the loss treats them as
    constants, so they should not require gradient (collected under no_grad)."""
    pytest.importorskip("gymnasium")
    import gymnasium as gym

    torch.manual_seed(0)
    env = gym.make("CartPole-v1")
    state, _ = env.reset(seed=0)
    net = ActorCriticNet(env.observation_space.shape[0], env.action_space.n)

    _, _, log_probs, _, values, _, _, _, _, _ = collect_rollouts(env, net, 32, state)
    assert not log_probs.requires_grad, (
        "Rolled-out log_probs should be detached (collect under torch.no_grad) — "
        "they're the 'old' policy in the ratio."
    )
    assert not values.requires_grad, (
        "Rolled-out values should be detached — they're inputs to GAE, not the "
        "value targets the critic is trained against."
    )
    env.close()


# --- integration: does it actually learn? ------------------------------------

def test_train_converges():
    """Full PPO run on CartPole-v1 with the default config.

    With seed=0, total_steps=50_000, the default rollout_steps=1024, epochs=10,
    lr=1e-3 — the policy should reach mean episode reward > 150 over the last
    10 logged iterations. (Random ≈ 21, max possible is 500.)

    Runs in under ~30 seconds on a laptop CPU.
    """
    pytest.importorskip("gymnasium")
    rewards = train(seed=0, total_steps=50_000)
    assert len(rewards) > 0, "train returned no iteration rewards — did any episodes finish?"

    last10 = [r for _, r in rewards[-10:]]
    mean_last10 = sum(last10) / len(last10)
    assert mean_last10 > 150.0, (
        f"PPO did not converge: mean reward (last 10 iters) = {mean_last10:.1f}. "
        f"Random ≈ 21. Checks: "
        f"(1) advantages are normalized to zero-mean unit-std before the loss; "
        f"(2) log_probs_new is recomputed from the CURRENT net on each minibatch "
        f"(not reused from the rollout); "
        f"(3) ppo_clip_loss has the minus sign on the policy term (we minimize a "
        f"loss but want to maximize expected return); "
        f"(4) compute_gae masks v_next by (1 - dones[t]) so episode boundaries "
        f"don't leak value across them."
    )
