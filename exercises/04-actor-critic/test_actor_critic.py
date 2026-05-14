"""Tests for exercise 04 (actor-critic on CartPole).

Run from the repo root:  pytest exercises/04-actor-critic/

These run against starter.py — so they fail until you fill in the TODOs. To
check the reference solution instead, copy solution/actor_critic.py over
starter.py (or import from solution in a scratch session).
"""

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from starter import (
    PolicyNet,
    ValueNet,
    select_action,
    compute_returns,
    actor_critic_loss,
    train,
)


# --- compute_returns ---------------------------------------------------------

def test_compute_returns_known_sequence():
    # gamma = 0.5, rewards [1, 1, 1]:
    #   G2 = 1
    #   G1 = 1 + 0.5*1   = 1.5
    #   G0 = 1 + 0.5*1.5 = 1.75
    out = compute_returns([1.0, 1.0, 1.0], gamma=0.5)
    assert torch.allclose(out, torch.tensor([1.75, 1.5, 1.0]))


def test_compute_returns_undiscounted():
    out = compute_returns([1.0, 1.0, 1.0], gamma=1.0)
    assert torch.allclose(out, torch.tensor([3.0, 2.0, 1.0]))


def test_compute_returns_single_step():
    out = compute_returns([5.0], gamma=0.99)
    assert out.shape == (1,)
    assert torch.allclose(out, torch.tensor([5.0]))


def test_compute_returns_length_matches():
    rewards = [1.0] * 17
    assert compute_returns(rewards).shape == (17,)


# --- PolicyNet ---------------------------------------------------------------

def test_policy_net_output_shape_single():
    net = PolicyNet(state_dim=4, n_actions=2)
    out = net(torch.zeros(4))
    assert out.shape == (2,)


def test_policy_net_output_shape_batched():
    net = PolicyNet(state_dim=4, n_actions=2)
    out = net(torch.zeros(8, 4))
    assert out.shape == (8, 2)


# --- ValueNet ----------------------------------------------------------------

def test_value_net_output_shape_single():
    net = ValueNet(state_dim=4)
    out = net(torch.zeros(4))
    assert out.shape == (), f"expected scalar (), got {out.shape}"


def test_value_net_output_shape_batched():
    net = ValueNet(state_dim=4)
    out = net(torch.zeros(8, 4))
    assert out.shape == (8,), f"expected (8,), got {out.shape}"


# --- select_action -----------------------------------------------------------

def test_select_action_returns_int_and_gradient_logprob():
    net = PolicyNet(state_dim=4, n_actions=2)
    action, log_prob = select_action(net, np.zeros(4, dtype=np.float32))
    assert isinstance(action, int) and action in (0, 1)
    assert isinstance(log_prob, torch.Tensor)
    assert log_prob.shape == ()           # scalar
    assert log_prob.requires_grad         # loss must be able to backprop through it
    assert log_prob.item() <= 0.0         # log of a probability


# --- actor_critic_loss -------------------------------------------------------

def test_actor_critic_loss_returns_two_scalars():
    policy = PolicyNet(state_dim=4, n_actions=2)
    critic = ValueNet(state_dim=4)
    states = [np.zeros(4, dtype=np.float32)] * 5
    log_probs = [select_action(policy, s)[1] for s in states]
    values = torch.stack([critic(torch.zeros(4)) for _ in range(5)])
    returns = compute_returns([1.0] * 5)
    actor_loss, critic_loss = actor_critic_loss(log_probs, returns, values)
    assert actor_loss.shape == ()
    assert critic_loss.shape == ()


def test_actor_loss_backprops_into_policy():
    """actor_loss.backward() must produce non-zero grads in PolicyNet."""
    policy = PolicyNet(state_dim=4, n_actions=2)
    critic = ValueNet(state_dim=4)
    log_probs = [select_action(policy, np.zeros(4, dtype=np.float32))[1] for _ in range(5)]
    values = torch.stack([critic(torch.zeros(4)) for _ in range(5)])
    returns = compute_returns([1.0] * 5)
    actor_loss, _ = actor_critic_loss(log_probs, returns, values)
    actor_loss.backward()
    grads = [p.grad for p in policy.parameters()]
    assert any(g is not None and torch.any(g != 0) for g in grads), (
        "actor_loss did not produce non-zero gradients in PolicyNet. "
        "Check that log_probs still carry gradient (don't .detach() or .item() them)."
    )


def test_critic_loss_backprops_into_value_net():
    """critic_loss.backward() must produce non-zero grads in ValueNet."""
    policy = PolicyNet(state_dim=4, n_actions=2)
    critic = ValueNet(state_dim=4)
    log_probs = [select_action(policy, np.zeros(4, dtype=np.float32))[1] for _ in range(5)]
    values = torch.stack([critic(torch.zeros(4)) for _ in range(5)])
    returns = compute_returns([1.0] * 5)
    _, critic_loss = actor_critic_loss(log_probs, returns, values)
    critic_loss.backward()
    grads = [p.grad for p in critic.parameters()]
    assert any(g is not None and torch.any(g != 0) for g in grads), (
        "critic_loss did not produce non-zero gradients in ValueNet. "
        "Check that values came from critic(state_t) and were not detached."
    )


def test_advantages_do_not_flow_into_value_net():
    """The advantage used in the actor loss must not carry gradient into ValueNet.

    This is the key structural property of actor-critic: the critic teaches the
    actor a baseline, but the actor loss doesn't train the critic.
    actor_critic_loss must call .detach() on (returns - values) before using
    it to weight the log-probs.
    """
    policy = PolicyNet(state_dim=4, n_actions=2)
    critic = ValueNet(state_dim=4)
    log_probs = [select_action(policy, np.zeros(4, dtype=np.float32))[1] for _ in range(5)]
    values = torch.stack([critic(torch.zeros(4)) for _ in range(5)])
    returns = compute_returns([1.0] * 5)
    actor_loss, _ = actor_critic_loss(log_probs, returns, values)
    actor_loss.backward()
    # The actor_loss backward should NOT have touched ValueNet params.
    grads = [p.grad for p in critic.parameters()]
    assert all(g is None or torch.all(g == 0) for g in grads), (
        "actor_loss.backward() wrote gradients into ValueNet — the advantages "
        "are not detached. Use (returns - values).detach() in actor_critic_loss."
    )


# --- integration: does it actually learn? ------------------------------------

def test_actor_critic_learns_cartpole():
    """Full training run on CartPole-v1 with the default config.

    Actor-critic should converge faster than REINFORCE (exercise 02) because the
    critic gives a state-dependent baseline instead of the episode mean. We check:
      - the best episode reached at least 195 (a random policy never does)
      - the final 100-episode mean is well above random (> 100)

    A random policy averages ~21. A policy that always falls right away scores ~10.
    """
    pytest.importorskip("gymnasium")
    returns = train(seed=0)
    assert len(returns) == 600
    assert max(returns) >= 195.0, (
        f"Actor-critic did not learn: best episode was {max(returns):.0f} "
        f"(random ≈ 21). Check the gradient sign in actor_critic_loss and that "
        f"log_probs still carry gradient."
    )
    assert float(np.mean(returns[-100:])) > 100.0, (
        f"Final 100-episode mean was {float(np.mean(returns[-100:])):.1f} — "
        f"should be > 100 (random ≈ 21). The critic baseline should make "
        f"training more stable than REINFORCE by the end."
    )
