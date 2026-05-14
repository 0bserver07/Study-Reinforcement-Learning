"""Tests for exercise 02 (REINFORCE on CartPole).

Run from the repo root:  pytest exercises/02-policy-gradients/

These run against starter.py — so they fail until you fill in the TODOs. To
check the reference solution instead, copy solution/reinforce.py over starter.py
(or import from solution in a scratch session).
"""

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from starter import PolicyNet, select_action, compute_returns, reinforce_loss, train


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


# --- select_action -----------------------------------------------------------

def test_select_action_returns_int_and_gradient_logprob():
    net = PolicyNet(state_dim=4, n_actions=2)
    action, log_prob = select_action(net, np.zeros(4, dtype=np.float32))
    assert isinstance(action, int) and action in (0, 1)
    assert isinstance(log_prob, torch.Tensor)
    assert log_prob.shape == ()           # scalar
    assert log_prob.requires_grad         # loss must be able to backprop through it
    assert log_prob.item() <= 0.0         # log of a probability


# --- reinforce_loss ----------------------------------------------------------

def test_reinforce_loss_is_scalar_and_differentiable():
    net = PolicyNet(state_dim=4, n_actions=2)
    log_probs = [select_action(net, np.zeros(4, dtype=np.float32))[1] for _ in range(5)]
    returns = compute_returns([1.0, 1.0, 1.0, 1.0, 1.0])
    loss = reinforce_loss(log_probs, returns)
    assert loss.shape == ()
    assert loss.requires_grad
    loss.backward()
    grads = [p.grad for p in net.parameters()]
    assert any(g is not None and torch.any(g != 0) for g in grads)


# --- integration: does it actually learn? ------------------------------------

def test_reinforce_learns_cartpole():
    """Full training run on CartPole-v1 with the default config (800 episodes,
    lr=1e-3, gradient clipping). Takes a couple of minutes.

    A random policy averages ~21 and never gets near 195; a collapsed policy is
    worse. So "the best episode reached at least 195" is a solid 'it learned'
    check. REINFORCE is noisy — it can hit 500 and then dip — so we don't demand
    a high *final* average, just that it learned and didn't fall apart.
    """
    pytest.importorskip("gymnasium")
    returns = train(seed=0)
    assert len(returns) == 800
    assert max(returns) >= 195.0, (
        f"REINFORCE did not learn: best episode was {max(returns):.0f} "
        f"(random ≈ 21). Check the gradient sign in reinforce_loss, the returns, "
        f"and that the log-probs you stored still carry gradient."
    )
    assert float(np.mean(returns[-100:])) > 50.0   # still well above random by the end
