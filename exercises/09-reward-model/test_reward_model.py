"""Tests for exercise 09 (Bradley-Terry reward model on synthetic preferences).

Run from the repo root:  pytest exercises/09-reward-model/

These run against starter.py — so they fail until you fill in the TODOs. To
check the reference solution, copy solution/reward_model.py over starter.py
or import from solution in a scratch session.
"""

import math
import pytest

torch = pytest.importorskip("torch")

from starter import (
    FEATURE_DIM,
    RewardModel,
    bradley_terry_loss,
    train_reward_model,
)


# ── forward shape contract ────────────────────────────────────────────────────

def test_reward_model_forward():
    """Single input → scalar (shape ()); batched input → (B, 1)."""
    torch.manual_seed(0)
    model = RewardModel()

    single = torch.randn(FEATURE_DIM)
    r_single = model(single)
    assert isinstance(r_single, torch.Tensor)
    assert r_single.shape == (), (
        f"single input should give a scalar tensor (shape ()), got {r_single.shape}"
    )

    B = 7
    batch = torch.randn(B, FEATURE_DIM)
    r_batch = model(batch)
    assert r_batch.shape == (B, 1), (
        f"batched input should give shape ({B}, 1), got {r_batch.shape}"
    )


# ── Bradley-Terry loss properties ─────────────────────────────────────────────

def test_bradley_terry_loss_perfect_separation():
    """When chosen >> rejected, -log σ(big positive) is ≈ 0."""
    chosen = torch.tensor([10.0, 12.0, 15.0, 9.0])
    rejected = torch.tensor([-10.0, -8.0, -12.0, -7.0])
    loss = bradley_terry_loss(chosen, rejected)
    assert loss.item() < 1e-5, (
        f"With chosen >> rejected the loss should be ~0, got {loss.item():.6f}"
    )


def test_bradley_terry_loss_tied():
    """When chosen == rejected, -log σ(0) = log(2) ≈ 0.6931."""
    chosen = torch.tensor([0.3, -1.7, 2.5, 0.0])
    rejected = chosen.clone()  # identical → diff = 0
    loss = bradley_terry_loss(chosen, rejected)
    assert abs(loss.item() - math.log(2)) < 1e-6, (
        f"Tied rewards should give loss = log(2) ≈ 0.693, got {loss.item():.6f}"
    )


def test_bradley_terry_loss_gradient():
    """Loss must propagate gradient into the reward model parameters."""
    torch.manual_seed(0)
    model = RewardModel()
    chosen_feats = torch.randn(8, FEATURE_DIM)
    rejected_feats = torch.randn(8, FEATURE_DIM)

    r_chosen = model(chosen_feats)
    r_rejected = model(rejected_feats)
    loss = bradley_terry_loss(r_chosen, r_rejected)

    assert loss.requires_grad, "loss should require gradient"
    loss.backward()

    grads = [p.grad for p in model.parameters()]
    assert len(grads) > 0, "model has no parameters"
    assert all(g is not None for g in grads), (
        "backward() left some parameters without gradients — make sure the loss "
        "depends on the model output"
    )
    assert any(torch.any(g != 0) for g in grads), (
        "all gradients are zero — check that bradley_terry_loss returns a "
        "non-constant function of the inputs (the diff, not e.g. just chosen)"
    )


# ── integration: does it learn? ───────────────────────────────────────────────

def test_train_recovers_signal():
    """After 2000 steps, held-out Spearman vs true reward should be > 0.85.

    Random init scores around 0; the only way to push past 0.85 is to learn
    the ranking from the preference labels — which means the BT loss is
    actually doing what it should. CPU only; runs in well under 10 seconds.
    """
    out = train_reward_model(seed=0, n_steps=2000)
    spearman = out["spearman"]
    losses = out["losses"]
    assert len(spearman) == 2000
    assert len(losses) == 2000

    final = spearman[-1]
    assert final > 0.85, (
        f"Held-out Spearman after 2000 steps = {final:.3f}; expected > 0.85. "
        f"Check: bradley_terry_loss uses (chosen - rejected), not (rejected - chosen); "
        f"loss.backward() runs before optimizer.step(); "
        f"the same model is used for both forward passes (no fresh init in the loop)."
    )
