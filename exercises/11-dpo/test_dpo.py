"""Tests for exercise 11 (DPO on a toy preference dataset).

Run from the repo root:  pytest exercises/11-dpo/

These run against starter.py — so they fail until you fill in the TODOs. To
check the reference solution instead, copy solution/dpo.py over starter.py
(or import from solution in a scratch session).
"""

import math
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from starter import (
    N_PROMPTS,
    N_ANSWERS,
    BETA,
    Policy,
    dpo_loss,
    make_true_reward,
    make_preference_dataset,
    train_dpo,
    greedy_mean_true_reward,
)


# ── Policy.forward ────────────────────────────────────────────────────────────

def test_policy_forward():
    """Single int input returns (n_answers,); batched input returns (B, n_answers)."""
    policy = Policy()
    logits_single = policy(0)
    assert logits_single.shape == (N_ANSWERS,), (
        f"Expected shape ({N_ANSWERS},) for a single prompt, got {logits_single.shape}"
    )
    idx = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    logits_batch = policy(idx)
    assert logits_batch.shape == (4, N_ANSWERS), (
        f"Expected shape (4, {N_ANSWERS}) for batched prompts, got {logits_batch.shape}"
    )


def test_policy_has_embedding_params():
    policy = Policy()
    params = list(policy.parameters())
    assert len(params) > 0, "Policy should have at least one parameter tensor"
    assert params[0].shape == (N_PROMPTS, N_ANSWERS), (
        f"Expected the embedding to have shape ({N_PROMPTS}, {N_ANSWERS}), got {params[0].shape}"
    )


# ── dpo_loss: starting point ──────────────────────────────────────────────────

def test_dpo_loss_starts_at_log2():
    """When policy == ref, the bracket inside sigmoid is zero, so loss == -log(0.5) == log(2)."""
    torch.manual_seed(0)
    B = 16
    logits = torch.randn(B, N_ANSWERS)
    # Same logits for both — policy and ref are identical, so log-ratios cancel
    chosen = torch.randint(N_ANSWERS, (B,))
    rejected = torch.randint(N_ANSWERS, (B,))
    loss = dpo_loss(logits, logits.clone(), chosen, rejected, beta=BETA)
    assert abs(loss.item() - math.log(2.0)) < 1e-5, (
        f"With policy == ref, DPO loss must be log(2) ≈ 0.6931, got {loss.item():.6f}. "
        f"Check that the bracket simplifies to zero and -log sigmoid(0) = log(2)."
    )


# ── dpo_loss: gradient pushes toward chosen ───────────────────────────────────

def test_dpo_loss_pushes_toward_chosen():
    """After one gradient step on a single pair, log pi(chosen) - log pi(rejected) should increase."""
    torch.manual_seed(0)
    policy = Policy()
    # Initialize ref by copying current policy so they're identical at step 0.
    ref = Policy()
    ref.load_state_dict(policy.state_dict())
    for p in ref.parameters():
        p.requires_grad_(False)

    prompt = torch.tensor([0], dtype=torch.long)
    chosen = torch.tensor([3], dtype=torch.long)
    rejected = torch.tensor([7], dtype=torch.long)

    def margin(pol):
        with torch.no_grad():
            lp = F.log_softmax(pol(prompt), dim=-1)
            return (lp[0, 3] - lp[0, 7]).item()

    before = margin(policy)

    opt = torch.optim.SGD(policy.parameters(), lr=1.0)
    opt.zero_grad()
    loss = dpo_loss(policy(prompt), ref(prompt), chosen, rejected, beta=BETA)
    loss.backward()
    opt.step()

    after = margin(policy)
    assert after > before, (
        f"log pi(chosen) - log pi(rejected) should increase after one DPO step, "
        f"got before={before:.6f}, after={after:.6f}. "
        f"Check the sign in the loss: we minimize -log sigmoid(...), and the bracket "
        f"must be (chosen_logratio - rejected_logratio), not the reverse."
    )


# ── dpo_loss: gradients exist and are non-zero ────────────────────────────────

def test_dpo_loss_gradient():
    """The loss backprops and produces non-zero gradients on the policy."""
    torch.manual_seed(0)
    policy = Policy()
    ref = Policy()
    ref.load_state_dict(policy.state_dict())
    # Slightly perturb the policy so the bracket isn't exactly zero — then
    # the gradient is unambiguously non-zero. (At policy == ref the gradient
    # is non-zero in general too, but small at init.)
    with torch.no_grad():
        for p in policy.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    for p in ref.parameters():
        p.requires_grad_(False)

    B = 8
    prompts = torch.randint(N_PROMPTS, (B,))
    chosen = torch.randint(N_ANSWERS, (B,))
    rejected = torch.randint(N_ANSWERS, (B,))

    loss = dpo_loss(policy(prompts), ref(prompts), chosen, rejected, beta=BETA)
    assert loss.requires_grad, "dpo_loss output must require gradient"
    assert loss.shape == (), f"dpo_loss must return a scalar, got shape {loss.shape}"

    loss.backward()
    grads = [p.grad for p in policy.parameters()]
    assert any(g is not None and torch.any(g != 0) for g in grads), (
        "backward() should leave non-zero gradients on the policy parameters"
    )
    # And no gradient leaked into the reference.
    for p in ref.parameters():
        assert p.grad is None or torch.all(p.grad == 0), (
            "Reference policy should not receive gradients (it must be frozen)"
        )


# ── integration: training pushes the greedy policy well above baseline ───────

def test_train_improves_reward():
    """After 2000 DPO steps from seed=0, the greedy policy should be much better than random.

    The 'true' reward table has values roughly in [-2, 2]. A uniform-random
    answer scores about 0.09 on average. A trained DPO policy should clear 0.6.

    Runs on CPU in well under 10 seconds.
    """
    true_reward = make_true_reward()
    dataset = make_preference_dataset(n_pairs=512, true_reward=true_reward)

    ref = Policy()
    losses, rewards, policy = train_dpo(
        dataset, ref, true_reward, beta=BETA, n_steps=2000, lr=0.05, seed=0
    )

    assert len(losses) == 2000
    assert len(rewards) == 2000

    # Sanity: first loss is near log(2) since policy starts equal to ref.
    assert abs(losses[0] - math.log(2.0)) < 0.05, (
        f"First loss should be ~log(2) since the policy is initialized from the reference, "
        f"got {losses[0]:.4f}"
    )

    final_reward = greedy_mean_true_reward(policy, true_reward)
    uniform_baseline = true_reward.mean().item()   # ~0.09
    assert final_reward > 0.6, (
        f"DPO did not learn the preferences: greedy mean true reward = {final_reward:.3f}. "
        f"Uniform-random baseline ≈ {uniform_baseline:.3f}. "
        f"Check: dpo_loss uses (chosen - rejected) inside sigmoid (not the reverse); "
        f"ref_policy is frozen (no gradient leaks); the policy is updated, not the reference."
    )
