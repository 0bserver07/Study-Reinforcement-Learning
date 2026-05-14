"""Tests for exercise 15 (GRPO on a verifiable toy task).

Run from the repo root:  pytest exercises/15-grpo-rlvr/

These run against starter.py — so they fail until you fill in the TODOs. To
check the reference solution instead, copy solution/grpo_rlvr.py over starter.py
(or import from solution in a scratch session).
"""

import pytest

torch = pytest.importorskip("torch")

from starter import (
    PROMPTS,
    N_PROMPTS,
    N_ANSWERS,
    verifier,
    GRPOPolicy,
    sample_group,
    group_advantages,
    grpo_loss,
    train,
)


# ── verifier ──────────────────────────────────────────────────────────────────

def test_verifier_correct():
    assert verifier((1, 1), 2) == 1.0
    assert verifier((2, 3), 5) == 1.0
    assert verifier((3, 3), 6) == 1.0


def test_verifier_wrong():
    assert verifier((1, 1), 0) == 0.0
    assert verifier((2, 3), 4) == 0.0
    assert verifier((1, 2), 9) == 0.0


# ── GRPOPolicy ────────────────────────────────────────────────────────────────

def test_policy_output_shape():
    policy = GRPOPolicy()
    logits = policy(0)
    assert logits.shape == (N_ANSWERS,), (
        f"Expected logits shape ({N_ANSWERS},), got {logits.shape}"
    )


def test_policy_has_parameters():
    policy = GRPOPolicy()
    params = list(policy.parameters())
    assert len(params) > 0, "GRPOPolicy should have at least one parameter tensor"
    assert params[0].shape == (N_PROMPTS, N_ANSWERS)


# ── sample_group ─────────────────────────────────────────────────────────────

def test_sample_group_shapes():
    torch.manual_seed(42)
    policy = GRPOPolicy()
    K = 8
    answers, log_probs = sample_group(policy, 0, K)
    assert answers.shape == (K,), f"answers should be shape ({K},), got {answers.shape}"
    assert log_probs.shape == (K,), f"log_probs should be shape ({K},), got {log_probs.shape}"


def test_sample_group_answers_in_range():
    torch.manual_seed(42)
    policy = GRPOPolicy()
    answers, _ = sample_group(policy, 0, K=32)
    assert answers.min() >= 0
    assert answers.max() < N_ANSWERS


def test_sample_group_log_probs_require_grad():
    torch.manual_seed(42)
    policy = GRPOPolicy()
    _, log_probs = sample_group(policy, 0, K=4)
    assert log_probs.requires_grad, (
        "log_probs from sample_group must require gradient — don't .detach() them here"
    )


def test_sample_group_log_probs_are_negative():
    torch.manual_seed(42)
    policy = GRPOPolicy()
    _, log_probs = sample_group(policy, 0, K=16)
    assert (log_probs <= 0).all(), "log-probabilities must be <= 0"


# ── group_advantages ──────────────────────────────────────────────────────────

def test_group_advantages_zero_mean():
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.0])
    adv = group_advantages(rewards)
    assert adv.sum().abs() < 1e-5, (
        f"Advantages should sum to ~0 (zero mean), got sum={adv.sum().item():.6f}"
    )


def test_group_advantages_approx_unit_std():
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.0])
    adv = group_advantages(rewards)
    # std will be slightly below 1 due to the +eps denominator, but very close
    assert abs(adv.std().item() - 1.0) < 0.01, (
        f"Advantages should have std ≈ 1.0, got {adv.std().item():.4f}"
    )


def test_group_advantages_sign():
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.0])
    adv = group_advantages(rewards)
    # First two (reward=1) should be positive; last two (reward=0) negative
    assert adv[0] > 0 and adv[1] > 0
    assert adv[2] < 0 and adv[3] < 0


def test_group_advantages_k1_returns_zeros():
    rewards = torch.tensor([1.0])
    adv = group_advantages(rewards)
    assert adv.shape == (1,)
    assert adv.item() == 0.0, "K==1: should return zeros (no group to compare against)"


def test_group_advantages_all_equal_returns_zeros():
    for val in [0.0, 1.0, 0.5]:
        rewards = torch.tensor([val, val, val, val])
        adv = group_advantages(rewards)
        assert (adv == 0).all(), (
            f"All-equal rewards (val={val}) should give zero advantages, got {adv}"
        )


# ── grpo_loss ─────────────────────────────────────────────────────────────────

def test_grpo_loss_is_scalar():
    torch.manual_seed(42)
    policy = GRPOPolicy()
    K = 8
    answers, log_probs_old = sample_group(policy, 0, K)
    log_probs_old = log_probs_old.detach()

    rewards = torch.tensor([verifier(PROMPTS[0], int(a)) for a in answers])
    adv = group_advantages(rewards)

    logits = policy(0)
    dist = torch.distributions.Categorical(logits=logits)
    log_probs_new = dist.log_prob(answers)

    loss = grpo_loss(log_probs_new, log_probs_old, adv)
    assert loss.shape == (), f"grpo_loss should return a scalar, got shape {loss.shape}"


def test_grpo_loss_requires_grad_and_backprops():
    torch.manual_seed(42)
    policy = GRPOPolicy()
    K = 8

    # Use a prompt where at least some answers are correct to get nonzero advantages
    prompt_idx = PROMPTS.index((1, 1))  # correct answer is 2
    answers, log_probs_old = sample_group(policy, prompt_idx, K)
    log_probs_old = log_probs_old.detach()

    rewards = torch.tensor([verifier(PROMPTS[prompt_idx], int(a)) for a in answers])
    # Force nonzero advantages by hand if all rewards are equal (unlikely but possible)
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5, 1.0, -1.0, 0.5, -0.5])

    logits = policy(prompt_idx)
    dist = torch.distributions.Categorical(logits=logits)
    log_probs_new = dist.log_prob(answers)

    loss = grpo_loss(log_probs_new, log_probs_old, adv)
    assert loss.requires_grad, "grpo_loss output should require gradient"

    loss.backward()
    grads = [p.grad for p in policy.parameters()]
    assert any(g is not None and torch.any(g != 0) for g in grads), (
        "backward() should populate non-zero gradients in the policy"
    )


# ── integration: does it actually learn? ─────────────────────────────────────

def test_grpo_learns_addition():
    """Full training run with the default config (seed=0, 1500 steps, K=8, lr=0.5).

    A random policy over 10 answers gets mean reward ≈ 0.10 (1/10).
    A trained policy should push well above 0.6 on the last 200 steps.

    This runs in well under 10 seconds — the policy is tiny and the task is tiny.
    """
    rewards = train(seed=0)
    assert len(rewards) == 1500

    last200_mean = sum(rewards[-200:]) / 200
    assert last200_mean > 0.6, (
        f"Policy did not converge: mean reward (last 200 steps) = {last200_mean:.3f}. "
        f"Random guessing ≈ 0.10. "
        f"Check: group_advantages returns zeros when all rewards are equal (no update — "
        f"that's correct, but you still need step_rewards.append()); "
        f"make sure loss.backward() happens before optimizer.step(); "
        f"check the sign in grpo_loss (gradient ascent via minimization → needs the minus)."
    )
