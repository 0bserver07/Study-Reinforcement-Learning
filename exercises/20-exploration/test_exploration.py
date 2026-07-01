"""Tests for exercise 20 (intrinsic-motivation exploration with RND).

Run from the repo root:  pytest exercises/20-exploration/

These run against starter.py — so they fail until you fill in the TODOs.
To check the reference solution instead, copy solution/exploration.py over
starter.py (or import from solution in a scratch session).
"""

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from starter import (
    ChainEnv,
    QLearningAgent,
    RNDIntrinsicReward,
    train_q_learning_alone,
    train_with_intrinsic,
)


# ── ChainEnv ──────────────────────────────────────────────────────────────────


def test_chain_env_basic():
    """reset returns 0, stepping right from 0 lands at 1 with no reward,
    and reaching the goal returns reward 1.0 with done=True."""
    env = ChainEnv(n_states=20, max_steps=100)
    s0 = env.reset()
    assert s0 == 0, f"reset() should return 0, got {s0}"

    s1, r, done, _ = env.step(1)  # right
    assert s1 == 1
    assert r == 0.0
    assert done is False

    # Walk to the goal — 18 more rights gets us from state 1 to state 19.
    for _ in range(17):
        s, r, done, _ = env.step(1)
        assert r == 0.0
        assert done is False
    # 18th right lands at the goal (state 19).
    s, r, done, _ = env.step(1)
    assert s == 19
    assert r == 1.0
    assert done is True


def test_chain_env_left_at_zero_stays_at_zero():
    """Boundary check: left at state 0 doesn't underflow."""
    env = ChainEnv(n_states=20)
    env.reset()
    s, r, done, _ = env.step(0)
    assert s == 0
    assert r == 0.0


def test_chain_env_timeout():
    """Episode terminates with done=True after max_steps even without reward."""
    env = ChainEnv(n_states=20, max_steps=5)
    env.reset()
    for _ in range(4):
        _, r, done, _ = env.step(0)  # left, stays at 0
        assert done is False
        assert r == 0.0
    _, r, done, _ = env.step(0)
    assert done is True
    assert r == 0.0


# ── QLearningAgent ────────────────────────────────────────────────────────────


def test_q_agent_act_returns_int_in_range():
    agent = QLearningAgent(n_states=20, seed=0)
    for _ in range(50):
        a = agent.act(0)
        assert isinstance(a, int)
        assert a in (0, 1)


def test_q_agent_update_terminal_uses_reward_only():
    """On a terminal transition, target should be reward, not reward + gamma*max(Q[s'])."""
    agent = QLearningAgent(n_states=20, alpha=1.0, gamma=0.99, epsilon=0.0, seed=0)
    # Plant a misleading next-state Q so the bug would be visible.
    agent.q[5, :] = np.array([100.0, 100.0])
    agent.update(s=4, a=1, r=1.0, s_next=5, done=True)
    # alpha=1.0, done=True → Q[4, 1] should jump to exactly r = 1.0.
    assert abs(agent.q[4, 1] - 1.0) < 1e-9, (
        f"on done=True, target should be r (1.0). got Q[4,1] = {agent.q[4, 1]}"
    )


def test_q_agent_update_bootstrap():
    """On a non-terminal transition, target = r + gamma * max(Q[s'])."""
    agent = QLearningAgent(n_states=20, alpha=1.0, gamma=0.99, epsilon=0.0, seed=0)
    agent.q[5, :] = np.array([2.0, 5.0])  # max is 5.0
    agent.update(s=4, a=1, r=0.0, s_next=5, done=False)
    expected = 0.0 + 0.99 * 5.0
    assert abs(agent.q[4, 1] - expected) < 1e-9, (
        f"target should be r + gamma * max(Q[s']) = {expected}, got Q[4,1] = {agent.q[4,1]}"
    )


# ── RNDIntrinsicReward ────────────────────────────────────────────────────────


def test_rnd_intrinsic_positive_for_novel_state():
    """A never-trained state should have positive intrinsic reward."""
    rnd = RNDIntrinsicReward(n_states=20, seed=0)
    r = rnd.intrinsic_reward(5)
    assert r > 0.0, f"intrinsic reward for novel state should be > 0, got {r}"


def test_rnd_intrinsic_decays_with_training():
    """Intrinsic reward for state s should drop substantially after many updates on s.

    The mechanism: the predictor learns to match the target output for inputs
    it's seen. After 200 gradient steps on the same one-hot input, the raw
    prediction error should be near zero. With normalization the post-training
    reward should still be much smaller than the pre-training reward.
    """
    rnd = RNDIntrinsicReward(n_states=20, seed=0)
    state = 10
    r_before = rnd.intrinsic_reward(state)
    for _ in range(200):
        rnd.update(state)
    r_after = rnd.intrinsic_reward(state)
    assert r_after < 0.1 * r_before, (
        f"intrinsic reward did not decay enough: before={r_before:.4f}, "
        f"after 200 updates={r_after:.4f}. Expected r_after < 0.1 * r_before."
    )


def test_rnd_predictor_trains_target_frozen():
    """target parameters should never move; predictor parameters should."""
    rnd = RNDIntrinsicReward(n_states=20, seed=0)
    target_before = [p.detach().clone() for p in rnd.target.parameters()]
    pred_before = [p.detach().clone() for p in rnd.predictor.parameters()]
    for s in range(10):
        rnd.update(s)
    target_after = list(rnd.target.parameters())
    pred_after = list(rnd.predictor.parameters())
    for a, b in zip(target_before, target_after):
        assert torch.allclose(a, b), "target network must stay frozen"
    moved = any(not torch.allclose(a, b) for a, b in zip(pred_before, pred_after))
    assert moved, "predictor should have moved after several update() calls"


# ── failure baseline: plain Q-learning never finds the goal ───────────────────


def test_q_learning_alone_fails():
    """Vanilla epsilon-greedy Q-learning on a 20-state chain never sees the goal
    in 200 episodes. This is the problem RND is meant to fix."""
    returns = train_q_learning_alone(seed=0, n_episodes=200)
    assert len(returns) == 200
    last50_mean = sum(returns[-50:]) / 50
    assert last50_mean < 0.1, (
        f"plain Q-learning was expected to flatline near 0 on this chain, "
        f"but mean(last 50) = {last50_mean:.3f}. "
        f"If this fails, the chain may be too easy — check ChainEnv.step."
    )


# ── integration: RND makes it work ────────────────────────────────────────────


def test_train_with_intrinsic_succeeds():
    """Q-learning + RND finds the goal and learns the path.

    With intrinsic_coef=0.1, 200 episodes, seed=0: mean extrinsic return over
    the last 50 episodes should exceed 0.5. On the reference solution this
    converges to ~1.0 within the first ~100 episodes.

    Runs in well under 15s on CPU — the env is tiny, the network is tiny.
    """
    returns, agent, rnd = train_with_intrinsic(
        seed=0, n_episodes=200, intrinsic_coef=0.1
    )
    assert len(returns) == 200
    last50_mean = sum(returns[-50:]) / 50
    assert last50_mean > 0.5, (
        f"RND-augmented Q-learning did not converge: mean extrinsic return "
        f"(last 50 episodes) = {last50_mean:.3f}, expected > 0.5. "
        f"Common causes: (1) you're returning intrinsic reward without "
        f"normalization, and its scale dominates extrinsic reward — try "
        f"smaller intrinsic_coef; (2) you're updating Q on r_combined but "
        f"recording r_combined as the episode return — make sure total_ext "
        f"accumulates r_ext only; (3) RNDIntrinsicReward.update isn't being "
        f"called, so the predictor never learns and the bonus stays maxed "
        f"out everywhere."
    )
