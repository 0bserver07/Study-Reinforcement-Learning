"""Tests for exercise 20 (exploration with an RND bonus on a combination lock).

Run from the repo root:  pytest exercises/20-exploration/

These run against starter.py, so they fail until you fill in the TODOs.
To check the reference solution instead, copy solution/exploration.py over
starter.py (or import from solution in a scratch session).
"""

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from starter import (
    CombinationLock,
    QLearningAgent,
    RNDIntrinsicReward,
    train_q_learning_alone,
    train_with_intrinsic,
)

N = 16  # squares 0..15; the +1 is on square 15

# Seeds for the two training tests. The thresholds leave a wide margin: the
# docstrings say what the reference solution did over many more seeds.
PLAIN_SEEDS = (0, 1, 2)
RND_SEEDS = (0, 1, 2)


def _first_goal(returns):
    """1-based episode of the first +1, or None if there wasn't one."""
    return next((i + 1 for i, r in enumerate(returns) if r > 0), None)


# ── CombinationLock ───────────────────────────────────────────────────────────


def test_lock_right_moves_open_it():
    """reset returns 0, each right moves one square with reward 0, and the 15th
    right lands on square 15 with reward 1.0 and done=True."""
    env = CombinationLock(n_states=N)
    s0 = env.reset()
    assert s0 == 0, f"reset() should return 0, got {s0}"

    for square in range(1, N - 1):
        s, r, done, _ = env.step(1)
        assert s == square, f"right should move to square {square}, got {s}"
        assert r == 0.0
        assert done is False, f"the episode shouldn't end on square {square}"

    s, r, done, _ = env.step(1)
    assert s == N - 1
    assert r == 1.0
    assert done is True


@pytest.mark.parametrize("square", [0, 7, N - 2])
def test_lock_wrong_move_ends_episode(square):
    """Left is the wrong move on every square, square 0 included: reward 0, the
    episode ends, and the agent stays where it was."""
    env = CombinationLock(n_states=N)
    env.reset()
    for _ in range(square):
        env.step(1)
    s, r, done, _ = env.step(0)
    assert r == 0.0
    assert done is True, f"left on square {square} should end the episode"
    assert s == square, f"after the wrong move the agent stays on square {square}, got {s}"


def test_lock_reset_starts_over():
    env = CombinationLock(n_states=N)
    env.reset()
    env.step(1)
    env.step(1)
    env.step(0)
    assert env.reset() == 0
    s, r, done, _ = env.step(1)
    assert (s, r, done) == (1, 0.0, False)


# ── QLearningAgent ────────────────────────────────────────────────────────────


def test_q_agent_act_returns_int_in_range():
    agent = QLearningAgent(n_states=N, seed=0)
    for _ in range(50):
        a = agent.act(0)
        assert isinstance(a, int)
        assert a in (0, 1)


def test_q_agent_act_breaks_ties_at_random():
    """With both Q-values equal the greedy choice is a tie, and a tie should be a
    coin flip. np.argmax would return action 0 (the wrong move) every time."""
    agent = QLearningAgent(n_states=N, epsilon=0.0, seed=0)
    picks = [agent.act(3) for _ in range(1000)]
    share_right = sum(picks) / len(picks)
    assert 0.4 < share_right < 0.6, (
        f"with Q[3] = [0, 0] and epsilon=0, right was picked {share_right:.1%} of the "
        f"time; a random tie-break picks it about half the time. Pick among all the "
        f"actions tied for the max, not the first one."
    )


def test_q_agent_act_is_greedy_without_ties():
    agent = QLearningAgent(n_states=N, epsilon=0.0, seed=0)
    agent.q[3] = [0.0, 0.2]
    assert all(agent.act(3) == 1 for _ in range(100))
    agent.q[3] = [0.2, 0.0]
    assert all(agent.act(3) == 0 for _ in range(100))


def test_q_agent_update_terminal_uses_reward_only():
    """A move that ends the episode has nothing after it: target = r, no bootstrap."""
    agent = QLearningAgent(n_states=N, alpha=1.0, gamma=0.99, epsilon=0.0, seed=0)

    # Reaching the goal. Plant a value on the goal square that the target must ignore.
    agent.q[N - 1, :] = [100.0, 100.0]
    agent.update(s=N - 2, a=1, r=1.0, s_next=N - 1, done=True)
    assert abs(agent.q[N - 2, 1] - 1.0) < 1e-9, (
        f"on done=True the target should be r (1.0), got Q[{N - 2},1] = {agent.q[N - 2, 1]}"
    )

    # The wrong move. The agent stays on square 4, whose right move is worth 5.
    agent.q[4, :] = [0.0, 5.0]
    agent.update(s=4, a=0, r=0.0, s_next=4, done=True)
    assert agent.q[4, 0] == 0.0, (
        f"the wrong move ends the episode with reward 0, so its target is 0, "
        f"got Q[4,0] = {agent.q[4, 0]}"
    )


def test_q_agent_update_bootstrap():
    """On a move that doesn't end the episode, target = r + gamma * max(Q[s'])."""
    agent = QLearningAgent(n_states=N, alpha=1.0, gamma=0.99, epsilon=0.0, seed=0)
    agent.q[5, :] = [2.0, 5.0]  # max is 5.0
    agent.update(s=4, a=1, r=0.0, s_next=5, done=False)
    expected = 0.0 + 0.99 * 5.0
    assert abs(agent.q[4, 1] - expected) < 1e-9, (
        f"target should be r + gamma * max(Q[s']) = {expected}, got Q[4,1] = {agent.q[4, 1]}"
    )

    # With alpha < 1 the update moves only part of the way, and returns the TD error.
    agent = QLearningAgent(n_states=N, alpha=0.5, gamma=0.99, epsilon=0.0, seed=0)
    agent.q[4, 1] = 1.0
    agent.q[5, :] = [2.0, 5.0]
    td = agent.update(s=4, a=1, r=0.1, s_next=5, done=False)
    expected_td = (0.1 + 0.99 * 5.0) - 1.0
    assert abs(td - expected_td) < 1e-9, f"update() should return the TD error {expected_td}, got {td}"
    assert abs(agent.q[4, 1] - (1.0 + 0.5 * expected_td)) < 1e-9


# ── RNDIntrinsicReward ────────────────────────────────────────────────────────


def test_rnd_intrinsic_positive_for_novel_state():
    """A never-trained state should have positive intrinsic reward."""
    rnd = RNDIntrinsicReward(n_states=N, seed=0)
    r = rnd.intrinsic_reward(5)
    assert r > 0.0, f"intrinsic reward for novel state should be > 0, got {r}"


def test_rnd_intrinsic_decays_with_training():
    """Intrinsic reward for state s should drop substantially after many updates on s.

    The predictor learns to match the target output for inputs it has seen.
    After 200 gradient steps on the same one-hot input, the raw prediction
    error should be near zero, and so the normalized reward much smaller than
    before training.
    """
    rnd = RNDIntrinsicReward(n_states=N, seed=0)
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
    rnd = RNDIntrinsicReward(n_states=N, seed=0)
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


# ── failure baseline: plain epsilon-greedy almost never opens the lock ────────


@pytest.mark.parametrize("seed", PLAIN_SEEDS)
def test_q_learning_alone_fails(seed):
    """Plain epsilon-greedy Q-learning almost never reaches the goal in 200 episodes.

    Until its first +1 every Q-value is exactly 0, so every move is a coin flip
    and fifteen rights in a row come once in 2^15 = 32,768 episodes: about 0.6%
    of 200-episode runs reach the goal even once. After one lucky +1 it still
    has to get lucky again to reach square 14, so a second success in the same
    run is rarer still. Over 2,000 seeds of the reference solution, 8 runs
    reached the goal once and none twice.
    """
    returns = train_q_learning_alone(seed=seed, n_episodes=200)
    assert len(returns) == 200
    n_goals = int(sum(returns))
    assert n_goals <= 1, (
        f"plain epsilon-greedy reached the goal in {n_goals} of 200 episodes "
        f"(seed {seed}); it should almost never get there. Check that "
        f"QLearningAgent.act breaks ties at random rather than always picking "
        f"right, and that CombinationLock.step ends the episode on a left."
    )


# ── integration: the bonus opens the lock ─────────────────────────────────────


@pytest.mark.parametrize("seed", RND_SEEDS)
def test_train_with_intrinsic_succeeds(seed):
    """Q-learning + RND reaches the goal early and keeps reaching it.

    Over 1,000 seeds of the reference solution the first +1 came at a median of
    episode 22 and never later than episode 53, and after 200 episodes the
    greedy move was right on every square. With epsilon = 0.1 still on, each of the 15 moves
    is a random left 5% of the time, so about 0.95^15 = 46% of episodes reach
    the goal: that's the ceiling for the mean return. The lowest mean over the
    last 50 of 200 episodes was 0.28 across the 1,000 seeds.
    """
    returns, agent, rnd = train_with_intrinsic(seed=seed, n_episodes=200, intrinsic_coef=0.1)
    assert len(returns) == 200

    assert np.all(agent.q[:, 0] == 0.0), (
        f"Q(s, left) should stay exactly 0 on every square, got {np.round(agent.q[:, 0], 4)}. "
        f"The wrong move pays 0 and ends the episode, so its target is 0. Either "
        f"the bonus is being added on moves that end the episode (only add it "
        f"when done is False), or QLearningAgent.update bootstraps on done=True."
    )

    fresh = RNDIntrinsicReward(n_states=N, seed=seed)  # same init as the one in the loop
    trained = any(
        not torch.equal(a, b) for a, b in zip(fresh.predictor.parameters(), rnd.predictor.parameters())
    )
    assert trained, (
        "the RND predictor never changed during training. Call rnd.update(s_next) "
        "after computing the bonus, so the bonus shrinks on squares the agent knows."
    )

    first = _first_goal(returns)
    assert first is not None and first <= 100, (
        f"first +1 at episode {first} (seed {seed}); the reference solution got "
        f"there by episode 53 on all 1,000 seeds tried. Check that the bonus is added "
        f"to the reward Q-learning sees (r_ext + intrinsic_coef * r_int), and that "
        f"act() breaks ties at random."
    )

    last50_mean = sum(returns[-50:]) / 50
    assert last50_mean >= 0.2, (
        f"mean extrinsic return over the last 50 episodes = {last50_mean:.2f}, "
        f"expected about 0.46 (0.95^15) and at least 0.2. Make sure total_ext "
        f"accumulates r_ext only."
    )

    greedy_right = [bool(agent.q[s, 1] > agent.q[s, 0]) for s in range(N - 1)]
    assert all(greedy_right), (
        f"after training the greedy move should be right on squares 0-{N - 2}, "
        f"got right on {greedy_right}"
    )
