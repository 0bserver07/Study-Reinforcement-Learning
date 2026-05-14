"""Tests for exercise 03 (tabular Q-learning on FrozenLake).

Run from the repo root:  pytest exercises/03-q-learning/

These run against starter.py — they fail until you fill in the TODOs.
"""

import pytest

np = pytest.importorskip("numpy")

from starter import epsilon_greedy, q_update, train, greedy_policy_return


# --- epsilon_greedy ----------------------------------------------------------

def test_epsilon_greedy_greedy_when_epsilon_zero():
    rng = np.random.default_rng(0)
    q_row = np.array([0.1, 0.9, 0.2, 0.0])
    for _ in range(20):
        a = epsilon_greedy(q_row, 0.0, 4, rng)
        assert a == 1                       # the argmax
        assert isinstance(a, int)           # a Python int, not np.int64


def test_epsilon_greedy_explores_when_epsilon_one():
    rng = np.random.default_rng(0)
    q_row = np.array([0.0, 0.0, 1.0, 0.0])  # action 2 is greedy, but epsilon=1
    seen = {epsilon_greedy(q_row, 1.0, 4, rng) for _ in range(300)}
    assert seen == {0, 1, 2, 3}             # every action shows up
    assert all(isinstance(a, int) for a in seen)


# --- q_update ----------------------------------------------------------------

def test_q_update_terminal():
    q = np.zeros((2, 2))
    td = q_update(q, state=0, action=1, reward=1.0, next_state=1, done=True, alpha=0.5)
    assert td == pytest.approx(1.0)         # target = reward, old Q = 0
    assert q[0, 1] == pytest.approx(0.5)    # 0 + 0.5 * 1.0
    assert q[0, 0] == 0.0                   # untouched


def test_q_update_bootstrap():
    q = np.array([[0.0, 0.0], [0.0, 2.0]])
    td = q_update(q, state=0, action=0, reward=1.0, next_state=1, done=False,
                  alpha=0.5, gamma=0.9)
    # target = 1.0 + 0.9 * max(q[1]) = 1.0 + 0.9 * 2.0 = 2.8
    assert td == pytest.approx(2.8)
    assert q[0, 0] == pytest.approx(1.4)    # 0 + 0.5 * 2.8


# --- integration: does it learn the optimal policy? --------------------------

def test_q_learning_solves_frozenlake():
    """Tabular Q-learning on the deterministic FrozenLake should learn the
    optimal policy: the greedy policy reaches the goal (return 1.0)."""
    pytest.importorskip("gymnasium")
    q_table, returns = train(n_episodes=2000, seed=0)
    assert len(returns) == 2000
    assert greedy_policy_return(q_table, seed=0) == 1.0, (
        "the greedy policy didn't reach the goal — check the TD target, the "
        "argmax over next-state Q-values, and that you update before moving on"
    )
    # by the end of training, plenty of episodes succeed (epsilon=0.1 still
    # sends the agent into a hole now and then; a random policy reaches the
    # goal ~1% of the time, so anything well above that means it learned):
    assert np.mean(returns[-200:]) > 0.3
