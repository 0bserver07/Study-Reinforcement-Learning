"""Tests for exercise 01 (value iteration on a gridworld).

Run from the repo root:  pytest exercises/01-mdps/

These run against starter.py — they fail until you fill in the TODOs.
"""

import pytest

np = pytest.importorskip("numpy")

from starter import GridWorldMDP, value_iteration, greedy_policy, follow_greedy


class _MiniMDP:
    """Tiny deterministic MDP for unit tests.

    `table` maps (state, action) -> (next_state, reward); `terminal` is the set
    of terminal states. Same interface GridWorldMDP exposes.
    """

    def __init__(self, n_states, n_actions, table, terminal):
        self.n_states = n_states
        self.n_actions = n_actions
        self._table = table
        self._terminal = set(terminal)

    def is_terminal(self, s):
        return s in self._terminal

    def transitions(self, s, a):
        if self.is_terminal(s):
            return [(1.0, s, 0.0)]
        ns, r = self._table[(s, a)]
        return [(1.0, ns, r)]


# --- value_iteration on tiny MDPs --------------------------------------------

def test_value_iteration_two_state():
    # State 0 non-terminal, state 1 terminal (V = 0).
    # From 0: action 0 -> (state 1, +1);  action 1 -> (state 0, 0) self-loop.
    # gamma = 0.5  =>  V[0] = max(1 + 0.5*0,  0 + 0.5*V[0]) = 1
    mdp = _MiniMDP(2, 2, {(0, 0): (1, 1.0), (0, 1): (0, 0.0)}, terminal={1})
    V = value_iteration(mdp, gamma=0.5)
    assert np.allclose(V, [1.0, 0.0], atol=1e-5)


def test_value_iteration_chain():
    # 0 -> 1 -> 2 (terminal). One action. Reward 5 on the 1 -> 2 step.
    # gamma = 0.9  =>  V[2] = 0, V[1] = 5, V[0] = 0.9 * 5 = 4.5
    mdp = _MiniMDP(3, 1, {(0, 0): (1, 0.0), (1, 0): (2, 5.0)}, terminal={2})
    V = value_iteration(mdp, gamma=0.9)
    assert np.allclose(V, [4.5, 5.0, 0.0], atol=1e-5)


def test_greedy_policy_picks_the_paying_action():
    mdp = _MiniMDP(2, 2, {(0, 0): (1, 1.0), (0, 1): (0, 0.0)}, terminal={1})
    V = value_iteration(mdp, gamma=0.5)
    policy = greedy_policy(mdp, V, gamma=0.5)
    assert int(policy[0]) == 0          # action 0 reaches the +1 terminal


# --- integration on the 5x5 gridworld ----------------------------------------

def test_gridworld_value_function_shape():
    mdp = GridWorldMDP(size=5)
    V = value_iteration(mdp)
    # cell (0,3) is one "right" step from the goal -> V == 10 (the +10 step,
    # then the goal is terminal with value 0)
    assert V[3] == pytest.approx(10.0, abs=1e-3)
    # values fall off as you move away from the goal along the top row:
    assert V[3] > V[2] > V[1] > V[0]
    # the goal cell is terminal, so it stays 0
    assert V[mdp.goal] == 0.0


def test_gridworld_greedy_policy_reaches_goal_from_everywhere():
    mdp = GridWorldMDP(size=5)
    V = value_iteration(mdp)
    policy = greedy_policy(mdp, V)
    for s in range(mdp.n_states):
        if mdp.is_terminal(s):
            continue
        assert follow_greedy(mdp, policy, start_state=s, max_steps=2 * mdp.n_states), (
            f"the greedy policy from state {s} didn't reach the goal — check the "
            f"argmax in greedy_policy and the backup in value_iteration"
        )
