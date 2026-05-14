"""Value iteration on a gridworld MDP — exercise starter.

Fill in the TODOs, then from the repo root:

    pip install -r exercises/requirements.txt
    pytest exercises/01-mdps/

See README.md for the task and HINTS.md if you're stuck. Reference solution:
solution/value_iteration.py — look after you've tried.
"""

import numpy as np

GAMMA = 0.99
_DELTAS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}  # up, right, down, left


class GridWorldMDP:
    """Deterministic gridworld. Reach the goal cheaply.

    States are grid cells, indexed `i * size + j` for cell (row i, col j). The
    goal is the top-right corner (0, size-1). Actions: 0=up, 1=right, 2=down,
    3=left, clamped at the walls. Reward: +10 for the step that lands on the
    goal, -1 for every other step. The goal is terminal (absorbing, no reward).

    Given — you don't implement this. It exposes the interface value_iteration /
    greedy_policy below should use:
        mdp.n_states, mdp.n_actions
        mdp.is_terminal(s) -> bool
        mdp.transitions(s, a) -> list of (prob, next_state, reward)
    """

    def __init__(self, size: int = 5):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        self.goal = size - 1  # cell (0, size-1) -> index 0*size + (size-1)

    def is_terminal(self, s: int) -> bool:
        return s == self.goal

    def transitions(self, s: int, a: int):
        if self.is_terminal(s):
            return [(1.0, s, 0.0)]
        i, j = divmod(s, self.size)
        di, dj = _DELTAS[a]
        i = min(self.size - 1, max(0, i + di))
        j = min(self.size - 1, max(0, j + dj))
        ns = i * self.size + j
        return [(1.0, ns, 10.0 if ns == self.goal else -1.0)]


def value_iteration(mdp, gamma: float = GAMMA, theta: float = 1e-8) -> np.ndarray:
    """Compute V* by repeated Bellman-optimality backups until it stops changing.

        V[s] <- max over a of  sum over (p, s', r) in mdp.transitions(s, a)  of  p * (r + gamma * V[s'])

    Skip terminal states — their value stays 0. After each sweep, stop if the
    largest change to any V[s] was below `theta`. Return V: a 1-D float array of
    length mdp.n_states.
    """
    # TODO
    raise NotImplementedError("value_iteration")


def greedy_policy(mdp, V, gamma: float = GAMMA) -> np.ndarray:
    """The policy greedy with respect to V: for each non-terminal state, the
    action maximizing  sum over (p, s', r)  of  p * (r + gamma * V[s']).

    Return an int array of length mdp.n_states (one action per state; the value
    at terminal states doesn't matter — leave it 0).
    """
    # TODO
    raise NotImplementedError("greedy_policy")


def follow_greedy(mdp, policy, start_state: int, max_steps: int = 100) -> bool:
    """Walk `policy` from `start_state` (assumes deterministic transitions).
    Return True if a terminal state is reached within `max_steps`. Given."""
    s = start_state
    for _ in range(max_steps):
        if mdp.is_terminal(s):
            return True
        _, s, _ = mdp.transitions(s, int(policy[s]))[0]
    return mdp.is_terminal(s)


if __name__ == "__main__":
    mdp = GridWorldMDP(size=5)
    V = value_iteration(mdp)
    policy = greedy_policy(mdp, V)
    print("V*:")
    print(np.round(V.reshape(mdp.size, mdp.size), 2))
    print("\nGreedy policy (G = goal):")
    arrows = {0: "^", 1: ">", 2: "v", 3: "<"}
    for i in range(mdp.size):
        row = []
        for j in range(mdp.size):
            s = i * mdp.size + j
            row.append("G" if mdp.is_terminal(s) else arrows[int(policy[s])])
        print(" ".join(row))
