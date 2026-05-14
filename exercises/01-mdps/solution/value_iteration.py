"""Value iteration on a gridworld MDP — reference solution.

Look at this after you've tried ../starter.py.

Run it directly:  python3 exercises/01-mdps/solution/value_iteration.py
"""

import numpy as np

GAMMA = 0.99
_DELTAS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}  # up, right, down, left


class GridWorldMDP:
    """Deterministic gridworld; see ../starter.py for the description."""

    def __init__(self, size: int = 5):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        self.goal = size - 1

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


def _q_value(mdp, V, s, a, gamma):
    return sum(p * (r + gamma * V[sp]) for (p, sp, r) in mdp.transitions(s, a))


def value_iteration(mdp, gamma: float = GAMMA, theta: float = 1e-8) -> np.ndarray:
    V = np.zeros(mdp.n_states)
    while True:
        delta = 0.0
        for s in range(mdp.n_states):
            if mdp.is_terminal(s):
                continue
            v_old = V[s]
            V[s] = max(_q_value(mdp, V, s, a, gamma) for a in range(mdp.n_actions))
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta:
            return V


def greedy_policy(mdp, V, gamma: float = GAMMA) -> np.ndarray:
    policy = np.zeros(mdp.n_states, dtype=int)
    for s in range(mdp.n_states):
        if mdp.is_terminal(s):
            continue
        policy[s] = int(np.argmax([_q_value(mdp, V, s, a, gamma)
                                   for a in range(mdp.n_actions)]))
    return policy


def follow_greedy(mdp, policy, start_state: int, max_steps: int = 100) -> bool:
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
