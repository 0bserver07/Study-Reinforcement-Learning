"""Tabular Q-learning on FrozenLake-v1 (non-slippery) — reference solution.

Look at this after you've tried ../starter.py.

Run it directly:  python3 exercises/03-q-learning/solution/q_learning.py
"""

import numpy as np

GAMMA = 0.99


def epsilon_greedy(q_row, epsilon: float, n_actions: int, rng) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(n_actions))
    return int(np.argmax(q_row))


def q_update(q_table, state, action, reward, next_state, done, alpha: float, gamma: float = GAMMA) -> float:
    target = reward if done else reward + gamma * np.max(q_table[next_state])
    td_error = target - q_table[state, action]
    q_table[state, action] += alpha * td_error
    return float(td_error)


def train(n_episodes: int = 2000, alpha: float = 0.8, epsilon: float = 0.1,
          gamma: float = GAMMA, seed: int = 0):
    import gymnasium as gym

    rng = np.random.default_rng(seed)
    env = gym.make("FrozenLake-v1", is_slippery=False)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    # Optimistic initialization (see starter.py for why): start Q above the true
    # values so every untried action looks attractive, which drives exploration.
    q_table = np.ones((n_states, n_actions))

    returns = []
    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        done = False
        total = 0.0
        while not done:
            action = epsilon_greedy(q_table[state], epsilon, n_actions, rng)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            q_update(q_table, state, action, reward, next_state, done, alpha, gamma)
            state = next_state
            total += float(reward)
        returns.append(total)
    env.close()
    return q_table, returns


def greedy_policy_return(q_table, seed: int = 0) -> float:
    import gymnasium as gym

    env = gym.make("FrozenLake-v1", is_slippery=False)
    state, _ = env.reset(seed=seed)
    done = False
    total = 0.0
    while not done:
        action = int(np.argmax(q_table[state]))
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total += float(reward)
    env.close()
    return total


if __name__ == "__main__":
    q, rets = train()
    print(f"trained {len(rets)} episodes; mean return last 100 = {np.mean(rets[-100:]):.2f}")
    print(f"greedy policy return from start = {greedy_policy_return(q)}  (1.0 means it reached the goal)")
