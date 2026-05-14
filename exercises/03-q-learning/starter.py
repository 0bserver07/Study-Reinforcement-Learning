"""Tabular Q-learning on FrozenLake-v1 (non-slippery) — exercise starter.

Fill in the TODOs, then from the repo root:

    pip install -r exercises/requirements.txt
    pytest exercises/03-q-learning/

See README.md for the task and HINTS.md if you're stuck. The reference
implementation is in solution/q_learning.py — look after you've tried.
"""

import numpy as np

GAMMA = 0.99


def epsilon_greedy(q_row, epsilon: float, n_actions: int, rng) -> int:
    """Pick an action ε-greedily from one state's row of Q-values.

    With probability `epsilon`: a uniformly random action.
    Otherwise: argmax of `q_row` (ties broken however argmax does — fine here).

    `rng` is a numpy Generator (np.random.default_rng(...)). Use `rng.random()`
    for the coin flip and `rng.integers(n_actions)` for the random action.
    Return a Python int.
    """
    # TODO
    raise NotImplementedError("epsilon_greedy")


def q_update(q_table, state, action, reward, next_state, done, alpha: float, gamma: float = GAMMA) -> float:
    """One Q-learning update, in place on `q_table`.

        target = reward                                         if done
                 reward + gamma * max_a' q_table[next_state, a']  otherwise
        q_table[state, action] += alpha * (target - q_table[state, action])

    Return the TD error: target - (old q_table[state, action]).
    """
    # TODO
    raise NotImplementedError("q_update")


def train(n_episodes: int = 2000, alpha: float = 0.8, epsilon: float = 0.1,
          gamma: float = GAMMA, seed: int = 0):
    """Train tabular Q-learning on FrozenLake-v1 with is_slippery=False.

    Returns (q_table, returns_per_episode).
    """
    import gymnasium as gym

    rng = np.random.default_rng(seed)
    env = gym.make("FrozenLake-v1", is_slippery=False)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    # Optimistic initialization: start Q above the true values, so every untried
    # action looks attractive. That drives the agent to try everything early on
    # — without it, argmax over an all-zeros row always picks action 0 and the
    # agent gets stuck against a wall. (1.0 matches FrozenLake's max return.)
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

            # TODO: one Q-learning update for this transition
            #   q_update(q_table, state, action, reward, next_state, done, alpha, gamma)
            raise NotImplementedError("train: the update step")

            state = next_state
            total += float(reward)
        returns.append(total)
    env.close()
    return q_table, returns


def greedy_policy_return(q_table, seed: int = 0) -> float:
    """Run one episode of the greedy policy from `q_table`; return total reward.

    On FrozenLake-v1 (non-slippery), reward is 1.0 only for reaching the goal,
    so this is 1.0 iff the greedy policy reaches the goal.
    """
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
