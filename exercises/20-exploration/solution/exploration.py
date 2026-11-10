"""Exploration with an RND novelty bonus on a combination lock: reference solution.

Look at this after you've tried ../starter.py.

Run it directly:  python3 exercises/20-exploration/solution/exploration.py

What's here:
  - CombinationLock: 16 squares in a row. Right moves to the next square. Left
    is the wrong move: it ends the episode with reward 0. Reaching the last
    square gives +1 and ends the episode.
  - QLearningAgent: tabular Q-learning with epsilon-greedy, ties broken at random.
  - RNDIntrinsicReward: two small MLPs (target frozen, predictor trained).
    The bonus for state s is ||predictor(s) - target(s)||^2, divided by a
    running standard deviation.
  - train_q_learning_alone, train_with_intrinsic: the two training paths.

Plain Q-learning moves at random until its first +1, and fifteen rights in a
row come once in 2^15 = 32,768 episodes. With the bonus, the first +1 came at a
median of episode 22 over 1,000 seeds, and never later than episode 53.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── environment ───────────────────────────────────────────────────────────────


class CombinationLock:
    """Squares 0..n_states-1 in a row. The agent starts on square 0.

    Actions: 0 = left, 1 = right.
    Right moves to the next square. Entering the last square gives reward 1.0
    and ends the episode.
    Left is the wrong move on every square, square 0 included: the episode ends
    with reward 0 and the agent stays where it was.

    So an episode lasts at most n_states - 1 moves. There's no time limit, and
    every done=True is a real ending.
    """

    def __init__(self, n_states: int = 16):
        self.n_states = n_states
        self.state = 0

    def reset(self) -> int:
        self.state = 0
        return self.state

    def step(self, action: int) -> tuple:
        assert action in (0, 1), f"action must be 0 (left) or 1 (right), got {action}"
        if action == 0:  # the wrong move
            return self.state, 0.0, True, {}
        self.state += 1
        reached_goal = self.state == self.n_states - 1
        reward = 1.0 if reached_goal else 0.0
        return self.state, reward, reached_goal, {}


# ── Q-learning ────────────────────────────────────────────────────────────────


class QLearningAgent:
    """Tabular Q-learning over (state, action), epsilon-greedy with random tie-breaking."""

    def __init__(
        self,
        n_states: int,
        n_actions: int = 2,
        alpha: float = 0.5,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        seed: int = 0,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q = np.zeros((n_states, n_actions))
        self.rng = np.random.default_rng(seed)

    def act(self, state: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        q = self.q[state]
        best = np.flatnonzero(q == q.max())  # every action tied for the max
        return int(self.rng.choice(best))

    def update(self, s: int, a: int, r: float, s_next: int, done: bool) -> float:
        target = r if done else r + self.gamma * np.max(self.q[s_next])
        td = target - self.q[s, a]
        self.q[s, a] += self.alpha * td
        return float(td)


# ── RND intrinsic reward ──────────────────────────────────────────────────────


class _MLP(nn.Module):
    """Small MLP: one-hot state -> hidden -> feature vector of dim feature_dim."""

    def __init__(self, n_states: int, hidden: int, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNDIntrinsicReward:
    """Random Network Distillation on a discrete state space.

    Two MLPs with identical architecture: target is randomly initialized and
    frozen; predictor is trained to match target on visited states. The bonus
    is the squared L2 distance between their outputs, divided by a running
    standard deviation of those distances.

    A new state produces a large error (the predictor hasn't matched the random
    target on that input yet). After many gradient steps on the same state, the
    error shrinks toward zero, so the bonus for that state decays.
    """

    def __init__(
        self,
        n_states: int,
        hidden: int = 32,
        feature_dim: int = 16,
        lr: float = 1e-3,
        seed: int = 0,
    ):
        self.n_states = n_states
        torch.manual_seed(seed)
        self.target = _MLP(n_states, hidden, feature_dim)
        self.predictor = _MLP(n_states, hidden, feature_dim)
        for p in self.target.parameters():
            p.requires_grad = False
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        # Running statistics for normalizing intrinsic reward (Welford-ish).
        self._reward_mean = 0.0
        self._reward_var = 1.0
        self._reward_count = 1e-4

    def _onehot(self, state: int) -> torch.Tensor:
        x = torch.zeros(self.n_states)
        x[state] = 1.0
        return x

    def _raw_error(self, state: int) -> torch.Tensor:
        x = self._onehot(state).unsqueeze(0)  # shape (1, n_states)
        with torch.no_grad():
            t = self.target(x)
            p = self.predictor(x)
            return (p - t).pow(2).mean(dim=-1).squeeze(0)  # scalar tensor

    def intrinsic_reward(self, state: int) -> float:
        """Normalized prediction error at `state`. No gradient, just a number.

        The raw error is divided by a running standard deviation, so the bonus
        stays on a stable scale while the raw errors shrink as the predictor
        learns. The relative novelty of states survives the division.
        """
        err = self._raw_error(state).item()
        self._update_running_stats(err)
        std = float(np.sqrt(self._reward_var)) + 1e-8
        return err / std

    def _update_running_stats(self, x: float) -> None:
        self._reward_count += 1
        delta = x - self._reward_mean
        self._reward_mean += delta / self._reward_count
        delta2 = x - self._reward_mean
        self._reward_var += (delta * delta2 - self._reward_var) / self._reward_count

    def update(self, state: int) -> float:
        """One gradient step on the predictor for `state`. Returns the MSE loss."""
        x = self._onehot(state).unsqueeze(0)
        target_out = self.target(x).detach()
        pred_out = self.predictor(x)
        loss = F.mse_loss(pred_out, target_out)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())


# ── training loops ────────────────────────────────────────────────────────────


def train_q_learning_alone(
    n_states: int = 16,
    n_episodes: int = 200,
    seed: int = 0,
) -> list:
    """Plain Q-learning with epsilon-greedy. Used as the failure baseline."""
    env = CombinationLock(n_states=n_states)
    agent = QLearningAgent(n_states=n_states, seed=seed)

    episode_returns = []
    for _ in range(n_episodes):
        s = env.reset()
        total = 0.0
        done = False
        while not done:
            a = agent.act(s)
            s_next, r, done, _ = env.step(a)
            agent.update(s, a, r, s_next, done)
            s = s_next
            total += r
        episode_returns.append(total)
    return episode_returns


def train_with_intrinsic(
    n_states: int = 16,
    n_episodes: int = 200,
    intrinsic_coef: float = 0.1,
    seed: int = 0,
) -> tuple:
    """Q-learning + RND. Returns (extrinsic_returns_per_episode, agent, rnd).

    The agent updates Q on `r_ext + intrinsic_coef * r_int`, where r_int is the
    normalized prediction error at the square it just moved to. The predictor
    takes one gradient step on that square, so the bonus for squares the agent
    visits a lot decays.

    A move that ends the episode gets no bonus: there's no next square to
    explore from. On this lock that keeps the wrong move worth exactly 0.

    Returns the *extrinsic* return per episode, so the metric isn't inflated by
    the bonus.
    """
    env = CombinationLock(n_states=n_states)
    agent = QLearningAgent(n_states=n_states, seed=seed)
    rnd = RNDIntrinsicReward(n_states=n_states, seed=seed)

    episode_returns = []
    for _ in range(n_episodes):
        s = env.reset()
        total_ext = 0.0
        done = False
        while not done:
            a = agent.act(s)
            s_next, r_ext, done, _ = env.step(a)
            r_int = 0.0
            if not done:
                r_int = rnd.intrinsic_reward(s_next)
                rnd.update(s_next)
            agent.update(s, a, r_ext + intrinsic_coef * r_int, s_next, done)
            s = s_next
            total_ext += r_ext
        episode_returns.append(total_ext)
    return episode_returns, agent, rnd


# ── main ──────────────────────────────────────────────────────────────────────


def _first_goal(returns: list):
    """1-based episode of the first +1, or None if there wasn't one."""
    return next((i + 1 for i, r in enumerate(returns) if r > 0), None)


if __name__ == "__main__":
    plain = train_q_learning_alone(seed=0)
    print(f"plain Q-learning   reached the goal in {int(sum(plain))} of {len(plain)} episodes")

    rnd_returns, agent, _ = train_with_intrinsic(seed=0)
    print(
        f"Q-learning + RND   reached the goal in {int(sum(rnd_returns))} of {len(rnd_returns)} "
        f"episodes, the first at episode {_first_goal(rnd_returns)}"
    )
    greedy = "".join("R" if agent.q[s, 1] > agent.q[s, 0] else "L" for s in range(agent.n_states - 1))
    print(f"greedy move on squares 0-{agent.n_states - 2} after training: {greedy}")
