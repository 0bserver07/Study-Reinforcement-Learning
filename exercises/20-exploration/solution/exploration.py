"""Intrinsic-motivation exploration on a sparse-reward chain MDP — reference solution.

Look at this after you've tried ../starter.py.

Run it directly:  python3 exercises/20-exploration/solution/exploration.py

What's here:
  - ChainEnv: a 1D chain of N states, reward only at the right end.
  - QLearningAgent: tabular Q-learning with epsilon-greedy.
  - RNDIntrinsicReward: two small MLPs (target frozen, predictor trained).
    Intrinsic reward for state s is ||predictor(s) - target(s)||^2 with a
    running standard-deviation normalization so the bonus magnitude stays
    roughly constant as the predictor learns.
  - train_q_learning_alone, train_with_intrinsic: the two training paths.

On the default 20-state chain, vanilla Q-learning never sees the goal in 200
episodes; Q-learning + RND finds the goal and learns the path.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── environment ───────────────────────────────────────────────────────────────


class ChainEnv:
    """1D chain MDP. States 0..n_states-1, agent starts at 0, goal at n_states-1.

    Actions: 0 = left, 1 = right. left at state 0 stays at 0.
    Reward: 1.0 on entering the goal state, 0.0 everywhere else.
    Episode ends on reaching the goal or after max_steps.

    With epsilon-greedy at epsilon=0.1, the chance of a random walk of length
    100 reaching the far end of a length-20 chain is tiny — that's the point.
    """

    def __init__(self, n_states: int = 20, max_steps: int = 100):
        self.n_states = n_states
        self.max_steps = max_steps
        self.state = 0
        self.steps = 0

    def reset(self) -> int:
        self.state = 0
        self.steps = 0
        return self.state

    def step(self, action: int) -> tuple:
        assert action in (0, 1), f"action must be 0 (left) or 1 (right), got {action}"
        if action == 1:
            self.state = min(self.state + 1, self.n_states - 1)
        else:
            self.state = max(self.state - 1, 0)
        self.steps += 1

        reached_goal = self.state == self.n_states - 1
        timed_out = self.steps >= self.max_steps
        reward = 1.0 if reached_goal else 0.0
        done = reached_goal or timed_out
        return self.state, reward, done, {}


# ── Q-learning ────────────────────────────────────────────────────────────────


class QLearningAgent:
    """Tabular Q-learning over (state, action). Standard epsilon-greedy."""

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
        return int(np.argmax(self.q[state]))

    def update(self, s: int, a: int, r: float, s_next: int, done: bool) -> float:
        target = r if done else r + self.gamma * np.max(self.q[s_next])
        td = target - self.q[s, a]
        self.q[s, a] += self.alpha * td
        return float(td)


# ── RND intrinsic reward ──────────────────────────────────────────────────────


class _MLP(nn.Module):
    """Tiny MLP: integer state -> one-hot -> feature vector of dim feature_dim."""

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
    frozen; predictor is trained to match target on visited states. Intrinsic
    reward is the squared L2 distance between their outputs, normalized by a
    running standard deviation so the bonus stays on a roughly constant scale
    as training progresses.

    A new state will produce a large error (predictor hasn't matched the random
    target for that input). After many gradient steps on the same state, the
    error shrinks toward zero — the bonus for that state decays.
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
        """Normalized prediction error at `state`. No gradient — just a number.

        The raw error is divided by a running standard deviation. This keeps
        the bonus on a stable scale: early on the absolute error magnitudes are
        large (the predictor hasn't learned anything), and they fall as
        training proceeds. Normalizing preserves the *relative* novelty
        signal across that decay.
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
    n_states: int = 20,
    n_episodes: int = 200,
    max_steps: int = 100,
    seed: int = 0,
) -> list:
    """Plain Q-learning with epsilon-greedy. Used as the failure baseline."""
    env = ChainEnv(n_states=n_states, max_steps=max_steps)
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
    n_states: int = 20,
    n_episodes: int = 200,
    max_steps: int = 100,
    intrinsic_coef: float = 0.1,
    seed: int = 0,
) -> tuple:
    """Q-learning + RND. Returns (extrinsic_returns_per_episode, agent, rnd).

    The agent updates Q on `r_combined = r_extrinsic + intrinsic_coef * r_intrinsic`,
    where r_intrinsic is the normalized prediction error at the *next* state.
    The RND predictor takes one gradient step on every visited state, so the
    bonus for states the agent visits a lot decays toward zero.

    Returns the *extrinsic* return per episode, so the metric isn't inflated by
    the intrinsic bonus.
    """
    env = ChainEnv(n_states=n_states, max_steps=max_steps)
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
            r_int = rnd.intrinsic_reward(s_next)
            r_combined = r_ext + intrinsic_coef * r_int
            agent.update(s, a, r_combined, s_next, done)
            rnd.update(s_next)
            s = s_next
            total_ext += r_ext
        episode_returns.append(total_ext)
    return episode_returns, agent, rnd


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plain = train_q_learning_alone(seed=0)
    last50_plain = sum(plain[-50:]) / 50
    print(f"plain Q-learning   mean(last 50) = {last50_plain:.3f}")

    rnd_returns, _, _ = train_with_intrinsic(seed=0)
    last50_rnd = sum(rnd_returns[-50:]) / 50
    print(f"Q-learning + RND   mean(last 50) = {last50_rnd:.3f}  (goal reward = 1.0)")
