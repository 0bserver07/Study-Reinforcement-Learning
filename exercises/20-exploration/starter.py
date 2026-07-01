"""Intrinsic-motivation exploration on a sparse-reward chain MDP — starter.

Fill in the TODOs, then from the repo root:

    pip install -r exercises/requirements.txt
    pytest exercises/20-exploration/

See README.md for the task and HINTS.md if you're stuck. The reference
implementation is in solution/exploration.py — look after you've tried.

The setup:
  - ChainEnv: a 1D chain of N states. Agent starts at state 0. Actions are
    {left, right}. Reward is 1.0 only on entering the goal at state N-1.
  - With epsilon-greedy exploration at epsilon=0.1 on a length-20 chain
    capped at 100 steps per episode, vanilla Q-learning never sees the
    goal in 200 episodes. The mean return stays at 0.
  - RND (Random Network Distillation): two MLPs that map a one-hot state
    to a feature vector. The target is randomly initialized and frozen;
    the predictor is trained to match the target's output on visited
    states. The intrinsic reward is the squared L2 distance between the
    two outputs. Novel states have large error; well-visited states have
    small error.
  - Q-learning trained on `r_extrinsic + intrinsic_coef * r_intrinsic`
    reliably reaches the goal.

You implement:
  1. ChainEnv.step  — the environment transition.
  2. QLearningAgent.act / .update — epsilon-greedy + Q-learning update.
  3. RNDIntrinsicReward.intrinsic_reward / .update — RND forward and one
     gradient step on the predictor.
  4. train_with_intrinsic — the loop that combines them.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── environment ───────────────────────────────────────────────────────────────


class ChainEnv:
    """1D chain MDP. States 0..n_states-1. Agent starts at 0. Goal at n_states-1.

    Actions: 0 = left, 1 = right.
    Reward: 1.0 on entering the goal state, 0.0 everywhere else.
    Episode ends on reaching the goal or after max_steps steps.

    Boundary: action 0 at state 0 stays at state 0 (don't fall off the chain).
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
        """Take one step. Return (next_state, reward, done, info).

        TODO:
          - if action == 1: new state = min(self.state + 1, self.n_states - 1)
          - if action == 0: new state = max(self.state - 1, 0)
          - self.steps += 1
          - reached_goal = (new state == self.n_states - 1)
          - timed_out   = (self.steps >= self.max_steps)
          - reward      = 1.0 if reached_goal else 0.0
          - done        = reached_goal or timed_out
          - self.state  = new state
          - return self.state, reward, done, {}
        """
        assert action in (0, 1), f"action must be 0 (left) or 1 (right), got {action}"
        # TODO
        raise NotImplementedError("ChainEnv.step")


# ── Q-learning ────────────────────────────────────────────────────────────────


class QLearningAgent:
    """Tabular Q-learning over (state, action) with epsilon-greedy.

    Q is initialized to zeros — on this env, optimistic init would mask the
    exploration problem you're supposed to fix with RND, so don't add it.
    """

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
        """epsilon-greedy action selection. Return a Python int.

        TODO:
          - with probability self.epsilon, return int(self.rng.integers(self.n_actions))
          - otherwise, return int(np.argmax(self.q[state]))
        """
        # TODO
        raise NotImplementedError("QLearningAgent.act")

    def update(self, s: int, a: int, r: float, s_next: int, done: bool) -> float:
        """One Q-learning update, in place on self.q. Return the TD error.

        target = r                                        if done
                 r + self.gamma * np.max(self.q[s_next])  otherwise
        td_error = target - self.q[s, a]
        self.q[s, a] += self.alpha * td_error

        Note `r` here is the COMBINED reward (extrinsic + intrinsic_coef * intrinsic).
        The agent doesn't know which part came from where — it just learns
        a Q-function over the combined signal.
        """
        # TODO
        raise NotImplementedError("QLearningAgent.update")


# ── RND intrinsic reward ──────────────────────────────────────────────────────


class _MLP(nn.Module):
    """Small MLP. One-hot state -> hidden -> feature vector.

    Don't change this — both the target and the predictor use it. The point
    of RND is that they have the *same* architecture but different weights;
    the target is frozen at its random init, the predictor learns to match.
    """

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

    target is randomly initialized and frozen.
    predictor is trained (Adam, MSE) to match target on visited states.
    Intrinsic reward at state s is the squared L2 distance between
    predictor(one_hot(s)) and target(one_hot(s)), normalized by a running
    standard deviation so the bonus magnitude stays roughly constant as
    the predictor learns and absolute errors shrink.
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
        # Freeze the target — never updates.
        for p in self.target.parameters():
            p.requires_grad = False
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        # Running stats for normalizing intrinsic reward.
        self._reward_mean = 0.0
        self._reward_var = 1.0
        self._reward_count = 1e-4

    def _onehot(self, state: int) -> torch.Tensor:
        """Build a one-hot vector of length self.n_states for `state`."""
        x = torch.zeros(self.n_states)
        x[state] = 1.0
        return x

    def _raw_error(self, state: int) -> torch.Tensor:
        """Squared L2 distance between predictor and target outputs at `state`.

        Returns a scalar tensor under torch.no_grad — the intrinsic reward
        is just a number plugged into Q-learning, not a backprop signal.
        """
        x = self._onehot(state).unsqueeze(0)  # (1, n_states)
        with torch.no_grad():
            t = self.target(x)
            p = self.predictor(x)
            return (p - t).pow(2).mean(dim=-1).squeeze(0)

    def intrinsic_reward(self, state: int) -> float:
        """Normalized prediction error at `state`. Returns a plain float.

        TODO:
          - err = self._raw_error(state).item()
          - self._update_running_stats(err)
          - std = float(np.sqrt(self._reward_var)) + 1e-8
          - return err / std

        Why normalize: early in training, raw errors can be O(1); after the
        predictor has learned a few states they fall to O(0.01). Without
        normalization the effective intrinsic-reward weight is moving without
        you controlling it. Dividing by a running std keeps the *relative*
        novelty signal stable.
        """
        # TODO
        raise NotImplementedError("RNDIntrinsicReward.intrinsic_reward")

    def _update_running_stats(self, x: float) -> None:
        """Online mean/variance update (Welford-ish). Given for you."""
        self._reward_count += 1
        delta = x - self._reward_mean
        self._reward_mean += delta / self._reward_count
        delta2 = x - self._reward_mean
        self._reward_var += (delta * delta2 - self._reward_var) / self._reward_count

    def update(self, state: int) -> float:
        """One gradient step on the predictor toward target on `state`.

        TODO:
          - x = self._onehot(state).unsqueeze(0)
          - target_out = self.target(x).detach()   # frozen — detach is belt and braces
          - pred_out   = self.predictor(x)
          - loss = F.mse_loss(pred_out, target_out)
          - self.optimizer.zero_grad()
          - loss.backward()
          - self.optimizer.step()
          - return float(loss.item())
        """
        # TODO
        raise NotImplementedError("RNDIntrinsicReward.update")


# ── training loops ────────────────────────────────────────────────────────────


def train_q_learning_alone(
    n_states: int = 20,
    n_episodes: int = 200,
    max_steps: int = 100,
    seed: int = 0,
) -> list:
    """Plain Q-learning, no intrinsic reward. The failure baseline.

    Given to you — you don't have to change this. Run it after you've
    finished ChainEnv and QLearningAgent to confirm vanilla Q-learning
    flatlines at 0 returns on this env.
    """
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
    """Q-learning where reward = extrinsic + intrinsic_coef * intrinsic.

    Returns:
        (extrinsic_returns_per_episode, agent, rnd)

    The intrinsic reward at each transition is computed at the *next* state
    (the state you just landed in — that's the one whose novelty you want
    to credit). Then take one RND predictor update on that state so its
    bonus shrinks over time.

    TODO — fill in the inner loop:
        a = agent.act(s)
        s_next, r_ext, done, _ = env.step(a)
        r_int = rnd.intrinsic_reward(s_next)
        r_combined = r_ext + intrinsic_coef * r_int
        agent.update(s, a, r_combined, s_next, done)
        rnd.update(s_next)
        s = s_next
        total_ext += r_ext

    Important: accumulate total_ext from r_ext, not r_combined. The
    integration test measures extrinsic-only return — you want to see
    that the agent actually reaches the goal, not that it's chasing the
    novelty bonus.
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
            # TODO: the inner step (see above)
            raise NotImplementedError("train_with_intrinsic: the inner loop")
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
