"""Exploration with an RND novelty bonus on a combination lock: starter.

Fill in the TODOs, then from the repo root:

    pip install -r exercises/requirements.txt
    pytest exercises/20-exploration/

See README.md for the task and HINTS.md if you're stuck. The reference
implementation is in solution/exploration.py; look after you've tried.

The setup:
  - CombinationLock: 16 squares in a row, start on square 0. Right moves to
    the next square. Left is the wrong move: it ends the episode with reward
    0. Reaching the last square gives +1 and ends the episode.
  - Plain epsilon-greedy Q-learning keeps every Q-value at exactly 0 until its
    first +1, so until then it moves at random, and fifteen rights in a row
    come once in 2^15 = 32,768 episodes.
  - RND (Random Network Distillation): two MLPs that map a one-hot state to
    a feature vector. The target is randomly initialized and frozen; the
    predictor is trained to match the target's output on visited states.
    The bonus is the squared L2 distance between the two outputs: large on
    new states, small on well-visited ones.
  - Q-learning trained on `r_extrinsic + intrinsic_coef * r_intrinsic` opens
    the lock in a few dozen episodes.

You implement:
  1. CombinationLock.step: the environment transition.
  2. QLearningAgent.act / .update: epsilon-greedy with random tie-breaking,
     and the Q-learning update.
  3. RNDIntrinsicReward.intrinsic_reward / .update: RND forward and one
     gradient step on the predictor.
  4. train_with_intrinsic: the loop that combines them.
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
        """Take one move. Return (next_state, reward, done, info).

        TODO:
          - if action == 0 (the wrong move): the episode ends where the agent
            stands, so return self.state, 0.0, True, {}
          - otherwise move right: self.state += 1
          - reached_goal = (self.state == self.n_states - 1)
          - reward       = 1.0 if reached_goal else 0.0
          - return self.state, reward, reached_goal, {}
        """
        assert action in (0, 1), f"action must be 0 (left) or 1 (right), got {action}"
        # TODO
        raise NotImplementedError("CombinationLock.step")


# ── Q-learning ────────────────────────────────────────────────────────────────


class QLearningAgent:
    """Tabular Q-learning over (state, action) with epsilon-greedy.

    Q is initialized to zeros. Optimistic init would do the exploring for you
    on this lock (every untried move would look good), which hides the
    problem you're supposed to fix with RND, so don't add it.
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
        """epsilon-greedy action selection, ties broken at random. Return a Python int.

        TODO:
          - with probability self.epsilon, return int(self.rng.integers(self.n_actions))
          - otherwise find every action whose Q-value equals max(self.q[state])
            (np.flatnonzero helps), pick one of them with self.rng.choice, and
            return it as an int

        Why not np.argmax: it returns the first of several tied actions. Q
        starts at 0, so at first every greedy choice is a tie, and np.argmax
        would pick action 0, the wrong move, every time.
        """
        # TODO
        raise NotImplementedError("QLearningAgent.act")

    def update(self, s: int, a: int, r: float, s_next: int, done: bool) -> float:
        """One Q-learning update, in place on self.q. Return the TD error.

        target = r                                        if done
                 r + self.gamma * np.max(self.q[s_next])  otherwise
        td_error = target - self.q[s, a]
        self.q[s, a] += self.alpha * td_error

        On this lock done=True always means the episode really ended (a wrong
        move or the goal), so there's no future to bootstrap from.

        Note `r` here is the COMBINED reward (extrinsic + intrinsic_coef * intrinsic).
        The agent doesn't know which part came from where; it just learns
        a Q-function over the combined signal.
        """
        # TODO
        raise NotImplementedError("QLearningAgent.update")


# ── RND intrinsic reward ──────────────────────────────────────────────────────


class _MLP(nn.Module):
    """Small MLP. One-hot state -> hidden -> feature vector.

    Don't change this: both the target and the predictor use it. The point
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
        # Freeze the target: it never updates.
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

        Returns a scalar tensor under torch.no_grad: the intrinsic reward
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
        normalization the effective intrinsic-reward weight moves without
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
          - target_out = self.target(x).detach()   # frozen; detach is belt and braces
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
    n_states: int = 16,
    n_episodes: int = 200,
    seed: int = 0,
) -> list:
    """Plain Q-learning, no intrinsic reward. The failure baseline.

    Given to you; you don't have to change this. Run it after you've
    finished CombinationLock and QLearningAgent to confirm that plain
    Q-learning almost never reaches the goal.
    """
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
    """Q-learning where reward = extrinsic + intrinsic_coef * intrinsic.

    Returns:
        (extrinsic_returns_per_episode, agent, rnd)

    The intrinsic reward for a move is computed at the *next* state (the
    square you just landed on: that's the one whose novelty you want to
    credit). Then take one RND predictor update on that square so its bonus
    shrinks over time.

    A move that ends the episode gets no bonus: there's no next square to
    explore from. On this lock that keeps the wrong move worth exactly 0.

    TODO: fill in the inner loop:
        a = agent.act(s)
        s_next, r_ext, done, _ = env.step(a)
        r_int = 0.0
        if not done:
            r_int = rnd.intrinsic_reward(s_next)
            rnd.update(s_next)
        agent.update(s, a, r_ext + intrinsic_coef * r_int, s_next, done)
        s = s_next
        total_ext += r_ext

    Important: accumulate total_ext from r_ext, not the combined reward. The
    integration test measures extrinsic-only return: you want to see that
    the agent actually reaches the goal, not that it's collecting bonus.
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
            # TODO: the inner step (see above)
            raise NotImplementedError("train_with_intrinsic: the inner loop")
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
