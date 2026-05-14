"""Actor-critic on CartPole-v1 — reference solution.

Look at this after you've made a real attempt at ../starter.py.

Run it directly:  python3 exercises/04-actor-critic/solution/actor_critic.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GAMMA = 0.99


class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)  # logits, shape [..., n_actions]


class ValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)  # scalar V(s), shape [...]


def select_action(policy: PolicyNet, state: np.ndarray):
    state_t = torch.as_tensor(state, dtype=torch.float32)
    logits = policy(state_t)
    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample()
    return int(action.item()), dist.log_prob(action)


def compute_returns(rewards: list, gamma: float = GAMMA) -> torch.Tensor:
    out = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        out.append(R)
    out.reverse()
    return torch.tensor(out, dtype=torch.float32)


def actor_critic_loss(log_probs: list, returns: torch.Tensor, values: torch.Tensor):
    """Compute actor and critic losses separately.

    The key move: advantages = (returns - values).detach() — the policy
    gradient does NOT flow through the critic. The two networks learn
    independently: the critic learns to predict returns (MSE), and the
    actor learns to prefer actions that had positive advantage.

    Returns:
        actor_loss: scalar, requires grad (flows into PolicyNet)
        critic_loss: scalar, requires grad (flows into ValueNet)
    """
    advantages = (returns - values).detach()
    lp = torch.stack(list(log_probs))
    actor_loss = -(lp * advantages).sum()
    critic_loss = F.mse_loss(values, returns)
    return actor_loss, critic_loss


def train(
    num_episodes: int = 600,
    lr: float = 3e-3,
    seed: int = 0,
    gamma: float = GAMMA,
    value_coef: float = 0.5,
) -> list:
    """Train actor-critic on CartPole-v1. Returns per-episode total rewards."""
    import gymnasium as gym

    torch.manual_seed(seed)
    np.random.seed(seed)
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = PolicyNet(state_dim, n_actions)
    critic = ValueNet(state_dim)

    # One optimizer over both nets — simpler than two separate ones and works fine.
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(critic.parameters()), lr=lr
    )

    episode_returns = []
    for ep in range(num_episodes):
        state, _ = env.reset(seed=seed + ep)
        log_probs, rewards, values = [], [], []
        done = False
        while not done:
            state_t = torch.as_tensor(state, dtype=torch.float32)
            action, log_prob = select_action(policy, state)
            value = critic(state_t)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            log_probs.append(log_prob)
            rewards.append(float(reward))
            values.append(value)

        returns = compute_returns(rewards, gamma)
        values_t = torch.stack(values)
        actor_loss, critic_loss = actor_critic_loss(log_probs, returns, values_t)
        loss = actor_loss + value_coef * critic_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(policy.parameters()) + list(critic.parameters()), 1.0
        )
        optimizer.step()

        episode_returns.append(sum(rewards))

    env.close()
    return episode_returns


if __name__ == "__main__":
    rets = train()
    last100 = rets[-100:]
    print(
        f"episodes: {len(rets)}  "
        f"mean(last 100): {sum(last100) / len(last100):.1f}  "
        f"max: {max(rets):.0f}  "
        f"(random policy ≈ 21, solved ≈ 475)"
    )
