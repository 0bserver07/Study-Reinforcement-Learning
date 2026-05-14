"""REINFORCE on CartPole-v1 — reference solution.

Look at this after you've made a real attempt at ../starter.py.

Run it directly:  python3 exercises/02-policy-gradients/solution/reinforce.py
"""

import numpy as np
import torch
import torch.nn as nn

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


def reinforce_loss(log_probs: list, returns: torch.Tensor) -> torch.Tensor:
    advantages = (returns - returns.mean()) / (returns.std() + 1e-8)
    lp = torch.stack(list(log_probs))
    return -(lp * advantages).sum()


def train(num_episodes: int = 800, lr: float = 1e-3, seed: int = 0) -> list:
    import gymnasium as gym

    torch.manual_seed(seed)
    np.random.seed(seed)
    env = gym.make("CartPole-v1")
    policy = PolicyNet(env.observation_space.shape[0], env.action_space.n)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    episode_returns = []
    for ep in range(num_episodes):
        state, _ = env.reset(seed=seed + ep)
        log_probs, rewards = [], []
        done = False
        while not done:
            action, log_prob = select_action(policy, state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            log_probs.append(log_prob)
            rewards.append(float(reward))

        loss = reinforce_loss(log_probs, compute_returns(rewards))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
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
