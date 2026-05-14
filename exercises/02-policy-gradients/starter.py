"""REINFORCE on CartPole-v1 — exercise starter.

Fill in the TODOs, then from the repo root:

    pip install -r exercises/requirements.txt
    pytest exercises/02-policy-gradients/

See README.md for the task and HINTS.md if you're stuck. The reference
implementation is in solution/reinforce.py — look after you've tried.
"""

import numpy as np
import torch
import torch.nn as nn

GAMMA = 0.99


class PolicyNet(nn.Module):
    """Maps a state vector to action logits (raw scores, NOT probabilities)."""

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        # TODO: build an MLP  state_dim -> hidden -> hidden -> n_actions  with
        # ReLU between the linear layers. Do NOT add a softmax here — return
        # logits. (Categorical(logits=...) is more numerically stable than
        # Categorical(probs=softmax(...)), and it's what select_action expects.)
        raise NotImplementedError("PolicyNet.__init__")

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # TODO: return logits, shape [..., n_actions]
        raise NotImplementedError("PolicyNet.forward")


def select_action(policy: PolicyNet, state: np.ndarray):
    """Sample an action from the policy.

    Returns (action, log_prob):
      - action: a Python int
      - log_prob: a scalar tensor for log π(action | state) that STILL CARRIES
        GRADIENT (don't call .item() on it — the loss has to backprop through it).
    """
    # TODO:
    #   - turn `state` into a float32 tensor
    #   - logits = policy(state_tensor)
    #   - dist = torch.distributions.Categorical(logits=logits)
    #   - action = dist.sample()
    #   - return int(action.item()), dist.log_prob(action)
    raise NotImplementedError("select_action")


def compute_returns(rewards: list, gamma: float = GAMMA) -> torch.Tensor:
    """Discounted return-to-go for each timestep.

        returns[t] = rewards[t] + gamma*rewards[t+1] + gamma**2*rewards[t+2] + ...

    Return a 1-D float32 tensor the same length as `rewards`. Do NOT normalize
    here — that happens in reinforce_loss.
    """
    # TODO: walk `rewards` backwards, accumulating  R = r + gamma * R,
    # building the list of returns, then reverse it and make it a tensor.
    raise NotImplementedError("compute_returns")


def reinforce_loss(log_probs: list, returns: torch.Tensor) -> torch.Tensor:
    """The REINFORCE objective, as a loss to MINIMIZE.

    Policy-gradient ASCENT maximizes  sum_t log π(a_t|s_t) * A_t,
    so we minimize  -( sum_t log π(a_t|s_t) * A_t ).

    Use the normalized returns as A_t: subtract the mean (a baseline) and divide
    by the std + 1e-8 (scaling). Subtracting a baseline doesn't bias the gradient
    in expectation; it just lowers variance.
    """
    # TODO:
    #   - advantages = (returns - returns.mean()) / (returns.std() + 1e-8)
    #   - stack log_probs into one tensor
    #   - return -(log_probs * advantages).sum()
    raise NotImplementedError("reinforce_loss")


def train(num_episodes: int = 800, lr: float = 1e-3, seed: int = 0) -> list:
    """Train REINFORCE on CartPole-v1. Returns the list of per-episode total rewards."""
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

        # TODO: one policy-gradient update for this episode:
        #   returns = compute_returns(rewards)
        #   loss = reinforce_loss(log_probs, returns)
        #   optimizer.zero_grad()
        #   loss.backward()
        #   torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)  # a noisy episode shouldn't blow up the step
        #   optimizer.step()
        raise NotImplementedError("train: the update step")

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
