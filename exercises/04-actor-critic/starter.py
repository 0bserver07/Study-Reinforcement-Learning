"""Actor-critic on CartPole-v1 — exercise starter.

Fill in the TODOs, then from the repo root:

    pip install -r exercises/requirements.txt
    pytest exercises/04-actor-critic/

See README.md for the task and HINTS.md if you're stuck. The reference
implementation is in solution/actor_critic.py — look after you've tried.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GAMMA = 0.99


class PolicyNet(nn.Module):
    """Maps a state vector to action logits (raw scores, NOT probabilities)."""

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        # TODO: build an MLP  state_dim -> hidden -> hidden -> n_actions  with
        # ReLU between the linear layers. Same shape as exercise 02 — no softmax,
        # return logits. (Categorical(logits=...) handles the softmax internally
        # and is numerically more stable.)
        raise NotImplementedError("PolicyNet.__init__")

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # TODO: return logits, shape [..., n_actions]
        raise NotImplementedError("PolicyNet.forward")


class ValueNet(nn.Module):
    """Maps a state vector to a scalar estimate of V(s)."""

    def __init__(self, state_dim: int, hidden: int = 128):
        super().__init__()
        # TODO: build an MLP  state_dim -> hidden -> hidden -> 1  with ReLU
        # between the linear layers. The output is a single scalar V(s) —
        # squeeze the trailing dimension in forward so the output shape is
        # [...] (not [..., 1]).
        raise NotImplementedError("ValueNet.__init__")

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # TODO: return V(s) — shape [] for a single state, [batch] for a batch.
        # Hint: nn.Linear(..., 1) gives shape [..., 1]; use .squeeze(-1) to drop
        # the last dim.
        raise NotImplementedError("ValueNet.forward")


def select_action(policy: PolicyNet, state: np.ndarray):
    """Sample an action from the policy.

    Returns (action, log_prob):
      - action: a Python int
      - log_prob: a scalar tensor for log π(action | state) that STILL CARRIES
        GRADIENT (don't call .item() on it — the loss has to backprop through it).
    """
    # TODO: same as exercise 02 —
    #   - turn `state` into a float32 tensor
    #   - logits = policy(state_tensor)
    #   - dist = torch.distributions.Categorical(logits=logits)
    #   - action = dist.sample()
    #   - return int(action.item()), dist.log_prob(action)
    raise NotImplementedError("select_action")


def compute_returns(rewards: list, gamma: float = GAMMA) -> torch.Tensor:
    """Discounted return-to-go for each timestep.

        returns[t] = rewards[t] + gamma*rewards[t+1] + gamma**2*rewards[t+2] + ...

    Return a 1-D float32 tensor the same length as `rewards`. Same function as
    exercise 02 — copy it if you like, but re-derive it; you'll need the
    intuition when you get to TD methods.
    """
    # TODO: walk `rewards` backwards, accumulating  R = r + gamma * R,
    # building the list of returns, then reverse it and make it a tensor.
    raise NotImplementedError("compute_returns")


def actor_critic_loss(log_probs: list, returns: torch.Tensor, values: torch.Tensor):
    """Compute actor and critic losses separately.

    The key difference from REINFORCE: instead of normalizing the raw returns
    as a crude baseline, we use the critic's estimate V(s_t) as a state-dependent
    baseline. The advantage A_t = G_t - V(s_t) tells the actor whether the
    outcome was better or worse than the critic predicted.

    Important: the policy gradient does NOT flow through the critic. The
    advantages must be detached from the critic's computation graph, or the
    actor loss will pull ValueNet's parameters in a policy direction, which
    is wrong. Use `(returns - values).detach()`.

    Returns:
        actor_loss: scalar tensor. Requires grad (flows into PolicyNet).
        critic_loss: scalar tensor. Requires grad (flows into ValueNet).
    """
    # TODO:
    #   - advantages = (returns - values).detach()     # stop gradient here
    #   - lp = torch.stack(list(log_probs))
    #   - actor_loss  = -(lp * advantages).sum()       # policy gradient, negated to minimize
    #   - critic_loss = F.mse_loss(values, returns)    # train the critic with MSE
    #   - return actor_loss, critic_loss
    raise NotImplementedError("actor_critic_loss")


def train(
    num_episodes: int = 600,
    lr: float = 3e-3,
    seed: int = 0,
    gamma: float = GAMMA,
    value_coef: float = 0.5,
) -> list:
    """Train actor-critic on CartPole-v1. Returns the list of per-episode total rewards."""
    import gymnasium as gym

    torch.manual_seed(seed)
    np.random.seed(seed)
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = PolicyNet(state_dim, n_actions)
    critic = ValueNet(state_dim)

    # One Adam optimizer over both networks — simpler than two separate ones.
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
            # TODO: compute value = critic(state_t) and append it to `values`.
            # This is V(s_t) before we take the action — the baseline for this step.
            raise NotImplementedError("train: call critic(state_t) and append value")

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            log_probs.append(log_prob)
            rewards.append(float(reward))

        # TODO: one actor-critic update for this episode:
        #   returns = compute_returns(rewards, gamma)
        #   values_t = torch.stack(values)
        #   actor_loss, critic_loss = actor_critic_loss(log_probs, returns, values_t)
        #   loss = actor_loss + value_coef * critic_loss
        #   optimizer.zero_grad()
        #   loss.backward()
        #   torch.nn.utils.clip_grad_norm_(
        #       list(policy.parameters()) + list(critic.parameters()), 1.0
        #   )
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
