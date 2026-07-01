"""PPO on CartPole-v1 — reference solution.

Look at this after you've made a real attempt at ../starter.py.

Run it directly:  python3 exercises/05-ppo/solution/ppo.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01


# ── network ──────────────────────────────────────────────────────────────────

class ActorCriticNet(nn.Module):
    """Shared-trunk MLP with a policy head (logits) and a value head (scalar).

    Single-state input shape:  (state_dim,)        → logits (n_actions,),  value ()
    Batched input shape:       (batch, state_dim)  → logits (batch, n_actions), value (batch,)
    """

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor):
        h = self.trunk(state)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value


# ── GAE ──────────────────────────────────────────────────────────────────────

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float = GAMMA,
    lam: float = LAMBDA,
):
    """Generalized Advantage Estimation.

    Args:
        rewards:    (T,) float tensor of rewards collected during the rollout.
        values:     (T,) float tensor of V(s_t) at each step.
        dones:      (T,) float tensor, 1.0 if the episode ended at step t else 0.0.
        next_value: scalar — V(s_T), the bootstrap value after the rollout.
        gamma:      discount.
        lam:        GAE λ; λ=0 → TD(0), λ=1 → Monte Carlo.

    Returns:
        advantages: (T,) tensor.
        returns:    (T,) tensor — advantages + values, the target for the critic.
    """
    T = rewards.shape[0]
    advantages = torch.zeros(T, dtype=torch.float32)
    gae = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        v_next = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * v_next * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


# ── PPO loss ─────────────────────────────────────────────────────────────────

def ppo_clip_loss(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    value_pred: torch.Tensor,
    value_target: torch.Tensor,
    entropy: torch.Tensor,
    clip_eps: float = CLIP_EPS,
    value_coef: float = VALUE_COEF,
    entropy_coef: float = ENTROPY_COEF,
):
    """Total PPO loss = clipped policy + value MSE − entropy bonus.

    The clipped surrogate (PPO-Clip):
        ratio = exp(log_probs_new − log_probs_old)
        L_policy = −E[min(ratio·A, clip(ratio, 1−ε, 1+ε)·A)]

    Value loss:
        L_value = MSE(value_pred, value_target)

    Entropy bonus is *subtracted* from the loss (we want HIGH entropy ⇒ low loss).

    Returns: scalar loss with gradient.
    """
    ratio = (log_probs_new - log_probs_old).exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()

    value_loss = F.mse_loss(value_pred, value_target)
    entropy_term = entropy.mean()

    return policy_loss + value_coef * value_loss - entropy_coef * entropy_term


# ── rollout collector ────────────────────────────────────────────────────────

def collect_rollouts(env, net: ActorCriticNet, n_steps: int, state, seed_offset: int = 0):
    """Run `n_steps` of env interaction with `net` and return tensors.

    Stateful: takes the current `state` and the seed_offset (for env.reset
    after a done) and returns the next state so the next call can continue.

    Returns:
        obs:        (n_steps, state_dim)
        actions:    (n_steps,)        LongTensor
        log_probs:  (n_steps,)        detached (these are the "old" log-probs)
        rewards:    (n_steps,)
        values:     (n_steps,)        detached
        dones:      (n_steps,)        1.0 if terminated/truncated at that step
        episode_rewards: list of floats — total reward for each episode that
                         FINISHED inside this rollout
        next_state: the env state to feed into the next collect_rollouts call
        next_value: V(next_state), detached — for GAE bootstrap
        ep_counter: number of episodes completed during this rollout (for seeding)
    """
    obs_buf = []
    act_buf = []
    logp_buf = []
    rew_buf = []
    val_buf = []
    done_buf = []
    episode_rewards = []

    current_ep_reward = 0.0
    ep_counter = 0

    for _ in range(n_steps):
        state_t = torch.as_tensor(state, dtype=torch.float32)
        with torch.no_grad():
            logits, value = net(state_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        next_state, reward, terminated, truncated, _ = env.step(int(action.item()))
        done = terminated or truncated

        obs_buf.append(state_t)
        act_buf.append(action)
        logp_buf.append(log_prob)
        rew_buf.append(float(reward))
        val_buf.append(value)
        done_buf.append(1.0 if done else 0.0)

        current_ep_reward += float(reward)
        state = next_state

        if done:
            episode_rewards.append(current_ep_reward)
            current_ep_reward = 0.0
            ep_counter += 1
            state, _ = env.reset(seed=seed_offset + ep_counter)

    obs = torch.stack(obs_buf)
    actions = torch.stack(act_buf)
    log_probs = torch.stack(logp_buf)
    rewards = torch.tensor(rew_buf, dtype=torch.float32)
    values = torch.stack(val_buf)
    dones = torch.tensor(done_buf, dtype=torch.float32)

    # Bootstrap value at the end of the rollout.
    with torch.no_grad():
        _, next_value = net(torch.as_tensor(state, dtype=torch.float32))

    return (
        obs,
        actions,
        log_probs,
        rewards,
        values,
        dones,
        episode_rewards,
        state,
        next_value,
        ep_counter,
    )


# ── training loop ────────────────────────────────────────────────────────────

def train(
    total_steps: int = 50_000,
    rollout_steps: int = 1024,
    epochs: int = 10,
    minibatch_size: int = 64,
    lr: float = 1e-3,
    clip_eps: float = CLIP_EPS,
    value_coef: float = VALUE_COEF,
    entropy_coef: float = ENTROPY_COEF,
    gamma: float = GAMMA,
    lam: float = LAMBDA,
    seed: int = 0,
) -> list:
    """Train PPO on CartPole-v1.

    Returns: list of (iteration_idx, mean_episode_reward) — one entry per
    rollout iteration, recording the mean reward over the episodes that
    finished inside that rollout. Empty rollouts (no full episode) are skipped.
    """
    import gymnasium as gym

    torch.manual_seed(seed)
    np.random.seed(seed)
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = ActorCriticNet(state_dim, n_actions)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    state, _ = env.reset(seed=seed)
    seed_counter = 0
    n_iters = total_steps // rollout_steps
    iter_rewards = []  # per-iteration mean episode reward

    for it in range(n_iters):
        (
            obs,
            actions,
            log_probs_old,
            rewards,
            values_old,
            dones,
            ep_rewards,
            state,
            next_value,
            n_eps,
        ) = collect_rollouts(env, net, rollout_steps, state, seed_offset=seed + seed_counter)
        seed_counter += n_eps

        # GAE on detached, no-grad values.
        advantages, returns = compute_gae(
            rewards, values_old, dones, next_value, gamma=gamma, lam=lam
        )
        # Normalize advantages — standard PPO trick, big stability win.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Multiple epochs of minibatch updates.
        N = obs.shape[0]
        idx = np.arange(N)
        for _ in range(epochs):
            np.random.shuffle(idx)
            for start in range(0, N, minibatch_size):
                mb = idx[start : start + minibatch_size]
                mb_t = torch.as_tensor(mb, dtype=torch.long)

                logits, value_pred = net(obs[mb_t])
                dist = torch.distributions.Categorical(logits=logits)
                log_probs_new = dist.log_prob(actions[mb_t])
                entropy = dist.entropy()

                loss = ppo_clip_loss(
                    log_probs_new=log_probs_new,
                    log_probs_old=log_probs_old[mb_t],
                    advantages=advantages[mb_t],
                    value_pred=value_pred,
                    value_target=returns[mb_t],
                    entropy=entropy,
                    clip_eps=clip_eps,
                    value_coef=value_coef,
                    entropy_coef=entropy_coef,
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                optimizer.step()

        if ep_rewards:
            iter_rewards.append((it, float(np.mean(ep_rewards))))

    env.close()
    return iter_rewards


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    t0 = time.time()
    rewards = train(seed=0)
    dt = time.time() - t0
    last10 = [r for _, r in rewards[-10:]]
    print(
        f"iterations: {len(rewards)}  "
        f"mean reward (last 10): {sum(last10) / max(len(last10), 1):.1f}  "
        f"(random ≈ 21, max 500)  "
        f"time: {dt:.1f}s"
    )
