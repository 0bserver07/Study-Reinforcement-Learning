"""PPO on CartPole-v1 — exercise starter.

Fill in the TODOs, then from the repo root:

    pip install -r exercises/requirements.txt
    pytest exercises/05-ppo/

See README.md for the task and HINTS.md if you're stuck. The reference
implementation is in solution/ppo.py — look after you've tried.

The five pieces to fill in:
  1. ActorCriticNet — small MLP with a shared trunk, a policy head (logits)
     and a scalar value head.
  2. compute_gae    — Generalized Advantage Estimation. The TD residuals
                      get blended into a multi-step advantage via λ.
  3. ppo_clip_loss  — the clipped surrogate + value MSE + entropy bonus.
  4. collect_rollouts — run the env for N steps with the current policy,
                      return tensors of obs/actions/log_probs/rewards/values/dones.
  5. train         — full loop: rollout → GAE → many epochs of minibatch
                      updates → repeat. Returns per-iteration mean episode reward.
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

    Single-state input  shape (state_dim,)        → logits (n_actions,),     value ()
    Batched input       shape (batch, state_dim)  → logits (batch, n_actions), value (batch,)

    Use tanh activations on the trunk (PPO papers use tanh; ReLU works too,
    but tanh is the common default).
    """

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        # TODO:
        #   self.trunk = nn.Sequential(
        #       nn.Linear(state_dim, hidden), nn.Tanh(),
        #       nn.Linear(hidden, hidden), nn.Tanh(),
        #   )
        #   self.policy_head = nn.Linear(hidden, n_actions)   # logits
        #   self.value_head  = nn.Linear(hidden, 1)           # scalar V(s)
        raise NotImplementedError("ActorCriticNet.__init__")

    def forward(self, state: torch.Tensor):
        # TODO: run state through self.trunk, then split into logits and value.
        # The value head outputs shape (..., 1); squeeze the last dim so it's
        # shape () for a single state and (batch,) for a batched input.
        raise NotImplementedError("ActorCriticNet.forward")


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

    The TD residual at step t is:
        δ_t = r_t + γ · V(s_{t+1}) · (1 − done_t) − V(s_t)

    The GAE advantage blends multi-step residuals:
        A_t = δ_t + (γλ) · (1 − done_t) · A_{t+1}

    λ=0 → A_t = δ_t (TD(0), low variance, biased).
    λ=1 → A_t ≈ G_t − V(s_t)  (Monte Carlo, unbiased, high variance).
    λ=0.95 is the standard default.

    Args:
        rewards:    (T,) float tensor.
        values:     (T,) float tensor of V(s_t) collected during the rollout
                    (detached; treat as constants).
        dones:      (T,) float tensor, 1.0 if the episode ended at step t else 0.0.
        next_value: scalar tensor — V(s_T), the bootstrap value after the rollout.
        gamma, lam: GAE hyperparameters.

    Returns:
        advantages: (T,) tensor.
        returns:    (T,) tensor = advantages + values. Use this as the critic target.
    """
    # TODO: walk t backwards from T-1 to 0, maintaining a running gae:
    #   T = rewards.shape[0]
    #   advantages = torch.zeros(T)
    #   gae = 0.0
    #   for t in reversed(range(T)):
    #       nonterminal = 1.0 - dones[t]
    #       v_next = next_value if t == T - 1 else values[t + 1]
    #       delta = rewards[t] + gamma * v_next * nonterminal - values[t]
    #       gae = delta + gamma * lam * nonterminal * gae
    #       advantages[t] = gae
    #   returns = advantages + values
    #   return advantages, returns
    raise NotImplementedError("compute_gae")


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

    Probability ratio:
        ratio = exp(log_probs_new − log_probs_old)         # π_new(a|s) / π_old(a|s)

    Clipped surrogate (PPO-Clip):
        L_policy = −mean( min( ratio · A,
                               clip(ratio, 1−ε, 1+ε) · A ) )

    Value loss:
        L_value = MSE(value_pred, value_target)

    Entropy bonus:
        H = mean(entropy)
        L_total = L_policy + value_coef · L_value − entropy_coef · H
                  (the entropy term is SUBTRACTED — we want high entropy,
                   so high entropy → low loss → good)

    Args:
        log_probs_new: (B,) log-probs from the CURRENT policy (has gradient).
        log_probs_old: (B,) log-probs from the rollout policy (detached, constant).
        advantages:    (B,) usually normalized to zero-mean unit-std by the caller.
        value_pred:    (B,) V(s) from the current network (has gradient).
        value_target:  (B,) target returns from GAE (detached, constant).
        entropy:       (B,) per-sample entropy of the current policy.

    Returns:
        Scalar tensor with gradient.
    """
    # TODO:
    #   ratio = (log_probs_new - log_probs_old).exp()
    #   unclipped = ratio * advantages
    #   clipped = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
    #   policy_loss = -torch.min(unclipped, clipped).mean()
    #   value_loss = F.mse_loss(value_pred, value_target)
    #   entropy_term = entropy.mean()
    #   return policy_loss + value_coef * value_loss - entropy_coef * entropy_term
    raise NotImplementedError("ppo_clip_loss")


# ── rollout collector ────────────────────────────────────────────────────────

def collect_rollouts(env, net: ActorCriticNet, n_steps: int, state, seed_offset: int = 0):
    """Run `n_steps` of env interaction with `net` and return tensors.

    Stateful: takes the current `state` and returns the next one so the next
    call continues from there.

    For each step t:
      - state_t = current state as a float32 tensor
      - logits, value = net(state_t)   under torch.no_grad()  (we don't want
        the rollout values in the policy graph; the loss will recompute them)
      - action = Categorical(logits=logits).sample()
      - log_prob = dist.log_prob(action)
      - step the env; if done, env.reset(seed=seed_offset + ep_counter)

    Returns:
        obs:        (n_steps, state_dim)   float32
        actions:    (n_steps,)              LongTensor
        log_probs:  (n_steps,)              float, DETACHED (these are "old")
        rewards:    (n_steps,)              float32
        values:     (n_steps,)              float, DETACHED
        dones:      (n_steps,)              float32, 1.0 if episode ended that step
        episode_rewards: list[float] — total reward for each episode that
                         finished inside this rollout
        next_state: env state to feed into the next collect_rollouts call
        next_value: scalar tensor V(next_state), DETACHED, for GAE bootstrap
        ep_counter: number of episodes completed during this rollout (for seeding)
    """
    # TODO: collect the buffers above. The structure looks like:
    #
    #   obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
    #   episode_rewards = []
    #   current_ep_reward = 0.0
    #   ep_counter = 0
    #
    #   for _ in range(n_steps):
    #       state_t = torch.as_tensor(state, dtype=torch.float32)
    #       with torch.no_grad():
    #           logits, value = net(state_t)
    #           dist = torch.distributions.Categorical(logits=logits)
    #           action = dist.sample()
    #           log_prob = dist.log_prob(action)
    #
    #       next_state, reward, terminated, truncated, _ = env.step(int(action.item()))
    #       done = terminated or truncated
    #
    #       obs_buf.append(state_t); act_buf.append(action)
    #       logp_buf.append(log_prob); rew_buf.append(float(reward))
    #       val_buf.append(value); done_buf.append(1.0 if done else 0.0)
    #
    #       current_ep_reward += float(reward)
    #       state = next_state
    #       if done:
    #           episode_rewards.append(current_ep_reward)
    #           current_ep_reward = 0.0
    #           ep_counter += 1
    #           state, _ = env.reset(seed=seed_offset + ep_counter)
    #
    #   obs = torch.stack(obs_buf); actions = torch.stack(act_buf)
    #   log_probs = torch.stack(logp_buf)
    #   rewards = torch.tensor(rew_buf, dtype=torch.float32)
    #   values = torch.stack(val_buf)
    #   dones = torch.tensor(done_buf, dtype=torch.float32)
    #
    #   with torch.no_grad():
    #       _, next_value = net(torch.as_tensor(state, dtype=torch.float32))
    #
    #   return obs, actions, log_probs, rewards, values, dones, \
    #          episode_rewards, state, next_value, ep_counter
    raise NotImplementedError("collect_rollouts")


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

    Outer loop (total_steps // rollout_steps iterations):
      1. Roll out rollout_steps env steps with the current policy.
      2. Compute GAE on the collected (rewards, values, dones, next_value).
      3. Normalize the advantages (zero mean, unit std). Standard PPO trick.
      4. For `epochs` passes, shuffle the batch and do minibatch SGD on
         ppo_clip_loss. Re-evaluate (logits, value) under the CURRENT net
         each minibatch — log_probs_new comes from those fresh logits, NOT
         from the rolled-out log_probs (which are the "old" ones in the ratio).
      5. Clip gradient norm at 0.5, optimizer.step().

    Returns:
        List of (iteration_index, mean_episode_reward) — one entry per
        rollout iteration where at least one full episode finished.
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
    iter_rewards = []

    for it in range(n_iters):
        # TODO: collect rollouts, compute GAE, run `epochs` of minibatch updates.
        #
        # 1) (obs, actions, log_probs_old, rewards, values_old, dones,
        #     ep_rewards, state, next_value, n_eps) = collect_rollouts(...)
        #    seed_counter += n_eps
        #
        # 2) advantages, returns = compute_gae(rewards, values_old, dones,
        #                                      next_value, gamma=gamma, lam=lam)
        #    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        #
        # 3) N = obs.shape[0]
        #    idx = np.arange(N)
        #    for _ in range(epochs):
        #        np.random.shuffle(idx)
        #        for start in range(0, N, minibatch_size):
        #            mb = idx[start : start + minibatch_size]
        #            mb_t = torch.as_tensor(mb, dtype=torch.long)
        #            logits, value_pred = net(obs[mb_t])
        #            dist = torch.distributions.Categorical(logits=logits)
        #            log_probs_new = dist.log_prob(actions[mb_t])
        #            entropy = dist.entropy()
        #
        #            loss = ppo_clip_loss(
        #                log_probs_new, log_probs_old[mb_t], advantages[mb_t],
        #                value_pred, returns[mb_t], entropy,
        #                clip_eps=clip_eps, value_coef=value_coef,
        #                entropy_coef=entropy_coef,
        #            )
        #            optimizer.zero_grad()
        #            loss.backward()
        #            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        #            optimizer.step()
        #
        # 4) if ep_rewards: iter_rewards.append((it, float(np.mean(ep_rewards))))
        raise NotImplementedError("train: the PPO update")

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
