# Hints: exercise 05

Read one at a time. Try after each before reading the next.

---

**Hint 1: `ActorCriticNet`.** It's an MLP with a shared trunk and two heads:

```python
self.trunk = nn.Sequential(
    nn.Linear(state_dim, hidden), nn.Tanh(),
    nn.Linear(hidden, hidden), nn.Tanh(),
)
self.policy_head = nn.Linear(hidden, n_actions)   # logits: NOT softmax
self.value_head  = nn.Linear(hidden, 1)           # scalar V(s)
```

In `forward`: run the state through the trunk, then split into `logits = self.policy_head(h)` and `value = self.value_head(h).squeeze(-1)`. The `.squeeze(-1)` is what makes `value` a scalar `()` for a single state and `(batch,)` for a batched input; the value head outputs `(..., 1)` and you want to drop that trailing 1.

Why tanh and not ReLU? Both work on CartPole, but the PPO paper uses tanh and the OpenAI implementations use tanh. Stick with the convention until you have a reason to deviate.

---

**Hint 2: `compute_gae`, the recursion.** GAE is a weighted sum of one-step TD residuals:

```
δ_t = r_t + γ · V(s_{t+1}) · (1 − done_t) − V(s_t)
A_t = δ_t + (γλ) · (1 − done_t) · A_{t+1}
```

You compute it backwards because `A_t` depends on `A_{t+1}`. The skeleton:

```python
T = rewards.shape[0]
advantages = torch.zeros(T)
gae = 0.0
for t in reversed(range(T)):
    nonterminal = 1.0 - dones[t]
    v_next = next_value if t == T - 1 else values[t + 1]
    delta = rewards[t] + gamma * v_next * nonterminal - values[t]
    gae = delta + gamma * lam * nonterminal * gae
    advantages[t] = gae
returns = advantages + values
return advantages, returns
```

Two things to get right:

- **`v_next` at the last step** is `next_value` (passed in, the bootstrap value V(s_T) after the rollout). For all earlier steps, it's `values[t + 1]`.
- **`nonterminal = 1 − dones[t]`** masks both the TD bootstrap (`v_next * nonterminal`) and the recursion (`gae * nonterminal`). If you forget either, the advantage "leaks" reward across episode boundaries that didn't actually exist.

---

**Hint 3: `ppo_clip_loss`.** Three pieces added together:

```python
# 1. Clipped policy surrogate.
ratio = (log_probs_new - log_probs_old).exp()              # π_new/π_old
unclipped = ratio * advantages
clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
policy_loss = -torch.min(unclipped, clipped).mean()

# 2. Value MSE.
value_loss = F.mse_loss(value_pred, value_target)

# 3. Entropy bonus (subtract from loss: high entropy is good).
entropy_term = entropy.mean()

return policy_loss + value_coef * value_loss - entropy_coef * entropy_term
```

The `min` is the pessimistic step: when both terms point in the same direction, you get the smaller (more conservative) one. Sign check: optimizer *minimizes*; we want to *maximize* expected reward, so policy_loss has a leading `−`. Entropy_coef is also a `−` for the same reason (we want to maximize entropy).

---

**Hint 4: `collect_rollouts`, the bookkeeping.** This is the longest piece but it's all mechanical:

```python
obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
episode_rewards = []
current_ep_reward = 0.0
ep_counter = 0

for _ in range(n_steps):
    state_t = torch.as_tensor(state, dtype=torch.float32)
    with torch.no_grad():                       # <- critical
        logits, value = net(state_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

    next_state, reward, terminated, truncated, _ = env.step(int(action.item()))
    done = terminated or truncated

    obs_buf.append(state_t); act_buf.append(action)
    logp_buf.append(log_prob); rew_buf.append(float(reward))
    val_buf.append(value); done_buf.append(1.0 if done else 0.0)

    current_ep_reward += float(reward)
    state = next_state
    if done:
        episode_rewards.append(current_ep_reward)
        current_ep_reward = 0.0
        ep_counter += 1
        state, _ = env.reset(seed=seed_offset + ep_counter)

obs       = torch.stack(obs_buf)
actions   = torch.stack(act_buf)
log_probs = torch.stack(logp_buf)         # already detached: no_grad above
rewards   = torch.tensor(rew_buf, dtype=torch.float32)
values    = torch.stack(val_buf)          # also detached
dones     = torch.tensor(done_buf, dtype=torch.float32)

with torch.no_grad():
    _, next_value = net(torch.as_tensor(state, dtype=torch.float32))

return (obs, actions, log_probs, rewards, values, dones,
        episode_rewards, state, next_value, ep_counter)
```

The `with torch.no_grad()` block is what makes `log_probs` and `values` the "old" rollout; they enter the loss as constants. The loss re-computes `log_probs_new` and `value_pred` from the (possibly updated) net on each minibatch.

The reason for re-seeding on done is reproducibility: the integration test sets `seed=0` and expects the same episode sequence every run.

---

**Hint 5: `train`, putting the loop together.** Inside the iteration loop:

```python
(obs, actions, log_probs_old, rewards, values_old, dones,
 ep_rewards, state, next_value, n_eps) = collect_rollouts(
    env, net, rollout_steps, state, seed_offset=seed + seed_counter,
)
seed_counter += n_eps

advantages, returns = compute_gae(
    rewards, values_old, dones, next_value, gamma=gamma, lam=lam,
)
# Normalize: this is critical. Without it, advantages can have huge scale
# differences across rollouts and learning destabilizes.
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

N = obs.shape[0]
idx = np.arange(N)
for _ in range(epochs):
    np.random.shuffle(idx)
    for start in range(0, N, minibatch_size):
        mb = idx[start : start + minibatch_size]
        mb_t = torch.as_tensor(mb, dtype=torch.long)

        logits, value_pred = net(obs[mb_t])               # CURRENT net
        dist = torch.distributions.Categorical(logits=logits)
        log_probs_new = dist.log_prob(actions[mb_t])      # has gradient
        entropy = dist.entropy()

        loss = ppo_clip_loss(
            log_probs_new, log_probs_old[mb_t], advantages[mb_t],
            value_pred, returns[mb_t], entropy,
            clip_eps=clip_eps, value_coef=value_coef, entropy_coef=entropy_coef,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        optimizer.step()

if ep_rewards:
    iter_rewards.append((it, float(np.mean(ep_rewards))))
```

The thing that makes PPO "PPO" rather than vanilla policy gradient is that inner double loop: `epochs` passes over the rollout, each with `minibatch_size` chunks. You're getting K updates out of every batch of data instead of one, and the clip is what makes that safe.

---

**Hint 6: if it runs but doesn't learn.** Common failure modes:

1. **No advantage normalization.** Without `(adv - adv.mean()) / (adv.std() + 1e-8)`, advantages can be on a wildly different scale every rollout and you'll see the loss bounce around without progress.
2. **Reusing `log_probs_old` as `log_probs_new`.** If `ratio == 1` always, the clip is never active, you're not doing PPO. The whole point of recomputing `log_probs_new` from the current net is that after the first epoch's updates, the policy has changed and `ratio ≠ 1`.
3. **Wrong sign on entropy or policy.** Policy is `−min(...)`, entropy term is `−entropy_coef · H`. Both negatives. If you forget either, you're either minimizing reward or minimizing entropy.
4. **Missing `(1 − done_t)` in GAE.** Episode boundaries leak value/advantage between independent episodes; the policy learns garbage signals at the boundary.
5. **`requires_grad` accidentally True on rollout `log_probs` or `values`.** They should come from a `with torch.no_grad():` block. If they're tracked, calling `loss.backward()` will try to backprop through old graph nodes and either error or compute the wrong thing.

If the test fails with `mean reward (last 10 iters) = 21.x` or similar, your policy hasn't improved at all; it's bug #1, #2, or #3 above. If it's around 40-100, you're learning but slowly. Try doubling `epochs` or `total_steps` first, then double-check #4 and #5.

---

If you're past hint 6 and still stuck, read [`solution/ppo.py`](./solution/ppo.py). Then come back and re-derive it without looking; you don't own it until you can.
