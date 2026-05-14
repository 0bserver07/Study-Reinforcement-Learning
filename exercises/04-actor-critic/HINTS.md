# Hints — exercise 04

Read one at a time. Try after each before reading the next.

---

**Hint 1 — `PolicyNet` and `ValueNet`.** Both are plain MLPs. For `PolicyNet`, it's the same `nn.Sequential(Linear, ReLU, Linear, ReLU, Linear)` as exercise 02 — `state_dim → hidden → hidden → n_actions`, no softmax. For `ValueNet`, same structure but the last layer outputs 1: `state_dim → hidden → hidden → 1`. In `ValueNet.forward`, call `.squeeze(-1)` on the output so the shape is `[]` for a single state and `[batch]` for a batch — not `[1]` or `[batch, 1]`. The test checks the shape exactly.

---

**Hint 2 — `select_action` and `compute_returns`.** These are identical to exercise 02. Copy them if you like, but re-read them — `compute_returns` in particular. Walking backwards (start `R = 0`, then `R = r + gamma * R` for each `r` in `reversed(rewards)`) gives you `G_t` because each step incorporates all future rewards. When you do it forward, you'd need two nested loops. Reverse is O(n) and clean.

---

**Hint 3 — `actor_critic_loss`, the detach.** This is the only structural difference from REINFORCE. You need:

```python
advantages = (returns - values).detach()
```

The `.detach()` cuts the computation graph so that when `actor_loss.backward()` runs, it produces gradients in `PolicyNet` (via `log_probs`) but not in `ValueNet` (via `values`). Without it, the actor loss would try to pull `ValueNet`'s parameters in the direction that makes advantages look bigger — that's not training the critic to predict returns, that's just breaking it. There's a test that checks this: `test_advantages_do_not_flow_into_value_net`.

---

**Hint 4 — `actor_critic_loss`, assembling the losses.** After the detach:

```python
lp = torch.stack(list(log_probs))
actor_loss  = -(lp * advantages).sum()
critic_loss = F.mse_loss(values, returns)
return actor_loss, critic_loss
```

The minus sign on `actor_loss` is the same reason as in REINFORCE: the optimizer minimizes, but the policy gradient is an ascent direction. `F.mse_loss(values, returns)` trains the critic to predict G_t — returns here are the Monte Carlo targets and the critic is learning to match them. Note that `critic_loss` uses `values` *without* detaching — it needs gradient to flow into `ValueNet`.

---

**Hint 5 — `train`, collecting values during rollout.** The loop structure is given; you need to add one line inside `while not done`:

```python
state_t = torch.as_tensor(state, dtype=torch.float32)   # already there
action, log_prob = select_action(policy, state)           # already there
value = critic(state_t)                                   # <-- add this
# ... env.step, append log_prob and reward ...
values.append(value)                                      # <-- and this
```

Call `critic` *before* `env.step` — you want V(s_t) at the state before the action, not the next state. Keep the tensors in the list (don't `.item()` them); you'll `torch.stack(values)` after the episode to get one tensor the critic loss can differentiate through.

---

**Hint 6 — `train`, the update step.** After the episode loop:

```python
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
```

`value_coef=0.5` scales the critic's contribution — without it the critic loss (which can be in the hundreds for early episodes, since V(s) starts near 0 and returns are ~10–500) would dominate and destabilize the actor update. The gradient clip (norm at 1.0) caps any single episode's impact on the weights.

---

If you're past hint 6 and still stuck, read [`solution/actor_critic.py`](./solution/actor_critic.py). Then come back and re-derive it without looking — you don't own it until you can.
