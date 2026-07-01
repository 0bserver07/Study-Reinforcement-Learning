# Hints — exercise 20

Read one at a time. Try after each before reading the next.

---

**Hint 1 — `ChainEnv.step`.** All of it:

```python
if action == 1:
    self.state = min(self.state + 1, self.n_states - 1)
else:
    self.state = max(self.state - 1, 0)
self.steps += 1
reached_goal = self.state == self.n_states - 1
timed_out = self.steps >= self.max_steps
reward = 1.0 if reached_goal else 0.0
done = reached_goal or timed_out
return self.state, reward, done, {}
```

Two things people get wrong here. (1) Forgetting the `min`/`max` clamps — without them, action 0 at state 0 goes to -1, then `self.q[-1]` becomes "the last row" because numpy negative indices are valid, and your bug looks like a hard-to-reproduce learning failure rather than an out-of-bounds error. (2) Computing `reached_goal` before updating `self.state`, so the goal-arrival is registered one step late. Update `self.state` first.

---

**Hint 2 — `QLearningAgent.act` and `.update`.** Two-liners:

```python
def act(self, state):
    if self.rng.random() < self.epsilon:
        return int(self.rng.integers(self.n_actions))
    return int(np.argmax(self.q[state]))

def update(self, s, a, r, s_next, done):
    target = r if done else r + self.gamma * np.max(self.q[s_next])
    td = target - self.q[s, a]
    self.q[s, a] += self.alpha * td
    return float(td)
```

The `done` branch matters. Bootstrapping off `max(Q[s_next])` on a terminal transition is the classic bug — `Q[s_next]` for the terminal state is whatever you initialized it to, and you'll either learn nonsense or accidentally learn fine (if the init happens to be 0 and you have no other terminal transitions, you're getting lucky). The `if done` branch makes the target correct even on the goal step.

---

**Hint 3 — `RNDIntrinsicReward.intrinsic_reward`.** The body is short:

```python
err = self._raw_error(state).item()
self._update_running_stats(err)
std = float(np.sqrt(self._reward_var)) + 1e-8
return err / std
```

The `_raw_error` helper already wraps the forward passes in `torch.no_grad()` and returns the scalar mean squared error. You don't backprop through this — it's just a number you feed into the Q-update. The reason to call `_update_running_stats` here (instead of in `update`) is that you want the running stats to reflect *every* error you ever query, not only the ones that get a gradient step.

Why not return the raw error and skip the normalization? Try it and see — at the start of training the raw errors are huge (the predictor knows nothing) and after a few states are learned they drop to nearly zero on those states but stay big elsewhere. The effective `intrinsic_coef` swings by orders of magnitude as a result. Normalizing keeps the relative novelty signal stable.

---

**Hint 4 — `RNDIntrinsicReward.update`.** Standard one-step MSE update on the predictor:

```python
x = self._onehot(state).unsqueeze(0)
target_out = self.target(x).detach()
pred_out = self.predictor(x)
loss = F.mse_loss(pred_out, target_out)
self.optimizer.zero_grad()
loss.backward()
self.optimizer.step()
return float(loss.item())
```

Why `.detach()` on `target_out` even though the target's parameters have `requires_grad=False`? It's not strictly necessary — but the detach makes the intent explicit and survives someone later forgetting to freeze the target. (And the freeze-via-`requires_grad=False` only stops gradients from flowing into the target's *parameters*; the graph still attaches up through the target's forward pass otherwise, which builds memory you don't need.)

---

**Hint 5 — `train_with_intrinsic`, the inner loop.** Inside `while not done`:

```python
a = agent.act(s)
s_next, r_ext, done, _ = env.step(a)
r_int = rnd.intrinsic_reward(s_next)
r_combined = r_ext + intrinsic_coef * r_int
agent.update(s, a, r_combined, s_next, done)
rnd.update(s_next)
s = s_next
total_ext += r_ext
```

Two design choices worth understanding. (1) Why intrinsic reward at `s_next`, not `s`? Because the "reward you got for taking action `a` in state `s`" includes "you landed in state `s_next`, which is novel" — the credit goes to the transition, and the novelty is a property of the landing state. (2) Why `total_ext += r_ext` and not `r_combined`? The test measures whether the agent actually reaches the goal. If you accumulate `r_combined`, you're reporting the bonus inflation as if it were real reward, and the test threshold becomes meaningless. Train on combined, report on extrinsic.

If your test fails with a returned value near zero, the most likely cause is forgetting to call `rnd.update(s_next)` — then the predictor never learns anything, the bonus stays maxed at every state forever, and Q-learning chases the bonus instead of the goal because the bonus rewards revisiting just as much as visiting new states.

---

If you're past hint 5 and the integration test still fails, read [`solution/exploration.py`](./solution/exploration.py). Then close it and re-derive the loop from scratch — there's only one thing happening per step, and the order matters.
