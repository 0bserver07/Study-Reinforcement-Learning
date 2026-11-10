# Hints: exercise 20

Read one at a time. Try after each before reading the next.

---

**Hint 1: `CombinationLock.step`.** All of it:

```python
if action == 0:  # the wrong move
    return self.state, 0.0, True, {}
self.state += 1
reached_goal = self.state == self.n_states - 1
reward = 1.0 if reached_goal else 0.0
return self.state, reward, reached_goal, {}
```

Two things people get wrong here. (1) Making left step back a square (`max(self.state - 1, 0)`), as on an ordinary chain. On the lock, left ends the episode on every square, square 0 included. With a step back instead, nothing ends an episode except the goal, so even plain ε-greedy wanders until it gets there. (2) Computing `reached_goal` before moving. Then arriving on square 15 doesn't register, and the +1 only comes with a 16th right, onto a square that doesn't exist. Move first.

---

**Hint 2: `QLearningAgent.act` and `.update`.** Short ones:

```python
def act(self, state):
    if self.rng.random() < self.epsilon:
        return int(self.rng.integers(self.n_actions))
    q = self.q[state]
    best = np.flatnonzero(q == q.max())  # every action tied for the max
    return int(self.rng.choice(best))

def update(self, s, a, r, s_next, done):
    target = r if done else r + self.gamma * np.max(self.q[s_next])
    td = target - self.q[s, a]
    self.q[s, a] += self.alpha * td
    return float(td)
```

The tie-break matters on this lock. Q starts at 0, so at first every greedy choice is a tie, and `int(np.argmax(self.q[state]))` returns the first tied action: 0, the wrong move. Plain ε-greedy with that tie rule goes left 95% of the time. The bonus agent still gets there, because ε sometimes picks right on the frontier square, but slowly: over 100 seeds its first +1 came at a median of episode 407, against 22 with random tie-breaking.

The `done` branch matters too. Bootstrapping off `max(Q[s_next])` on a move that ends the episode is the classic bug. After the wrong move `s_next` is the square the agent failed on, so the bug pushes Q(s, left) toward γ · Q(s, right): the wrong move ends up looking nearly as good as the right one. The integration test checks that Q(s, left) stays exactly 0.

This lock has no time limit, so `done` always means the episode really ended. If you add a step limit to an environment, a timeout is a different case: the episode stops, but the state still has a future, so the update should bootstrap. Keep the two apart. (gymnasium's `step` returns them separately, as `terminated` and `truncated`.)

---

**Hint 3: `RNDIntrinsicReward.intrinsic_reward`.** The body is short:

```python
err = self._raw_error(state).item()
self._update_running_stats(err)
std = float(np.sqrt(self._reward_var)) + 1e-8
return err / std
```

The `_raw_error` helper already wraps the forward passes in `torch.no_grad()` and returns the scalar mean squared error. You don't backprop through this: it's just a number you feed into the Q-update. The reason to call `_update_running_stats` here (instead of in `update`) is that you want the running stats to reflect *every* error you ever query, not only the ones that get a gradient step.

Why normalize at all? At the start of training the raw errors are large everywhere (the predictor knows nothing). After a few states are learned, the errors on those states drop by orders of magnitude, while on states the predictor hasn't trained on they fall much less. Without normalization, the weight of the bonus against the extrinsic reward moves with that drift, and you don't control it. On this lock the weight happens not to matter, since any positive bonus opens it (see the README), but it matters whenever the bonus has to be traded off against the extrinsic reward.

---

**Hint 4: `RNDIntrinsicReward.update`.** Standard one-step MSE update on the predictor:

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

Why `.detach()` on `target_out` even though the target's parameters have `requires_grad=False`? It's not strictly necessary, but the detach makes the intent explicit and survives someone later forgetting to freeze the target. (And the freeze-via-`requires_grad=False` only stops gradients from flowing into the target's *parameters*; the graph still attaches up through the target's forward pass otherwise, which builds memory you don't need.)

---

**Hint 5: `train_with_intrinsic`, the inner loop.** Inside `while not done`:

```python
a = agent.act(s)
s_next, r_ext, done, _ = env.step(a)
r_int = 0.0
if not done:
    r_int = rnd.intrinsic_reward(s_next)
    rnd.update(s_next)
agent.update(s, a, r_ext + intrinsic_coef * r_int, s_next, done)
s = s_next
total_ext += r_ext
```

Three design choices worth understanding. (1) Why the bonus at `s_next`, not `s`? The credit goes to the move, and what makes a move worth taking is that it lands somewhere new: the novelty belongs to the landing square. (2) Why no bonus when `done`? A move that ends the episode leads nowhere, so there's nothing to explore from. On this lock it's also what keeps the wrong move worth exactly 0. If you score the square the agent failed on as the wrong move's bonus, Q(s, left) turns positive, and on the frontier square, where right hasn't been tried and is still worth 0, the greedy move becomes the wrong one. Over 200 seeds, paying the bonus on moves that end the episode got the agent to the goal within 200 episodes in 16.5% of runs, against 100% without. (3) Why `total_ext += r_ext` and not the combined reward? The test measures whether the agent actually reaches the goal. If you accumulate the combined reward, you're reporting the bonus as if it were real reward, and the test threshold becomes meaningless. Train on combined, report on extrinsic.

If the integration test says Q(s, left) isn't 0, either the bonus is being paid when `done` is True, or `update` bootstraps on `done=True` (hint 2). If it says the predictor never changed, you forgot `rnd.update(s_next)`. The agent opens the lock without it, because any positive bonus does here, but the bonus no longer shrinks with visits, so RND is reduced to a random but fixed bonus per square.

---

If you're past hint 5 and the integration test still fails, read [`solution/exploration.py`](./solution/exploration.py). Then close it and re-derive the loop from scratch: there's only one thing happening per step, and the order matters.
