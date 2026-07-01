# Hints: exercise 02

Read one at a time. Try after each before reading the next.

---

**Hint 1: `compute_returns`.** Build it back to front. Start `R = 0`, walk `rewards` in reverse, and at each step `R = r + gamma * R`; that `R` is the return-to-go for that timestep. Collect those, reverse the list, make it a `torch.tensor(..., dtype=torch.float32)`. Check it against the worked example in the test (`gamma=0.5`, `[1,1,1]` → `[1.75, 1.5, 1.0]`).

---

**Hint 2: `PolicyNet`.** It's just an MLP. `nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))`. `forward` returns that straight through. No softmax, no activation on the last layer: these are logits.

---

**Hint 3: `select_action`.** `torch.as_tensor(state, dtype=torch.float32)` → feed to the policy → `dist = torch.distributions.Categorical(logits=logits)` → `action = dist.sample()`. Return `int(action.item())` and `dist.log_prob(action)`. The log-prob is a tensor that came from the network's parameters, so it carries gradient: return it as-is; don't `.item()` it.

---

**Hint 4: `reinforce_loss`.** Two steps. First, normalize the returns to use as the advantage: `adv = (returns - returns.mean()) / (returns.std() + 1e-8)`. Then `lp = torch.stack(list(log_probs))` and `return -(lp * adv).sum()`. The minus sign is because the optimizer minimizes but the policy gradient is an ascent direction. Watch out: if you forget `torch.stack` and try to multiply a Python list by a tensor, it won't broadcast the way you want.

---

**Hint 5: `train`, the update step.** Inside the episode loop, after you've collected `log_probs` and `rewards`:

```python
returns = compute_returns(rewards)
loss = reinforce_loss(log_probs, returns)
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)   # cap the gradient
optimizer.step()
```

That's it. Don't accumulate across episodes: update once per episode, then the next episode starts with fresh `log_probs` / `rewards` lists (the loop already does that). The gradient clip isn't optional: without it, an unusually long or short episode now and then produces a huge update and the policy collapses to always picking one action.

---

**Hint 6: if it runs but doesn't learn.** Print `np.mean(episode_returns[-50:])` every 50 episodes. If it's flat near 20–30: check the *sign* of the loss (ascent vs. descent: it should be a `-`), and check that `select_action` returns the log-prob *with* gradient (if you accidentally `.detach()`ed it or `.item()`ed it, `loss.backward()` does nothing to the policy). If it climbs and then crashes to ~10 and stays there: that's the policy collapsing. Make sure the gradient clip is in (Hint 5), and even then expect some wobble; it's the kind of thing a value-function baseline (Lecture 04) smooths out.

---

If you're past hint 6 and still stuck, read [`solution/reinforce.py`](./solution/reinforce.py). Then come back and re-derive it without looking. You don't own it until you can.
