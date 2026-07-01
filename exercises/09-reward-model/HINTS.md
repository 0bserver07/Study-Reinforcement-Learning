# Hints: exercise 09

Read one at a time. Try after each before reading the next.

---

**Hint 1: `RewardModel.__init__`.** A 3-layer MLP is plenty. The reference is literally:

```python
self.net = nn.Sequential(
    nn.Linear(feature_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, 1),
)
```

Hidden dim 32 works fine. The true reward is linear in the features, so a single linear layer would also work, but the MLP is closer to what a real reward head looks like.

---

**Hint 2: `RewardModel.forward`.** Two shape contracts, one branch:

```python
def forward(self, features):
    if features.dim() == 1:
        return self.net(features).squeeze(-1)   # scalar, shape ()
    return self.net(features)                   # (B, 1)
```

`.squeeze(-1)` on a 1-D input of shape `(FEATURE_DIM,)` will fail with a `Linear(... → 1)` because the output is shape `(1,)` and squeezing the last dim gives `()`, a scalar tensor. That's what `test_reward_model_forward` expects for the single-input case.

---

**Hint 3: `bradley_terry_loss`.** Two lines after flattening:

```python
chosen = chosen_rewards.flatten()
rejected = rejected_rewards.flatten()
return -F.logsigmoid(chosen - rejected).mean()
```

Why `F.logsigmoid` and not `torch.log(torch.sigmoid(x))`: when `x` is very negative, `sigmoid(x)` underflows to 0 and `log(0)` is `-inf`. `F.logsigmoid` does the computation in one numerically stable step.

Quick sanity check by hand: if `chosen == rejected`, the diff is 0, `σ(0) = 0.5`, `-log(0.5) = log(2) ≈ 0.6931`. That's the "no signal" loss; anything below that means the model is doing better than chance on this batch.

---

**Hint 4: `train_reward_model`, the update step.** The loop body:

```python
idx = torch.randint(n_pairs, (batch_size,), generator=g)
r_chosen = model(chosen[idx])      # (B, 1)
r_rejected = model(rejected[idx])  # (B, 1)

loss = bradley_terry_loss(r_chosen, r_rejected)

optimizer.zero_grad()
loss.backward()
optimizer.step()

losses.append(loss.item())

with torch.no_grad():
    pred_r = model(eval_feats).squeeze(-1)
spearmans.append(spearman_corr(pred_r, eval_true_r))
```

Common mistakes:

- Calling `RewardModel()` inside the loop: you reinitialize the weights every step, nothing learns. The model and optimizer are created above the loop; just use them.
- `optimizer.step()` before `loss.backward()`: silent no-op, no learning.
- Swapping `chosen` and `rejected` in the loss: the loss will *grow* and Spearman will go *negative*. If you see that, the sign is flipped.
- Forgetting `.detach()` or `torch.no_grad()` on the eval forward pass: not wrong, but it builds and discards a graph 2000 times. The wrapper exists for a reason.

---

**Hint 5: if Spearman plateaus below 0.85.** Three likely causes:

1. **The model isn't actually training.** Print `losses[0]` and `losses[-1]`. The first one should be near `log(2) ≈ 0.69` (random init, no separation yet). The last should be well under 0.2. If the loss is flat across all steps, see hint 4; you probably re-init the model in the loop, or you never call `optimizer.step()`.
2. **Sign flipped in `bradley_terry_loss`.** If Spearman trends *negative*, you swapped `chosen` and `rejected` somewhere, either in the loss, or when constructing the dataset. The dataset code is given to you, so check the loss.
3. **Learning rate way off.** The reference uses Adam at `lr=3e-3`. At `lr=0.1` the model overshoots and Spearman oscillates wildly; at `lr=1e-5` it just hasn't converged in 2000 steps. The starter defaults are tuned; don't fight them unless you have a reason.

---

If you're past hint 5 and still stuck, read [`solution/reward_model.py`](./solution/reward_model.py). Then re-derive `bradley_terry_loss` without looking; that's the one piece of math you should be able to write from memory by the time you're done.
