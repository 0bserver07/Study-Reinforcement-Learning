# Hints — exercise 11

Read one at a time. Try after each before reading the next.

---

**Hint 1 — `Policy`.** Two lines, same shape as the GRPO policy. In `__init__`:

```python
super().__init__()
self.embed = nn.Embedding(n_prompts, n_answers)
```

In `forward`, you need to handle both a single int (single-prompt call from a test) and a `LongTensor` (batched call from `train_dpo`):

```python
if isinstance(prompt_idx, int):
    idx = torch.tensor(prompt_idx, dtype=torch.long)
else:
    idx = prompt_idx.long()
return self.embed(idx)
```

`nn.Embedding(N, M)` is a lookup table — row `i` is a length-`M` vector. Indexing with a 0-d tensor gives a `(M,)` output; indexing with a `(B,)` tensor gives `(B, M)`. That's exactly what we want.

---

**Hint 2 — `dpo_loss`, step by step.** Two logit tensors of shape `(B, n_answers)` come in. Get log-probs first:

```python
policy_log_probs = F.log_softmax(policy_logits, dim=-1)
ref_log_probs    = F.log_softmax(ref_logits, dim=-1)
```

Now you need four scalars per batch element: `log pi(y_w)`, `log pi(y_l)`, `log pi_ref(y_w)`, `log pi_ref(y_l)`. `chosen_answer` is `(B,)`. `gather` is the cleanest way:

```python
pi_w  = policy_log_probs.gather(-1, chosen_answer.unsqueeze(-1)).squeeze(-1)
pi_l  = policy_log_probs.gather(-1, rejected_answer.unsqueeze(-1)).squeeze(-1)
ref_w = ref_log_probs.gather(-1, chosen_answer.unsqueeze(-1)).squeeze(-1)
ref_l = ref_log_probs.gather(-1, rejected_answer.unsqueeze(-1)).squeeze(-1)
```

The `unsqueeze` makes the indices `(B, 1)` to match the source's last dim; the `squeeze` strips the singleton back off.

---

**Hint 3 — finish `dpo_loss`.** Now the loss is one line of arithmetic and one line of reduction:

```python
logits = beta * ((pi_w - ref_w) - (pi_l - ref_l))
return -F.logsigmoid(logits).mean()
```

Sanity check: if `policy_logits == ref_logits`, then `pi_w == ref_w` and `pi_l == ref_l`, so `logits` is the zero vector. `logsigmoid(0) = log(0.5) = -log(2)`. The loss is then `-(-log(2)) = log(2) ≈ 0.6931`. The `test_dpo_loss_starts_at_log2` test checks this exactly.

Use `F.logsigmoid` rather than `torch.log(torch.sigmoid(...))` — it's much more numerically stable for large negative inputs. With `beta=0.1` and a tiny task this doesn't matter for correctness, but it's the right habit.

---

**Hint 4 — sign check.** The single most common bug in DPO is getting the sign backwards inside the bracket. The bracket must be `(chosen_logratio - rejected_logratio)`, not the reverse. Concretely: you want the loss to *decrease* as `pi(y_w)` rises and `pi(y_l)` falls. That means the argument to `sigmoid` should go up, and `-log sigmoid(...)` should go down. The `test_dpo_loss_pushes_toward_chosen` test verifies this directly — it computes the log-margin between chosen and rejected before and after one gradient step.

If that test fails, you almost certainly have the bracket flipped.

---

**Hint 5 — `train_dpo`, the update step.** Most of the loop is bookkeeping (already done for you). The update step itself is:

```python
idx = torch.randint(n, (batch_size,), generator=rng)
p_batch = prompts[idx]
w_batch = chosens[idx]
l_batch = rejecteds[idx]

policy_logits = policy(p_batch)
with torch.no_grad():
    ref_logits = ref_policy(p_batch)

loss = dpo_loss(policy_logits, ref_logits, w_batch, l_batch, beta)

optimizer.zero_grad()
loss.backward()
optimizer.step()

losses.append(loss.item())
mean_rewards.append(greedy_mean_true_reward(policy, true_reward))
```

Three things to be careful about:

1. The reference call is wrapped in `torch.no_grad()`. The reference is already frozen at the top of the function (`requires_grad_(False)`), but the `no_grad` makes it explicit and skips graph construction.
2. The policy and reference were initialized to the *same* weights (top of `train_dpo`). That's why the first loss is `log(2)`.
3. `mean_rewards` uses the *greedy* policy (argmax over answers), not a sampled one. The greedy reward is the right thing to check the policy converges toward the high-reward answers, not just that it shifts mass a bit.

---

**Hint 6 — why no reward model?** This is the conceptual hint. Standard RLHF needs three networks: SFT, reward model, policy. DPO collapses two of them: there is no separate reward model, because the DPO derivation shows that the optimal KL-constrained policy already encodes the reward, so reward differences can be written as log-ratios of policy probabilities. The preference labels (chosen, rejected) are the only "reward signal" you need.

In this exercise, the true reward exists (we use it to generate labels and to evaluate the trained policy), but the training pipeline never reads it. The DPO loss takes only logits and preference indices.

---

If you're past hint 6 and still stuck, read [`solution/dpo.py`](./solution/dpo.py). Then re-derive it without looking.
