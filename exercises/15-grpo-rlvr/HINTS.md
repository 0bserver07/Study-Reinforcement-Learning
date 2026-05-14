# Hints — exercise 15

Read one at a time. Try after each before reading the next.

---

**Hint 1 — `GRPOPolicy`.** It's two lines. `self.embed = nn.Embedding(n_prompts, n_answers)` in `__init__`, and in `forward`: `return self.embed(torch.tensor(prompt_idx, dtype=torch.long))`. The embedding is a lookup table: row `i` is the logit vector for prompt `i`. No activation, no softmax — logits go straight to `Categorical(logits=...)` later. One row per prompt, one column per answer.

---

**Hint 2 — `sample_group`.** Four lines:

```python
logits = policy(prompt_idx)                          # (N_ANSWERS,)
dist = torch.distributions.Categorical(logits=logits)
answers = dist.sample((K,))                          # (K,) LongTensor
log_probs = dist.log_prob(answers)                   # (K,) floats, with grad
return answers, log_probs
```

Do **not** call `.detach()` on `log_probs` here — the caller does that explicitly after sampling (to freeze the old policy). If you detach here, `grpo_loss` has nothing to differentiate.

---

**Hint 3 — `group_advantages`.** Handle the two edge cases first, then the normal case:

```python
K = rewards.shape[0]
if K == 1 or rewards.std() < eps:
    return torch.zeros_like(rewards)
return (rewards - rewards.mean()) / (rewards.std() + eps)
```

When all rewards are equal, `rewards.std()` is 0.0, so the condition catches it. The advantage of a step where every completion got the same reward is genuinely zero — nothing tells the policy which answers were better.

---

**Hint 4 — `grpo_loss`.** Work in log-space for the ratio (more numerically stable than dividing raw probabilities):

```python
ratio = (log_probs_new - log_probs_old).exp()   # π_new / π_old, shape (K,)
clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
per_sample = -torch.min(ratio * advantages, clipped * advantages)
return per_sample.mean()
```

The minus sign is the key: the optimizer *minimizes*, but we want to *maximize* expected reward. The clip is why this is "PPO-style": if the ratio drifts far from 1.0, the clipped term limits how much the loss can improve, which stops the policy from taking a huge jump based on one noisy group.

---

**Hint 5 — `train`, putting it together.** The structure inside the loop is:

```python
prompt_idx = torch.randint(N_PROMPTS, (1,), generator=rng).item()
prompt = PROMPTS[prompt_idx]

answers, log_probs_old = sample_group(policy, prompt_idx, K)
log_probs_old = log_probs_old.detach()  # freeze sampling policy

rewards = torch.tensor(
    [verifier(prompt, int(a)) for a in answers], dtype=torch.float32
)
adv = group_advantages(rewards)

# Re-evaluate current policy's log-probs on the same answers.
# The policy hasn't updated yet, so this is numerically identical to
# log_probs_old — but it's connected to the computation graph.
logits = policy(prompt_idx)
dist = torch.distributions.Categorical(logits=logits)
log_probs_new = dist.log_prob(answers)

loss = grpo_loss(log_probs_new, log_probs_old, adv)

optimizer.zero_grad()
loss.backward()
optimizer.step()

step_rewards.append(rewards.mean().item())
```

Two common mistakes: (1) forgetting `log_probs_old = log_probs_old.detach()` — then PyTorch tries to differentiate through the sampling step, which doesn't work with discrete samples; (2) putting `raise NotImplementedError` before `step_rewards.append(...)` — remove the raise and the NotImplementedError together, replace the whole block.

---

If you're past hint 5 and still stuck, read [`solution/grpo_rlvr.py`](./solution/grpo_rlvr.py). Then re-derive it without looking.
