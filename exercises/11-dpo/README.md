# Exercise 11 — DPO on a toy preference dataset

Goes with [lecture 11 (DPO)](../../notes/lectures/11-dpo.md).

You'll implement Direct Preference Optimization from scratch on a tiny dataset of (prompt, chosen, rejected) triples. No transformer, no tokenizer — the policy is a per-prompt categorical distribution, the same shape as the GRPO exercise. Everything runs in seconds on CPU, and the DPO loss and gradient flow are real.

## The point

DPO (Rafailov et al. 2023, arXiv:2305.18290) drops the reward-model step of RLHF: given pairwise preferences and a frozen reference policy, you can write down a closed-form loss that directly improves the policy on those preferences. The loss is:

```
L_DPO = -log sigmoid( beta * [(log pi(y_w|x) - log pi_ref(y_w|x))
                            - (log pi(y_l|x) - log pi_ref(y_l|x))] )
```

`y_w` is the chosen (winning) answer, `y_l` is the rejected one, `pi` is the policy being trained, `pi_ref` is a frozen reference (in the LLM setting, the SFT model). Beta controls how far the policy is allowed to drift from the reference.

You'll see directly that at initialisation, when the policy is a copy of the reference, the loss is exactly `log(2) ≈ 0.6931` — because the bracket is zero and `sigmoid(0) = 0.5`. As training progresses, the policy pushes log-prob mass from rejected answers to chosen ones.

## The task

Setup:
- `N_PROMPTS = 8` prompts (just indices).
- `N_ANSWERS = 10` answers per prompt (also indices).
- A fixed "true" reward table `r_true[prompt, answer]` (Gaussian) you don't use during training.
- A preference dataset of 512 `(prompt, chosen, rejected)` triples, labelled with Bradley-Terry noise on the true reward gap.

Fill in the TODOs in [`starter.py`](./starter.py). Three pieces:

1. `Policy` — `nn.Embedding(n_prompts, n_answers)` outputting logits. `forward` should accept a single `int` (returns `(n_answers,)`) or a `LongTensor` of shape `(B,)` (returns `(B, n_answers)`).
2. `dpo_loss(policy_logits, ref_logits, chosen, rejected, beta)` — the formula above. Use `F.log_softmax`, `.gather`, and `F.logsigmoid`.
3. `train_dpo(...)` — the loop: sample a batch of triples, forward policy and ref, compute the loss, step. Log per-step loss and per-step greedy mean true reward.

The reference, dataset generator, true-reward table, and evaluation helper are given. You don't fit a reward model.

## Setup

```bash
pip install -r exercises/requirements.txt
pytest exercises/11-dpo/
```

## Acceptance criteria

`pytest exercises/11-dpo/` passes. That's:

- `Policy.forward` handles both a single int and a `LongTensor` of indices, with the expected output shapes.
- `dpo_loss` returns `log(2)` when the policy equals the reference, has a non-zero gradient on the policy, and (crucially) the gradient pushes log-prob mass toward the chosen answer.
- Integration test: after `train_dpo(seed=0, n_steps=2000)`, the greedy (argmax) policy's mean true reward across the 8 prompts is above 0.6, versus a uniform-random baseline around 0.09 and a per-prompt oracle around 1.79. CPU, under 10 seconds.

You're done when the tests pass and you can explain: why the loss is `log(2)` at init, why no reward model is needed, and what happens if you set `beta` very small or very large.

## If you get stuck

Read [`HINTS.md`](./HINTS.md) — one hint at a time. The reference implementation is in [`solution/dpo.py`](./solution/dpo.py); look at it after you've made a real attempt.

## Going further (optional)

- Sweep `beta` over `{0.01, 0.1, 1.0, 5.0}`. Watch what happens to the loss curve and to the final greedy reward. Where does the policy underfit? Where does it collapse onto a single answer regardless of prompt?
- Replace the Adam optimizer with plain SGD. How does the lr need to change?
- Use a non-identical reference: train the reference for a few steps of SFT-like maximum-likelihood on `chosen` answers first, then run DPO with that reference. Does the final policy get further than starting from a uniform reference?
- Compare the DPO-trained policy to the "supervised" baseline of just maximising the log-prob of every chosen answer (i.e. ignoring the rejected ones). On this toy task with 512 pairs, which wins?
- Read the DPO paper and match each line of `dpo_loss` to the equations in section 4. The reference implementation in `solution/dpo.py` is named to mirror the paper.
