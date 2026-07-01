<!-- status: unreviewed | last-reviewed: never -->

# KL control in RL: cheat sheet

_Unreviewed: no one has checked this end to end. Treat the math and citations as unverified._

KL divergence between two distributions is the most-tuned knob in LLM-RL. It shows up as a hard constraint in TRPO, an adaptive penalty in PPO, an anchor to the SFT model in PPO-RLHF and GRPO, and as an implicit regularizer baked into the DPO loss. People get it wrong a lot. This sheet collects the formulas, where each one appears, how to estimate it cheaply, and what to watch on the dashboard.

---

## Table of contents

1. [Where KL appears in each method](#1-where-kl-appears-in-each-method)
2. [Formula sheet](#2-formula-sheet)
3. [Forward vs reverse KL](#3-forward-vs-reverse-kl)
4. [Estimators (K1, K2, K3)](#4-estimators-k1-k2-k3)
5. [Per-token KL in PyTorch](#5-per-token-kl-in-pytorch)
6. [β tuning rules of thumb](#6-β-tuning-rules-of-thumb)
7. [Common failure modes](#7-common-failure-modes)
8. [Dashboard checklist](#8-dashboard-checklist)
9. [References](#9-references)

---

## 1. Where KL appears in each method

| Method | KL role | Between | Type | Reference |
|---|---|---|---|---|
| TRPO | Hard constraint per update | π_new vs π_old (trajectory) | Reverse KL, on rollout states | arXiv:1502.05477 |
| PPO-clip | None explicit; clipping bounds the ratio | N/A | N/A | arXiv:1707.06347 |
| PPO-adaptive | Soft penalty, β tunes to target KL | π_new vs π_old | Reverse KL, per update | arXiv:1707.06347 |
| Ziegler RLHF | Per-token anchor penalty | π vs π_ref (SFT init) | Reverse, integrated into reward | arXiv:1909.08593 |
| InstructGPT / PPO-RLHF | Per-token anchor penalty | π vs π_ref (SFT init) | Reverse, integrated into reward | arXiv:2203.02155 |
| DPO | Implicit, via the analytic reparam | π vs π_ref | Reverse, baked into loss; β = inverse temperature | arXiv:2305.18290 |
| GRPO | Per-token or per-sequence anchor penalty + PPO clip | π vs π_ref | Reverse, added to surrogate loss | arXiv:2402.03300 |
| RLVR / DeepSeek-R1 | Same as GRPO; reported omitted in some R1-Zero runs | π vs π_ref | Reverse | arXiv:2501.12948 (verify the specific run) |

Two different things are both called "the KL penalty," and confusing them is the most common bug in this area:

- **Per-update KL** (TRPO, PPO-adaptive): keeps a single optimization step from moving the policy too far from the *previous* policy. Resets each iteration.
- **Anchor KL** (PPO-RLHF, GRPO, DPO): keeps the policy from drifting too far from a *fixed* reference (the SFT model) over the *whole run*. Cumulative.

Per-update KL is a stability tool for the optimizer. Anchor KL is an alignment tool: it's there because the reward model is a brittle proxy and you don't want the policy to wander away from the prior the SFT model encodes.

---

## 2. Formula sheet

Notation:
- π_θ: the current (policy being trained) policy
- π_old: the behavior policy used to collect the current batch
- π_ref: the frozen reference policy (typically the SFT-tuned model)
- x: prompt; y: full response; t: token index in y
- D: KL operator
- β: temperature/weight on the KL term (different per method)
- δ: TRPO trust-region size (typically 0.01)
- ε: PPO clip range (typically 0.2)

### Discrete KL definitions

```
D_KL(p || q) = Σ_a p(a) log( p(a) / q(a) )
```

For two LM next-token distributions over vocabulary V at context c:

```
D_KL(π_θ(·|c) || π_ref(·|c)) = Σ_{v in V} π_θ(v|c) log( π_θ(v|c) / π_ref(v|c) )
```

This is the **full distributional KL** at one context. Costs |V| × forward-pass logits, but you already have the logits, so it's a softmax + reduction.

### TRPO: hard KL constraint per update

```
maximize_θ  E_{(s,a)~π_old} [ (π_θ(a|s) / π_old(a|s)) · A(s,a) ]
subject to  E_{s~ρ_π_old} [ D_KL( π_old(·|s) || π_θ(·|s) ) ] ≤ δ
```

Linearize objective, quadraticize constraint via Fisher H, conjugate-gradient solve, line-search to satisfy the constraint. δ ≈ 0.01 is the default in the paper. Schulman et al. 2015, arXiv:1502.05477.

### PPO-adaptive KL penalty (Algorithm 1, "adaptive KL" variant)

Per-update soft penalty in place of the clip:

```
L^KLPEN(θ) = E [ (π_θ(a|s) / π_old(a|s)) · A(s,a) − β · D_KL( π_old(·|s) || π_θ(·|s) ) ]
```

After each update, compute d = E[D_KL]:

```
if d < d_target / 1.5:  β ← β / 2
if d > d_target × 1.5:  β ← β × 2
```

Schulman et al. 2017, arXiv:1707.06347, §4. The clipped surrogate (PPO-clip) is the more popular variant and has no KL term at all: the ratio clip plays the same stabilizing role implicitly.

### PPO-RLHF anchor KL (Ziegler, InstructGPT)

The reward used inside PPO is the RM score minus a per-token KL to the reference:

```
R(x, y) = r_φ(x, y) − β · Σ_t  [ log π_θ(y_t | x, y_<t) − log π_ref(y_t | x, y_<t) ]
```

Equivalently, treat the per-token bracket as a per-token reward shaping term; this is what InstructGPT does, so that GAE can credit-assign the KL cost across tokens:

```
r_t = − β · ( log π_θ(y_t | x, y_<t) − log π_ref(y_t | x, y_<t) )       for t < T
r_T = r_φ(x, y) − β · ( log π_θ(y_T | x, y_<T) − log π_ref(y_T | x, y_<T) )
```

Note the form inside the brackets is the **per-sample log-ratio**, which is an unbiased estimator of the per-token reverse KL; it's not the full distributional KL. The reverse-KL approximation comes from the K1 estimator (§4).

References: Ziegler et al. 2019 (arXiv:1909.08593) introduced this exact structure for LM fine-tuning; Ouyang et al. 2022 (arXiv:2203.02155) used it for InstructGPT with β ≈ 0.02 (verify exact value against the paper before quoting).

### DPO: implicit KL via the closed-form optimum

The optimum of the KL-constrained RLHF objective

```
max_π  E_{x, y~π} [ r(x, y) ]  −  β · D_KL( π(·|x) || π_ref(·|x) )
```

has the closed form

```
π*(y|x) = (1 / Z(x)) · π_ref(y|x) · exp( r(x, y) / β )
```

Rearranging for r and substituting into the Bradley-Terry preference model gives the DPO loss:

```
L_DPO(θ) = − E_{(x, y_w, y_l)} [ log σ( β · ( log[π_θ(y_w|x)/π_ref(y_w|x)] − log[π_θ(y_l|x)/π_ref(y_l|x)] ) ) ]
```

β plays the same role as in PPO-RLHF: bigger β = stay closer to π_ref. Rafailov et al. 2023, arXiv:2305.18290. The KL term is not added; it is implicit in the form of the loss. Typical values: β ∈ [0.1, 0.5].

### GRPO: group-relative + anchor KL

For prompt x, sample K completions y_1..y_K from π_old. Compute reward r_i for each, then the group-relative advantage:

```
A_i = (r_i − mean(r_1..r_K)) / (std(r_1..r_K) + ε_std)
```

Per-completion PPO clip:

```
ρ_i = exp( log π_θ(y_i|x) − log π_old(y_i|x) )
L_CLIP_i = min( ρ_i · A_i,  clip(ρ_i, 1−ε, 1+ε) · A_i )
```

Anchor KL added to the loss:

```
L_GRPO = − (1 / (B·K)) Σ_{b,i} L_CLIP_{b,i}  +  β · KL_term(π_θ, π_ref)
```

The DeepSeekMath paper uses a per-token estimator for KL_term so it scales sensibly across response lengths. See Shao et al. 2024, arXiv:2402.03300. RLVR / DeepSeek-R1 (arXiv:2501.12948) keeps the same structure; check the specific run before claiming KL was on or off.

---

## 3. Forward vs reverse KL

```
D_KL(p || q)   "forward": finite only where q > 0 wherever p > 0.   Mean-seeking.
D_KL(q || p)   "reverse": penalizes q putting mass where p is small. Mode-seeking.
```

For RLHF the conventional placement is **reverse KL** with q = π (policy) and p = π_ref:

```
D_KL( π || π_ref ) = E_{y ~ π} [ log π(y) − log π_ref(y) ]
```

The expectation is over samples from π, which matters: when you generate from π and compute log-ratios, you get an unbiased estimate of this quantity (see K1 in §4) for free.

Practical consequences:

- **Reverse KL** with a peaky π_ref produces a peaky π (mode-seeking). The policy is allowed to ignore modes of π_ref it doesn't like, as long as the modes it does keep stay close. This is what you want for RLHF: collapse to high-reward modes.
- **Forward KL** with a peaky π_ref produces a diffuse π that covers all modes of π_ref (mean-seeking). This is what SFT does implicitly, since the cross-entropy SFT loss equals forward KL up to a constant: `D_KL(p_data || π) = H(p_data, π) − H(p_data)`.

A subtle one: the "KL to reference" computed in DPO is also reverse KL (q=π, p=π_ref), even though the loss form doesn't make this obvious; see Rafailov et al. §4 for the derivation.

---

## 4. Estimators (K1, K2, K3)

In LLM training you usually do **not** evaluate the full Σ over the vocabulary at every token; you have one sample, and you want an unbiased estimate of `D_KL(π_θ || π_ref) = E_{y~π_θ} [ log π_θ(y) − log π_ref(y) ]`.

From Schulman's "Approximating KL Divergence" blog post (joschu.net/blog/kl-approx.html: blog, not a paper). Let `r = π_θ(y) / π_ref(y)` for a single sample y drawn from π_θ. Then:

| Name | Formula | Bias | Variance | Sign |
|---|---|---|---|---|
| K1 | `log r` | Unbiased | High | Can be negative |
| K2 | `0.5 · (log r)²` | Biased | Lower than K1 | Always ≥ 0 |
| K3 | `(r − 1) − log r` | Unbiased | Lower than K1 | Always ≥ 0 |

K3 is the recommended default for LLM training: unbiased like K1, but always non-negative (a property of the true KL it estimates) and lower-variance than K1.

```python
# Per-token, given log-probs of the actually sampled token y_t under each model:
# logp = log π_θ(y_t | context),  logp_ref = log π_ref(y_t | context)
log_r = logp - logp_ref           # K1
k2    = 0.5 * log_r ** 2          # K2
k3    = (log_r.exp() - 1) - log_r # K3
```

K1 is what almost everything in the wild actually uses (including the per-token KL in InstructGPT-style RLHF): it's the simplest, you already have the log-probs, and the variance is acceptable when averaged over a batch. K3 is worth switching to if you're seeing negative KL values in your logs (a giveaway that you're using K1 with too small a batch).

---

## 5. Per-token KL in PyTorch

The cheap path: you already have policy and reference logits from the forward passes you needed for the loss anyway. Don't recompute them.

```python
import torch
import torch.nn.functional as F

def per_token_kl_estimators(
    policy_logits: torch.Tensor,    # [B, T, V], current policy
    ref_logits:    torch.Tensor,    # [B, T, V], frozen reference
    sampled_ids:   torch.Tensor,    # [B, T], tokens actually drawn from π_θ
    mask:          torch.Tensor,    # [B, T], 1 on response tokens, 0 on prompt/pad
) -> dict[str, torch.Tensor]:
    """
    Returns per-token K1, K2, K3 estimates and the exact distributional KL.
    Shapes are [B, T]; multiply by `mask` and sum/mean as you like.

    K1 = log(π_θ/π_ref): unbiased, can be negative on a single sample
    K2 = 0.5 (log r)^2: biased, ≥ 0
    K3 = (r - 1) - log r: unbiased, ≥ 0  ← prefer this for logging
    """
    logp_policy = F.log_softmax(policy_logits, dim=-1)
    logp_ref    = F.log_softmax(ref_logits,    dim=-1)

    # Gather log-probs of the sampled tokens.
    idx = sampled_ids.unsqueeze(-1)                          # [B, T, 1]
    logp     = logp_policy.gather(-1, idx).squeeze(-1)       # [B, T]
    logp_r   = logp_ref.gather(-1, idx).squeeze(-1)          # [B, T]

    log_r = logp - logp_r                                    # K1
    k1 = log_r
    k2 = 0.5 * log_r.pow(2)
    k3 = (log_r.exp() - 1.0) - log_r

    # Exact distributional KL (uses every vocab entry). Use this for
    # validation logging; per-sample estimators above are what you
    # backprop through.
    probs_policy = logp_policy.exp()
    kl_exact = (probs_policy * (logp_policy - logp_ref)).sum(dim=-1)  # [B, T]

    return {
        "k1": k1 * mask,
        "k2": k2 * mask,
        "k3": k3 * mask,
        "kl_exact": kl_exact * mask,
    }


def sequence_kl(per_token: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sum a per-token KL estimate over the response and average over batch."""
    return (per_token * mask).sum(dim=-1).mean()
```

Notes:

- The `kl_exact` line is `O(B·T·V)`: fine to log every few steps, expensive every step. Use it as a sanity check that your K1/K3 estimators aren't drifting.
- If memory is tight, compute `kl_exact` without materializing `probs_policy`: `(F.softmax(policy_logits, -1) * (logp_policy - logp_ref)).sum(-1)`.
- The penalty term that goes *into the reward* in PPO-RLHF is usually K1, `β · (logp - logp_ref)`, applied per token. If you switch to K3 in the reward, you change the gradient.
- For DPO, you don't add a KL term: the loss already contains the log-ratios. Computing KL is for monitoring only.

---

## 6. β tuning rules of thumb

**β too low** (`< 0.01` in PPO-RLHF; `< 0.05` in DPO):
- Policy drifts arbitrarily far from π_ref.
- KL grows without bound on the dashboard.
- Policy finds reward-model exploits: repetition, keyword stuffing, formatting tricks that score high but read poorly.
- Loss of general capability outside the reward-model's domain. Forgets multilingual or coding skills it had at SFT.

**β too high** (`> 0.5` in PPO-RLHF; `> 1.0` in DPO):
- Policy can barely move. Training reward plateaus near the SFT baseline.
- Held-out reward barely changes from initialization.
- KL stays near zero throughout. The optimizer is fighting the penalty.

**Adaptive β (PPO-adaptive)**: target a fixed per-update KL budget. InstructGPT used a target KL around **6 nats** (verify the exact number in the paper before quoting). Adjust β with the rule from §2.

**Typical starting values:**

| Setting | β |
|---|---|
| PPO-RLHF token-level KL in reward | 0.02 (InstructGPT-style, verify) |
| GRPO KL penalty (DeepSeekMath) | 0.04 |
| DPO | 0.1 (start here, range 0.1–0.5) |
| TRPO trust-region δ | 0.01 |
| PPO-adaptive d_target | ≈ 0.01 (per state), paper-specific |

Tune by sweeping a factor of 2 in each direction from the starting value and watching held-out reward + KL trace together.

---

## 7. Common failure modes

### Per-trajectory KL ≪ distributional KL

The KL of one well-chosen rollout (one sample from π) can be small while the KL of the *full distribution* at that prompt has grown a lot. Per-trajectory log-ratios are sample estimates, and they vary. Symptoms:

- Average per-token KL on training rollouts looks fine (~ 1–5 nats over the whole response).
- Held-out evaluations show the model is producing repetitive, narrow-distribution outputs.
- Distributional KL (the full Σ over vocab) is 10× the sampled value.

Fix: log the exact distributional KL on a small validation prompt set every N steps, not just the sampled estimate from training rollouts. This is what catches the issue.

### Negative KL in logs

K1 is unbiased but can be negative on a single sample (especially short sequences). If the *average* per-token K1 over the batch is consistently negative, something is wrong: usually it means π_ref is no longer actually frozen (its gradients are being updated) or you swapped the two log-probs.

Fix:
- Confirm `for p in ref_model.parameters(): p.requires_grad = False` and that ref forward passes are inside `torch.no_grad()`.
- Switch the logged KL to K3, which is always ≥ 0. The training reward keeps using K1; only the logged value changes.

### Reward and KL both climb together → reward hacking

This is the classic RLHF pathology. Reward goes up, KL goes up, generations look worse. The reward model is being gamed.

Fix: raise β, shorten the run, train a better reward model, add held-out preference checks. Goodhart's law: when a measure becomes a target, it ceases to be a good measure (Goodhart 1975, not in arXiv; the broader literature in alignment is summarized well in Gao, Schulman, Hilton 2022, arXiv:2210.10760).

### KL spikes after K_epochs increase

If you raise PPO's K_epochs (the number of gradient passes per rollout batch), the policy moves further per batch and per-update KL spikes. The clip prevents catastrophic moves, but the anchor KL still drifts.

Fix: when raising K_epochs, lower the learning rate or raise β to compensate.

### DPO with β too small → collapsed log-ratios

In DPO, β too small (say 0.01) makes the loss flat in the log-ratio dimension and the model overfits to the preference dataset by sharpening preferences without bounded movement from π_ref. The implicit reward `r̂ = β · log(π_θ/π_ref)` becomes huge, but the policy hasn't actually improved on held-out preferences.

Fix: start β at 0.1. The "reward margin" `β · (log r_w − log r_l)` should grow gradually during training, not in early steps.

### Ratio explosion from log-prob compounding

For long completions, even small per-token differences in `log π_θ − log π_old` compound. With T = 512 tokens and an average per-token log-ratio of 0.01, the sequence-level ratio is `exp(5.12) ≈ 167`. PPO's per-completion clip at `1+ε = 1.2` will be exercised on every sample.

Fix: this is one reason GRPO often uses per-token clipping rather than per-sequence clipping. Check whether your implementation is computing the ratio per token or per sequence.

---

## 8. Dashboard checklist

Log every training step:

- **mean KL**: average per-token K1 or K3 over the batch, response tokens only. Should grow gradually and plateau. If it keeps climbing past your target, β is too low.
- **mean reward**: moving average over the batch. Should grow. If it grows while KL also grows fast, suspect reward hacking.
- **clip fraction**: fraction of tokens (or completions, depending on impl) where the PPO ratio was clipped. Healthy: 10–40%. Too low (<5%): ε is too large to constrain. Too high (>60%): ε too small or policy moving too fast.
- **ratio mean / max / min**: `exp(log π_θ − log π_old)` summary stats. Max should be in `[1, 1+ε]` after clipping; if it's much larger, the clip isn't being applied where you think.

Log every N (say 50) steps:

- **distributional KL**: the exact `Σ π_θ log(π_θ/π_ref)` computed on a fixed held-out prompt set. The most reliable measure of how far the policy has drifted. Compare to mean K1 from training. If there's a 2-3× gap, your training estimate is misleading.
- **max-token KL**: the highest per-token K1 in the batch. A single token at K1=20 means the policy is putting probability where π_ref is essentially zero. Often a sign of mode collapse on that token.
- **ref-model NLL gap**: `−log π_ref(y)` on a held-out reference set vs `−log π_θ(y)` on the same set. If the gap widens dramatically, the policy has lost coverage of the reference distribution.
- **entropy**: `H(π_θ)` per token, averaged over response tokens. A sharp drop (e.g., 3.5 → 2.0 nats over a few hundred steps) signals collapse.

Sanity check before training starts:

- KL at step 0 should be ≈ 0 (the policy *is* the reference at init).
- If not: π_ref isn't actually the same as the initialization, or one of the model copies isn't frozen.

Healthy KL ranges for sequence-level KL (sum of per-token K1 over the full response) on typical RLHF runs:

```
End of training:    10–60 nats (depends on response length, β, task)
Per-token average:  0.05–0.5 nats
```

These are very rough. The shape of the curve (steady vs blowing up) matters more than the absolute value.

---

## 9. References

All arXiv IDs verified against arxiv.org.

**Trust regions and PPO**

- Schulman, Levine, Moritz, Jordan, Abbeel. 2015. "Trust Region Policy Optimization." ICML 2015. arXiv:1502.05477.
- Schulman, Wolski, Dhariwal, Radford, Klimov. 2017. "Proximal Policy Optimization Algorithms." arXiv:1707.06347. PPO-clip and PPO-adaptive-KL both introduced here.

**RLHF for language models**

- Ziegler, Stiennon, Wu, Brown, Radford, Amodei, Christiano, Irving. 2019. "Fine-Tuning Language Models from Human Preferences." arXiv:1909.08593. First systematic application of PPO + per-token KL-to-reference for LM fine-tuning.
- Stiennon, Ouyang, Wu, Ziegler, Lowe, Voss, Radford, Amodei, Christiano. 2020. "Learning to summarize from human feedback." arXiv:2009.01325. Same KL structure, longer-form generation.
- Christiano, Leike, Brown, Martic, Legg, Amodei. 2017. "Deep reinforcement learning from human preferences." arXiv:1706.03741. The earliest end-to-end RLHF, predates LM application.
- Ouyang, Wu, Jiang, et al. 2022. "Training language models to follow instructions with human feedback." NeurIPS 2022. arXiv:2203.02155. InstructGPT. The KL-to-reference β is reported in the paper. Verify the exact value before quoting.

**DPO and relatives**

- Rafailov, Sharma, Mitchell, Ermon, Manning, Finn. 2023. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." arXiv:2305.18290. Closed-form derivation that converts the KL-constrained RLHF objective into a supervised loss on preference pairs.

**GRPO and RLVR**

- Shao, Wang, Zhu, et al. (DeepSeek-AI). 2024. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." arXiv:2402.03300. GRPO is §3.
- DeepSeek-AI. 2025. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv:2501.12948. Verify against the paper which runs include the KL penalty and which omit it.

**Reward hacking / overoptimization**

- Gao, Schulman, Hilton. 2022. "Scaling Laws for Reward Model Overoptimization." arXiv:2210.10760. Empirical scaling of how much policy KL it takes to overoptimize a reward model.

**KL estimators (blog, not a paper)**

- Schulman. "Approximating KL Divergence." http://joschu.net/blog/kl-approx.html. Describes K1, K2, K3 estimators and their bias/variance tradeoffs. Cite the URL; no DOI, no arXiv ID.

---

## What to read first

If you only have an hour: PPO paper §4 (KL-penalty variant), then the Ziegler et al. 2019 paper (clearest description of the per-token KL in the reward), then the DPO paper §4 (KL-constrained derivation). That covers all four KL roles in this sheet: per-update soft penalty, per-update hard constraint (skip to TRPO §2 for that), anchor penalty in the reward, and implicit KL in a closed-form loss.
