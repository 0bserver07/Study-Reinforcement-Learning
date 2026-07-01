<!-- status: unreviewed | last-reviewed: never -->

# RLHF vs DPO vs GRPO vs RLVR — cheat sheet

_Unreviewed — no one has checked this end to end. Treat the math and citations as unverified._

Side-by-side reference for the main RL methods used to align LLMs: PPO-RLHF, DPO and its preference-optimization cousins (IPO, KTO, ORPO, SimPO), GRPO, and the verifiable-reward GRPO loop that drives DeepSeek-R1. Use this when picking a method, not when learning one — the lectures it points back to do the teaching.

Related: [`./RL-Quick-Reference.md`](./RL-Quick-Reference.md) for general deep-RL algorithms, [`./RL-Math-Formulas.md`](./RL-Math-Formulas.md) for the underlying math.

---

## Notation used throughout

| Symbol | Meaning |
|---|---|
| `x` | prompt |
| `y` | completion (full sequence of tokens) |
| `y_w`, `y_l` | preferred ("win") and rejected ("lose") completions |
| `π_θ` | policy being trained |
| `π_ref` | frozen reference policy (usually the SFT model) |
| `π_old` | policy snapshot used to collect the rollout batch |
| `r_φ` | learned reward model |
| `β` | KL or strength coefficient (per-method scale differs) |
| `σ` | sigmoid; `logσ` is `log(sigmoid(.))` |
| `K` | number of completions sampled per prompt (group methods) |

All log-probabilities are summed over response tokens only — prompt tokens are masked. Forgetting this mask is the most common silent bug; see the debugging note at the end.

---

## The comparison table

| Method | Reward model? | Reference policy? | Data form | On/off-policy | Cost vs PPO-RLHF | Known failure | Reach for it when |
|---|---|---|---|---|---|---|---|
| **PPO-RLHF** (Ouyang 2022, arXiv:2203.02155) | yes (learned `r_φ`) | yes (KL penalty) | preference pairs (for `r_φ`) + prompts (for PPO) | on-policy | 1.0× (baseline; 3–4 models live) | reward hacking; high hyperparameter load | maximum performance; you can afford the compute and the engineering |
| **DPO** (Rafailov 2023, arXiv:2305.18290) | no (implicit) | yes | paired preferences `(x, y_w, y_l)` | off-policy | ~0.25–0.35× | overconfidence; drift if β too small; underfits hard tasks | small budget, paired preferences, want a one-knob method |
| **IPO** (Azar 2023, arXiv:2310.12036) | no | yes | paired preferences | off-policy | ~0.25–0.35× | slower to fit; needs careful β | DPO overfits or saturates at high KL |
| **KTO** (Ethayarajh 2024, arXiv:2402.01306) | no | yes | unpaired binary labels (good / bad) | off-policy | ~0.25–0.35× | sensitive to `λ_D` / `λ_U` ratio; class imbalance | you only have thumbs up/down, not pairs |
| **ORPO** (Hong 2024, arXiv:2403.07691) | no | **no** | paired preferences | off-policy | ~0.20–0.30× | no KL anchor; pref signal can swamp the SFT term | you want to fold SFT and alignment into one stage |
| **SimPO** (Meng 2024, arXiv:2405.14734) | no | **no** | paired preferences | off-policy | ~0.20–0.30× | length-bias re-emerges without the length norm; collapse if `γ` too low | no SFT model loaded; want length-normalized DPO |
| **GRPO** (Shao 2024, arXiv:2402.03300) | optional (`r_φ` or rule) | yes (KL penalty) | prompt + K sampled completions, each scored | on-policy | ~0.45–0.55× (no critic; K extra forward passes) | baseline collapse when all-K-same; entropy collapse | objective scorer available; want PPO without a critic |
| **RLVR / R1-style GRPO** (DeepSeek-R1, arXiv:2501.12948) | no (rule-based checker) | yes | prompt + K completions, verifier-scored | on-policy | ~0.45–0.55× + verifier cost | verifier hacking; format-vs-correctness reward imbalance | verifiable domain (math, code, proofs); want reasoning to emerge |

Cost numbers are rough order-of-magnitude estimates from the lecture notes and from informal practitioner reports — they swing widely with model size, K, and how aggressively you batch.

---

## PPO-RLHF (Ouyang 2022, arXiv:2203.02155)

The original three-stage pipeline: SFT, then reward model on preference pairs, then PPO to maximize the reward model with a KL anchor to the SFT model.

**Objective** (per generated completion `y` from prompt `x`):

```
max_θ  E[r_φ(x, y)]  −  β · KL(π_θ(·|x) || π_ref(·|x))
```

The PPO surrogate that actually gets optimized (per token):

```
L_PPO = -E[ min( ratio · A,  clip(ratio, 1-ε, 1+ε) · A ) ]
ratio = π_θ(a_t | s_t) / π_old(a_t | s_t)
```

where `A` is a GAE advantage from a learned value head. The reward `r_φ(x, y)` is sparse — it lands at the last token of the completion. The KL term contributes a per-token shaping reward `−β · (log π_θ − log π_ref)`.

**Data**: ~10–50k SFT demos, ~30–100k preference pairs for `r_φ`, ~30k+ prompts for the PPO rollout phase. (InstructGPT scale.)

**Failure modes**:
- Reward hacking: model finds inputs that score high on `r_φ` but are gibberish, repetition, or off-topic.
- KL blow-up: `β` too small → policy drifts, KL > 100, outputs degrade.
- Value head instability: critic loss dominates, policy stops updating.
- Mode collapse: entropy bonus too low; same response for everything.

**Code sketch** (one PPO step, with the four models live):

```python
# Models: policy π_θ, ref (frozen SFT), reward_model r_φ, value_head V
y = policy.generate(x)                                  # rollout
logp = policy.log_prob(y, x)                            # current
logp_old = logp.detach()                                # cache
logp_ref = ref.log_prob(y, x)                           # frozen
rm = reward_model(x, y)                                 # scalar, last token

token_reward = -beta * (logp - logp_ref)                # per-token KL shape
token_reward[-1] += rm                                  # sparse RM reward
advantages, returns = gae(token_reward, V(states), gamma, lam)

ratio = torch.exp(logp - logp_old)
surr1 = ratio * advantages
surr2 = ratio.clamp(1 - eps, 1 + eps) * advantages
policy_loss = -torch.min(surr1, surr2).mean()
value_loss  = F.mse_loss(V(states), returns)
loss = policy_loss + vf_coef * value_loss
loss.backward(); optimizer.step()
```

See [`../lectures/10-ppo-for-llms.md`](../lectures/10-ppo-for-llms.md) for the full pipeline.

---

## DPO (Rafailov 2023, arXiv:2305.18290)

Closed-form solution of the same KL-constrained RLHF objective. Reward differences become log-probability ratios; no `r_φ` and no rollouts needed.

**Loss**:

```
L_DPO = -E_{(x, y_w, y_l)}[ log σ( β · ( log π_θ(y_w|x) − log π_ref(y_w|x)
                                       − log π_θ(y_l|x) + log π_ref(y_l|x) ) ) ]
```

**Data**: paired preferences `(x, y_w, y_l)`. No rollout, no scoring; offline.

**Failure modes**:
- `β` too small → policy stays glued to the reference; learns nothing.
- `β` too large (e.g. 1.0) → drifts; outputs go incoherent.
- Reference not frozen → gradients flow through it; KL anchor disappears.
- Log-probs computed over the prompt → model just learns to echo the prompt.
- Overconfidence: pushes `π_θ(y_w)` arbitrarily high once it beats `y_l`, even when the underlying signal is weak.

**Code sketch**:

```python
def dpo_loss(logp_chosen, logp_rej, ref_logp_chosen, ref_logp_rej, beta=0.1):
    # All inputs are sum of per-token log-probs over RESPONSE tokens only.
    chosen_ratio = logp_chosen - ref_logp_chosen        # log π_θ/π_ref for y_w
    rej_ratio    = logp_rej    - ref_logp_rej           # log π_θ/π_ref for y_l
    logits = beta * (chosen_ratio - rej_ratio)
    return -F.logsigmoid(logits).mean()
```

See [`../lectures/11-dpo.md`](../lectures/11-dpo.md).

---

## IPO (Azar 2023, arXiv:2310.12036)

A regression-style objective that fixes DPO's tendency to push the log-ratio gap arbitrarily large. The Bradley-Terry assumption that DPO inherits can drive the policy to push `logσ(.)` toward 1 even when the preference is noisy; IPO swaps the log-sigmoid for a squared loss with a fixed target.

**Loss**:

```
L_IPO = E_{(x, y_w, y_l)}[ ( (log π_θ(y_w|x) − log π_ref(y_w|x))
                           − (log π_θ(y_l|x) − log π_ref(y_l|x))
                           − 1 / (2β) )^2 ]
```

The `1/(2β)` term is the fixed target margin for the log-ratio gap.

**Data**: paired preferences. Same as DPO.

**Failure modes**:
- Slower convergence than DPO; needs more steps.
- Underfits if the preference data is actually well-separated.
- `β` plays a different role than in DPO — it sets the target margin, not the loss steepness.

**Code sketch**:

```python
def ipo_loss(logp_chosen, logp_rej, ref_logp_chosen, ref_logp_rej, beta=0.1):
    chosen_ratio = logp_chosen - ref_logp_chosen
    rej_ratio    = logp_rej    - ref_logp_rej
    gap = chosen_ratio - rej_ratio
    target = 1.0 / (2.0 * beta)
    return ((gap - target) ** 2).mean()
```

(The exact functional form in the Azar paper has several equivalent presentations; the lecture notes in [`../lectures/12-beyond-dpo.md`](../lectures/12-beyond-dpo.md) use a slightly different parameterization. Verify against §4 of arXiv:2310.12036 before relying on the exact constants.)

---

## KTO (Ethayarajh 2024, arXiv:2402.01306)

Drops the pairwise constraint. Each example is a single `(x, y, label)` with `label ∈ {desirable, undesirable}`. Loss is built from a prospect-theory-style value function (loss aversion: people weight losses more than gains).

**Loss** (per example, one of two branches):

```
For desirable y:
    L = 1 − σ( β · (log π_θ(y|x) − log π_ref(y|x)) − z_ref )

For undesirable y:
    L = 1 − σ( z_ref − β · (log π_θ(y|x) − log π_ref(y|x)) )
```

where `z_ref` is a per-batch reference point (the KL between policy and reference, averaged over a contrast batch). Class weights `λ_D` and `λ_U` rebalance if you have more of one type.

**Data**: unpaired binary feedback. Thumbs-up/down logs from a deployed product fit naturally. Roughly: collect `(prompt, completion, was_thumbed_up)` and you're done.

**Failure modes**:
- Class imbalance: if `λ_D` and `λ_U` aren't tuned, the majority class dominates.
- Reference KL estimate is noisy at small batch size.
- The implicit reward is per-example, so noisy single-example labels hurt more than in DPO where the pair gives some anchoring.

**Code sketch**:

```python
def kto_loss(logp, ref_logp, is_desirable, z_ref, beta=0.1,
             lambda_d=1.0, lambda_u=1.0):
    # logp, ref_logp: sums over response tokens. z_ref: scalar KL anchor.
    delta = beta * (logp - ref_logp) - z_ref
    desirable_loss   = lambda_d * (1 - torch.sigmoid(delta))
    undesirable_loss = lambda_u * (1 - torch.sigmoid(-delta))
    return torch.where(is_desirable, desirable_loss, undesirable_loss).mean()
```

(The published KTO loss has additional bookkeeping for how `z_ref` is computed within a minibatch. Treat the sketch as the shape, not the exact algorithm — check §4 of the paper.)

---

## ORPO (Hong 2024, arXiv:2403.07691)

Folds SFT and preference alignment into one loss. **No reference model.** The SFT term provides the anchor; the odds-ratio term sharpens the gap between chosen and rejected.

**Loss**:

```
L_ORPO = L_SFT(y_w | x)  +  λ · L_OR
L_OR   = -log σ( log( odds(y_w | x) / odds(y_l | x) ) )
odds(y | x) = π_θ(y|x) / (1 − π_θ(y|x))
```

In practice the odds are computed in log-space using `log(1 − exp(logp))` (numerically stable via `log1mexp`).

**Data**: paired preferences. But you skip the SFT stage — ORPO does both at once.

**Failure modes**:
- No KL anchor → policy can drift if `λ` is too high or SFT signal is weak.
- Numerical issues in `log(1 − exp(logp))` when `logp` is near zero.
- Length bias: longer chosen responses inflate `log π_θ(y_w)` and look better.

**Code sketch**:

```python
def orpo_loss(logp_chosen, logp_rejected, chosen_labels_logp, lam=0.1):
    # chosen_labels_logp = SFT cross-entropy loss on y_w
    sft_loss = -chosen_labels_logp.mean()

    # log-odds: log(p / (1-p)) = logp - log1mexp(logp)
    log_odds_w = logp_chosen   - log1mexp(logp_chosen)
    log_odds_l = logp_rejected - log1mexp(logp_rejected)
    or_loss = -F.logsigmoid(log_odds_w - log_odds_l).mean()

    return sft_loss + lam * or_loss
```

---

## SimPO (Meng 2024, arXiv:2405.14734)

Length-normalized DPO without a reference model. Replaces `log π_θ(y|x)` with the **average** per-token log-probability and introduces a target reward margin `γ`.

**Loss**:

```
L_SimPO = -E[ log σ( (β/|y_w|) · log π_θ(y_w|x)
                   − (β/|y_l|) · log π_θ(y_l|x)
                   − γ ) ]
```

where `|y|` is the number of response tokens. The length normalization is what kills the bias toward longer responses that vanilla DPO exhibits; `γ` (typical 0.5–1.5) sets the minimum margin the model has to clear.

**Data**: paired preferences. No reference model needed.

**Failure modes**:
- `γ` too low → margin collapses; model can't distinguish chosen from rejected.
- No reference → policy can drift far from the SFT prior; some catastrophic forgetting reports.
- The length norm assumes preferences aren't actually about length; for tasks where length is a real signal (e.g. detailed explanations), this hurts.

**Code sketch**:

```python
def simpo_loss(logp_chosen, len_chosen, logp_rej, len_rej, beta=2.0, gamma=1.0):
    # logp_chosen, logp_rej: sums of token log-probs over the response.
    # len_chosen, len_rej:   response token counts (excluding prompt).
    chosen = (beta / len_chosen) * logp_chosen
    rej    = (beta / len_rej)    * logp_rej
    return -F.logsigmoid(chosen - rej - gamma).mean()
```

---

## GRPO (Shao 2024, arXiv:2402.03300)

PPO without a critic. Sample K completions per prompt, score them, use the group's mean as the baseline. Connect it back to REINFORCE-with-baseline: `b = mean(r_1..r_K)` and the group is the minibatch.

**Advantages**:

```
A_i = (r_i − mean(r_1..r_K)) / (std(r_1..r_K) + eps)        # i = 1..K, per prompt
```

**Loss** (per completion, then averaged):

```
ratio_i = exp( log π_θ(y_i|x) − log π_old(y_i|x) )
L_CLIP_i = -min( ratio_i · A_i,  clip(ratio_i, 1-ε, 1+ε) · A_i )
L_KL     = β · ( log π_θ(y_i|x) − log π_ref(y_i|x) )
L_GRPO   = mean_i ( L_CLIP_i + L_KL_i )
```

The reward `r_i` is a single scalar per completion. All tokens in `y_i` share the same advantage — there is no per-token credit assignment.

**Data**: prompts plus a scorer. The scorer can be a learned `r_φ`, a rule (math correctness, code tests), or an LLM judge. K is typically 4–16.

**Failure modes**:
- Baseline collapse: all K completions score the same → `A_i ≈ 0` → no gradient. Common when the model can't solve the problem at all (all wrong) or has memorized it (all right). Curriculum problems with 10–80% pass rate work best.
- Entropy collapse: once the model finds one strategy that works, exploration dies. Add an entropy bonus or watch entropy explicitly.
- KL drift: `β` too low → policy walks away from the reference.

**Code sketch** (advantage + loss; rollout omitted):

```python
def group_advantages(rewards, eps=1e-8):
    # rewards: [B, K] — one scalar per completion per prompt
    mean = rewards.mean(dim=1, keepdim=True)
    std  = rewards.std(dim=1, keepdim=True) + eps
    return (rewards - mean) / std

def grpo_loss(logp_cur, logp_old, logp_ref, adv, clip_eps=0.2, kl_beta=0.04):
    # All log-prob tensors are [B, K], summed over response tokens.
    ratio = torch.exp(logp_cur - logp_old.detach())
    surr1 = ratio * adv
    surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv
    policy_loss = -torch.min(surr1, surr2).mean()
    kl = (logp_cur - logp_ref.detach()).mean()
    return policy_loss + kl_beta * kl
```

See [`../lectures/12-beyond-dpo.md`](../lectures/12-beyond-dpo.md) and [`../lectures/15-rl-verifiable-rewards.md`](../lectures/15-rl-verifiable-rewards.md).

---

## RLVR / R1-style GRPO (DeepSeek-R1, arXiv:2501.12948)

Same GRPO algorithm, with the reward coming from a **rule-based verifier** instead of a learned `r_φ`. Two reward components:

```
reward(y, y*) = format_reward(y) + correctness_reward(y, y*)
```

- `format_reward`: did the model emit `<think>...</think>` followed by an answer? (Usually 0 or 0.1.)
- `correctness_reward`: does the extracted answer match `y*`? (Usually 0 or 1.0 — exact match on normalized numeric answer for math; pass rate on a test suite for code.)

The early phase of training is carried by the format reward — it's the only dense signal before the model can solve anything. As correctness reward grows, format reward saturates and contributes little.

**R1-Zero**: GRPO directly on a base model (no SFT). Reasoning behaviors (longer chains on harder problems, backtracking, self-check) emerged without being explicitly rewarded.

**R1**: cold-start SFT on a few thousand legible reasoning traces → GRPO → rejection sampling SFT → final RLHF pass for general helpfulness.

**Data**: prompts with verifiable answers (GSM8K, MATH, code with test suites, formal proofs).

**Failure modes**:
- Verifier hacking: model special-cases visible test inputs; gap between train and held-out reward.
- Answer matching without reasoning: model pattern-matches the question, guesses an answer in the right range, and the `<think>` block is decorative.
- Format reward too high relative to correctness → empty reasoning gets near-max reward.
- All the GRPO failure modes (baseline collapse, entropy collapse, KL drift).

**Code sketch** (reward function only; the GRPO loss is the same as above):

```python
def rlvr_reward(response: str, expected: str) -> float:
    fmt = 0.1 if has_think_tags(response) and has_answer_box(response) else 0.0
    ans = extract_answer(response)
    correct = ans is not None and numeric_equal(ans, expected)
    return fmt + (1.0 if correct else 0.0)

# Then for each prompt x with expected answer y*:
#   ys = [policy.generate(x) for _ in range(K)]
#   rs = torch.tensor([rlvr_reward(y, y_star) for y in ys])
#   adv = group_advantages(rs.unsqueeze(0))           # [1, K]
#   ... feed into grpo_loss() above ...
```

See [`../lectures/15-rl-verifiable-rewards.md`](../lectures/15-rl-verifiable-rewards.md).

---

## Decision tree — which method to reach for

Walk top to bottom, take the first match.

```
1. Does your task have a verifiable correct answer
   (math, code with tests, formal proof, exact-match)?
   ├── YES → Does the base model sometimes succeed (10–80% pass rate)?
   │         ├── YES → RLVR + GRPO
   │         │         (R1-style: format + correctness reward)
   │         └── NO  → Bootstrap first with STaR / ReST or SFT on
   │                   curated solutions, then GRPO.
   └── NO  → continue

2. What data do you have?
   ├── Paired preferences (chosen / rejected)
   │   ├── Big budget, want max perf, can run a reward model
   │   │      → PPO-RLHF
   │   ├── Small budget, want simple                → DPO
   │   ├── DPO is overfitting / overconfident       → IPO
   │   ├── No reference model loaded, want simpler  → SimPO
   │   └── Want to skip the SFT stage               → ORPO
   │
   ├── Binary labels only (thumbs up/down per response)
   │      → KTO
   │
   ├── Multiple ranked / scored completions per prompt
   │   (objective metric available)
   │      → GRPO (with whatever scorer you have)
   │
   └── No structured feedback yet
          → Collect preferences first, then DPO; OR
            use an LLM judge to label pairs and run DPO/SimPO.
```

A few common situations spelled out:

- **"I have human preference pairs and a small budget"** → DPO.
- **"I have lots of human preference data and compute"** → PPO-RLHF.
- **"I have a verifier (code / math / format)"** → GRPO with verifiable reward (RLVR).
- **"I have only positive examples and a few negatives"** → KTO (treat the positives as desirable, negatives as undesirable).
- **"I don't have a reference model loaded"** → ORPO or SimPO.
- **"My DPO model is overconfident and outputs are degrading"** → IPO, or lower `β`, or add length normalization (SimPO).
- **"I want chain-of-thought reasoning to emerge"** → R1-style RLVR + GRPO on a domain with a checker.
- **"I have a domain with no checkable answers (writing, dialogue, style)"** → DPO or PPO-RLHF. RLVR doesn't apply.
- **"I have thumbs-up/thumbs-down from a deployed product"** → KTO.

---

## At-a-glance shape of the losses

For quick recall. All losses minimize what's written.

```
PPO-RLHF :  -E[ min( ρ · A,  clip(ρ, 1-ε, 1+ε) · A ) ]      ρ = π_θ/π_old
DPO      :  -E[ log σ( β · ( Δ_w - Δ_l ) ) ]                 Δ = log π_θ - log π_ref
IPO      :   E[ ( (Δ_w - Δ_l) - 1/(2β) )^2 ]
KTO      :  per-example value function in prospect-theory shape, anchored to z_ref
ORPO     :  L_SFT(y_w)  +  λ · -log σ( log OR(y_w) - log OR(y_l) )
SimPO    :  -E[ log σ( (β/|y_w|) log π_θ(y_w) - (β/|y_l|) log π_θ(y_l) - γ ) ]
GRPO     :  -E[ min( ρ · A_i,  clip(ρ, 1-ε, 1+ε) · A_i ) ] + β · KL(π_θ || π_ref)
              with A_i = (r_i - mean_K) / (std_K + eps)
RLVR     :  same as GRPO; r_i comes from a rule-based checker, not r_φ
```

---

## Hyperparameter starting points

Defaults from the lecture notes. Tune from here.

| Method | β (or equiv) | Other | Notes |
|---|---|---|---|
| PPO-RLHF | β = 0.1 | ε = 0.2, vf_coef = 0.1, lr = 1e-6 | InstructGPT-style; β is the KL coefficient |
| DPO | β = 0.1 | lr = 1e-6 (AdamW) | One real knob; range 0.1–0.5 |
| IPO | β = 0.1 | lr = 1e-6 | β sets the target margin `1/(2β)`, not loss steepness |
| KTO | β = 0.1 | λ_D = λ_U = 1.0; tune for class imbalance | |
| ORPO | λ = 0.1–0.5 | lr = 5e-6 | No reference model; SFT cross-entropy is the anchor |
| SimPO | β ≈ 2.0, γ ≈ 1.0 | lr = 1e-6 | β is different scale from DPO because of length normalization |
| GRPO | β = 0.04, ε = 0.2 | K = 4–16, lr = 1e-6 | Smaller β than DPO because gradient signal is noisier |
| RLVR | β = 0.04, ε = 0.2 | K = 4–16, lr = 1e-6 | Same as GRPO; verifier replaces reward model |

`lr` here is the policy learning rate. Reward-model training and value-head training use higher lrs (1e-5 to 1e-4) when they apply.

---

## What you'll see go wrong (quick diagnoses)

| Symptom | Likely cause | First thing to try |
|---|---|---|
| Reward climbs, outputs degrade | Reward hacking (PPO, GRPO, RLVR) | Raise β; private held-out scorer; check for repetition / keyword stuffing |
| KL > 100, model "forgot how to write" | β too small | Raise β; lower lr; restart from checkpoint |
| DPO reward margin stays at 0 | β too small or ref model unfrozen | Raise β to 0.1; check `requires_grad` on ref |
| GRPO advantages all ~0 | All K completions scored the same | Add harder problems; raise K; check for verifier-saturation |
| Entropy drops sharply mid-training | Mode collapse | Add entropy bonus (~0.001–0.01); lower lr |
| SimPO outputs collapse | γ too low | Raise γ to 1.0+; consider IPO if instability persists |
| Loss is NaN | log(1 - exp(logp)) blew up (ORPO); std=0 (GRPO) | Use `log1mexp`; add `eps` to `std` |
| Training reward up, held-out reward flat | Verifier hacking (RLVR) or RM overfit (PPO) | Hold out problems from the reward checker; rotate RM training data |

**The bug that bites everyone**: forgetting to mask prompt tokens when summing log-probs over the response. Symptom: the model "learns" to repeat the prompt verbatim. Fix:

```python
prompt_len = len(tokenizer(prompt).input_ids)
response_logp = per_token_logp[prompt_len:].sum()      # response only
```

---

## What's not in here

- **PPO for non-LLM RL** (Atari, MuJoCo, etc.) → [`./RL-Quick-Reference.md`](./RL-Quick-Reference.md).
- **Reward model architecture and training** → [`../lectures/09-reward-modeling.md`](../lectures/09-reward-modeling.md).
- **Constitutional AI / RLAIF** (replacing human labels with AI labels) → [`../lectures/27-rlaif.md`](../lectures/27-rlaif.md).
- **RRHF** (Yuan 2023, arXiv:2304.05302) — listwise ranking loss; covered briefly in [`../lectures/12-beyond-dpo.md`](../lectures/12-beyond-dpo.md), omitted here because it's largely subsumed by GRPO in practice.
- **Process reward models** (Lightman 2023, arXiv:2305.20050) → [`../lectures/15-rl-verifiable-rewards.md`](../lectures/15-rl-verifiable-rewards.md).
- **Reward hacking theory** (Gao, Schulman, Hilton 2022, arXiv:2210.10760) → referenced in [`../lectures/12-beyond-dpo.md`](../lectures/12-beyond-dpo.md).

---

## Citations (verified against the lecture notes and arXiv IDs)

- **PPO-RLHF / InstructGPT** — Ouyang et al. 2022, "Training language models to follow instructions with human feedback." arXiv:2203.02155. NeurIPS 2022.
- **DPO** — Rafailov et al. 2023, "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." arXiv:2305.18290. NeurIPS 2023.
- **IPO** — Azar et al. 2023, "A General Theoretical Paradigm to Understand Learning from Human Preferences." arXiv:2310.12036.
- **KTO** — Ethayarajh et al. 2024, "KTO: Model Alignment as Prospect Theoretic Optimization." arXiv:2402.01306. ICML 2024.
- **ORPO** — Hong et al. 2024, "ORPO: Monolithic Preference Optimization without Reference Model." arXiv:2403.07691. EMNLP 2024.
- **SimPO** — Meng et al. 2024, "SimPO: Simple Preference Optimization with a Reference-Free Reward." arXiv:2405.14734.
- **GRPO / DeepSeekMath** — Shao et al. 2024, "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." arXiv:2402.03300.
- **DeepSeek-R1** — DeepSeek-AI 2025, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv:2501.12948.

The exact functional forms of IPO and KTO above are simplifications of the published losses (different parameterizations are equivalent). Before relying on the constants in production code, verify against §4 of each paper.
