<!-- status: unreviewed | last-reviewed: never -->

# Lecture 15: RL with Verifiable Rewards and Reasoning Models

_Unreviewed — no one has checked this end to end. Treat the math, code, and citations as unverified._

**Time**: ~3-4 h · **Prerequisites**: Lectures 02, 06, 12

---

## Where this fits

Lectures 10–12 covered the standard RLHF pipeline: train a reward model on human preference comparisons, then run PPO against it. The reward model is a learned approximation of "what humans prefer," and its quality bottlenecks the whole system. Reward hacking, overoptimization, and distribution shift in the reward model are all real failure modes covered in those lectures.

In some domains you can skip the learned reward model entirely. If the task has a verifiable correct answer — a math problem with a numeric result, a coding problem with a test suite, a formal proof checked by a theorem prover — you can evaluate each response directly and hand that binary or scalar signal to the RL algorithm. This is RL with verifiable rewards, or RLVR.

RLVR is not a fundamentally new algorithm. The RL machinery (policy gradient, KL penalties, clipping) is the same as in [Lecture 06: PPO](./06-ppo.md) and [Lecture 12: Beyond DPO](./12-beyond-dpo.md). What changes is the reward source: a checker replaces a learned model. That single substitution removes several failure modes and, in the 2024–2025 period, produced the most visible jump in reasoning capability of any alignment technique.

The verifiable domains that matter most in practice are mathematical reasoning (where a symbolic solver or regex can check numeric answers), code generation (where a test harness can execute completions), and formal proofs (where a proof assistant like Lean or Coq can verify each step mechanically). These domains share a property: the ground truth is computable, not just human-judged. That computability is what makes the reward signal stable.

---

## Verifiable rewards

### The reward structure in practice

A typical RLVR reward function has two components.

The first is a **correctness reward**: the checker says the answer is right or wrong. For math, this is usually exact match on a normalized numeric answer (strip units, strip formatting, compare floats). For code, it's pass rate on a private test suite. Binary is simplest; a partial-credit scalar is possible but introduces its own calibration questions.

The second is a **format reward**: the model used the expected output structure. In DeepSeek-R1 and related work, the model is expected to emit a `<think>...</think>` block followed by an answer, and the format reward confirms that structure is present and well-formed.

The format reward matters early in training for a specific reason. Before the policy learns to produce correct answers often enough to get meaningful correctness signal, the format reward is the only dense signal available. A model that produces badly-structured output can't be scored for correctness reliably anyway, so training the format first creates the scaffold for the correctness signal to land. Once the policy reliably produces structured output, the format reward contributes little (its variance drops to near zero) and the correctness reward carries the learning.

A combined reward for one response might look like:

```python
def rlvr_reward(response: str, expected_answer: str) -> float:
    format_ok = has_think_tags(response) and has_answer_box(response)
    format_reward = 0.1 if format_ok else 0.0

    extracted = extract_answer(response)
    correct = extracted is not None and numeric_equal(extracted, expected_answer)
    correctness_reward = 1.0 if correct else 0.0

    return format_reward + correctness_reward
```

The weights are illustrative — different implementations vary, and the exact schedule (whether to anneal the format component) is an engineering choice.

### Reward hacking

Any reward function can be gamed, and RLVR is no exception. The gameable surface here is the verifier, not a learned model.

Common failure modes:

- **Weak test suites for code**: a model learns to special-case the visible test inputs rather than solving the problem generally. Pass rate goes to 1.0 on the training tests, 0.5 on held-out. This is the same problem as overfitting to a validation set but at the reward level.
- **Answer-matching without reasoning**: the model learns to emit the correct string in the answer box by pattern-matching the question format, without the intermediate reasoning being causally connected to the answer. This can happen when the dataset has systematic patterns (e.g., answers in arithmetic problems cluster by problem type).
- **Format exploitation**: generating very long `<think>` blocks even when reasoning is trivially short, if length correlates with score somewhere in the pipeline.

The practical mitigation is to hold out a test set of problems from the reward checker's view, just as you hold out test data in supervised learning. Using a stronger and more private verifier (e.g., a separate symbolic checker rather than string-matching) also helps.

A more subtle form of reward hacking: the model learns to produce correct-looking structure without doing any reasoning. For math, this can look like:

```
<think>
Let me solve this problem about the ratio of apples to oranges.
The answer is probably a whole number between 1 and 10.
I'll guess 4.
</think>
Answer: 4
```

If 4 happens to be correct (and on short arithmetic problems it sometimes is, given small integer answers), this string gets full reward. The model is essentially doing shallow pattern-matching on the problem type, not reasoning. Distinguishing this from genuine reasoning at training time is hard — it requires evaluating whether the chain is causally connected to the answer, which is expensive to automate. One proxy: hold out a set of problems where the answer distribution doesn't match training (e.g., where typical problem types have unusual answers) and measure whether reward degrades. If it does, the model was surface-matching, not reasoning.

---

## GRPO

### Background and motivation

Standard PPO ([Lecture 06](./06-ppo.md)) needs a value function (critic) that estimates expected return from each state. In LLM settings, the "state" is the full context so far, and the action space is the vocabulary. Training a separate critic network that matches the policy network in scale is expensive — roughly doubling memory and compute. You need to maintain the reference policy, the active policy, the reward model, and the critic simultaneously.

GRPO (Group Relative Policy Optimization), introduced in the DeepSeekMath paper (Shao et al. 2024, arXiv:2402.03300), removes the critic. The insight is that for a given prompt, you can approximate the baseline needed for advantage estimation by sampling multiple completions and using their mean reward as the baseline. No separate network required.

This is essentially a return to the REINFORCE-with-baseline structure from [Lecture 02: Policy Gradients](./02-policy-gradients.md). The difference is that the baseline is not a global average or a learned value function — it's the within-group mean for that specific prompt. Because you're computing advantages relative to other responses to the same question, you automatically control for prompt difficulty. A hard problem where everyone scores 0.1 and an easy problem where everyone scores 0.9 both produce similar advantage distributions; the policy is pushed toward what works better than the group average on each prompt.

### The algorithm

For a batch of prompts, GRPO does the following:

1. For each prompt `x`, sample K completions from the current policy: `y_1, y_2, ..., y_K`.
2. Score each completion with the reward function: `r_1, r_2, ..., r_K`.
3. Compute the group-relative advantage for each completion:

```
A_i = (r_i - mean(r_1..r_K)) / (std(r_1..r_K) + eps)
```

4. Treat these advantages as fixed (no gradient through them) and compute the PPO-style clipped objective.
5. Add a KL penalty to the reference policy (the original SFT model or the initial policy before RL began).

### The loss

For a completion `y_i` of length `T` tokens, the log-probability under a policy is the sum of per-token log-probs:

```
log pi(y_i | x) = sum_{t=1}^{T} log pi(y_i[t] | x, y_i[1..t-1])
```

Let `pi_theta(y_i | x)` be the current policy's log-probability of completion `y_i`, and `pi_old(y_i | x)` be the policy that was used to sample (the "behavior policy" for the batch). The probability ratio is:

```
r_i(theta) = exp(log pi_theta(y_i | x) - log pi_old(y_i | x))
```

This ratio is a single scalar per completion — it compresses the whole sequence into one number. Note that for long completions, even small per-token differences in log-prob can compound into large ratios, which is one reason the clipping is necessary.

The GRPO objective per sample:

```
L_CLIP_i = min(r_i * A_i,  clip(r_i, 1 - eps, 1 + eps) * A_i)
```

The clipping works the same way as in standard PPO: if the advantage is positive (this completion was better than the group average), the objective is capped so the policy can't increase `r_i` beyond `1 + eps` to claim extra credit. If the advantage is negative, the policy can't decrease `r_i` below `1 - eps` to get extra penalty credit. In both cases the clip is conservative — it limits how much the update can move in a direction the advantage supports.

The full policy loss averages over all K completions and all B prompts in the batch:

```
L_CLIP = -(1 / (B * K)) * sum_b sum_i L_CLIP_i(b, i)
```

(negated because we maximize the surrogate objective by minimizing the loss).

The KL term penalizes deviation from the reference policy `pi_ref`:

```
L_KL = beta * (1 / (B * K)) * sum_b sum_i (log pi_theta(y_i|x_b) - log pi_ref(y_i|x_b))
```

An alternative formulation (used in some implementations) computes the KL token-by-token and sums over the completion length before averaging over batch and group. This gives the KL a scale that grows with sequence length, which some practitioners prefer because it makes `kl_beta` more comparable across different completion lengths.

Total loss:

```
L_GRPO = L_CLIP + L_KL
```

The original DeepSeekMath paper includes an additional term that averages the KL computed per-token over the completion length, so that longer completions don't contribute disproportionately more to the KL than shorter ones. Whether to use sequence-level or token-level KL is an implementation choice; both are defensible.

### Contrast with PPO

| Aspect | PPO | GRPO |
|---|---|---|
| Advantage estimate | `r + gamma * V(s') - V(s)` (GAE) | `(r_i - group_mean) / group_std` |
| Critic network | Required | Not needed |
| Memory | Policy + critic + ref + RM | Policy + ref (+ RM if used) |
| Baseline type | Learned value function | Within-group mean reward |
| Gradient through baseline | Yes (critic trains) | No (baseline is a constant) |

GRPO's advantage estimate has higher variance than a well-trained critic because the group mean from K samples is a noisy estimate of the true expected return. In practice K is set to 4–16 for language model training; larger K reduces variance but multiplies forward-pass cost. On math tasks the binary nature of the reward means most variance reduction comes from having diverse completions (some correct, some not), which happens naturally when the problem is in the right difficulty range for the model.

A subtle point: the PPO critic in standard RLHF estimates `V(s_t)` at every token step, giving per-token advantages via GAE. GRPO instead assigns a single reward to the whole completion and treats all tokens in that completion as equally responsible for it. This is the outcome reward assumption built into the algorithm — the whole sequence is credited or debited as a unit. Per-token credit assignment is not attempted. That's both a simplification (no critic needed) and a limitation (tokens early in the completion get the same advantage as tokens late in the completion, even though early tokens might not be causally important for whether the answer is correct).

### The connection back to lecture 02

REINFORCE ([Lecture 02](./02-policy-gradients.md)) optimizes:

```
gradient = E[ (sum_t log pi(a_t | s_t)) * (R - b) ]
```

where `R` is the episode return and `b` is a baseline subtracted to reduce variance. GRPO is this formula applied to whole-completion generation, with `R = r_i` (the reward from the verifiable checker) and `b = mean(r_1..r_K)` (the within-group mean). The group is your minibatch of rollouts from the same prompt; the baseline is their average.

The PPO-style clipping on top is the modification from Lecture 06: it bounds how much the policy ratio can deviate from 1 before the objective stops improving. Without clipping, large gradient steps could destabilize training. With clipping, the update is conservative.

So the full GRPO loss is: REINFORCE baseline estimator (from Lecture 02) + PPO clipping (from Lecture 06) + KL penalty to reference (from Lecture 10/12) − critic (removed, replaced by group mean). That's the whole algorithm.

### A reference implementation

The following code shows the core of a GRPO update — the advantage computation and the loss. It assumes you have a scoring function and can compute log-probabilities under the policy and reference policy. This is the reference for the `exercises/15-grpo-rlvr/` exercise set (planned).

```python
import torch
import torch.nn.functional as F


def compute_group_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute group-relative advantages.

    Args:
        rewards: shape [batch, K] — reward for each of K completions per prompt.
        eps: stability constant for std normalization.

    Returns:
        advantages: shape [batch, K], zero-mean and unit-std within each group.
    """
    group_mean = rewards.mean(dim=1, keepdim=True)          # [batch, 1]
    group_std = rewards.std(dim=1, keepdim=True) + eps      # [batch, 1]
    advantages = (rewards - group_mean) / group_std          # [batch, K]
    return advantages


def grpo_loss(
    log_probs_current: torch.Tensor,   # [batch, K]  — current policy, with grad
    log_probs_old: torch.Tensor,       # [batch, K]  — behavior policy, no grad
    log_probs_ref: torch.Tensor,       # [batch, K]  — reference policy, no grad
    advantages: torch.Tensor,          # [batch, K]  — from compute_group_advantages
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
) -> tuple[torch.Tensor, dict]:
    """
    GRPO loss: PPO-clipped objective with group-relative advantages,
    plus a KL penalty to the reference policy.

    Returns the scalar loss and a dict of metrics.
    """
    # Probability ratio: pi_current / pi_old
    ratio = torch.exp(log_probs_current - log_probs_old.detach())

    # PPO clipped surrogate (maximize, so negate at the end)
    surr_unclipped = ratio * advantages
    surr_clipped   = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss    = -torch.min(surr_unclipped, surr_clipped).mean()

    # KL penalty: E[log pi_current - log pi_ref]
    # This is an approximate forward KL; the sign encourages staying near pi_ref.
    kl = (log_probs_current - log_probs_ref.detach()).mean()
    kl_loss = kl_beta * kl

    loss = policy_loss + kl_loss

    metrics = {
        "policy_loss": policy_loss.item(),
        "kl": kl.item(),
        "mean_ratio": ratio.mean().item(),
        "clip_fraction": ((ratio - 1).abs() > clip_eps).float().mean().item(),
    }
    return loss, metrics


# ── Toy end-to-end example ──────────────────────────────────────────────────
#
# Suppose we have a batch of 4 prompts, each with K=4 completions.
# Rewards come from a verifiable checker (e.g., does the numeric answer match?).

def demo_grpo_step():
    torch.manual_seed(0)
    batch_size, K = 4, 4

    # Dummy log-probs (in practice, computed by forward-passing the policy
    # over each prompt+completion pair and summing token log-probs).
    log_probs_current = torch.randn(batch_size, K, requires_grad=True)
    log_probs_old     = log_probs_current.detach() + 0.05 * torch.randn(batch_size, K)
    log_probs_ref     = log_probs_current.detach() + 0.10 * torch.randn(batch_size, K)

    # Binary rewards from a verifiable checker: 1.0 = correct, 0.0 = wrong.
    # Each row is one prompt; columns are K different completions.
    rewards = torch.tensor([
        [1.0, 0.0, 1.0, 0.0],   # prompt 0: half correct
        [0.0, 0.0, 0.0, 1.0],   # prompt 1: one correct
        [1.0, 1.0, 1.0, 0.0],   # prompt 2: mostly correct
        [0.0, 0.0, 0.0, 0.0],   # prompt 3: all wrong — zero gradient (see note)
    ])

    advantages = compute_group_advantages(rewards)
    print("Advantages:\n", advantages)
    # Prompt 3 advantages are all (0-0)/std ≈ 0 — correct, no learning signal.
    # That's expected: if every completion is equally (wrong), we can't tell
    # which direction to move.

    loss, metrics = grpo_loss(
        log_probs_current, log_probs_old, log_probs_ref, advantages
    )
    print("Loss:", loss.item())
    print("Metrics:", metrics)

    loss.backward()
    print("Gradient norm:", log_probs_current.grad.norm().item())


if __name__ == "__main__":
    demo_grpo_step()
```

A few things to notice in the toy example. When all K completions score identically (prompt 3 above, all zeros), the group standard deviation is zero (or near-zero, caught by eps), and all advantages collapse to zero. There is no gradient. This is correct behavior: if the model can't produce any correct answers, reward-relative training can't tell it which direction to go. In practice this means you need problems where the model sometimes succeeds — typically 10–80% pass rate. Problems where the model never gets the right answer, or always does, provide no gradient.

Also notice that `log_probs_current` is the only tensor with `requires_grad=True`. The advantages, `log_probs_old`, and `log_probs_ref` are all treated as fixed constants — they were computed without gradients (or detached). Gradients only flow through the current policy parameters. This is important: the baseline (the group mean) does not backpropagate, which is what makes GRPO cheap. In PPO, the critic also backpropagates, adding a second optimization problem on top of the policy update.

In a real LLM training loop, `log_probs_current` would be computed by a forward pass of the policy model over the concatenated prompt+completion, summing log-softmax values at the response token positions. The `log_probs_old` values would be computed once (before the update step) and cached, and the `log_probs_ref` values would come from a frozen copy of the initial model. The `rewards` tensor comes from running each completion through the verifiable checker — in math, calling a symbolic solver or a regex-based answer extractor; in code, running the completion through a sandboxed test harness.

---

## The DeepSeek-R1 recipe

### R1-Zero: pure RL from a base model

DeepSeek-R1-Zero (DeepSeek-AI 2025, arXiv:2501.12948) applies GRPO directly to a base language model — not an SFT-tuned instruction-following model. The reward function is rule-based: correctness (exact match on normalized answer) plus the format reward for `<think>...</think>` structure.

No curated reasoning demonstrations. No chain-of-thought SFT warm-up. Just RL on a base model.

What emerged: the model spontaneously learned to produce longer reasoning chains on harder problems, to backtrack and reconsider intermediate steps (the "aha moment" described in the paper, where the model writes something like "wait, let me reconsider"), and to self-check its answers before committing. These behaviors were not explicitly rewarded — they emerged because they turned out to be instrumentally useful for getting the correctness reward.

R1-Zero also developed failure modes. Language mixing (switching between languages mid-reasoning), low readability, and occasionally circular or repetitive chains. These are downstream of training a base model without format SFT — the model has no strong prior on what a legible output looks like, so the format reward alone is insufficient to enforce coherent structure.

A second observation from the paper: reasoning chain length increased over training. The model learned to allocate more tokens to harder problems and fewer to easier ones, without any explicit length reward. This happened because longer reasoning chains were, on average, more likely to reach correct answers on the hard problems in the training set. Token length became instrumentally correlated with correctness, so the policy naturally extended chains where extension helped. This emergent length-scaling is part of what people refer to as "test-time compute scaling" — the model itself learned to use inference compute adaptively.

### R1: cold-start SFT + RL + final alignment

DeepSeek-R1 adds stages around the RL training to address R1-Zero's failure modes:

1. **Cold-start SFT**: a small dataset of high-quality reasoning traces (few thousand examples) is used for a light SFT pass before RL begins. This gives the model a prior on what legible reasoning looks like, avoiding R1-Zero's language mixing and readability failures. The traces include worked solutions with explicit `<think>` blocks and follow a consistent format.

2. **RL with GRPO**: same reward function as R1-Zero, but starting from the SFT-initialized policy. The reasoning capability sharpens substantially more quickly than R1-Zero because the policy starts from a better initial point.

3. **Rejection sampling + SFT + RLHF pass**: after the main RL stage, the policy is used to generate reasoning traces on a broader set of tasks — math, code, writing, instruction following. Correct or high-quality outputs are filtered and combined into a second SFT dataset. A second SFT pass trains the model on this combined data. A final RLHF stage (using a learned reward model over human preferences) then aligns output style for general helpfulness and safety.

The resulting model matches or exceeds OpenAI o1-class performance on math and coding benchmarks while being fully open-weight.

A useful way to read the pipeline schematically:

```
Base model
    │
    ▼ [cold-start SFT — format + reasoning traces, ~thousands of examples]
    │
    ▼ [GRPO RL — correctness + format reward, many steps]
    │
    ▼ [rejection sampling — generate at scale, filter correct, collect diverse data]
    │
    ▼ [second SFT — math + code + general capabilities]
    │
    ▼ [RLHF — learned reward model, preference data, PPO or similar]
    │
  R1 (deployed)
```

The R1 paper also reports that reasoning-capable behavior can be distilled from R1 back into smaller models (1.5B, 7B, etc.) through SFT on R1-generated traces. The distilled models perform better than models trained from scratch with RL at the smaller scale. This suggests that for small models, distilling from a large reasoning model is more compute-efficient than running GRPO independently.

### The "RL amplifies the base model" interpretation

A common takeaway from R1-Zero: the reasoning behaviors that emerge from RL were already latent in the base model. RL did not teach the model to reason from nothing — it selected and amplified reasoning patterns that the base model could already produce, pushing them from occasional to systematic.

This matters for expectations. A base model trained on essentially no mathematical reasoning data will not suddenly reason well after GRPO training. The RL training adjusts the distribution of what the model produces; it cannot add capabilities that are entirely absent. The quality of the pre-training corpus and the scale of the base model set the ceiling.

---

## o1 and inference-time scaling

OpenAI's o1 (released September 2024) has no public technical paper. The official blog post ("Learning to reason with LLMs") describes RL on chain-of-thought and the observation that performance improves as the model is given more time to "think" — more tokens in the reasoning trace — before producing its final answer. The blog post explicitly connects o1 performance to training compute and to inference compute as two separable levers.

What's publicly observable:

- o1 produces structured, extended reasoning traces before answering.
- Performance on math and coding benchmarks scales with inference token budget (longer traces do better, up to a point).
- o1 models are slower and more expensive per query than equivalently-sized models without extended reasoning.

What's not publicly documented: the exact training algorithm, the reward function structure, whether a learned process reward model is used at inference time, and how the inference compute is allocated (fixed budget, adaptive, search-based).

The general principle underneath — **test-time compute as a lever** — is well-established in the literature independently of o1. For any task where you can verify candidate answers:

- **Pass@K with voting**: generate K completions, take the majority answer. Performance increases with K. This is the cheapest version of inference-time compute.
- **Best-of-N with a reward model**: generate N completions, score with a reward model, return the top-scoring one. Requires a reliable scorer.
- **Beam search over reasoning steps**: maintain a beam of partial reasoning traces, score each step with a process reward model, prune low-score branches. Requires a PRM.
- **MCTS over proof steps**: for formal domains (theorem proving), use Monte Carlo tree search with a learned policy and value function. Used in AlphaProof.

The lesson that carries across all of these: a model that can sometimes reach the right answer, combined with a way to identify which attempt succeeded, can be made arbitrarily more reliable by sampling more. This is complementary to training-time RL — RL improves the per-sample quality; inference compute multiplies the chances of getting a good sample.

The interaction with RLVR is direct: the verifier you train against is also usable at inference time to identify good completions. A model trained with RLVR already knows "correctness is verifiable here," and the verifier is a natural scoring function for inference-time search. Models that were not trained with RLVR (e.g., models trained only on preference data) don't have this as readily because there's no external checker to identify the best completion.

---

## Process vs outcome reward models

RLVR with a rule-based checker is a pure outcome signal: the model gets feedback only after the complete answer, and only on whether the final answer is right. This is simple and effective when answers are short and verifiers are reliable.

Two papers give context for when outcome supervision falls short.

Uesato et al. 2022 (arXiv:2211.14275) ran the first systematic comparison of process-based and outcome-based feedback on GSM8K math problems. Their finding: outcome supervision produces similar final-answer accuracy with less annotation cost. However, outcome-supervised models make more reasoning errors that happen to cancel out — wrong steps that lead to the right answer. Process supervision is necessary when you care about the correctness of intermediate steps, not just the final answer.

Lightman et al. 2023 (arXiv:2305.20050) trained a process reward model (PRM) that scored each reasoning step rather than only the final answer, using human-annotated step-level labels on the MATH benchmark. Their process-supervised model substantially outperformed outcome-supervised models at the same scale. The PRM allows verification at each step, which is useful both for training (denser reward signal) and for inference-time search (prune branches with low step scores).

A **process reward model** scores individual reasoning steps; an **outcome reward model** scores the final answer. RLVR with a rule-based checker is outcome supervision — the checker only knows if the last answer is correct. PRMs require human-labeled step data (expensive) or synthetic labels from a verifier that can evaluate partial solutions (limited to domains with that structure, like formal proofs).

The practical tradeoff: outcome supervision is cheap and works well when correct reasoning usually leads to correct answers. Process supervision is more expensive to collect but provides a richer training signal and can detect subtler reasoning failures.

### A concrete example of where they diverge

Consider a multi-step algebra problem. An outcome-supervised model might learn to produce:

```
Step 1: Let x = the unknown.
Step 2: Multiply both sides by 4.   ← actually wrong
Step 3: Simplify to get x = 12.     ← happens to be the right answer because
                                        an error in step 2 cancelled an error in step 1
```

Outcome reward: 1.0 (correct final answer). The model gets rewarded for a flawed reasoning path.

A process-supervised model would score step 2 as incorrect (it doesn't follow from step 1), giving a low intermediate reward. Training against this signal discourages the erroneous chain even when it accidentally reaches the right answer.

In practice, building a PRM requires deciding what a "step" is (sentence boundary? newline? mathematical expression boundary?), collecting human labels at that granularity, and training a model to predict step quality. For the MATH dataset Lightman et al. used Amazon Mechanical Turk annotators who rated each step as correct, neutral, or incorrect. For domains without clear step structure (open-ended reasoning, creative tasks), step-level labels are hard to define and collect.

A useful middle ground: **automatic process labels** generated by a strong model or a symbolic solver. For math, you can sometimes check whether each equation follows algebraically from the previous one using a CAS (computer algebra system). This gives synthetic step-level labels without human annotation, at the cost of being limited to formally checkable steps.

---

## The cheap cousins: rejection sampling fine-tuning

Two methods produce reasoning-capable models without running full RL. They're worth understanding both as baselines and as practical alternatives when RL infrastructure is not available.

### STaR

STaR (Self-Taught Reasoner, Zelikman et al. 2022, arXiv:2203.14465) is a simple loop:

1. Sample chains of thought from the current model on a labeled dataset.
2. Keep the chains where the final answer is correct.
3. Fine-tune on those chains (standard SFT).
4. Repeat.

For problems the model fails on, a "rationalization" step generates a chain conditioned on the correct answer, which prevents the model from only learning easy examples. STaR can also generate rationales for otherwise-failing cases to ensure data diversity.

The resulting model gets better at reasoning across iterations because the training data progressively includes harder problems solved correctly. STaR is EM-like (expectation-maximization): E-step samples and filters, M-step trains.

### ReST

ReST (Reinforced Self-Training, Gulcehre et al. 2023, arXiv:2308.08998) generalizes STaR slightly. It generates a large offline dataset from the current policy, filters by reward threshold, and fine-tunes. The key distinction from online RL methods like PPO or GRPO is that the data is generated offline and reused — no live rollouts during training. This is more compute-efficient and, as Gulcehre et al. note, less susceptible to reward hacking because the reward is applied at filter time, not as a live training signal.

Both STaR and ReST are best-of-N distillation rather than policy gradient. They push the model toward correct solutions without computing gradients through the reward signal. This makes them simpler to implement but also less data-efficient for hard problems where the model rarely succeeds: if the model gets the right answer 0.1% of the time, you need enormous sample budgets to collect enough training signal.

The comparison between STaR/ReST and GRPO comes down to where the credit assignment happens. In STaR/ReST, credit assignment is binary at filter time: a chain is either kept (correct) or discarded (incorrect). There's no gradient signal about which parts of the chain were useful. In GRPO, the policy gradient flows through log-probabilities of specific tokens, giving some per-token signal (coarsely: the whole completion gets the same group-relative advantage, but at least the update direction is informed by the actual token choices). GRPO should, in theory, be more sample-efficient for hard problems because it squeezes more information out of each rollout.

A common workflow in practice: use STaR or ReST to bootstrap a reasoning policy to reasonable accuracy (say, 30–50% pass rate on the target problem set), then switch to GRPO for the final stage where the harder gradient-based optimization is worth the infrastructure cost. The cold-start SFT in R1 serves the same bootstrapping function as one round of STaR — it gets the policy to a starting point where some correctness signal is available before RL begins.

---

## Code as a verifiable domain

Code with test suites is as natural a home for RLVR as math. The reward function is: run the completion through a sandboxed test harness, return pass rate on the test cases. This is described in more detail in [Lecture 13: RLHF for Code Generation](./13-rlhf-code-generation.md).

The differences from math that matter for RLVR:

- **Partial credit is natural**: a code submission can pass 3 of 5 tests. Mapping this to a scalar reward (0.0–1.0) is straightforward, whereas math usually requires a binary threshold.
- **Test suite coverage matters more**: a math verifier checks the answer against a known value — there's no "coverage" issue. A code test suite may fail to exercise important edge cases, making reward hacking much easier. Private, unseen test cases (held out from the training reward) are even more important than in math.
- **Execution is expensive**: running code in a sandbox for each of K completions per prompt is slower than checking a numeric answer. This makes large K more costly for code than for math, and pushes practitioners toward smaller group sizes or asynchronous evaluation pipelines.

The GRPO loss and advantage computation are identical between math and code. The only difference is what produces the `rewards` tensor — and how long that computation takes.

---

## Practical training dynamics

A few patterns show up repeatedly in GRPO runs on math and code, worth knowing before you start.

**Early training**: the format reward dominates. The policy quickly learns to produce structured output (the `<think>` tags, the answer box). Correctness reward stays near zero. This phase usually ends within a few hundred steps.

**Mid training**: the correctness reward starts climbing. The policy finds some problems it can solve reliably and begins generalizing. This is also when entropy starts dropping — the model is converging toward specific reasoning strategies. Watch entropy here. If it drops sharply (say, from 3.5 to 2.0 nats per token within 500 steps), the policy may be collapsing to a narrow style.

**Late training**: correctness reward plateaus. You're in the regime where problems on the training distribution are being solved but generalization to harder problems is marginal. Adding harder problems (curriculum), increasing KL beta to slow convergence, or switching to a distillation step are all options.

### Hyperparameter guidance

| Parameter | Typical range | Effect of going higher |
|---|---|---|
| K (group size) | 4–16 | Lower variance, higher cost |
| `clip_eps` | 0.1–0.3 | Higher: more aggressive updates |
| `kl_beta` | 0.01–0.1 | Higher: stays closer to reference |
| Learning rate | 1e-6–5e-6 | Higher: faster but less stable |
| Batch size (prompts) | 64–512 | Higher: smoother gradients |

`kl_beta` is the most sensitive hyperparameter in RLVR settings. Too low and the policy drifts far from the reference, which can cause the policy to produce outputs that score well on the training verifier but are incoherent on other inputs. Too high and the policy barely moves. A common diagnostic: plot `KL(pi_theta || pi_ref)` over training. It should increase gradually and then plateau. A KL that keeps climbing after thousands of steps suggests `kl_beta` needs to be raised.

The learning rate for RLVR is usually lower than for SFT because the gradient signal is noisier (rewards are binary, and group-relative advantages depend on the sample that happened to be drawn). Starting at 1e-6 and tuning up is safer than starting at 1e-4.

---

## When to use this

A rough decision tree:

```
Does the task have a verifiable correct answer?
├── No  → Use preference-based methods: DPO, PPO+RM, RLHF.
│         RLVR will not help; there's no checker to provide signal.
└── Yes → Can you build a reliable, un-gameable verifier?
    ├── No  → Fix the verifier first, or use a PRM for step-level feedback,
    │         or fall back to SFT on curated examples.
    └── Yes → Does the base model sometimes produce correct answers?
        ├── No  → Use STaR or ReST to bootstrap first.
        │         GRPO needs some correct signal in the group; if pass rate
        │         is near 0%, there's nothing to learn from.
        └── Yes → RLVR + GRPO. Start with K=4–8, kl_beta=0.04, lr=1e-6.
```

RLVR is the right approach when:

- The task has a reliable, un-gameable verifier. Math with a symbolic checker, code with a private test suite that has good coverage, formal proofs. If your "verifier" is just string-matching or a shallow heuristic, you'll get reward hacking.
- You want reasoning behavior to emerge or sharpen. The training signal from a verifier is strong enough to shape long-horizon generation in ways that SFT alone typically doesn't.
- A learned reward model would be a bottleneck. If labeling preferences is expensive, or if the preference data doesn't cover the tail of hard problems, a rule-based verifier sidesteps those problems.

RLVR is probably not the right approach when:

- The domain is subjective. Writing quality, tone, helpfulness — these don't have checkable correct answers. Use preference-based methods (RLHF with a learned reward model, DPO).
- You don't have a reliable verifier. A weak test suite is more dangerous than no test suite, because it trains reward hacking rather than reasoning.

**Failure modes to watch for:**

- **Entropy collapse**: the policy converges to one reasoning style, losing diversity. Monitor entropy over training and add an entropy bonus term if it drops too fast. GRPO training with binary rewards is especially prone to this because a policy that always produces the same correct solution is locally optimal. The policy has no incentive to explore once it reliably gets `r=1.0` on most training problems.
- **KL overrun**: even with a KL penalty, the policy can drift far from the reference if `kl_beta` is too low. The KL should be monitored per training step; a large spike usually means the learning rate or clip epsilon is too aggressive.
- **Reward hacking on the verifier**: log training reward vs held-out reward separately from the start. A gap that grows over time is evidence of hacking.
- **Baseline collapse**: when all completions in a group score the same (all right or all wrong), the group mean equals each reward, advantages are zero, and there is no gradient. This is correct behavior, not a bug, but it means you need a curriculum of problems in the model's "zone of proximal development." Problems where the model gets 10–80% of completions correct are the most useful.
- **Weak base model**: RL does not create reasoning from nothing. If the base model has no plausible path to a correct solution on a problem type, GRPO will not find one either. The ceiling is set by pre-training.
- **Format reward masking correctness**: if the format reward is set too high relative to correctness reward, the policy can learn to produce perfectly-formatted empty reasoning and still achieve near-maximum reward on easy problems. Keep the format reward small (0.1 vs 1.0 for correctness is reasonable) and consider annealing it to zero after the first few hundred steps.

---

## Exercises

A tested exercise implementing GRPO on a verifiable toy arithmetic task is planned at `exercises/15-grpo-rlvr/`. That exercise will walk through setting up a small policy (a tiny transformer or MLP over digit tokens), a ground-truth arithmetic checker as the reward function, and the full GRPO training loop from the reference code above.

Other things worth implementing:

- **Process vs outcome reward on a small math set**: take 200 problems from GSM8K, score them with outcome reward (final answer only) and with a step-checking heuristic (does each line follow from the previous?). Train two models and compare error types, not just final accuracy. The Uesato et al. paper (arXiv:2211.14275) is the reference for this experiment.
- **STaR on GSM8K-style problems**: implement the sample → filter → SFT loop. Measure how many iterations are needed before improvement plateaus. Compare to a single SFT run on all correct solutions from iteration 1. This experiment reveals the data efficiency cost of the EM-style approach.
- **Entropy monitoring during GRPO**: add a logging hook to track the entropy of the policy distribution over the vocabulary at each training step. Plot it against the training reward curve. Notice when it drops and whether reward improvement and entropy collapse happen together or sequentially.
- **Vary K in GRPO**: train with K=2, K=4, K=8 (completions per prompt) at fixed total forward-pass compute (so K=2 gets 2x more prompts than K=8 per step). Compare variance of advantage estimates and held-out reward. The question: does more diversity per prompt (K=8) beat more prompts with less diversity (K=2)?
- **Reward hacking detection**: design a verifier that can be gamed (e.g., an arithmetic checker that only checks the last digit of the answer). Train with GRPO against it and observe when training reward diverges from true accuracy. Then fix the verifier and retrain. This is a controlled demonstration of how verifier quality shapes what the model learns.

---

## Debugging checklist

**Training reward not increasing after 500 steps**: check whether the model is producing the expected output format at all. If `format_reward` is also stuck at zero, the format check may be broken or the model isn't generating the expected tags. Print a sample completion every 50 steps.

**Training reward increasing but held-out reward flat or declining**: reward hacking on the verifier. Compare training accuracy to a separate held-out set using the same checker. If there's a gap, the verifier has exploitable patterns — fix the verifier or hold out more test problems from the reward signal.

**KL exploding**: `kl_beta` is too low, or the learning rate is too high. Reduce learning rate first (try 5e-7), then raise `kl_beta` (try 0.1). Check whether gradient norms are reasonable (clip at 1.0).

**All advantages near zero every step**: either K is too small (K=1 trivially gives zero variance), or every prompt has all-correct or all-incorrect completions. Check the distribution of pass rates across prompts. Ideal: a mix, not all 0s or all 1s. Add harder problems if the model is saturating.

**Policy entropy dropping to near zero**: add an entropy bonus term to the loss. A coefficient of 0.001–0.01 on `-entropy(pi_theta)` is usually enough to maintain diversity without hurting reward.

**Gradient norms spiking occasionally**: this often happens when a group of completions has one very high-reward outlier, producing a large advantage for that sample. Add gradient clipping (`max_norm=1.0`). Also consider outlier clipping on advantages before computing the loss.

```python
# Add this diagnostic at the end of each training step:
print(f"step={step} | reward_mean={rewards.mean():.3f} | "
      f"kl={metrics['kl']:.4f} | clip_frac={metrics['clip_fraction']:.3f} | "
      f"adv_std={advantages.std():.3f}")

# Healthy ranges (rough):
#   reward_mean: increasing over time
#   kl: 0.01–0.5 (not exploding, not stuck at 0)
#   clip_frac: 0.1–0.5 (some clipping but not all)
#   adv_std: > 0.1 (advantages are not collapsed to zero)
```

---

## References

All IDs verified against arxiv.org.

**GRPO / DeepSeekMath**

- Shao, Wang, Zhu et al. 2024. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." DeepSeek-AI. arXiv:2402.03300. — Introduces GRPO. The GRPO section (§3) is self-contained.

**DeepSeek-R1**

- DeepSeek-AI. 2025. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv:2501.12948. — Describes R1-Zero (pure RL on base model) and R1 (cold-start SFT + RL + RLHF). Primary source for the recipe in this lecture.

**Process vs outcome supervision**

- Uesato, Kushman, Kumar et al. 2022. "Solving math word problems with process- and outcome-based feedback." DeepMind. arXiv:2211.14275. — First systematic GSM8K comparison; outcome supervision gets similar accuracy with less annotation.
- Lightman, Kosaraju, Burda et al. 2023. "Let's Verify Step by Step." OpenAI. arXiv:2305.20050. — Human-annotated step-level labels on MATH; process supervision outperforms outcome supervision at matched scale.

**STaR and ReST**

- Zelikman, Wu, Mu, Goodman. 2022. "STaR: Bootstrapping Reasoning With Reasoning." NeurIPS 2022. arXiv:2203.14465. — Iterative sample → filter → SFT loop for chain-of-thought.
- Gulcehre, Le Paine, Srinivasan et al. 2023. "Reinforced Self-Training (ReST) for Language Modeling." Google DeepMind. arXiv:2308.08998. — Offline generate → filter → train loop; more compute-efficient than online RL.

**PPO (prerequisite)**

- Schulman, Wolski, Dhariwal et al. 2017. "Proximal Policy Optimization Algorithms." OpenAI. arXiv:1707.06347. — See [Lecture 06](./06-ppo.md) for the derivation and implementation. GRPO inherits the clipped surrogate objective directly from this paper.

**o1 (no paper)**

- OpenAI. September 2024. "Learning to reason with LLMs." OpenAI blog post. No technical paper available. The blog post describes inference-time scaling behavior and high-level training philosophy; the specific algorithm and reward function are not disclosed. Claims about the training procedure or internal search strategy that are not in the official blog post are speculative.

**Reading order suggestion**

If you're reading the primary sources: start with §3 of DeepSeekMath (arXiv:2402.03300) for the GRPO algorithm, then read the R1 paper (arXiv:2501.12948) for the full training pipeline. The Lightman et al. PRM paper (arXiv:2305.20050) is the clearest treatment of why process supervision helps and how step-level labels are collected. STaR (arXiv:2203.14465) is short and worth reading in full — it's one of the cleaner experimental papers in this space.

---

## Next lecture

[Lecture 16: Agentic RL](./16-agentic-rl.md)
