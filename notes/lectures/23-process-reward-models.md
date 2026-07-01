<!-- status: unreviewed | last-reviewed: never -->

# Lecture 23: Process reward models vs outcome reward models

_Unreviewed — no one has checked this end to end. Treat the math, code, and citations as unverified._

**Time**: ~2-3 h · **Prerequisites**: Lectures 09, 15

---

## Where this fits

Lecture 09 covered learned reward models trained on preference comparisons. Lecture 15 covered verifiable rewards — rule-based checkers that score the final answer of a reasoning chain. Both are *outcome* signals: the model writes a complete response, you score the whole thing, and the gradient credits or debits every token in the chain by the same amount.

That's fine when chains are short and the verifier is reliable. It gets worse as chains get longer. A 30-step proof or a 2000-token math solution gets a single scalar at the end; the gradient signal that updates step 7 is identical to the signal that updates step 28. Credit assignment across a long chain from a single bit at the end is noisy.

The alternative is to score every step. A *process reward model* (PRM) takes a partial solution and assigns a score after each step. The reward signal is dense, the credit assignment is local, and at inference time you can prune branches mid-search instead of waiting for completion.

The cost is the labels. Step-level supervision needs either humans annotating every line (expensive — Lightman et al. 2023 collected 800K labels for one math dataset) or an automated proxy (cheaper, but the proxy itself can be wrong or gameable).

This lecture works through three things: where the idea came from, how the modern automated variant (Math-Shepherd) generates step labels without humans, and why DeepSeek-R1's authors looked at all of this and decided not to use it. Then a code sketch for scoring a chain-of-thought with a PRM and using it for best-of-N decoding. Then what's still open as of early 2026.

---

## Outcome rewards: a quick recap

The setup from Lecture 15. The model emits a reasoning chain `y = (y_1, y_2, ..., y_T)` for a prompt `x`. A verifier `v(x, y)` returns a scalar — usually binary, 1 if the final answer is correct, 0 otherwise. The RL update treats this scalar as the reward for the whole completion. In GRPO terms, every token in `y` gets the same group-relative advantage.

```
prompt ────► policy ────► chain of thought ────► answer ────► v(answer) ────► reward
                                                                                │
                                                                                ▼
                                                            (one scalar credits every token)
```

What you give up: per-step information. The verifier doesn't know that step 7 had a sign error and steps 8–28 worked around it accidentally. It only knows the answer came out right.

What you gain: simplicity. No step-level labels to collect. No reward model to train. No moving target — the verifier is a fixed program. Rule-based checkers on math and code are essentially impossible to game in the way a learned reward model can be gamed (cf. Lecture 09 on reward hacking), because they directly compute correctness rather than learning a proxy for it.

Two things break this picture as chains get longer:

1. **Sparse signal.** If the model takes 50 steps and gets the answer right once in 16 attempts, you have one positive example per prompt per rollout. The gradient is informative on average, but for any individual token it's a tiny coefficient on a noisy estimate.
2. **Correct answers from wrong reasoning.** The model can pattern-match the answer format, guess from a small answer space, or accidentally cancel two errors. The verifier rewards all of these. Uesato et al. 2022 (arXiv:2211.14275) measured this directly on GSM8K and reduced reasoning errors among correct solutions from 14.0% to 3.4% by switching from outcome to process supervision.

A way to feel the sparse-signal problem. Suppose the policy emits a 40-step chain and gets one binary reward at the end. Treat each step as a Bernoulli decision the policy "made" — a choice that might be right or wrong on its own. The reward at the end is a function of all 40 decisions. From a single bit at the end, you cannot in general tell which of the 40 decisions deserves credit or blame. The signal-to-noise ratio per-decision is at best `O(1/sqrt(T))` for a chain of length `T` — and only that good if the decisions are independent, which they're not.

The classical RL response to sparse delayed rewards is to learn a value function and bootstrap (Lecture 03). For LLMs that's expensive — the value function is another large network — and that's exactly the reason GRPO dropped the critic in the first place (Lecture 15). Process supervision is a different response: instead of bootstrapping a value function from outcomes, get the per-step labels directly, either from humans or from rollouts.

Process supervision is the response to both of these problems.

---

## Process rewards: the idea

A PRM is a function `r(x, y_{1..k})` that takes a prompt and the first `k` steps of a reasoning chain and returns a scalar score for step `k`. You apply it after each step. A chain of 8 steps gives you 8 scores instead of 1.

Two ways to use the scores:

**At training time.** Replace the single end-of-chain reward with a sequence of per-step rewards. The advantage for tokens in step `k` depends on `r(x, y_{1..k})`, not on the final outcome alone. The credit assignment is local: a bad step gets a low score immediately, and the gradient targets the tokens that produced that step.

**At inference time.** Score candidate completions step by step. Prune branches where intermediate scores drop. In its simplest form: generate N complete chains, score every step in each, and rerank by the minimum (or product, or sum) of step scores. In a richer form: do beam search where the beam is over partial chains and the score is the PRM's running estimate.

A picture of the inference-time use:

```
        ┌── step 1a (score 0.92) ── step 2a (score 0.88) ── ... ── final
prompt ─┼── step 1b (score 0.85) ── step 2b (score 0.40)  ✗ prune
        ├── step 1c (score 0.70) ── step 2c (score 0.65) ── ... ── final
        └── step 1d (score 0.30)  ✗ prune
```

Lightman et al. 2023 ("Let's Verify Step by Step," OpenAI, arXiv:2305.20050) reported 78% accuracy on a representative MATH test subset using a PRM-reranked best-of-N, outperforming an outcome-supervised reranker at the same scale. Their dataset, PRM800K, is 800,000 step-level human labels collected over thousands of MATH solutions, and is the largest open process-supervision dataset.

The training signal in the PRM is simple: each step gets a label in `{good, neutral, bad}` (Lightman et al. used three classes; many follow-ups use binary), and the model is trained with cross-entropy to predict the class from the prompt and the chain so far.

---

## What is a "step"?

The whole approach hinges on this and there is no clean answer.

Lightman et al. chose newline-separated lines in the model's chain-of-thought as steps. This works on MATH because the model is prompted into a solution format with one logical move per line. It would not work on free-form prose where line breaks are arbitrary. Math-Shepherd (Wang et al. 2024) uses the same line-based convention because their setting is the same — MATH and GSM8K with prompted chain-of-thought.

What "step" means in domains without a natural line structure:

- **Code.** A function body, a statement, an expression? Statement-level is probably the smallest unit where you can say "this is right or wrong" in isolation. But many statements only make sense in the context of earlier ones, so step-by-step labeling forces you to score in context anyway.
- **Formal proofs.** Lean/Coq give you tactics as the natural step. A tactic either succeeds or fails when applied, which is a free verifier — no learned PRM needed.
- **Free-form reasoning.** Sentence boundaries, paragraph boundaries, "checkpoints" the model is prompted to emit. All of these are arbitrary, and the granularity choice changes what the PRM is learning.

The lesson: PRMs work best when the domain has a natural step structure, ideally one where step boundaries are unambiguous and individual steps can be evaluated locally. Math fits. Code fits, with caveats. Long-form writing or open-ended reasoning does not.

DeepSeek-AI 2025 (arXiv:2501.12948) lists "difficult to define a fine-grain step in general reasoning" as the first of three reasons they didn't adopt PRMs for R1. We'll come back to the other two.

---

## Lightman et al. 2023: human-labeled process supervision

The setup. Train a generator (GPT-4 era model) on MATH problems. Sample candidate solutions. Hand each solution to human annotators who label every step as `positive` (clearly correct), `neutral` (ambiguous or harmless), or `negative` (clearly wrong, or a step that follows from a wrong premise). Aggregate enough of these to train a PRM with cross-entropy on the three-class label.

The annotation effort. PRM800K is 800,000 step labels. The annotators were instructed to be strict — neutral was for actual ambiguity, not for "I'm not sure," and negative meant the step did not follow from the prior context, even if the final answer turned out to be correct. This is the labor that makes the dataset valuable and the approach expensive.

The result. At test time, the PRM was used to rerank best-of-N samples. For each candidate solution, the per-step scores were aggregated (the paper used the product of step probabilities, which is equivalent to the minimum in log-space for normalized scores) and the highest-scoring candidate selected. On MATH, PRM-reranked best-of-1860 reached 78.2% on a representative test subset, vs. lower numbers for ORM-reranked or majority-vote baselines at the same N. Active learning — where the annotators were sent the steps the current PRM was most uncertain about — substantially improved the labeling efficiency.

Why it works better than an ORM at the same N. The ORM only knows whether the final answer is right; it can't tell a lucky guess from solid reasoning. The PRM can. So when the ORM ranks a chain highly, it might be a chain where steps 1–7 were nonsense but step 8 happened to produce the right number. The PRM penalizes that chain at step 3 and the reranker drops it. As N grows, the PRM's ability to discriminate among many candidate chains pays off more than the ORM's coarser judgment.

What it doesn't tell you. The headline number is reranking accuracy, not policy improvement from PRM-supervised RL. Lightman et al. did the training-time use too — training a generator with PRM-derived rewards — and got smaller but consistent gains. But the most-cited result is the inference-time best-of-N, because that's where the gap is biggest.

A subtle bit of methodology worth knowing. To compare PRMs and ORMs fairly, Lightman et al. trained both on the same generator outputs. The ORM was trained on chain-level correctness labels — "this whole chain ended right" or "this whole chain ended wrong." The PRM was trained on the step-level labels described above. Both models had the same architecture and parameter count. The fair comparison is "what does step-level signal buy you over chain-level signal, holding everything else equal?" That's a cleaner question than "PRM800K vs. some ORM somewhere," and the gap they report is the answer to the clean version.

Another thing worth flagging. The labels in PRM800K are not just "did this step lead to the right answer." Annotators were instructed to evaluate whether the step *follows from the prior steps*, regardless of how the chain ends. So a step that doesn't follow logically but happens to land on the right answer gets a `negative` label, and a step that follows correctly but the chain eventually goes wrong elsewhere still gets a `positive` label for that specific step. This is the philosophical distinction between process and outcome at the label level: outcomes are about the chain, process is about the step itself in context. Math-Shepherd's automated labels (next section) blur this line, because they label steps by what the rollouts from them produce — which is closer to a chain-level signal applied at each step than to a true step-in-isolation judgment.

---

## Math-Shepherd 2024: automated process labels via Monte Carlo rollouts

The expense of PRM800K motivated the next paper. Wang et al. 2024 ("Math-Shepherd," ACL 2024, arXiv:2312.08935) replaces the human annotators with Monte Carlo rollouts from an LLM "completer."

The idea: for each step in a candidate solution, run `N` rollouts that continue from that step to a final answer. If the rollouts reach the correct answer, the step probably wasn't bad. If they don't, it probably was. The empirical success rate from the step becomes the step's label.

Two labeling schemes in the paper:

**Hard estimation (HE).** Label the step `1` if *any* of the `N` rollouts from it reach the correct answer, else `0`.

```
y_HE(s_k) = 1 if ∃ j ∈ {1..N} : answer(rollout_j from s_k) = ground_truth
         else 0
```

**Soft estimation (SE).** Label the step as the empirical success rate.

```
y_SE(s_k) = (1/N) · Σ_{j=1..N} 1[answer(rollout_j from s_k) = ground_truth]
```

The PRM is then trained with binary cross-entropy on these labels (treating SE labels as soft targets in `[0, 1]`):

```
L_PRM = -Σ_k [ y(s_k) · log r_θ(x, y_{1..k}) + (1 - y(s_k)) · log(1 - r_θ(x, y_{1..k})) ]
```

The completer in their experiments was LLemma-7B, with `N=8` rollouts per step. They produced ~170K labeled solutions for GSM8K and ~270K for MATH this way — comparable in scale to PRM800K, without the human labor.

The trick is subtle and worth pausing on. The label for step `k` is *not* "is this step correct in isolation?" It's "starting from this step, how often does a continuation reach the right answer?" That's a value-function-like signal: the label estimates the expected return from the state, where the state is "the partial chain so far."

This collapses two things into one: a step is "good" both when it's locally correct *and* when the model can recover from it. A locally-correct step that the model can't follow up on gets a lower label than a sloppy but recoverable one. Whether this is the right thing depends on what you're using the PRM for. For reranking complete chains it works well, because what you care about is whether the chain ended correctly. For doing tree search it's less clean, because you don't want to over-credit recoverable but locally-wrong steps.

The results in the paper: Mistral-7B trained with PPO using Math-Shepherd's PRM as the reward signal went from 77.9% to 84.1% on GSM8K, and further to 89.1% when combined with PRM-based verification at inference time. Comparable to using human PRM labels at smaller scale, at zero human annotation cost.

The cost. `N=8` rollouts per step, per training problem. For a chain of 10 steps, that's 80 generations to label one solution. Multiply by however many training solutions. The compute cost is not trivial, but it scales with compute (which you can buy) rather than with labelers (which you have to coordinate).

The catch. The completer is itself an LLM, and an imperfect one. If the completer is bad at math, its rollouts are noisy estimators of step quality. A step that a strong solver would recover from gets labeled `0` because the weak completer can't continue from it. This is the same problem as having a noisy value function in actor-critic — your baseline is wrong. The PRM trained on those labels inherits the completer's blind spots.

A diagnostic: if you label the same step with different completers and the labels swing wildly, your labels are dominated by completer noise rather than by step quality. A small ablation in any PRM-training pipeline: take a held-out set of steps, label each with two different completers, and report the inter-completer agreement. If agreement is below ~80% on steps where humans agree, the labels are too noisy to learn from reliably.

A connection worth making. The Math-Shepherd MC-rollout label is structurally similar to a Monte Carlo value estimate from Lecture 03. There, you estimated `V(s)` by averaging the returns from many rollouts starting at `s`. Here, you're estimating `P(reach correct answer | partial chain)` by averaging rollouts from the partial chain. Both are MC estimates. Both have variance that scales with `1/sqrt(N)`. Both are biased by the rollout policy: if the rollout policy is the same as the policy being trained, the estimate is on-policy and consistent; if the rollout policy is different (a separate "completer"), the estimate is off-policy and biased. Math-Shepherd uses a fixed completer that is *not* the policy being trained, which makes the labels off-policy and stale as the policy improves. This is one reason the labels get less informative over the course of training and need refreshing.

A cheaper variant that some replications use: instead of running `N` independent rollouts per step, run a single rollout from each of the `N` candidate chains the policy already produced, and use the cross-chain step alignment to share rollouts. This is messier to implement (steps don't always align across chains) but eliminates the dedicated completer.

---

## Why DeepSeek didn't use PRMs

The DeepSeek-AI papers — DeepSeekMath (arXiv:2402.03300) and DeepSeek-R1 (arXiv:2501.12948) — went the other way. They scaled outcome-only RL with GRPO on rule-based verifiers and reached reasoning performance competitive with the closed o1-class models. The R1 paper has a section explicitly titled "Unsuccessful Attempts" where the first item is process reward models. Their reasons are worth quoting carefully because they're rare to see written down.

The three reasons in the R1 paper (paraphrased from §4.2):

1. **Defining a step is hard outside narrow domains.** Math problems have natural step boundaries; general reasoning does not. As soon as you try to use a PRM beyond math (e.g., for coding agents, open-ended research, multi-turn dialog), the granularity question becomes the dominant problem.
2. **Step-level correctness is hard to verify cheaply.** Human annotation is expensive and slow. Automated annotation (the Math-Shepherd style) is unsatisfactory in their experience — the rollout-based labels are too noisy when the completer is much weaker than the target.
3. **PRMs introduce a learned model into the reward signal, and learned reward models get hacked.** The R1 paper states: "once a model-based PRM is introduced, it inevitably leads to reward hacking, and retraining the reward model needs additional training resources and it complicates the whole training pipeline."

The third point is the most general one and the one most worth internalizing. A rule-based outcome verifier (run the test suite, check the regex, run the symbolic solver) is not learned and not hackable in the same way. A PRM is a neural network trained to predict step quality; the policy can learn to produce text that the PRM scores highly without that text actually being good reasoning, in the same way RLHF policies overoptimize learned reward models (Lecture 09).

The R1 conclusion: PRMs may still help for inference-time reranking on a fixed model, but as a training reward signal "its advantages are limited compared to the additional computational overhead it introduces."

DeepSeek-R1-Zero — trained with pure GRPO on rule-based correctness rewards, no PRM at all — was the existence proof that this position is at least defensible.

A more careful reading. The R1 result doesn't show that PRMs are wrong, only that for math and code (where the rule-based outcome signal is reliable) the PRM machinery isn't pulling its weight. If you're in a domain where the outcome is hard to verify cheaply — long-horizon agentic tasks, scientific reasoning, anything where you can't write a clean checker — the calculation changes. Process supervision might still be the right answer when the alternative is no signal at all.

There's an asymmetry here that's worth naming. A rule-based outcome verifier on a math problem is binary and immediate: the answer is `7` or it isn't. The check is one regex. A PRM trained to score steps is a continuous function over an open-ended text space, and a policy with millions of parameters has many degrees of freedom to drift toward high-PRM regions that aren't actually higher-quality reasoning. The asymmetry is that the verifier sees the world the way the world is (right answer or not), while the PRM sees the world the way it was trained to see it (which is correlated with quality but isn't quality itself). The first is a fact; the second is a model. The DeepSeek argument is essentially that when a fact is available, you should use the fact and not the model. The PRM-supportive argument is that the fact is sometimes only available at the end of the chain, and the model fills in the per-step structure that the fact lacks.

A useful summary of the tradeoffs:

| Aspect | Outcome reward (rule-based) | Process reward (learned PRM) |
|---|---|---|
| Label cost | Free (ground truth available) | High (humans) or moderate (MC rollouts) |
| Signal density | One scalar per chain | One scalar per step |
| Credit assignment | Whole chain credited as a unit | Per-step credit |
| Gameable? | Verifier-specific; rule checkers are mostly safe | Yes, like any learned reward model |
| Domain fit | Strong when verifier exists | Stronger when chains are long and steps are decomposable |
| Inference use | Best-of-N with verifier, majority vote | Best-of-N with PRM rerank, beam search guided by PRM |
| Infrastructure | One model + verifier | Policy + reference + PRM (+ completer for labels) |

The right answer depends on which row dominates for your problem. R1's bet was that for math reasoning the simplicity of the first column outweighed the per-step signal of the second. Whether the same bet holds in other domains is open.

---

## Inference-time use: best-of-N and beam search

Even when you don't use a PRM as a training signal, it can be useful as a re-ranker over completed chains, or as a guide for tree search at inference. The training cost is amortized over many inferences, and the inference-time use doesn't expose the PRM to reward hacking — the policy doesn't update against it.

### Best-of-N with PRM reranking

Generate N candidate chains. Score each one with the PRM step-by-step. Aggregate the step scores into a chain score. Return the highest-scoring chain.

The aggregation function matters. Common choices:

- **Min:** the chain's score is its weakest step. Sensitive to a single bad step anywhere in the chain.
- **Product:** multiplicative across steps (equivalent to sum in log-space). Penalizes long chains more, because each step's `[0, 1]` score chips away at the product.
- **Mean:** the average over steps. Insensitive to step length but also insensitive to a single catastrophic step.
- **Last-step score:** equivalent to using an ORM, because the last step's PRM score tends to be correlated with the final answer's correctness. Defeats most of the point of the PRM.

Lightman et al. used the product. Math-Shepherd reports using the minimum. In practice you tune this against held-out data.

### PRM-guided beam search

Maintain a beam of `B` partial chains. At each step, expand each partial chain by sampling `K` continuations, scoring each new step with the PRM, and keeping the top `B` partial chains by aggregated score. This is straightforward beam search where the per-token model log-probabilities are replaced (or augmented) with PRM scores at step boundaries.

Pseudocode:

```python
def beam_search_prm(prompt, policy, prm, beam_width=4, n_expansions=4,
                    max_steps=20, agg=min):
    """Greedy beam search guided by per-step PRM scores.

    Each beam entry is (partial_chain, aggregated_score, finished_bool).
    A 'step' is whatever the policy's step-emission convention is —
    e.g., a newline-terminated segment.
    """
    beams = [(prompt, 0.0, False)]  # initial beam: just the prompt
    for _ in range(max_steps):
        candidates = []
        for chain, score, finished in beams:
            if finished:
                candidates.append((chain, score, True))
                continue
            # Sample n_expansions next steps from the policy
            next_steps = policy.sample_next_step(chain, n=n_expansions)
            for step_text, is_terminal in next_steps:
                new_chain = chain + step_text
                step_score = prm.score(prompt, new_chain)  # scalar in [0,1]
                new_score = combine_scores(score, step_score, agg)
                candidates.append((new_chain, new_score, is_terminal))

        # Keep the top beam_width candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_width]

        if all(finished for _, _, finished in beams):
            break

    return max(beams, key=lambda x: x[1])  # best finished chain


def combine_scores(running, new_step, agg):
    """Fold the new step's score into the running aggregate.

    For agg=min, this is min(running, new_step) — but the initial running
    score is 0.0 (which would dominate), so use the new step directly for
    the first call. A real implementation would pass a counter.
    """
    if agg is min:
        return min(running, new_step) if running > 0 else new_step
    elif agg is sum:
        return running + new_step  # mean = sum / n_steps, tracked externally
    raise ValueError(f"unknown agg: {agg}")
```

The pseudocode glosses over a few real-world headaches: extracting "the next step" from a model that emits tokens (you have to detect the step boundary — usually a newline — and stop generation there), avoiding redundant rollouts when multiple beams share a prefix, and dealing with the cost of running the PRM `B * K` times per step.

The compute cost is the main constraint. A beam of 4 with 4 expansions and 20 steps is 320 step evaluations per query, plus the PRM forward pass each time. For a 7B-parameter PRM, this is significantly more inference compute than just sampling a chain. Whether it's worth it depends on the value of the gain — for one-shot solutions to hard problems, often yes; for high-volume serving, usually no.

### A simpler PRM scoring sketch

Before beam search, the most basic use: take a chain you've already generated, score it step by step, and decide whether to keep it. This is what reranking best-of-N looks like in code.

```python
import torch
from typing import Sequence


def score_chain_with_prm(
    prm_model,
    tokenizer,
    prompt: str,
    chain: str,
    step_separator: str = "\n",
) -> tuple[list[float], float]:
    """Score a chain-of-thought step by step.

    Returns:
        step_scores: list of per-step probabilities in [0, 1].
        chain_score: aggregated score (here: min across steps).

    Assumptions:
        - `prm_model` is a classifier that maps (prompt, partial chain) → P(good step).
        - Steps in `chain` are separated by `step_separator` (newline by default).
        - The model has a step-end token that we'd score in a token-level PRM;
          this sketch uses a sequence-classifier interface for clarity.
    """
    steps = chain.split(step_separator)
    step_scores = []
    partial = ""

    for step in steps:
        if not step.strip():
            continue
        partial = (partial + step_separator + step) if partial else step
        # Build the input the PRM expects: prompt + partial chain so far.
        input_text = f"{prompt}\n{partial}"
        encoded = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        with torch.no_grad():
            logits = prm_model(**encoded).logits  # [1, 2] or [1, 1]
            # Two-class PRM: index 1 = "good step"
            prob_good = torch.softmax(logits, dim=-1)[0, 1].item()

        step_scores.append(prob_good)

    chain_score = min(step_scores) if step_scores else 0.0
    return step_scores, chain_score


def best_of_n_with_prm(
    prm_model,
    tokenizer,
    prompt: str,
    candidates: Sequence[str],
) -> tuple[str, float, list[list[float]]]:
    """Pick the best candidate chain by minimum step PRM score."""
    all_step_scores = []
    chain_scores = []
    for chain in candidates:
        step_scores, chain_score = score_chain_with_prm(
            prm_model, tokenizer, prompt, chain
        )
        all_step_scores.append(step_scores)
        chain_scores.append(chain_score)

    best_idx = max(range(len(candidates)), key=lambda i: chain_scores[i])
    return candidates[best_idx], chain_scores[best_idx], all_step_scores
```

A few notes on what's hidden here. In a real PRM trained the Lightman or Math-Shepherd way, the per-step score is the probability emitted by the PRM at the step-boundary token (often a special `<step>` token, or just a newline that the PRM was trained to predict on). The "two-class classifier" sketch above is a simplification — in practice the PRM is a language model with a classification head over a small vocabulary, and you pull the score from the specific position where the step ends. The structure of the code is right; the tokenization details depend on how the PRM was trained.

Also: the `min` aggregation is one choice. For best-of-N reranking on MATH-style problems, both Lightman et al. and Math-Shepherd report that aggregation choice changes results by a few percentage points but doesn't change the qualitative answer (PRM beats ORM, PRM-product close to PRM-min).

---

## Training with process rewards

If you want to use the PRM as a training signal, not just a re-ranker, the wiring is straightforward in principle. Replace the single end-of-chain reward in GRPO with a sequence of per-step rewards.

In the GRPO loss from Lecture 15, the advantage `A_i` for the `i`-th completion is computed from a single scalar reward `r_i`. The whole completion gets that one advantage. For process supervision, you want each step `k` in completion `i` to get its own advantage `A_{i,k}` based on the PRM's score for that step.

Pseudocode for the modification:

```python
def compute_process_advantages(
    prm_scores: torch.Tensor,  # [batch, K, max_steps] — masked to actual step count
    step_mask: torch.Tensor,    # [batch, K, max_steps] — 1 where a step exists
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-step group-relative advantages.

    For each (batch, step_index), compute the group mean and std across the
    K completions at that step index. Step indices that don't align across
    completions are masked out.
    """
    # group_mean: [batch, 1, max_steps]
    masked = prm_scores * step_mask
    counts = step_mask.sum(dim=1, keepdim=True).clamp(min=1)
    group_mean = masked.sum(dim=1, keepdim=True) / counts
    diff = (prm_scores - group_mean) * step_mask
    group_var = (diff ** 2).sum(dim=1, keepdim=True) / counts
    group_std = group_var.sqrt() + eps

    advantages = (prm_scores - group_mean) / group_std
    advantages = advantages * step_mask
    return advantages
```

Two complications worth flagging:

1. **Step alignment across the group.** GRPO compares K completions for the same prompt. If completion 1 has 6 steps and completion 2 has 9 steps, comparing "step 3 of completion 1" with "step 3 of completion 2" is comparing different content. There's no clean way to align steps across completions, so the within-group baseline is noisier than in outcome-GRPO. One workaround: only compare advantages at matching depth, and accept that the baseline is dirty.
2. **The PRM is a learned model, and it's being optimized against.** This is the reward-hacking concern DeepSeek raised. The policy can drift into a region where it produces text the PRM scores well but that isn't actually good reasoning. The standard mitigations from Lecture 09 apply: hold out problems from the PRM's training set, monitor true accuracy on the held-out set separately from PRM-scored "accuracy," and retrain the PRM periodically against the current policy's outputs (online PRM training).

The standard outcome-GRPO update from Lecture 15 has the structural advantage that the reward source is fixed and known. Process-GRPO doesn't — the PRM is in the loop and can be gamed. That's the practical concern, and it's the reason most modern reasoning training pipelines (R1, the open replications, the DeepSeekMath paper itself) use outcome rewards for the policy update and reserve PRMs (if at all) for inference-time reranking.

### Mixed objectives

One way to keep some of the dense-signal benefit of process rewards without committing fully to PRM-as-reward: combine them. Use the outcome reward as the dominant signal and the PRM score as a small additional bonus. Roughly:

```
r_total(chain) = r_outcome(chain) + lambda * mean(prm_scores_along_chain)
```

with `lambda` small (say `0.1`). The outcome reward keeps the policy anchored to actually solving problems; the PRM bonus gives a denser gradient on the way. This is similar in spirit to reward shaping in classical RL — adding a potential-based bonus to speed learning without changing the optimal policy.

The catch is the same as in classical reward shaping: if the bonus isn't potential-based, the optimal policy under the shaped reward isn't the same as under the original. In LLM RL, where we don't optimize to convergence anyway, this matters less in theory but matters in practice for what the policy drifts toward. A common diagnostic: train with `lambda = 0` and `lambda > 0`, both for the same number of steps, and compare held-out outcome accuracy. If the `lambda > 0` run is faster to a given accuracy but eventually plateaus lower, the bonus was helpful early and harmful late — anneal it.

### Where the per-step gradient lands

A practical detail in the code. When you have per-step advantages, you need to assign them to per-token log-probabilities in a sensible way. The simplest scheme: every token in step `k` gets advantage `A_k`. So if step `k` spans tokens `t_start^k` to `t_end^k`, the per-token loss contribution is `A_k * log pi(token_t | context_t)` for each `t` in that range.

```python
# Sketch — per-step advantages assigned to per-token log-probs.
# token_step_index[t] = which step token t belongs to (0-indexed).
# advantages_per_step[k] = the advantage for step k.

token_advantages = advantages_per_step[token_step_index]  # [seq_len]
per_token_loss = -(token_advantages * token_log_probs).sum()
```

This is a small but real change from outcome GRPO, where you'd have `token_advantages = advantages_per_chain[chain_index]` (a single value broadcast across the whole completion). The per-step version gives a sharper gradient — tokens in low-advantage steps get pushed away from the policy's current behavior at exactly the points where the PRM thinks the chain went wrong. Whether the gradient is sharper *in a useful direction* depends on whether the PRM's per-step scores are causally informative, which is the question that all of this hinges on.

---

## When to use each

A rough decision tree, narrower than the one in Lecture 15.

```
Is there a reliable rule-based verifier for the final answer?
├── Yes  → Use outcome supervision (GRPO with rule-based reward).
│         Reserve PRMs for inference-time best-of-N if you can afford it.
└── No   → Is the domain decomposable into well-defined steps?
    ├── Yes → Process supervision is worth considering. Decide on labels:
    │         ├── Step-level checker exists (e.g., theorem prover) → use it directly,
    │         │   no learned PRM needed.
    │         ├── Otherwise → train a PRM. Choose between:
    │         │   ├── Human labels (Lightman-style): high quality, high cost.
    │         │   └── MC rollouts (Math-Shepherd-style): cheaper, noisier,
    │         │       limited by completer quality.
    │         │
    │         └── Then decide use:
    │             ├── Inference-time reranking only — safe, expensive at serve time.
    │             └── Training reward signal — fast learning but watch for hacking.
    │
    └── No  → Outcome supervision (preference-based, Lecture 09–10).
              Or: redesign the task to expose decomposable structure if you can.
```

The places where PRMs are most likely to be the right answer in early 2026:

- **Long-horizon math or symbolic reasoning** where the outcome signal is too sparse — the chain has 50+ steps and the model rarely gets to the end. A PRM gives you a dense signal so you can train at all.
- **Inference-time search over reasoning** where you can afford the extra compute and the PRM is held out from policy optimization (so reward hacking isn't a concern).
- **Domains with naturally checkable steps** like formal proof search (Lean, Coq, Isabelle), where each tactic application is locally verified and you get the "PRM" for free from the proof assistant.

The places where outcome supervision is still doing the heavy lifting:

- **General math and code training** for reasoning models. R1, the open replications, the post-R1 wave of papers in 2025 — almost all of them use outcome-only GRPO. The simplicity wins.
- **RLHF-style alignment** where the "reward" is a preference judgment over a complete response. Per-step preferences are hard to ask humans about coherently.

---

## Open questions in early 2026

A few directions that were active when this was written, and that the reader should expect to see move:

**Generative reward models.** Instead of training a PRM as a classifier that outputs a scalar, train it as a generator that emits a critique or a chain-of-thought judgment about the step. The "score" is derived from the model's verdict. This style of reward modeling shifts the reward into the same medium as the policy's output — both are text — which has some appeal because the reward model can explain its judgments and can in principle be improved by the same techniques (CoT, self-consistency) as the policy. Whether generative reward models are more robust to gaming than scalar-output ones is an open empirical question.

**LLM-as-judge as a PRM.** A line related to the Constitutional AI work (Lecture 14) and to RLAIF: use a frontier model as the step-level judge, prompted to evaluate each step. No training, just prompting. The cost is the inference cost of running a large judge model many times per training step or per inference, and the judge's biases (positional bias, length bias, the things Lecture 14 covered) leak into the reward. The cost will probably come down as judges get cheaper; the bias problem is more fundamental.

**Process rewards from execution traces in code.** A clean case where automated process supervision is natural: if the policy emits code with intermediate states (variable values after each statement, output after each print), an interpreter or sandbox provides ground-truth per-step information. You don't need a learned PRM; the runtime is the PRM. Recent work has used this for code repair (per-statement variable-state matching), for agentic tasks (per-action tool-call success), and for theorem proving. This may be where process supervision pays off most in practice, precisely because the "labels" come from execution rather than from human or LLM judgment, sidestepping the reward-hacking concern.

**Hybrid outcome + process objectives.** Combine a small per-step PRM bonus with a larger end-of-chain outcome reward. The PRM gives the dense gradient signal early in training when the outcome signal is too sparse to learn from; the outcome reward ensures the policy doesn't overoptimize the PRM. The literature on the right mixing schedule is thin as of early 2026.

**Better step segmentation.** Maybe the biggest practical bottleneck on PRMs. Most of the existing work uses newline-separated steps, which is a hack. Learning to segment a chain into meaningful units, possibly as a joint task with reward modeling, is an obvious direction.

**Process supervision under distribution shift.** Almost all the published PRM results train and evaluate on the same problem distribution (GSM8K, MATH). The robustness of PRMs to out-of-distribution problems — different math sub-areas, problems harder than training, problems in different formats — is not well-characterized. The ORM/outcome side has the same issue, but the PRM has more places to break because the per-step judgment is itself a learned function, not just the final-answer judgment.

**The "do you need a value function" question, restated.** Process rewards are essentially a way to import the value function from classical RL into LLM training without calling it a value function. A Math-Shepherd-style PRM is approximately `V(s) = P(reach correct answer | state)`. The DeepSeek argument is that maintaining this value function isn't worth it relative to higher-variance outcome-only estimators with enough samples. Whether this argument holds as chains get even longer (agentic settings with hundreds of steps) is one of the live empirical questions. At some chain length, the variance of outcome-only estimators becomes the binding constraint and a learned value function (PRM by another name) becomes worth the trouble. Where that crossover is, in 2026, is unclear.

---

## A concrete worked example

Putting the pieces together on a small case. Suppose the problem is: "Mia has 12 apples. She gives 1/3 of them to her sister, then eats 2. How many does she have left?" The correct answer is `6`.

Three candidate chains the policy produced:

**Chain A** (correct reasoning, correct answer):
```
Step 1: Mia starts with 12 apples.
Step 2: She gives 1/3 to her sister: 12 × 1/3 = 4 apples given away.
Step 3: She has 12 - 4 = 8 apples left after giving.
Step 4: She eats 2, so she has 8 - 2 = 6 apples remaining.
Answer: 6
```

**Chain B** (wrong middle step, right answer by luck):
```
Step 1: Mia starts with 12 apples.
Step 2: She gives 1/3 away: 1/3 of 12 = 3 apples given.   ← wrong: 1/3 of 12 is 4
Step 3: She has 12 - 3 = 9 apples after giving.            ← wrong: should be 8
Step 4: She eats 2: 9 - 2 = 7.                              ← wrong arithmetic chain
Step 5: Wait, let me reconsider. Actually 1/3 of 12 is 4, so 12-4 = 8, then 8-2 = 6.
Answer: 6
```

**Chain C** (right setup, wrong arithmetic):
```
Step 1: Mia starts with 12 apples.
Step 2: She gives 1/3 of 12 = 4 apples to her sister.
Step 3: She has 12 - 4 = 9 apples left.                    ← wrong: 12 - 4 = 8
Step 4: She eats 2: 9 - 2 = 7.
Answer: 7
```

What the different signals say:

| Signal | Chain A | Chain B | Chain C |
|---|---|---|---|
| Outcome reward (correct answer?) | 1 | 1 | 0 |
| ORM score (trained on chain-level) | high | high | low |
| PRM scores (per step, Lightman-style) | all `positive` | `positive, negative, negative, negative, positive` | `positive, positive, negative, negative` |
| Min PRM score | high | low (the `negative` steps) | low |
| Math-Shepherd HE (any rollout from step → correct?) | all 1 | 1 for step 1, possibly 0 for steps 2-4, 1 for step 5 | 1 for steps 1-2, 0 for steps 3-4 |
| Math-Shepherd SE (fraction of rollouts → correct) | all near 1 | depends on completer's ability to recover | 1.0, 1.0, then drops |

The outcome reward and the ORM both rank Chain B as good as Chain A. The PRM (under both Lightman and Math-Shepherd) catches that Chain B's middle steps are wrong, even though it recovered. Chain C is correctly punished by both signals.

The interesting case is Chain B. Outcome-trained policies get reinforced for producing chains like B — sloppy reasoning that occasionally lucks into the right answer with a late correction. Process-trained policies don't, because the per-step signal flags steps 2-4 as bad. The Uesato et al. finding (3.4% vs 14.0% reasoning errors among correct solutions) is what shows up in aggregate when you train against signals that distinguish these cases vs. signals that don't.

The case for DeepSeek's position: at scale, with enough rollouts, the policy figures out that Chain A-style reasoning is more reliable than Chain B-style reasoning because Chain A wins more often than Chain B across the full problem distribution. The outcome signal is noisy on a per-chain basis but informative in expectation. The PRM signal is denser but adds a learned model that itself can be wrong (e.g., it might fail to recognize a valid alternative reasoning path it didn't see in training).

The case for the PRM position: even at scale, you spend many more rollouts learning what the PRM could have told you with one. The compute advantage of dense signal vs sparse signal compounds as chains get longer.

The honest answer is that for the chain lengths and problem domains the field has measured at scale through early 2026, the outcome side is doing fine. For chain lengths and problem domains beyond that, the field doesn't really know.

---

## Exercises

A planned exercise lives at `exercises/23-prm-rerank/` (not yet built). Suggested implementations:

- **Best-of-N reranking on GSM8K.** Generate 16 chains with a small model. Score them three ways: by majority vote (no model), by an ORM (trained on whole-chain correctness), and by a PRM (Math-Shepherd style, using MC rollouts from the same generator as the completer). Compare accuracy at N=1, 4, 16. Measure how much of the PRM's win is sample efficiency vs. catching specific bad-reasoning failure modes.
- **Build a Math-Shepherd-style label generator.** For a small math problem set (say, 200 problems from GSM8K), implement the MC-rollout labeling. Use a small open model as the completer with `N=4`. Record the labels and inspect them. Find a step where the soft-estimate label is `~0.5` and look at why: is it genuinely ambiguous, or is the completer just bad?
- **Compare aggregation functions.** Take a PRM (Math-Shepherd's released weights work) and score 100 GSM8K candidate chains. Rank them by min, product, and mean. Compare top-1 accuracy across aggregations. Find a chain where min and mean disagree about ranking and write down why.
- **Reward-hacking demo on a PRM.** Train a tiny PRM on a small step-label dataset. Generate adversarial chains designed to score highly under that PRM (e.g., by repeating high-scoring step patterns). Measure how much the PRM-rewarded chains diverge from genuinely correct chains. This is the concrete demonstration of the failure mode the DeepSeek paper warns about.

---

## Debugging notes

**PRM scores are clustered near 0.5.** The PRM hasn't learned to discriminate. Either the training labels are too noisy (often the case with MC rollouts from a weak completer) or the model is undertrained. Inspect the label distribution first; if labels themselves are mostly `0.5` (with soft estimation), the completer is the bottleneck.

**PRM scores correlate with chain length and nothing else.** Length bias has crept in. Common cause: long chains with many easy steps accumulate small positive PRM scores that aggregate up. Solution: use length-normalized aggregation (mean instead of product) or train the PRM with length-stratified data.

**Best-of-N with PRM doesn't beat majority vote.** Either the PRM is weak, or your N is too small for the PRM's discrimination to pay off, or your aggregation is dominated by a single high-variance step. Try different N (PRMs help more at large N), try different aggregations, and verify the PRM has above-chance accuracy on individual step labels in a held-out set first.

**Training with PRM rewards diverges or hacks.** The classic learned-reward-model failure. Hold out a large set of problems from the PRM's training set. Track held-out outcome accuracy separately from PRM-scored accuracy during training. A growing gap is the hack. Mitigations: lower learning rate, higher KL penalty to the reference, retrain the PRM against the current policy periodically.

**Step boundaries are misaligned between the policy and the PRM.** The PRM was trained on chains where steps are newline-separated; the policy emits chains where steps are sometimes split across multiple lines (or vice versa). Either retrain the PRM on the policy's distribution of step boundaries, or post-process the policy's output to normalize boundaries before scoring.

**PRM scores aren't calibrated.** The raw score is supposed to be a probability that the step is good, but the PRM often outputs values that are systematically too confident (all scores near 0 or 1) or too cautious (all scores near 0.5). For reranking this matters less because you're just comparing scores; for using the PRM as a value function in tree search it matters a lot. The standard fix is post-hoc calibration on a held-out set: fit a temperature parameter `T` and use `softmax(logits / T)` instead of the raw softmax. Platt scaling and isotonic regression also work.

**The PRM is great on the training distribution and terrible on anything else.** Especially common when the PRM was trained on outputs from a single generator. The PRM has learned the surface patterns of that generator's chains rather than the underlying step quality. Symptom: the PRM scores a hand-written correct solution lower than a generator-produced wrong one. Fix: train the PRM on chains from a diverse set of generators, or include adversarial chains (perturbations of correct solutions) in the training set.

**Best-of-N with PRM is worse than ORM at small N.** This is reported in some replications. The intuition: at small N, the PRM's per-step noise dominates; at large N, the discrimination advantage of the per-step signal kicks in. If your N is small for compute reasons, an ORM may genuinely be the better choice. Lightman et al.'s headline number used very large N (`N=1860`); the gap shrinks at small N.

---

## A note on terminology

"Process reward model," "process-supervised reward model," and "PRM" are used interchangeably in the literature. Some papers use "step-level reward model" or "stepwise verifier." Lightman et al. used "PRM" and the term has stuck.

"Outcome reward model" or "ORM" specifically means a learned model trained on chain-level correctness labels. A rule-based outcome verifier (regex match against the ground truth) is sometimes also called outcome supervision, but it's not a model — it's a function. In careful writing, "outcome reward" covers both learned and rule-based, while "ORM" specifically means the learned variant. Lecture 15 used "verifier" for the rule-based variant; this lecture follows the same convention.

"Process supervision" describes the training regime; "PRM" describes the artifact the regime produces. You can have process supervision without a learned PRM if the per-step signal comes from a checker (e.g., proof assistant tactics). That's not common in practice for the math/code reasoning settings most of the literature is about, but it's the cleanest case where process supervision works.

---

## References

All arXiv IDs verified against arxiv.org. Venue/year cross-checked where given.

**Process vs outcome supervision (the comparison)**

- Uesato, Kushman, Kumar, Song, Siegel, Wang, Creswell, Irving, Higgins. 2022. "Solving math word problems with process- and outcome-based feedback." DeepMind. arXiv:2211.14275. — First systematic GSM8K comparison. Outcome supervision matches process supervision on final-answer accuracy but produces more wrong-reasoning-right-answer cases. Reduced reasoning errors among correct solutions from 14.0% to 3.4% with process supervision.

- Lightman, Kosaraju, Burda, Edwards, Baker, Lee, Leike, Schulman, Sutskever, Cobbe. 2023. "Let's Verify Step by Step." OpenAI. arXiv:2305.20050. — Released PRM800K (800K step-level human labels on MATH). PRM-reranked best-of-N reaches 78% on a representative MATH subset, outperforming ORM-reranking and majority vote at matched N. The clearest treatment of how to collect step-level labels at scale.

**Automated process supervision**

- Wang, Li, Shao, Xu, Dai, Li, Chen, Wu, Sui. 2024. "Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations." ACL 2024. arXiv:2312.08935. — Replaces human step labels with MC rollouts from a "completer" LLM. Hard estimation and soft estimation as labeling schemes. Mistral-7B from 77.9% to 89.1% on GSM8K (PPO + PRM verification).

**The DeepSeek perspective**

- Shao, Wang, Zhu, Xu, Song, Bi, Zhang, Zhang, Li, Wu, Guo. 2024. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." DeepSeek-AI. arXiv:2402.03300. — Introduces GRPO. Uses outcome-only rewards from a rule-based verifier. Discussed in Lecture 15.

- DeepSeek-AI. 2025. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv:2501.12948. — §4.2 "Unsuccessful Attempts" explicitly discusses why PRMs were not adopted: hard to define fine-grain steps in general reasoning, automated step-level annotation unsatisfactory, learned PRMs lead to reward hacking and complicate the training pipeline.

**Reading order suggestion**

If you're reading the sources: Uesato et al. (arXiv:2211.14275) is the cleanest experimental setup for the outcome-vs-process comparison and short enough to read in full. Then Lightman et al. (arXiv:2305.20050) for the canonical human-labeled PRM and the most-cited best-of-N result. Then Math-Shepherd (arXiv:2312.08935) for the automated alternative. Finish with §4.2 of the DeepSeek-R1 paper (arXiv:2501.12948) for the counter-perspective. Reading them in order gives you the arc: PRMs help → automating the labels makes them affordable → but the simpler outcome-only approach worked just as well at scale in the domain where both were tested.

---

## Next lecture

There is no Lecture 24 yet. Topics that would extend this naturally: generative reward models, LLM-as-judge in detail (the bias side beyond what Lecture 14 covered), or process supervision via execution traces for code-generating agents.
