<!-- status: unreviewed | last-reviewed: never -->

# Lecture 17: Online and Iterative Preference Optimization

_Unreviewed: no one has checked this end to end. Treat the math, code, and citations as unverified._

**Time**: ~2-3 h · **Prerequisites**: Lectures 09, 11, 12

---

## Where this fits

[Lecture 11: Direct Preference Optimization](./11-dpo.md) derived DPO as a one-shot, offline procedure: collect a fixed dataset of (prompt, chosen, rejected) triples, run the DPO loss until convergence, done. No reward model, no generation loop. [Lecture 12: Beyond DPO](./12-beyond-dpo.md) covered variants (IPO, KTO, ORPO) that change the loss but keep the same offline structure.

That structure has a gap. In practice, offline DPO tends to underperform PPO-based RLHF, especially on tasks where quality differences are large (code, math, long-form reasoning). Xu et al. 2024 ran a careful controlled comparison (arXiv:2404.10719) and found PPO consistently outperforms DPO across dialogue and code generation benchmarks when both are tuned properly. The gap isn't a minor bug; it's structural.

The field's response split roughly in two directions. One direction: fix the loss (IPO, KTO, ORPO; Lecture 12). The other direction, covered here: fix the data distribution by keeping it on-policy. Both directions were explored simultaneously through 2023–2024, and the practical consensus shifted toward the data-distribution fix, since it works with any loss (you can run iterative IPO or iterative DPO) and the gains are often larger.

This lecture is about why that gap exists and what the field did about it.

---

## Why offline DPO underperforms

The core problem is that offline DPO's preference data is collected once, from some policy that is not the policy being trained. Call the data-collection policy π_data. After a few gradient steps, π_θ has moved away from π_data. The preference pairs were labeled based on responses from π_data, but now you're training a different distribution. DPO's implicit reward function is:

```
r_DPO(x, y) = β log(π_θ(y|x) / π_ref(y|x))
```

This is only well-calibrated where the data has coverage. When π_θ starts generating responses in regions of response space that π_data never reached, the implicit reward extrapolates without constraint. The model can find high-DPO-loss minimizers that are off-distribution, exploiting gaps in the data rather than actually improving.

To see why: the DPO loss pushes `log π_θ(y_w|x) - log π_θ(y_l|x)` (the log-ratio difference) to be large. Once the training data is fixed, this can be satisfied by simultaneously lowering `π_θ(y_l)` below what the data distribution would expect, even for pairs where the true quality gap is small. The gradient doesn't know that those pairs were labeled under π_data, not the current π_θ. It just minimizes the loss.

PPO avoids this because it samples responses from the current policy at every step. The reward model is applied to responses the policy actually generates. If the policy drifts to a new region and starts producing bad responses, the reward model scores those bad responses and the policy gets negative signal immediately. DPO has no such closed loop.

Xu et al. 2024 (arXiv:2404.10719) ran a careful controlled comparison. Their finding: well-tuned PPO outperforms DPO across all their test cases, including dialogue and competitive programming. The gap is larger on tasks with high variance in response quality, exactly where the policy is most likely to generate responses outside the training data's coverage.

The fix is to keep the preference data in distribution by regenerating it as the policy changes. That's the idea behind iterative DPO.

---

## Iterative DPO

Iterative DPO runs DPO in rounds. At the start of round `r`, you have a policy π_r (initialized to the SFT model for round 0). The loop is:

1. For each prompt in your dataset, sample K responses from π_r.
2. Label the responses with a preference signal: a learned reward model, an LLM judge, or a verifiable checker (e.g., unit tests, exact-match answers).
3. Form (chosen, rejected) pairs from the labeled responses.
4. Run DPO on those pairs to get π_{r+1}.
5. Repeat from step 1 with π_{r+1}.

Because the responses in step 1 come from the current policy, the preference data is always on-distribution. The policy can't exploit coverage gaps that no longer exist. When it drifts, you regenerate data from wherever it drifted to.

### Pair selection strategies

How you form pairs from K responses per prompt matters. Three common approaches:

**Best-vs-worst**: pick the highest-scored and lowest-scored responses. Maximum signal per pair; can be noisy if the score gap is large and the worst response is obviously bad.

**Contrastive**: pick adjacent responses in the score ranking (best vs. second-best, second-best vs. third-best, etc.). More pairs per prompt; signal is finer-grained. Works well when K is large (K ≥ 8) and the scorer is reliable.

**Threshold-based**: label responses above a threshold as "chosen" and below as "rejected," then sample pairs across the threshold. Good when your scorer is binary (correct/incorrect) and you want to use all positive examples.

The choice interacts with K: with K=4 and a binary scorer (math answer check), you might get 0, 1, 2, 3, or 4 correct responses. If 0 or all 4 are correct, you have no signal for that prompt in that round. Larger K reduces the probability of this "all-same" outcome.

Three papers show this loop working at different scales:

**Self-Rewarding Language Models** (Yuan et al. 2024, arXiv:2401.10020) uses the model itself as the judge: it generates candidate responses, evaluates them with LLM-as-a-Judge prompting, forms DPO pairs from the evaluations, and trains on those. Iterating the loop improves both instruction-following quality and the model's ability to judge responses. The key observation is that both capabilities improve together, since the judge is the same model as the policy.

**Iterative Reasoning Preference Optimization** (Pang et al. 2024, arXiv:2404.19733, NeurIPS 2024) applies the iterative loop specifically to chain-of-thought reasoning. It generates competing reasoning chains, scores them by whether they reach the correct answer, and runs DPO+NLL on the resulting pairs. Starting from Llama-2-70B-Chat, two rounds of iteration push GSM8K accuracy from 55.6% to 81.6%. The NLL term (maximizing log-likelihood of chosen responses alongside the DPO objective) turns out to matter. Without it, the policy degrades fluency.

**RSO: Statistical Rejection Sampling Improves Preference Optimization** (Liu et al. 2023, arXiv:2309.06657, ICLR 2024) makes a formal argument about why you need on-policy data: the maximum likelihood estimator of the optimal policy under the RLHF objective requires preference pairs sampled from that optimal policy, not from a fixed offline distribution. RSO approximates this using rejection sampling: sample many candidate responses, keep only those whose likelihood under the current policy is high enough (reject the rest), and train DPO on those retained responses. This is a lighter-weight approximation to full on-policy sampling. You don't need to update the policy between samples, just filter to responses that are plausible under it. Empirically RSO outperforms both offline DPO and SLiC (a related offline method).

RSO's rejection sampling is also the training-time equivalent of best-of-n inference (sample n responses, pick the highest-scored one to return to the user). Best-of-n is a useful baseline: for a given compute budget, is it better to train the policy harder (DPO/PPO) or to sample more responses at inference and filter? Gao et al. 2022 (arXiv:2210.10760) measured this: best-of-n scales quadratically in gold reward vs KL, while RL scales logarithmically. Best-of-n is more sample-efficient for small KL budgets; RL dominates at large KL. In practice you usually train and then apply best-of-n on top.

The Llama 3 post-training pipeline (Dubey et al. 2024, arXiv:2407.21783) also uses iterative preference optimization. The report describes several rounds of DPO training, where each round uses preference data generated from the model produced by the previous round. The most recent batch of preference data is always collected from the current policy. The report notes that DPO required less compute than PPO at large scale and performed better on instruction-following benchmarks for their setup, though the full training details aren't made public.

One open question in the iterative literature: should the reference policy π_ref update between rounds? The default (and what the DPO derivation assumes) is to keep π_ref fixed as the original SFT model throughout all rounds. Updating π_ref to the previous round's checkpoint changes the KL anchor: the policy is now regularized toward its most recent self, not the SFT model. This relaxes the KL constraint and lets the policy move further per round, which can accelerate improvement but also accelerates over-optimization. Most published implementations keep π_ref fixed.

---

## Online preference optimization

Iterative DPO does rounds: generate data, train for a while, repeat. Take the limit where you generate data and update continuously, and you get online DPO.

Online DPO looks like this: on each step, sample two responses from the current policy for a given prompt, score them with a reward model or LLM judge, treat the higher-scored one as chosen, and immediately run a DPO-style update. No waiting for a full round. The preference data is always fresh from the current policy.

Guo et al. 2024 (arXiv:2402.04792, "Direct Language Model Alignment from Online AI Feedback") call their version OAIF (Online AI Feedback). They use a large LLM annotator (PaLM 2-L) to label pairs of on-policy responses each step and train with DPO on those labels. The annotating model is much larger than the training model (PaLM 2-XS), so the judge has higher quality than the policy. Human evaluation across several tasks shows OAIF outperforms both offline DPO and PPO-style RLHF in their setup.

The Zephyr recipe (Tunstall et al. 2023, arXiv:2310.16944, cited in Lecture 11) is an early version of the iterative idea applied to AI feedback: they used UltraFeedback (AI-labeled preference data generated by GPT-4 rating model outputs) and ran one round of DPO on it. It's not iterative in the multi-round sense (one round, static data), but the preference data came from AI labeling of diverse model outputs rather than from human annotators. The result matched much larger PPO-trained models on MT-Bench. The multi-round version of this recipe is what labs moved to in 2024.

The online DPO / OAIF setup also shows that you don't need the judge to be the model being trained. In OAIF, the training model is PaLM 2-XS and the judge is PaLM 2-L, two different models. This is generally safer: the judge isn't accumulating biases from the training signal, and its quality stays stable across rounds. When the judge and policy are the same model (self-rewarding), you get a feedback loop where both improve together, which is more sample-efficient but also means the judge's errors get baked in to the policy faster.

This creates a spectrum:

```
offline DPO
  ↓   (add one refresh of data mid-training)
iterative DPO (few rounds)
  ↓   (shrink round length toward zero)
online DPO
  ↓   (switch from DPO loss to PPO/GRPO clipped objective)
PPO / GRPO
```

Moving right along this spectrum increases cost: you need a generation pass on every training step, plus a scoring pass. But the policy is always close to the data distribution, so the implicit reward is well-calibrated throughout. The tradeoff is compute versus on-policy-ness.

The DPO loss doesn't change as you move along this spectrum. The functional form is the same whether the pairs came from a six-month-old static dataset or were sampled thirty seconds ago. What changes is the distribution of (chosen, rejected) relative to the current policy. Online DPO just ensures that distribution is always fresh.

One practical difference between online DPO and PPO: PPO has a clipped ratio `clip(π_new/π_old, 1-ε, 1+ε)` that explicitly limits the policy update size per step. DPO's β parameter does something similar (it's a KL penalty), but it doesn't directly bound the per-step ratio. If you're updating frequently on very fresh on-policy data, the DPO loss can still take large steps. Some online DPO implementations add gradient clipping or schedule β to compensate.

Another difference: PPO maintains a value function (a critic) that estimates expected future reward. The critic gives PPO a lower-variance gradient estimate and lets it handle long-horizon credit assignment. DPO has no critic: the signal comes entirely from the log-ratio difference between chosen and rejected. For tasks where responses are short and the quality signal is immediate (single-turn instruction following), this is fine. For longer tasks with sparse signal, the critic matters more, which is part of why PPO still has an advantage on complex reasoning and multi-turn tasks.

GRPO (Lecture 12, Shao et al. 2024, arXiv:2402.03300) sits between online DPO and PPO in this regard: it computes group-relative advantages from multiple samples per prompt, giving a lower-variance signal than a single pair, but it still doesn't require a learned value function. This makes it cheaper than PPO while being more signal-efficient than online DPO on tasks with objective scoring.

In terms of implementation complexity: offline DPO is a standard supervised training loop. Iterative DPO adds an outer generation loop. Online DPO moves that generation inside every step. GRPO adds group sampling and advantage normalization. PPO adds a value network, GAE computation, and multiple minibatch epochs per rollout. Each step up requires more infrastructure: more GPU memory, more hyperparameters, more monitoring.

---

## Generative reward models

All of the above needs a preference signal at labeling time. Lecture 09 covered the classical approach: train a Bradley-Terry reward model with a scalar head on top of a pretrained LLM, then call `reward_model(prompt, response)` to get a float. That works, but it has costs: you need to collect enough preference data to train the RM, the RM can overfit or be miscalibrated, and a scalar head throws away reasoning.

An alternative is to prompt an LLM directly to judge responses. Two common interfaces:

**Pairwise judgment**: present both responses and ask the judge which is better and why. Chain-of-thought happens naturally. The model reasons through the comparison before committing.

**Pointwise scoring**: present one response at a time and ask for a rating (1–10, or a rubric-based score). Easier to calibrate across prompts; no inter-response comparison needed.

A minimal pairwise judge prompt looks like:

```
You are evaluating two responses to a user prompt.

[User Prompt]
{prompt}

[Response A]
{response_a}

[Response B]
{response_b}

Which response is more helpful, accurate, and well-reasoned?
Do not let response length influence your judgment.
First briefly explain your reasoning, then output "A" or "B".
```

The explicit "do not let response length influence" instruction is a direct mitigation for verbosity bias. It doesn't eliminate the bias but reduces it.

Zheng et al. 2023 (arXiv:2306.05685, NeurIPS 2023) studied this systematically via MT-Bench and Chatbot Arena. They found that GPT-4 used as a judge matches human preference labels at over 80% agreement, roughly the same as inter-human agreement, making it a viable substitute for human labeling. They also identified the main failure modes:

- **Position bias**: the judge tends to prefer whichever response appears first (or second) in a pairwise prompt. Fix: randomize order across two calls, check for consistency.
- **Verbosity bias**: longer responses score higher even when they aren't better. Fix: instruct the judge explicitly to penalize unnecessary length.
- **Self-preference**: a model used as its own judge tends to prefer responses that match its own style. Known issue in self-rewarding setups; use a different model as judge when possible.

The term "generative reward model" refers to a reward model that produces a natural-language rationale alongside its score, rather than a raw scalar. The rationale can encode information a scalar loses (e.g., why one response is safer, which factual claim is wrong), and the generation process can use chain-of-thought to reason before committing to a score. The downside is cost: a scalar RM is one forward pass per response; a generative RM generates tens to hundreds of tokens per comparison.

Pros of LLM-as-judge over a scalar RM: no RM training needed, can adapt to new criteria via prompting, output is interpretable.
Cons: slower (a full generation per comparison, not just a forward pass), non-deterministic, position and verbosity biases compound across rounds, and if the judge is the same model you're training, its biases get reinforced round after round.

---

## Reward over-optimization

Any time you optimize a policy against a proxy reward, whether that proxy is a learned RM or an LLM judge, you can push past the optimum. The proxy keeps going up; the true objective starts going down. This is Goodhart's Law applied to RL.

Gao, Schulman, and Hilton 2022 (arXiv:2210.10760, ICML 2023) measured this carefully using a synthetic setup where a "gold" reward model plays the role of humans. They found:

- The gold reward increases as you optimize against a proxy RM up to a point, then decreases even as the proxy keeps increasing.
- The rate of over-optimization scales predictably with proxy RM size: smaller RMs over-optimize faster per unit of KL divergence from the reference.
- For reinforcement learning, the gold reward as a function of KL grows roughly logarithmically before turning over. For best-of-n sampling, the shape is different (approximately quadratic).

This means every RL-from-preferences pipeline needs protection against over-optimization:

- **KL-to-reference penalty**: keeping π_θ close to π_ref limits how far the policy can exploit proxy reward gaps. DPO's β parameter does this implicitly.
- **Early stopping**: monitor a held-out metric that isn't the proxy reward. Stop when it stops improving.
- **RM ensembles**: if multiple independent RMs all agree a response is good, it's less likely to be a proxy artifact.
- **Fresh preference data**: resetting the RM or re-collecting preference data after several rounds of training resets the proxy, so accumulated exploitation disappears.

That last point is an argument for the iterative loop: each round regenerates data from the current policy, so the "proxy gap" that accumulated over the round gets reset. Over-optimization within a round is bounded by how many gradient steps you take before regenerating.

A rough mental model of what the curve looks like in practice:

```python
# Pseudocode: not runnable, illustrating the shape

kl_values = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]  # KL(policy || reference)
proxy_reward = [0.0, 0.4, 0.7, 1.0, 1.4, 1.8, 2.2]  # proxy keeps going up
gold_reward  = [0.0, 0.3, 0.6, 0.8, 0.7, 0.4, 0.0]  # gold peaks and falls

# The gap between proxy and gold starts small,
# grows as the policy exploits the proxy's blind spots,
# and eventually the gold reward turns negative.
# You want to stop somewhere in the 2.0–5.0 KL range in this sketch.
```

The shape varies by RM size and task, but the qualitative pattern is consistent with Gao et al.'s results: there's a peak, and you overshoot it if you don't monitor a separate signal.

---

## The 2024–2025 RLHF stack

The broad pattern across labs is:

1. **Pretraining**: large-scale next-token prediction on text.
2. **SFT**: fine-tune on demonstrations of desired behavior (instruction following, helpfulness).
3. **Preference optimization**: one or more rounds of PPO, DPO, or GRPO against a preference signal.

The preference signal is increasingly AI-generated rather than human-labeled. See [Lecture 14: Constitutional AI, RLAIF](./14-constitutional-ai-rlaif.md) for the RLAIF family. For reasoning tasks, verifiable rewards (math answers, code execution) are common. See [Lecture 15: RL with Verifiable Rewards](./15-rl-verifiable-rewards.md). A reasoning model adds RLVR on top of the alignment stack: the preference signal is correct-or-not rather than a human-labeled scalar.

The preference optimization step is often iterated. The Llama 3 report (arXiv:2407.21783) describes multiple rounds of DPO with fresh on-policy preference data each round. The exact number of rounds, data volumes, and filtering procedures aren't fully public.

There's no single canonical recipe. What the labs agree on:

- SFT before preference optimization (skip it and the preference step is much weaker).
- KL regularization to the reference throughout (β in DPO; the KL penalty in PPO).
- On-policy data is better than static data, and multiple rounds are better than one.
- The preference signal, whatever it is, will be Goodharted if you push hard enough.

What they disagree on (or at least don't publish clearly): the exact ratio of human to AI preference labels, how many rounds, when to switch from DPO to PPO or GRPO, and how to handle multi-turn preference data. Multi-turn is genuinely harder: a preference label on a final response in a conversation doesn't tell you which of the assistant's earlier turns were responsible for the quality difference. Some labs collect turn-level preference labels; others label at the conversation level and accept the credit-assignment noise.

A few things that are known from public reports and that differ from the earliest RLHF papers (InstructGPT, 2022):

- The preference signal is increasingly AI-generated. Human labelers set the initial calibration and handle edge cases, but the volume of labeled data comes from AI feedback (constitutional AI, LLM-as-judge, or verifiable checkers). This shifts cost from labeler time to inference compute.
- Filtering matters as much as the loss. Before DPO or PPO training, most pipelines filter the preference dataset: remove pairs where the score gap is too small, remove prompts where the model already gets high scores on all K responses, deduplicate near-duplicate prompts. The filtering criteria are rarely published.
- Data curation for SFT is still the highest-leverage intervention. A better SFT model produces better π_ref, which gives the preference optimization step a better starting point and a more meaningful KL anchor. The DPO loss measures deviation from π_ref. If π_ref is already producing decent responses, the KL constraint keeps the policy near a good distribution. If π_ref is weak, the KL constraint anchors to a weak distribution and you need higher β (tighter constraint) or more rounds just to maintain coherence.
- Prompt curation matters too. If your training prompts don't cover the skills you care about, no amount of iterating will help. The iterative loop can only improve on the prompts it sees.

---

## Iterative DPO: a code sketch

The following shows the loop structure and the LLM-judge labeling interface. The DPO loss itself is the same function from Lecture 11. Refer there for its derivation. What this code adds is the sampling loop, the scoring interface, and order-randomization to reduce position bias in pairwise judgment.

```python
import random
import torch
import torch.nn.functional as F
from typing import Callable


# The DPO loss from Lecture 11 (referenced, not re-derived here)
def dpo_loss(
    policy_chosen_logprobs: torch.Tensor,
    policy_rejected_logprobs: torch.Tensor,
    ref_chosen_logprobs: torch.Tensor,
    ref_rejected_logprobs: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    chosen_ratio = policy_chosen_logprobs - ref_chosen_logprobs
    rejected_ratio = policy_rejected_logprobs - ref_rejected_logprobs
    logits = beta * (chosen_ratio - rejected_ratio)
    return -F.logsigmoid(logits).mean()


# --- LLM-judge interface ---

def llm_judge_pairwise(
    prompt: str,
    response_a: str,
    response_b: str,
    judge_fn: Callable[[str], str],
) -> str:
    """
    Call an LLM judge to compare two responses.
    Randomizes order to reduce position bias.
    Returns "a" or "b" (the winner in the *original* ordering).

    judge_fn: takes a formatted string, returns the judge's text output.
    """
    # Randomize which response appears first in the prompt
    flip = random.random() < 0.5
    if flip:
        first, second = response_b, response_a
        first_label, second_label = "Response 2", "Response 1"
    else:
        first, second = response_a, response_b
        first_label, second_label = "Response 1", "Response 2"

    judge_prompt = (
        f"Prompt: {prompt}\n\n"
        f"{first_label}:\n{first}\n\n"
        f"{second_label}:\n{second}\n\n"
        "Which response is better? Answer with '1' or '2' and a brief reason."
    )
    output = judge_fn(judge_prompt)

    # Parse the judge's choice back to original labels
    if "1" in output and "2" not in output:
        position_winner = "first"
    elif "2" in output and "1" not in output:
        position_winner = "second"
    else:
        # Ambiguous: skip this pair
        return "tie"

    if flip:
        # first=b, second=a; if judge picked "first", winner is b
        return "b" if position_winner == "first" else "a"
    else:
        return "a" if position_winner == "first" else "b"


# --- Iterative DPO loop ---

def iterative_dpo(
    policy,            # current policy (LLM, mutable)
    ref_policy,        # reference policy (frozen SFT model)
    prompts: list,     # training prompts
    scorer,            # callable: (prompt, response) -> float, or use judge_fn
    compute_logprobs,  # callable: (model, prompt, response) -> float (log prob)
    optimizer,
    num_rounds: int = 3,
    K: int = 4,        # responses per prompt per round
    beta: float = 0.1,
    steps_per_round: int = 100,
):
    """
    Iterative DPO training loop.

    Each round:
      1. Sample K responses per prompt from the current policy.
      2. Score responses (reward model or LLM judge).
      3. Build (chosen, rejected) pairs.
      4. Run DPO updates on those pairs.
    """
    for round_idx in range(num_rounds):
        print(f"\n--- Round {round_idx + 1} / {num_rounds} ---")

        # Step 1: generate responses from current policy
        pairs = []
        for prompt in prompts:
            responses = [policy.generate(prompt) for _ in range(K)]

            # Step 2: score responses
            scores = [scorer(prompt, r) for r in responses]

            # Step 3: pick best and worst as (chosen, rejected)
            best_idx = max(range(K), key=lambda i: scores[i])
            worst_idx = min(range(K), key=lambda i: scores[i])

            if scores[best_idx] == scores[worst_idx]:
                # No signal; skip this prompt this round
                continue

            pairs.append({
                "prompt": prompt,
                "chosen": responses[best_idx],
                "rejected": responses[worst_idx],
            })

        print(f"  Built {len(pairs)} preference pairs from {len(prompts)} prompts")

        # Step 4: DPO updates on the freshly collected pairs
        policy.train()
        for step in range(steps_per_round):
            batch = random.choices(pairs, k=8)

            chosen_lp = torch.stack([
                compute_logprobs(policy, p["prompt"], p["chosen"]) for p in batch
            ])
            rejected_lp = torch.stack([
                compute_logprobs(policy, p["prompt"], p["rejected"]) for p in batch
            ])

            with torch.no_grad():
                ref_chosen_lp = torch.stack([
                    compute_logprobs(ref_policy, p["prompt"], p["chosen"]) for p in batch
                ])
                ref_rejected_lp = torch.stack([
                    compute_logprobs(ref_policy, p["prompt"], p["rejected"]) for p in batch
                ])

            loss = dpo_loss(chosen_lp, rejected_lp, ref_chosen_lp, ref_rejected_lp, beta)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            if step % 20 == 0:
                print(f"  step {step:3d}  loss={loss.item():.4f}")

        # After each round the policy is π_{r+1};
        # the next round will sample from it.

        # Useful diagnostic: average score gap between chosen and rejected.
        # If this shrinks across rounds, the scorer isn't differentiating well
        # or the policy has learned to cluster near the scorer's sweet spot.
        avg_gap = sum(
            scorer(p["prompt"], p["chosen"]) - scorer(p["prompt"], p["rejected"])
            for p in pairs
        ) / max(len(pairs), 1)
        print(f"  Round {round_idx + 1} complete. Avg score gap: {avg_gap:.3f}")


# --- Example scorer using an LLM judge ---

def make_judge_scorer(judge_fn: Callable[[str], str]):
    """
    Wrap llm_judge_pairwise into a format compatible with the loop above.

    For the loop we need a pointwise score, not a pairwise winner.
    Simplest approach: call the judge multiple times against a fixed reference
    response ("the previous best") and count wins. Or use a pointwise prompt.
    """
    def pointwise_scorer(prompt: str, response: str) -> float:
        rating_prompt = (
            f"Prompt: {prompt}\n\n"
            f"Response: {response}\n\n"
            "Rate this response from 1 (poor) to 10 (excellent). Output only the number."
        )
        output = judge_fn(rating_prompt).strip()
        try:
            return float(output.split()[0])
        except (ValueError, IndexError):
            return 5.0  # fallback: neutral score on parse failure
    return pointwise_scorer
```

A few notes on the code above:

- `scorer` can be a reward model call or the `make_judge_scorer` wrapper. The loop structure is the same either way.
- The code picks best-vs-worst from K samples. An alternative is to use all valid pairs: for K=4 responses, you can form up to 6 pairs (4 choose 2) if you score all of them. More pairs per prompt means more gradient signal per round, but also more correlation between pairs (they share a prompt and sometimes share reasoning patterns).
- Order randomization in `llm_judge_pairwise` cuts position bias roughly in half in practice; running two calls with flipped order and requiring agreement cuts it further.
- `steps_per_round` is a hyperparameter. Too many steps per round and you over-optimize on the current batch before refreshing; too few and each round makes little progress. In practice, 100–500 steps per round with early stopping on a held-out set.
- The scorer (RM or judge) is in the inner loop of every round. Its biases accumulate across rounds. If the judge prefers verbose responses, the policy will trend verbose over rounds. This is not a bug you can easily see round-to-round, but it compounds.
- The reference policy (`ref_policy`) stays frozen for the entire training run, not just one round. This is correct: the KL term is always measured against the original SFT model, not against the previous round's checkpoint.

---

## When to use this

**Offline DPO**: when you have a good static preference dataset and can't afford a generation loop. Easiest to run; weakest on-policy signal. Acceptable for general instruction following when the dataset is large and diverse, worse for tasks where the policy needs to explore (math, code, long reasoning).

**Iterative DPO**: when you have a scorer (RM or judge) and can afford K × |prompts| generation calls per round. This is the default choice for production alignment pipelines as of 2024–2025. Two to four rounds give most of the gain in most reported experiments: Pang et al. 2024 show two rounds going from 55.6% to 81.6% on GSM8K; Yuan et al. 2024 show improvement over three iterations before plateauing. More rounds risk compounding scorer biases and require monitoring a held-out metric to catch saturation.

**Online DPO**: when you can afford continuous on-policy generation and scoring. The preference data is always fresh. Practical if your scorer is fast (e.g., a small RM). Gets expensive with a large LLM judge.

**PPO / GRPO**: when you need maximum on-policy-ness, have a verifiable reward (math, code execution), or are doing RLVR-style training. PPO's clipped objective and value function add engineering complexity but give tighter control over the update. GRPO (Lecture 12) is a lighter alternative that skips the value function.

All of these should run with a KL-to-reference term. Without it, the policy drifts far enough that the reference's calibration becomes useless and over-optimization accelerates.

**Practical gotchas**:

- The scorer is in the critical path. A slow LLM judge makes every round expensive. Cache judge calls where you can.
- Distribution shift between rounds can be large. If the policy changes a lot in one round, the preference pairs from the previous round are stale by the next. Keep steps_per_round moderate.
- Knowing when to stop iterating is non-trivial. Proxy reward (what the scorer says) will keep going up. Track a separate held-out metric (human evaluation, a separate RM, or a verifiable test set) to detect over-optimization.
- LLM-judge non-determinism: temperature and sampling mean the same pair can get different labels on different calls. Use temperature=0 or average over multiple calls if you can afford it.
- If your judge and your policy are the same model family, self-preference bias will grow over rounds. Use a different model as judge when possible.
- The score gap between chosen and rejected tends to shrink over rounds, as the policy learns the scorer's preferences. Watch for rounds where nearly all K responses get the same score. That means the scorer has saturated and the loop has stopped producing useful signal. You either need a harder scorer, harder prompts, or more rounds with a different data mix.
- Memory cost: if you're keeping the reference policy in GPU memory alongside the training policy, you're paying ~2x the model size in VRAM. For large models, quantize the reference or offload it to CPU between scoring steps.

---

## Exercises

1. Run two rounds of iterative DPO on a small preference dataset (50–100 prompts, any open-instruction set). Compare the round-2 policy against the one-shot DPO baseline on a held-out set. Does the iterative version improve, and does it matter how many steps you run per round? Try 50 steps and 200 steps per round and compare.

2. Build an LLM-judge labeler that calls a model twice per pair (with flipped order) and records both judgments. Measure its position bias: what fraction of pairs get a different winner when you flip order? Plot this against temperature. Does higher temperature increase or decrease position bias?

3. Train a policy for several rounds while logging both the proxy reward (from your scorer) and a separate held-out metric (human eval, a different RM, or test-set accuracy). Plot both over rounds. Find the point where the proxy keeps rising but the held-out metric levels off or drops. That's over-optimization becoming visible.

4. Take the `iterative_dpo` code sketch above and swap in a binary scorer (correct/incorrect on math problems). Change the pair-selection from best-vs-worst to threshold-based: any correct response is "chosen," any incorrect is "rejected." How does the number of usable pairs per round change with K=4 vs K=8?

5. Read the Gao et al. 2022 scaling laws paper (arXiv:2210.10760) and reproduce the qualitative shape of the proxy-vs-gold curves using a toy setup: train a small reward model on 1000 preference pairs, use it as the proxy, use a held-out 200-pair test set as the "gold" evaluator, and run 500 steps of best-of-n selection. Plot proxy reward and gold reward against the number of best-of-n samples. Where does the gold reward peak?

---

## References

**Xu et al. 2024**. "Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study." arXiv:2404.10719. Verified. Systematic comparison showing PPO outperforms DPO across dialogue and code generation when both are well-tuned; characterizes DPO's off-distribution exploitation failure mode.

**Yuan et al. 2024**. "Self-Rewarding Language Models." arXiv:2401.10020. Verified. Iterative DPO loop where the model acts as its own LLM judge; both instruction-following and judging quality improve across iterations.

**Pang et al. 2024**. "Iterative Reasoning Preference Optimization." arXiv:2404.19733. NeurIPS 2024. Verified. Applies iterative DPO to chain-of-thought reasoning; uses a DPO+NLL objective and correctness-based scoring; shows large gains on GSM8K and MATH from Llama-2-70B-Chat.

**Liu et al. 2023**. "Statistical Rejection Sampling Improves Preference Optimization." arXiv:2309.06657. ICLR 2024. Verified. Formal motivation for on-policy preference data; rejection sampling to approximate the optimal policy's distribution; RSO outperforms offline DPO and SLiC.

**Tunstall et al. 2023**. "Zephyr: Direct Distillation of LM Alignment." arXiv:2310.16944. Referenced in Lecture 11. One round of DPO on AI-labeled preference data from GPT-4 (UltraFeedback); competitive with much larger PPO-trained models on MT-Bench. Cited here as the single-round precursor to the multi-round iterative recipes.

**Guo et al. 2024**. "Direct Language Model Alignment from Online AI Feedback." arXiv:2402.04792. Verified. Online DPO using an LLM annotator to label on-policy response pairs each step (OAIF); outperforms offline DPO and PPO-RLHF on human evaluation.

**Zheng et al. 2023**. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." arXiv:2306.05685. NeurIPS 2023. Verified. Establishes LLM-as-judge as viable (GPT-4 matches human agreement at >80%); documents position, verbosity, and self-preference biases and mitigations.

**Gao, Schulman, Hilton 2022**. "Scaling Laws for Reward Model Overoptimization." arXiv:2210.10760. ICML 2023. Verified. Measures proxy-vs-gold reward divergence as a function of KL from reference; gold reward peaks then falls as proxy keeps rising; overoptimization rate scales with RM size. Compares RL vs. best-of-n: best-of-n is more efficient at small KL, RL dominates at large KL.

**Dubey et al. 2024**. "The Llama 3 Herd of Models." arXiv:2407.21783. Verified. Describes Llama 3's post-training pipeline including iterative DPO with fresh on-policy preference data each round. The exact round count, data volumes, and filtering details are not fully described in the public report.

---

## Next lecture

[Lecture 18: Distillation of Reasoning Models](./18-distillation-reasoning.md): how the big RL-trained reasoning models get compressed into much smaller ones via SFT on the teacher's chain-of-thought traces.
