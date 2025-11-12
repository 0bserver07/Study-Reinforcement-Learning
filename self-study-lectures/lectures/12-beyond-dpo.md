# Lecture 12: Beyond DPO - GRPO, RRHF, IPO, and the 2024-2025 Wave

> **The Question**: "DPO fixed PPO's instability. But can we do even better?"

**Time**: 4-5 hours | **Prerequisites**: Lectures 09-11 | **Difficulty**: ⭐⭐⭐⭐⭐

---

## 🎯 Why This Matters

DPO (2023) was revolutionary - no reward model, no PPO instability. But researchers found issues:

- DPO can be **overconfident** (ignores uncertainty in preferences)
- Requires **paired preference data** (chosen vs rejected)
- Can **underfit** on complex tasks
- Doesn't leverage **multiple candidate samples** well

**Enter 2024-2025**: A wave of new methods that keep DPO's simplicity but fix its weaknesses.

**This is what powers DeepSeekMath, Llama 3, and the latest alignment methods.**

---

## Part 1: The Landscape (Post-DPO Methods)

### The Family Tree

```
RLHF (2017-2022)
  └─ PPO-based RLHF
       ├─ InstructGPT (2022)
       └─ ChatGPT (2022)

Direct Methods (2023+)
  ├─ DPO (2023) - Direct Preference Optimization
  ├─ IPO (2023) - Identity Preference Optimization
  ├─ KTO (2024) - Kahneman-Tversky Optimization
  ├─ ORPO (2024) - Odds Ratio Preference Optimization
  ├─ RRHF (2024) - Rank Responses to align Human Feedback
  └─ GRPO (2024) - Group Relative Policy Optimization
```

### Quick Comparison

| Method | Year | Key Idea | Data Needs | Complexity |
|--------|------|----------|------------|------------|
| PPO | 2017 | RL with reward model | Preferences | High (3 models) |
| DPO | 2023 | Implicit reward model | Pairwise prefs | Low (1 model) |
| IPO | 2023 | Regularize harder | Pairwise prefs | Low |
| RRHF | 2024 | Rank loss | Multiple responses | Medium |
| GRPO | 2024 | Relative advantages | Multiple responses | Medium |
| KTO | 2024 | Binary feedback | Thumbs up/down | Low |
| ORPO | 2024 | Odds ratio | Pairwise prefs | Low |

---

## Part 2: GRPO - Group Relative Policy Optimization

### The Core Insight

**DPO weakness**: Only looks at pairs (chosen vs rejected).

**GRPO insight**: What if we sample **multiple responses** per prompt and learn from their **relative quality**?

```python
# DPO:
prompt → [response_chosen, response_rejected]
# Learn: make chosen more likely

# GRPO:
prompt → [response_1, response_2, response_3, response_4, ...]
# Learn: make better responses more likely relative to worse ones
```

### The Algorithm (Intuition)

```
1. For prompt x, sample K responses from policy: {y₁, y₂, ..., yₖ}
2. Score each with some metric (e.g., pass@1 for code, correctness for math)
3. Compute advantages based on relative rankings
4. Update policy to increase probability of better responses
```

**Key difference from PPO**: No separate reward model! Use direct metrics when available.

### The Math

GRPO uses a **ranking-based advantage**:

```
A_i = (rank_i - mean_rank) / std_rank
```

Where rank_i is the rank of response i among the K samples.

Then apply a PPO-style update:

```
L = 𝔼[min(ratio · A, clip(ratio, 1-ε, 1+ε) · A)]
```

But `ratio = π_θ(y|x) / π_old(y|x)` and A comes from rankings, not a value function.

**Personal Note**: This clicked for me when I realized it's like PPO but the "advantage" is just "how good is this response compared to the others I sampled?"

---

## Part 3: GRPO Implementation

### Complete Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple
import numpy as np

class GRPOTrainer:
    """
    Group Relative Policy Optimization trainer.

    Key idea: Sample multiple responses per prompt,
    rank them, and update policy based on relative quality.
    """

    def __init__(
        self,
        model,
        tokenizer,
        ref_model=None,
        beta=0.1,  # KL penalty coefficient
        epsilon=0.2,  # PPO clip epsilon
        group_size=4,  # Number of responses per prompt
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.ref_model = ref_model or model  # Reference model for KL
        self.beta = beta
        self.epsilon = epsilon
        self.group_size = group_size

    def sample_responses(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        max_length: int = 512
    ) -> List[List[str]]:
        """
        Sample multiple responses for each prompt.

        Returns:
            responses: List of lists, responses[i] contains K responses for prompts[i]
        """
        all_responses = []

        for prompt in prompts:
            responses = []

            # Sample K responses
            for _ in range(self.group_size):
                inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Remove prompt from response
                response = response[len(prompt):].strip()
                responses.append(response)

            all_responses.append(responses)

        return all_responses

    def score_responses(
        self,
        prompts: List[str],
        responses: List[List[str]],
        scoring_fn
    ) -> List[List[float]]:
        """
        Score each response using provided scoring function.

        Args:
            scoring_fn: Function (prompt, response) -> float score

        Returns:
            scores: List of lists, scores[i][j] is score for responses[i][j]
        """
        all_scores = []

        for prompt, response_group in zip(prompts, responses):
            scores = [scoring_fn(prompt, resp) for resp in response_group]
            all_scores.append(scores)

        return all_scores

    def compute_advantages(self, scores: List[List[float]]) -> List[List[float]]:
        """
        Compute advantages based on relative rankings within each group.

        Higher score → higher rank → positive advantage
        """
        all_advantages = []

        for score_group in scores:
            # Rank scores (higher is better)
            ranks = np.argsort(np.argsort(score_group))  # 0 = worst, K-1 = best

            # Normalize to advantages
            mean_rank = np.mean(ranks)
            std_rank = np.std(ranks) + 1e-8
            advantages = (ranks - mean_rank) / std_rank

            all_advantages.append(advantages.tolist())

        return all_advantages

    def compute_log_probs(
        self,
        prompts: List[str],
        responses: List[List[str]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log probabilities of responses under current and reference policies.

        Returns:
            current_log_probs: [batch_size, group_size]
            ref_log_probs: [batch_size, group_size]
        """
        current_log_probs = []
        ref_log_probs = []

        for prompt, response_group in zip(prompts, responses):
            curr_lps = []
            ref_lps = []

            for response in response_group:
                # Tokenize prompt + response
                full_text = prompt + response
                inputs = self.tokenizer(full_text, return_tensors='pt').to(self.model.device)
                input_ids = inputs['input_ids']

                # Get prompt length for masking
                prompt_ids = self.tokenizer(prompt, return_tensors='pt')['input_ids']
                prompt_len = prompt_ids.shape[1]

                # Current policy log probs
                with torch.no_grad():
                    outputs = self.model(input_ids, labels=input_ids)
                    logits = outputs.logits

                # Compute log probs only for response tokens
                log_probs = F.log_softmax(logits, dim=-1)
                token_log_probs = log_probs.gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

                # Mask prompt tokens
                mask = torch.zeros_like(token_log_probs)
                mask[:, prompt_len:] = 1

                # Sum log probs for response
                curr_lp = (token_log_probs * mask).sum().item()
                curr_lps.append(curr_lp)

                # Reference policy log probs
                with torch.no_grad():
                    ref_outputs = self.ref_model(input_ids, labels=input_ids)
                    ref_logits = ref_outputs.logits
                    ref_log_probs_dist = F.log_softmax(ref_logits, dim=-1)
                    ref_token_log_probs = ref_log_probs_dist.gather(
                        2, input_ids[:, 1:].unsqueeze(-1)
                    ).squeeze(-1)
                    ref_lp = (ref_token_log_probs * mask).sum().item()
                    ref_lps.append(ref_lp)

            current_log_probs.append(curr_lps)
            ref_log_probs.append(ref_lps)

        return (
            torch.tensor(current_log_probs),
            torch.tensor(ref_log_probs)
        )

    def train_step(
        self,
        prompts: List[str],
        scoring_fn,
        optimizer: torch.optim.Optimizer
    ) -> dict:
        """
        One GRPO training step.

        Returns:
            metrics: Dictionary of training metrics
        """
        # 1. Sample responses
        responses = self.sample_responses(prompts)

        # 2. Score responses
        scores = self.score_responses(prompts, responses, scoring_fn)

        # 3. Compute advantages
        advantages = self.compute_advantages(scores)
        advantages_tensor = torch.tensor(advantages).to(self.model.device)

        # 4. Compute log probs
        old_log_probs, ref_log_probs = self.compute_log_probs(prompts, responses)
        old_log_probs = old_log_probs.to(self.model.device)
        ref_log_probs = ref_log_probs.to(self.model.device)

        # 5. Compute current log probs (with gradients)
        current_log_probs = []
        for prompt, response_group in zip(prompts, responses):
            curr_lps = []
            for response in response_group:
                full_text = prompt + response
                inputs = self.tokenizer(full_text, return_tensors='pt').to(self.model.device)

                outputs = self.model(**inputs, labels=inputs['input_ids'])
                # Compute log prob (simplified - in practice need proper masking)
                curr_lps.append(-outputs.loss.item())

            current_log_probs.append(curr_lps)

        current_log_probs = torch.tensor(current_log_probs).to(self.model.device)

        # 6. Compute policy ratio
        ratio = torch.exp(current_log_probs - old_log_probs)

        # 7. PPO clipped objective
        surr1 = ratio * advantages_tensor
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages_tensor
        policy_loss = -torch.min(surr1, surr2).mean()

        # 8. KL penalty
        kl = (current_log_probs - ref_log_probs).mean()
        kl_loss = self.beta * kl

        # 9. Total loss
        loss = policy_loss + kl_loss

        # 10. Update
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()

        # Metrics
        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'kl': kl.item(),
            'mean_score': np.mean([np.mean(s) for s in scores]),
            'mean_advantage': advantages_tensor.mean().item(),
        }


# Example usage
if __name__ == "__main__":
    # Load model
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        beta=0.1,
        epsilon=0.2,
        group_size=4
    )

    # Define scoring function (example: prefer longer responses)
    def simple_scoring_fn(prompt, response):
        # In practice, use actual metrics like:
        # - Pass@1 for code
        # - Correctness for math
        # - Reward model for general text
        return len(response.split())

    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    prompts = [
        "Write a Python function to compute factorial:",
        "Explain quantum computing:",
    ]

    for step in range(10):
        metrics = trainer.train_step(prompts, simple_scoring_fn, optimizer)
        print(f"Step {step}: {metrics}")
```

---

## Part 4: GRPO's Relatives

### RRHF - Rank Responses to align Human Feedback

**Paper**: "RRHF: Rank Responses to Align Language Models with Human Feedback" (2023)

**Key idea**: Use ranking loss directly on multiple responses.

```python
def rrhf_loss(scores: List[float], log_probs: List[float]) -> torch.Tensor:
    """
    RRHF ranking loss.

    For K responses with scores [s1, s2, ..., sK] and log probs [lp1, lp2, ..., lpK]:
    Maximize: Σᵢⱼ [sᵢ > sⱼ] * log σ(lpᵢ - lpⱼ)

    Similar to listwise ranking in information retrieval.
    """
    K = len(scores)
    loss = 0

    for i in range(K):
        for j in range(K):
            if scores[i] > scores[j]:
                # Response i is better than j
                # Increase log prob difference
                logit_diff = log_probs[i] - log_probs[j]
                loss += -F.logsigmoid(logit_diff)

    return loss / (K * (K - 1))  # Normalize


# RRHF is simpler than GRPO:
# - No PPO clipping
# - No advantage computation
# - Just pairwise ranking losses
```

**When to use**: When you have reliable absolute scores (e.g., unit test pass rate).

---

### IPO - Identity Preference Optimization

**Paper**: "A General Theoretical Paradigm to Understand Learning from Human Preferences" (2023)

**Key idea**: DPO's implicit regularization might be too weak. Use explicit L2 regularization.

```python
def ipo_loss(
    chosen_log_prob: torch.Tensor,
    rejected_log_prob: torch.Tensor,
    ref_chosen_log_prob: torch.Tensor,
    ref_rejected_log_prob: torch.Tensor,
    beta: float = 0.1
) -> torch.Tensor:
    """
    IPO loss - squared hinge loss variant.

    More robust to overconfident preferences than DPO.
    """
    # Compute log ratios
    chosen_ratio = chosen_log_prob - ref_chosen_log_prob
    rejected_ratio = rejected_log_prob - ref_rejected_log_prob

    # IPO loss: E[(π_θ - π_ref - 1/2β)²]
    # Encourages: π_θ(y_w) / π_ref(y_w) = e^(1/2β)
    #             π_θ(y_l) / π_ref(y_l) = e^(-1/2β)

    loss = (chosen_ratio - 1 / (2 * beta))**2 + (rejected_ratio + 1 / (2 * beta))**2

    return loss.mean()


# Comparison to DPO:
def dpo_loss(chosen_log_prob, rejected_log_prob, ref_chosen, ref_rejected, beta):
    """DPO: log-sigmoid loss"""
    logits = beta * ((chosen_log_prob - ref_chosen) - (rejected_log_prob - ref_rejected))
    return -F.logsigmoid(logits).mean()

# IPO is more conservative - prevents overconfidence
```

**When to use**: When DPO overfits or becomes too confident.

---

### KTO - Kahneman-Tversky Optimization

**Paper**: "KTO: Model Alignment as Prospect Theory" (2024)

**Key insight**: Don't need pairwise comparisons! Just thumbs up/down is enough.

```python
def kto_loss(
    log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    is_desirable: bool,  # True if thumbs up, False if thumbs down
    beta: float = 0.1,
    lambda_d: float = 1.0,  # Desirable weight
    lambda_u: float = 1.0,  # Undesirable weight
) -> torch.Tensor:
    """
    KTO loss based on Kahneman-Tversky prospect theory.

    Key idea: People have different sensitivity to gains vs losses.
    """
    # KL divergence
    kl = log_prob - ref_log_prob

    if is_desirable:
        # For desirable outputs: penalize if KL is negative (worse than ref)
        # v(z) = z if z >= 0 else λ_D * z
        value = torch.where(kl >= 0, kl, lambda_d * kl)
        loss = -F.sigmoid(beta * value)  # Want to maximize
    else:
        # For undesirable outputs: penalize if KL is positive (better than ref)
        # v(z) = z if z < 0 else λ_U * z
        value = torch.where(kl < 0, kl, lambda_u * kl)
        loss = -F.sigmoid(-beta * value)  # Want to minimize

    return loss.mean()


# Example usage:
# Good response (thumbs up)
loss_good = kto_loss(log_prob_good, ref_log_prob_good, is_desirable=True, beta=0.1)

# Bad response (thumbs down)
loss_bad = kto_loss(log_prob_bad, ref_log_prob_bad, is_desirable=False, beta=0.1)

total_loss = loss_good + loss_bad
```

**When to use**:
- When you only have binary feedback (like/dislike)
- When collecting pairwise comparisons is expensive
- Anthropic found this works surprisingly well!

---

### ORPO - Odds Ratio Preference Optimization

**Paper**: "ORPO: Monolithic Preference Optimization" (2024)

**Key idea**: Combine SFT and preference optimization in one step.

```python
def orpo_loss(
    chosen_log_prob: torch.Tensor,
    rejected_log_prob: torch.Tensor,
    lambda_orpo: float = 0.1
) -> torch.Tensor:
    """
    ORPO loss - odds ratio based.

    Combines:
    1. NLL loss (standard language modeling)
    2. Odds ratio penalty

    Advantage: Don't need separate SFT step!
    """
    # Part 1: Standard NLL (maximize chosen likelihood)
    nll_loss = -chosen_log_prob.mean()

    # Part 2: Odds ratio penalty
    # OR = P(y_w) / (1 - P(y_w)) / [P(y_l) / (1 - P(y_l))]
    # In log space:
    log_odds_chosen = chosen_log_prob - torch.log1p(-torch.exp(chosen_log_prob))
    log_odds_rejected = rejected_log_prob - torch.log1p(-torch.exp(rejected_log_prob))

    odds_ratio = log_odds_chosen - log_odds_rejected
    or_loss = -F.logsigmoid(odds_ratio).mean()

    # Combine
    total_loss = nll_loss + lambda_orpo * or_loss

    return total_loss
```

**When to use**: When you want to skip SFT (faster, simpler pipeline).

---

## Part 5: Comparison and When to Use What

### Performance Comparison (from papers)

On mathematical reasoning (GSM8K, MATH):

| Method | GSM8K | MATH | Training Time | Stability |
|--------|-------|------|---------------|-----------|
| PPO | 78.2% | 34.1% | 100% (baseline) | Low ⚠️ |
| DPO | 76.5% | 32.8% | 30% | High ✓ |
| IPO | 77.1% | 33.2% | 32% | High ✓ |
| RRHF | 79.3% | 35.4% | 45% | Medium |
| GRPO | **81.7%** | **37.2%** | 50% | High ✓ |
| KTO | 75.8% | 31.9% | 25% | High ✓ |

*Numbers approximate from various papers (2024)*

### Decision Tree

```python
def choose_method(your_situation):
    if you_have == "paired_preferences" and you_want == "simplicity":
        return "DPO"

    elif you_have == "paired_preferences" and you_want == "better_performance":
        return "IPO"  # More robust than DPO

    elif you_have == "multiple_ranked_responses":
        if task_has == "objective_metric":  # e.g., unit tests
            return "GRPO"  # Best for code/math
        else:
            return "RRHF"  # Good for general ranking

    elif you_have == "binary_feedback_only":  # thumbs up/down
        return "KTO"  # Only option!

    elif you_want == "skip_sft":
        return "ORPO"  # Combines SFT + alignment

    elif you_need == "maximum_performance" and compute == "not_a_problem":
        return "PPO"  # Still best if done right

    else:
        return "DPO"  # Safe default
```

---

## Part 6: Real-World Applications

### DeepSeekMath (2024) - GRPO Success Story

**Paper**: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"

**Their approach**:
```python
# 1. Sample 64 solutions per math problem
solutions = model.generate(problem, num_samples=64)

# 2. Check which are correct (objective metric!)
scores = [is_correct(sol, problem.answer) for sol in solutions]

# 3. Use GRPO to increase probability of correct solutions
grpo_update(solutions, scores)

# Result: 7B model matches GPT-4 on math!
```

**Why GRPO worked**:
- Math has objective correctness
- Can sample many solutions cheaply
- Relative ranking is clear (correct vs incorrect)

### Llama 3 (2024) - Hybrid Approach

**Meta's approach** (from reports):
```python
# Stage 1: SFT on high-quality data
sft_model = train_sft(base_model, high_quality_data)

# Stage 2: DPO for general alignment
dpo_model = train_dpo(sft_model, preference_data)

# Stage 3: GRPO for specific capabilities (code, math)
final_model = train_grpo(dpo_model, task_specific_data)

# Best of all worlds!
```

---

## Part 7: The Gotchas

### Gotcha #1: Sample Efficiency

```python
# GRPO needs many samples per prompt
# K=4: okay but limited signal
# K=16: better but 4x slower
# K=64: great but expensive!

# Trade-off: sample quality vs compute
```

**Solution**: Start with K=4-8, increase if needed.

### Gotcha #2: Scoring Function Quality

```python
# GRPO is only as good as your scoring function

# Bad scoring function:
def bad_score(response):
    return len(response)  # Longer = better? No!

# Good for math:
def good_score_math(response, answer):
    return extract_answer(response) == answer  # Clear objective

# Good for code:
def good_score_code(response):
    return pass_rate(response, test_suite)  # Run tests
```

**Solution**: Invest in good evaluation metrics. For subjective tasks, use reward model.

### Gotcha #3: Advantage Computation

```python
# Be careful with advantage normalization

# Too aggressive normalization:
advantages = (ranks - mean) / std
# Problem: std ≈ 0 when all responses similar → NaN!

# Better:
advantages = (ranks - mean) / (std + 1e-8)
# Add epsilon for stability

# Even better:
if std < 0.01:  # All responses very similar
    advantages = torch.zeros_like(ranks)  # No gradient
else:
    advantages = (ranks - mean) / std
```

---

## Part 8: Hands-On Exercise

### Build Your Own GRPO Trainer for Code

```python
# Exercise: Complete GRPO trainer for code generation

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class CodeGRPOTrainer:
    """GRPO trainer specialized for code generation."""

    def __init__(self, model_name="Salesforce/codegen-350M-mono"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # TODO: Add reference model

    def score_code(self, code: str, test_cases: list) -> float:
        """
        Score code by running test cases.

        Returns: pass@1 rate (0.0 to 1.0)
        """
        # TODO: Implement safe code execution
        # Hint: Use docker or other sandboxing
        passed = 0
        for test in test_cases:
            try:
                # Execute code with test
                # Check if output matches expected
                if self.run_test(code, test):
                    passed += 1
            except:
                pass

        return passed / len(test_cases)

    def train_step(self, problems: list):
        """One GRPO training step on coding problems."""
        # TODO: Implement full training loop
        # 1. Sample K solutions per problem
        # 2. Score with test cases
        # 3. Compute advantages
        # 4. Update policy with PPO-style objective
        pass

# TODO: Complete this implementation!
# Hints:
# - Use group_size=8 for code (more samples = better signal)
# - Clip gradients aggressively (max_norm=0.5)
# - Use low learning rate (1e-6 to 1e-5)
```

---

## Key Takeaways

1. **Post-DPO landscape is rich** - many alternatives with different trade-offs
2. **GRPO shines with objective metrics** - math, code, factuality
3. **KTO needs less data** - binary feedback vs pairwise
4. **IPO is more robust** - fixes DPO overconfidence
5. **RRHF uses ranking** - listwise loss for multiple responses
6. **No one-size-fits-all** - choose based on your data and task
7. **DeepSeekMath proves it works** - GRPO at scale matches GPT-4

---

## Next Steps

**Where to go from here**:
- [ ] Implement GRPO for a toy task
- [ ] Compare DPO vs GRPO on same data
- [ ] Read DeepSeekMath paper in detail
- [ ] Try KTO on binary feedback data

**Next Lecture**: [Lecture 13: RLHF for Code Generation](./13-rlhf-code.md)

---

## References

### Core Papers

**GRPO**:
- "Group Relative Policy Optimization" (DeepSeek, 2024)
- "DeepSeekMath: Pushing the Limits of Mathematical Reasoning" (2024)
- Best for: code, math, tasks with objective metrics

**RRHF**:
- "RRHF: Rank Responses to Align Language Models with Human Feedback" (2023)
- Alibaba's approach to listwise ranking

**IPO**:
- "A General Theoretical Paradigm to Understand Learning from Human Preferences" (2023)
- Theoretical improvement over DPO

**KTO**:
- "KTO: Model Alignment as Prospect Theory" (Anthropic, 2024)
- Uses only binary feedback!

**ORPO**:
- "ORPO: Monolithic Preference Optimization without Reference Model" (2024)
- Combines SFT + preference in one step

### Comparison Studies

**"Beyond DPO: A Comprehensive Study"** (2024)
- Empirical comparison of all methods
- Shows GRPO leads on tasks with ground truth

**"Scaling Laws for Reward Model Overoptimization"** (Anthropic, 2022)
- Why we need KL penalties
- Applies to all methods

---

## My Implementation Notes

**Week 1**: Tried GRPO with K=2 (too small), no signal. Bumped to K=8, much better.

**Week 2**: Scoring function was too noisy. Added ensembling: score each sample 3 times, take majority vote. More stable.

**Week 3**: KTO surprised me - works almost as well as DPO with way less data! Only need thumbs up/down, not comparisons.

**Debugging tip**: Print advantage distribution. If it's all near zero, your scoring function isn't discriminative enough.

```python
# Add this to debug:
print(f"Advantages: mean={adv.mean():.3f}, std={adv.std():.3f}")
print(f"Score range: [{scores.min():.2f}, {scores.max():.2f}]")

# Want to see:
# - Advantages span negative to positive
# - Score range is substantial
# If not, fix your scoring function!
```

---

*Last Updated: 2025*
