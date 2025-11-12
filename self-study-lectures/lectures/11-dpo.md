# Lecture 11: Direct Preference Optimization (DPO)

> **The Breakthrough**: "What if the optimal policy IS the reward model?"

**Time**: 3-4 hours | **Prerequisites**: Lectures 09-10 | **Difficulty**: ⭐⭐⭐⭐☆

---

## 🎯 Why This Matters

PPO-based RLHF works but is **complex and unstable**:
- Train 3 models (SFT, reward model, policy)
- PPO hyperparameters are finicky
- Reward model can be gamed
- Expensive to run

**DPO's insight**: You don't need a separate reward model. The optimal policy contains all the information!

**This simplified RLHF and made it accessible. It's how many modern models are aligned.**

---

## Part 1: The Problem with PPO-RLHF

### What's Wrong with PPO?

```python
# PPO-RLHF pipeline:
# 1. Train reward model r_φ on preferences
reward_model = train_reward_model(preferences)

# 2. Use PPO to optimize policy against reward model
for step in range(training_steps):
    responses = policy.generate(prompts)
    rewards = reward_model.score(responses)

    # PPO update (complex!)
    advantages = compute_advantages(rewards, values)
    ratio = new_policy_prob / old_policy_prob
    clipped_ratio = clip(ratio, 1-epsilon, 1+epsilon)
    loss = -min(ratio * advantages, clipped_ratio * advantages)

    # Plus value function loss, entropy bonus, KL penalty...
    # So many hyperparameters! 😰
```

**Problems**:
1. **Reward model errors compound** - if r_φ is wrong, PPO amplifies it
2. **Hyperparameter sensitivity** - ε, KL coeff, learning rate all critical
3. **Training instability** - PPO can diverge or collapse
4. **Computational cost** - 3 models, many forward/backward passes

**Personal note**: I spent weeks tuning PPO. Every time I changed data, had to retune everything.

---

## Part 2: The Key Insight

### Reparameterization of the Reward

**Standard RLHF objective**:
```
Maximize: E[r_φ(x,y)] - β KL(π_θ || π_ref)
```

Where:
- r_φ = learned reward model
- π_θ = policy we're training
- π_ref = reference policy (usually SFT model)
- β = KL penalty coefficient

**DPO's insight**: This has a closed-form optimal solution!

```
π*(y|x) = (1/Z(x)) π_ref(y|x) exp(r(x,y) / β)
```

Where Z(x) is a partition function.

**Rearranging** to solve for reward:
```
r(x,y) = β log(π*(y|x) / π_ref(y|x)) + β log Z(x)
```

**The magic**: We can express reward in terms of the optimal policy!

### Why This Helps

Instead of:
1. Learn r_φ
2. Optimize π_θ to maximize r_φ

We do:
1. Directly optimize π_θ on preferences

**Intuition**: The preferences already tell us what good outputs look like. Why learn an intermediate reward model?

---

## Part 3: The DPO Objective

### From Bradley-Terry to DPO

**Bradley-Terry** for reward model:
```
P(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))
```

**Substitute DPO reparameterization**:
```
P(y_w > y_l | x) = σ(β log(π_θ(y_w|x)/π_ref(y_w|x)) - β log(π_θ(y_l|x)/π_ref(y_l|x)))
```

**Simplify**:
```
P(y_w > y_l | x) = σ(β log(π_θ(y_w|x) π_ref(y_l|x) / π_ref(y_w|x) π_θ(y_l|x)))
```

**DPO Loss** (maximize likelihood of preferences):
```
L_DPO = -E[log σ(β log(π_θ(y_w|x)/π_ref(y_w|x)) - β log(π_θ(y_l|x)/π_ref(y_l|x)))]
```

**In code**:
```python
def dpo_loss(policy_chosen_logprobs, policy_rejected_logprobs,
             reference_chosen_logprobs, reference_rejected_logprobs,
             beta=0.1):
    """
    DPO loss function.

    Args:
        policy_chosen_logprobs: log π_θ(y_w|x)
        policy_rejected_logprobs: log π_θ(y_l|x)
        reference_chosen_logprobs: log π_ref(y_w|x)
        reference_rejected_logprobs: log π_ref(y_l|x)
        beta: KL penalty coefficient
    """
    # Compute log ratios
    policy_chosen_logratios = policy_chosen_logprobs - reference_chosen_logprobs
    policy_rejected_logratios = policy_rejected_logprobs - reference_rejected_logprobs

    # DPO loss
    logits = beta * (policy_chosen_logratios - policy_rejected_logratios)
    loss = -F.logsigmoid(logits).mean()

    return loss
```

**That's it!** No reward model, no PPO, no value functions.

---

## Part 4: Complete DPO Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List
import copy

class PreferenceDataset(Dataset):
    """Dataset for preference pairs."""

    def __init__(self, data: List[Dict], tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize chosen
        chosen_text = item['prompt'] + item['chosen']
        chosen_tokens = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        # Tokenize rejected
        rejected_text = item['prompt'] + item['rejected']
        rejected_tokens = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'chosen_input_ids': chosen_tokens['input_ids'].squeeze(0),
            'chosen_attention_mask': chosen_tokens['attention_mask'].squeeze(0),
            'rejected_input_ids': rejected_tokens['input_ids'].squeeze(0),
            'rejected_attention_mask': rejected_tokens['attention_mask'].squeeze(0),
            'prompt': item['prompt']
        }


class DPOTrainer:
    """Direct Preference Optimization trainer."""

    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        beta=0.1,
        lr=1e-6
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.beta = beta

        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

    def compute_logprobs(
        self,
        model,
        input_ids,
        attention_mask
    ):
        """
        Compute log probabilities of sequence under model.

        Returns average log probability per token.
        """
        with torch.set_grad_enabled(model.training):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )

            # Get log probabilities
            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=-1)

            # Gather log probs of actual tokens
            # Shift by 1 (predict next token)
            labels = input_ids[:, 1:].unsqueeze(-1)
            gathered_log_probs = log_probs[:, :-1].gather(2, labels).squeeze(-1)

            # Mask padding tokens
            mask = attention_mask[:, 1:].float()

            # Average log prob per sequence
            sequence_log_probs = (gathered_log_probs * mask).sum(-1) / mask.sum(-1)

            return sequence_log_probs

    def train_step(self, batch):
        """Single training step."""

        # Move to device
        device = next(self.model.parameters()).device
        chosen_ids = batch['chosen_input_ids'].to(device)
        chosen_mask = batch['chosen_attention_mask'].to(device)
        rejected_ids = batch['rejected_input_ids'].to(device)
        rejected_mask = batch['rejected_attention_mask'].to(device)

        # Compute log probs for policy model
        policy_chosen_logprobs = self.compute_logprobs(
            self.model, chosen_ids, chosen_mask
        )
        policy_rejected_logprobs = self.compute_logprobs(
            self.model, rejected_ids, rejected_mask
        )

        # Compute log probs for reference model
        with torch.no_grad():
            ref_chosen_logprobs = self.compute_logprobs(
                self.ref_model, chosen_ids, chosen_mask
            )
            ref_rejected_logprobs = self.compute_logprobs(
                self.ref_model, rejected_ids, rejected_mask
            )

        # DPO loss
        policy_chosen_logratios = policy_chosen_logprobs - ref_chosen_logprobs
        policy_rejected_logratios = policy_rejected_logprobs - ref_rejected_logprobs

        logits = self.beta * (policy_chosen_logratios - policy_rejected_logratios)
        loss = -F.logsigmoid(logits).mean()

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Metrics
        with torch.no_grad():
            # Implicit reward (for logging)
            chosen_rewards = self.beta * policy_chosen_logratios
            rejected_rewards = self.beta * policy_rejected_logratios

            # Accuracy: how often is chosen preferred?
            accuracy = (chosen_rewards > rejected_rewards).float().mean()

            # Reward margin
            reward_margin = (chosen_rewards - rejected_rewards).mean()

        return {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'reward_margin': reward_margin.item(),
            'chosen_reward': chosen_rewards.mean().item(),
            'rejected_reward': rejected_rewards.mean().item(),
        }

    def train(self, train_loader, epochs=3):
        """Full training loop."""
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            total_acc = 0

            for i, batch in enumerate(train_loader):
                metrics = self.train_step(batch)

                total_loss += metrics['loss']
                total_acc += metrics['accuracy']

                if (i + 1) % 10 == 0:
                    avg_loss = total_loss / (i + 1)
                    avg_acc = total_acc / (i + 1)
                    print(f"Epoch {epoch+1}, Step {i+1}: "
                          f"Loss={avg_loss:.4f}, Acc={avg_acc:.4f}, "
                          f"Margin={metrics['reward_margin']:.4f}")

            print(f"\nEpoch {epoch+1} complete: "
                  f"Avg Loss={total_loss/len(train_loader):.4f}, "
                  f"Avg Acc={total_acc/len(train_loader):.4f}\n")


# Example usage
if __name__ == "__main__":
    # Load model and tokenizer
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Create reference model (frozen copy)
    ref_model = copy.deepcopy(model)

    # Prepare data
    preference_data = [
        {
            'prompt': "Explain quantum computing:",
            'chosen': "Quantum computers use qubits that can be in superposition...",
            'rejected': "Quantum computers are just faster regular computers."
        },
        # ... more examples
    ]

    dataset = PreferenceDataset(preference_data, tokenizer)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Train with DPO
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        beta=0.1,
        lr=1e-6
    )

    trainer.train(train_loader, epochs=3)

    # Save fine-tuned model
    model.save_pretrained("./dpo_model")
    print("DPO training complete!")
```

---

## Part 5: DPO vs PPO - The Real Comparison

### What You Gain

**Simplicity**:
```python
# PPO: ~1000 lines of code
# DPO: ~200 lines of code
```

**Stability**:
```python
# PPO hyperparameters (all critical):
- epsilon (clip range)
- GAE lambda
- value function coefficient
- entropy coefficient
- KL penalty
- learning rate (policy)
- learning rate (value)
- minibatch size
- gradient accumulation

# DPO hyperparameters:
- beta (one parameter!)
- learning rate
```

**Speed**:
```python
# PPO:
#   - Forward pass: policy + value function
#   - Backward pass: both networks
#   - Multiple epochs per batch

# DPO:
#   - Forward pass: just policy
#   - Backward pass: just policy
#   - Single pass per batch

# DPO is ~3x faster in practice
```

### What You Lose

**Flexibility**:
- PPO can use any reward signal (RM, rules, human-in-loop)
- DPO requires paired preferences

**Theoretical Guarantees**:
- PPO has convergence proofs
- DPO is more heuristic

**Performance** (sometimes):
- On some tasks, well-tuned PPO > DPO
- But DPO is "good enough" for most

---

## Part 6: The Gotchas

### Gotcha #1: β is Critical

```python
# β too small (e.g., 0.01):
#   - Policy stays too close to reference
#   - Doesn't learn much from preferences
#   - Underfit

# β too large (e.g., 1.0):
#   - Policy drifts far from reference
#   - Can overfit to preferences
#   - May forget language modeling

# Sweet spot: 0.1 to 0.5
# Start with 0.1, tune if needed
```

### Gotcha #2: Reference Model Matters

```python
# Bad: Random initialization as reference
ref_model = AutoModelForCausalLM.from_pretrained("gpt2")
# Policy will drift arbitrarily far

# Good: Use SFT model as reference
sft_model = train_sft(base_model, demonstrations)
ref_model = copy.deepcopy(sft_model)
# Policy stays close to good generations
```

### Gotcha #3: Log Prob Computation

```python
# Common bug: Including prompt tokens in log prob
# Wrong:
logprob = logprobs.sum()  # Includes prompt!

# Right: Mask prompt tokens
prompt_len = len(tokenizer(prompt)['input_ids'])
logprob = logprobs[:, prompt_len:].sum()  # Only response
```

---

## Part 7: When to Use DPO

### Use DPO When:

✅ You have pairwise preferences
✅ You want simplicity and stability
✅ You're okay with "good enough" performance
✅ You want fast iteration

### Use PPO When:

✅ You need maximum performance
✅ You have complex reward signals
✅ You can afford the engineering effort
✅ You have compute to spare

### Real-World Usage:

**Llama 2** (Meta): Used PPO (they have the resources)
**Zephyr** (HuggingFace): Used DPO (open source, simpler)
**Most startups**: Use DPO (pragmatic choice)

---

## Key Takeaways

1. **DPO eliminates reward model** - directly optimizes on preferences
2. **Much simpler than PPO** - one parameter (β) vs many
3. **3x faster training** - fewer forward/backward passes
4. **Closed-form derivation** - based on optimal policy equation
5. **Reference model is KL regularizer** - prevents drift
6. **Good enough for most uses** - 90% of PPO performance, 10% of complexity

---

## Next Lecture

**[Lecture 12: Beyond DPO - GRPO and Relatives](./12-beyond-dpo.md)**

Now that you know DPO, learn about the methods that improve it!

Before moving on:
- [ ] Implement DPO on toy dataset
- [ ] Compare to reward model approach
- [ ] Understand β parameter deeply
- [ ] Know when to use vs PPO

---

## References

### The Paper

**Rafailov et al. (2023)** - "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
- **arXiv**: [2305.18290](https://arxiv.org/abs/2305.18290)
- Revolutionary simplification of RLHF
- Shows reward is implicit in policy
- Empirically matches PPO performance

### Follow-up Work

**IPO (2023)** - "A General Theoretical Paradigm to Understand Learning from Human Preferences"
- Addresses DPO's overconfidence
- Better theoretical grounding

**ORPO (2024)** - "ORPO: Monolithic Preference Optimization"
- Combines SFT + DPO
- Even simpler pipeline

### Applications

**Zephyr-7B** (HuggingFace, 2023)
- Used DPO for alignment
- Matched larger models
- Open source implementation

**Anthropic's Work** (2024)
- Constitutional AI with DPO
- AI-generated preferences
- Scalable alignment

---

## My Implementation Notes

**Week 1**: First tried β=1.0, model went crazy. Outputs became gibberish. Lesson: Start with β=0.1!

**Week 2**: Forgot to freeze reference model. Gradients flowed through, defeats the purpose. Always `ref_model.requires_grad_(False)`.

**Week 3**: Computed log probs wrong (included prompt). Model "learned" to repeat prompt. Mask is essential!

**Debugging tip**:
```python
# Always check reward difference
reward_diff = chosen_reward - rejected_reward
print(f"Reward margin: {reward_diff.mean():.3f}")

# Should be positive and growing
# If negative or shrinking, something's wrong
```

---

*Last Updated: 2025*
