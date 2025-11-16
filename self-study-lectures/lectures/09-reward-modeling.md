# Lecture 09: Reward Modeling for RLHF

> **The Realization**: "Wait, if we don't know what reward function we want, can we just... learn it from examples?"

**Time**: 3-4 hours | **Prerequisites**: Lectures 01-02, basic NLP | **Difficulty**: ⭐⭐⭐⭐☆

---

## 🎯 Why This Matters

You know how to do RL now. But for LLMs, there's a problem:

**What's the reward for "good" text generation?**

- Not BLEU score (correlation with quality is weak)
- Not perplexity (measures memorization, not helpfulness)
- Not exact match (stifles creativity)

**Solution**: Learn the reward function from human preferences.

**This is the first half of RLHF. Master this, and you understand how ChatGPT learns "helpfulness."**

---

## Part 1: The Problem (Why Reward Modeling?)

### The Traditional RL Setup

```python
# Normal RL:
def reward(state, action, next_state):
    if next_state.is_goal():
        return 1.0
    return 0.0

# Easy! Reward is known and programmatic.
```

### The LLM Alignment Problem

```python
# For LLMs:
def reward(prompt, response):
    # Is this response "good"? How do I measure:
    # - Helpfulness?
    # - Harmlessness?
    # - Honesty?
    # - Following instructions?
    return ???  # We don't know!
```

**Examples that break simple metrics**:

```python
# Example 1: BLEU score fails
prompt = "How do I make a cake?"
response_A = "To make a cake, you need flour, eggs, sugar..."  # Helpful
response_B = "Cake flour eggs sugar butter vanilla extract"   # Word salad

# BLEU might rate B higher (more keyword matches)!

# Example 2: Safety
prompt = "How do I break into a car?"
response_A = "Call a locksmith or AAA for help."  # Helpful + Safe
response_B = "Use a slim jim to..."  # Helpful but UNSAFE

# No automatic metric captures this!
```

---

## Part 2: Human Preferences (The Data We Actually Have)

### What We Can Get from Humans

Instead of asking "what's the reward?", we ask:

> "Which response is better: A or B?"

```python
@dataclass
class Comparison:
    prompt: str
    response_A: str
    response_B: str
    preferred: str  # "A" or "B"

# Example:
comparison = Comparison(
    prompt="Explain quantum computing to a 10-year-old",
    response_A="Quantum computers use qubits that can be 0 and 1 at the same time, like Schrödinger's cat!",
    response_B="Quantum computers leverage quantum superposition and entanglement to achieve computational speedup via quantum parallelism.",
    preferred="A"  # A is simpler, age-appropriate
)
```

**Key insight**: Humans are better at comparisons than absolute ratings.

```python
# Hard: "Rate this response 1-10"
# (What's the difference between 7 and 8?)

# Easy: "Which is better: A or B?"
# (Clear preference, consistent)
```

---

## Part 3: The Bradley-Terry Model (Theory)

### The Setup

We have:
- Prompt x
- Two responses y₁ and y₂
- Human says y₁ > y₂

We want to learn a reward function r(x, y) such that:
- If y₁ > y₂, then r(x, y₁) > r(x, y₂)

### The Bradley-Terry (BT) Model

**Assumption**: Probability of preferring y₁ over y₂ follows a logistic:

```
P(y₁ > y₂ | x) = σ(r(x, y₁) - r(x, y₂))
```

Where σ(z) = 1 / (1 + e^(-z)) is the sigmoid function.

**Intuition**:
- If r(y₁) >> r(y₂): P(y₁ > y₂) ≈ 1 (almost certainly prefer y₁)
- If r(y₁) ≈ r(y₂): P(y₁ > y₂) ≈ 0.5 (toss-up)
- If r(y₁) << r(y₂): P(y₁ > y₂) ≈ 0 (almost certainly prefer y₂)

### The Loss Function

We want to maximize likelihood of observed preferences:

```
L = Σ log P(y_win > y_lose | x)
  = Σ log σ(r(x, y_win) - r(x, y_lose))
```

In practice, we minimize negative log-likelihood:

```
Loss = -Σ log σ(r(x, y_win) - r(x, y_lose))
```

**Personal Note**: This is just cross-entropy loss! We're treating preference as binary classification.

---

## Part 4: Reward Model Architecture

### The Network

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class RewardModel(nn.Module):
    """
    Reward model for scoring text generations.

    Architecture:
      prompt + response → transformer → scalar reward
    """
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()

        # Base transformer
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size

        # Reward head (scalar output)
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)  # Scalar reward
        )

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            rewards: [batch_size] - scalar reward per example
        """
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use [CLS] token or mean pooling
        # Here we use [CLS] (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]

        # Compute reward
        reward = self.reward_head(cls_embedding).squeeze(-1)  # [batch]

        return reward
```

### Input Format

```python
def format_input(prompt, response, tokenizer):
    """
    Format prompt + response for reward model.

    Format: [CLS] prompt [SEP] response [SEP]
    """
    text = f"{prompt} {tokenizer.sep_token} {response}"

    encoding = tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding='max_length',
        return_tensors='pt'
    )

    return encoding['input_ids'], encoding['attention_mask']
```

---

## Part 5: Training the Reward Model

### The Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class PreferenceDataset(Dataset):
    """
    Dataset of preference comparisons.
    """
    def __init__(self, comparisons, tokenizer, max_length=512):
        self.comparisons = comparisons
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.comparisons)

    def __getitem__(self, idx):
        comp = self.comparisons[idx]

        # Tokenize chosen response
        chosen_text = f"{comp['prompt']} {self.tokenizer.sep_token} {comp['chosen']}"
        chosen_enc = self.tokenizer(
            chosen_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        # Tokenize rejected response
        rejected_text = f"{comp['prompt']} {self.tokenizer.sep_token} {comp['rejected']}"
        rejected_enc = self.tokenizer(
            rejected_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'chosen_input_ids': chosen_enc['input_ids'].squeeze(0),
            'chosen_attention_mask': chosen_enc['attention_mask'].squeeze(0),
            'rejected_input_ids': rejected_enc['input_ids'].squeeze(0),
            'rejected_attention_mask': rejected_enc['attention_mask'].squeeze(0),
        }


def train_reward_model(model, train_loader, val_loader, epochs=3, lr=1e-5):
    """
    Train reward model on preference data.
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            # Move to device
            chosen_ids = batch['chosen_input_ids'].to(device)
            chosen_mask = batch['chosen_attention_mask'].to(device)
            rejected_ids = batch['rejected_input_ids'].to(device)
            rejected_mask = batch['rejected_attention_mask'].to(device)

            # Compute rewards
            reward_chosen = model(chosen_ids, chosen_mask)
            reward_rejected = model(rejected_ids, rejected_mask)

            # Bradley-Terry loss
            # We want reward_chosen > reward_rejected
            # P(chosen > rejected) = sigmoid(reward_chosen - reward_rejected)
            # Loss = -log P(chosen > rejected)
            logits = reward_chosen - reward_rejected
            loss = -nn.functional.logsigmoid(logits).mean()

            # Backprop
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (important for stability!)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Metrics
            total_loss += loss.item()
            predictions = (logits > 0).float()  # Predict chosen if logit > 0
            correct += predictions.sum().item()
            total += len(predictions)

        # Epoch stats
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Accuracy: {accuracy:.4f}")

        # Validation
        if val_loader is not None:
            val_acc = evaluate_reward_model(model, val_loader, device)
            print(f"  Val Accuracy: {val_acc:.4f}")

    return model


def evaluate_reward_model(model, val_loader, device):
    """Evaluate reward model accuracy on validation set."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            chosen_ids = batch['chosen_input_ids'].to(device)
            chosen_mask = batch['chosen_attention_mask'].to(device)
            rejected_ids = batch['rejected_input_ids'].to(device)
            rejected_mask = batch['rejected_attention_mask'].to(device)

            reward_chosen = model(chosen_ids, chosen_mask)
            reward_rejected = model(rejected_ids, rejected_mask)

            # Count correct predictions
            correct += (reward_chosen > reward_rejected).sum().item()
            total += len(reward_chosen)

    return correct / total
```

---

## Part 6: Gotchas (The Devil in the Details)

### Gotcha #1: Reward Hacking

**Problem**: Model learns to exploit reward model weaknesses.

```python
# Example:
# Reward model trained on short comparisons
# Learns: "longer = better"

# During RL, policy generates:
response = "This is great! " * 1000  # Just repeats
# Gets high reward (long), but useless!
```

**Solution**: Diverse training data, regularization, length normalization.

```python
def compute_reward_normalized(model, prompt, response):
    """Normalize reward by response length."""
    raw_reward = model(prompt, response)
    length_penalty = len(response.split()) * 0.01
    return raw_reward - length_penalty
```

### Gotcha #2: Overoptimization

**Problem**: RL policy becomes too good at fooling reward model.

**The KL Penalty**:
```python
# During RL, penalize deviation from reference policy:
reward_final = reward_model(response) - β * KL(π_θ || π_ref)

# β controls how much we trust the reward model
# Higher β = stay closer to reference policy
```

**Personal Note**: This took me forever to understand. The intuition is: the reward model is imperfect. Don't optimize it too hard or you'll find adversarial examples.

### Gotcha #3: Position Bias

**Problem**: Humans have biases (prefer response shown first, etc.)

```python
# Bad: Always show preferred response as "A"
# Model learns: "A is always better"

# Good: Randomize positions
if random.random() > 0.5:
    comparison = (response_A, response_B, "A")
else:
    comparison = (response_B, response_A, "B")
```

### Gotcha #4: Reward Scale

**Problem**: Reward values can explode/vanish.

```python
# Before training, normalize rewards:
def normalize_rewards(rewards):
    """Normalize to zero mean, unit variance."""
    return (rewards - rewards.mean()) / (rewards.std() + 1e-8)
```

---

## Part 7: Connection to Papers

### InstructGPT (OpenAI, 2022)

**Paper**: "Training language models to follow instructions with human feedback"

**Key contributions**:
1. Collect 40k preference comparisons from human labelers
2. Train reward model using Bradley-Terry
3. Fine-tune GPT-3 with PPO using reward model
4. Show models are more helpful, honest, harmless

**Architecture**: GPT-3 6B as reward model backbone

```python
# Their reward model setup:
# - Start from SFT model (not random init)
# - Remove LM head, add scalar reward head
# - Train on comparisons
# - Use in PPO with KL penalty
```

### Anthropic's Constitutional AI (2022)

**Paper**: "Constitutional AI: Harmlessness from AI Feedback"

**Key idea**: Use AI feedback instead of only human feedback.

```python
# Process:
1. Generate responses with base model
2. Ask model to critique itself: "Is this harmful?"
3. Revise response based on critique
4. Train reward model on AI-generated preferences
5. RLHF with this reward model
```

**Why this matters**: Cheaper to scale than pure human feedback.

### DPO (2023) - "Direct Preference Optimization"

**Key insight**: You don't need a separate reward model!

```python
# Instead of:
#   1. Train reward model
#   2. Use PPO with reward model

# Do:
#   1. Directly optimize policy on preferences

# We'll cover this in Lecture 11
```

---

## Part 8: Practical Tips

### Data Collection

**How much data?**
- Minimum: ~1k comparisons (toy experiments)
- Good: ~10k comparisons (research)
- Production: ~100k+ comparisons (ChatGPT scale)

**Who labels?**
- Domain experts (expensive, high quality)
- Crowd workers (cheap, need quality control)
- AI assistance (scalable, needs validation)

### Model Selection

**Base model choices**:
```python
# For reward model, use similar size to policy model
# - Policy: GPT-2 (124M) → Reward: BERT-base (110M)
# - Policy: GPT-3 (1.3B) → Reward: RoBERTa-large (355M)
# - Policy: GPT-3.5 (175B) → Reward: GPT-2 (1.5B)

# Bigger reward model ≠ always better
# Diminishing returns past a certain size
```

### Training Tricks

**Learning rate**: Start small (1e-5 to 1e-6)

```python
# Reward models are sensitive to LR
# Too high → overfits to training preferences
# Too low → doesn't learn generalizable preferences
```

**Gradient clipping**: Always clip!

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Ensemble**: Train multiple reward models

```python
# Use 3-5 models, average predictions
def ensemble_reward(models, prompt, response):
    rewards = [model(prompt, response) for model in models]
    return np.mean(rewards)
```

---

## Part 9: Evaluation

### How to Tell if Your Reward Model is Good

**Metric 1: Accuracy on held-out comparisons**

```python
# Target: >60-70% (random is 50%)
# InstructGPT achieved ~72%
```

**Metric 2: Agreement with humans**

```python
# Show humans:
#   - Prompt
#   - Two responses
#   - Which does reward model prefer?

# Measure: % human agrees with model
```

**Metric 3: Out-of-distribution generalization**

```python
# Test on:
#   - Different prompts
#   - Different response lengths
#   - Different topics
#   - Adversarial examples
```

### Red Flags

```python
# Bad signs:
# 1. Val accuracy << train accuracy (overfitting)
# 2. Prefers always longer/shorter responses (shallow heuristic)
# 3. Inconsistent (flips preference randomly)
# 4. Extreme reward values (unstable)
```

---

## Part 10: Full Example

```python
# Complete reward model training script

from transformers import AutoModel, AutoTokenizer
import torch

# 1. Load data
comparisons = [
    {
        'prompt': "Explain quantum computing",
        'chosen': "Quantum computers use quantum bits that can be 0 and 1 simultaneously...",
        'rejected': "Quantum computers are fast computers."
    },
    # ... more comparisons
]

# 2. Create dataset
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
dataset = PreferenceDataset(comparisons, tokenizer)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 3. Initialize model
reward_model = RewardModel('bert-base-uncased')

# 4. Train
trained_model = train_reward_model(
    reward_model,
    train_loader,
    val_loader=None,
    epochs=3,
    lr=1e-5
)

# 5. Use for inference
def score_response(prompt, response):
    """Score a response with trained reward model."""
    input_ids, attn_mask = format_input(prompt, response, tokenizer)
    reward = trained_model(input_ids, attn_mask)
    return reward.item()

# Example:
reward_A = score_response(
    "How do I bake a cake?",
    "Mix flour, eggs, sugar, and bake at 350°F for 30 minutes."
)
reward_B = score_response(
    "How do I bake a cake?",
    "Cake is made from flour."
)

print(f"Response A reward: {reward_A:.3f}")
print(f"Response B reward: {reward_B:.3f}")
# Expected: reward_A > reward_B
```

---

## Key Takeaways

1. **Human preferences > hand-crafted rewards** for alignment
2. **Bradley-Terry model** - core of reward modeling
3. **Reward model = classifier** trained on comparisons
4. **Gotchas**: reward hacking, overoptimization, biases
5. **This is step 1 of RLHF** - next is PPO training (Lecture 10)
6. **InstructGPT** - showed this works at scale

---

## Next Lecture

**[Lecture 10: PPO for Language Models](./10-ppo-for-llms.md)**

Where we'll:
- Take the reward model we just trained
- Use it to fine-tune an LLM with PPO
- Implement full RLHF pipeline
- Understand why this makes ChatGPT "helpful"

Before that, make sure you:
- [ ] Understand Bradley-Terry model
- [ ] Can implement reward model training
- [ ] Know the main gotchas
- [ ] Understand KL penalty intuition

---

## References

### Core Papers

**Christiano et al. (2017)** - "Deep reinforcement learning from human preferences"
- First to do RL from human feedback at scale
- Atari games from preferences
- Foundation for RLHF

**Ouyang et al. (2022)** - "Training language models to follow instructions with human feedback" (InstructGPT)
- Applied RLHF to GPT-3
- 40k human comparisons
- Created ChatGPT's predecessor

**Bai et al. (2022)** - "Constitutional AI: Harmlessness from AI Feedback" (Anthropic)
- AI-generated feedback
- Scalable alternative to human labeling
- Powers Claude's alignment

**Rafailov et al. (2023)** - "Direct Preference Optimization"
- Eliminates reward model
- Simpler RLHF pipeline
- Coming in Lecture 11!

---

## My Debug Log

**Week 1**: Reward model always gave same score. Bug: forgot to unfreeze transformer layers. Fixed by setting `model.transformer.requires_grad = True`.

**Week 2**: Val accuracy stuck at 50%. Bug: position bias in data. Fixed by randomizing response order.

**Week 3**: Rewards exploded during PPO. Bug: no reward normalization. Added `(r - r.mean()) / r.std()`.

**Lesson**: Print everything. Check shapes. Normalize inputs and outputs.

---

*Last Updated: 2025*
