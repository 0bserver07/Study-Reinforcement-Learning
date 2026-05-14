<!-- status: unreviewed | last-reviewed: never -->

# Lecture 09: Reward modeling for RLHF

_Unreviewed — no one has checked this end to end. Treat the math, code, and citations as unverified._

**Time:** 3–4 hours | **Prerequisites:** Lectures 01–02, basic NLP

---

## Why reward modeling

You know how to do RL. But for LLMs, there's a foundational problem: what is the reward for "good" text generation?

- Not BLEU score — correlation with quality is weak.
- Not perplexity — it measures memorization, not helpfulness.
- Not exact match — it penalizes valid paraphrases.

The solution is to learn the reward function from human preferences. This is the first half of RLHF. Get this right and you understand how ChatGPT learns "helpfulness."

---

## Part 1: The problem

### Normal RL

```python
def reward(state, action, next_state):
    if next_state.is_goal():
        return 1.0
    return 0.0

# Reward is known and programmatic.
```

### LLM alignment

```python
def reward(prompt, response):
    # Is this response "good"? How do I measure:
    # - Helpfulness?
    # - Harmlessness?
    # - Honesty?
    # - Following instructions?
    return ???  # We don't know.
```

**Examples that break simple metrics:**

```python
# BLEU score fails
prompt = "How do I make a cake?"
response_A = "To make a cake, you need flour, eggs, sugar..."  # Helpful
response_B = "Cake flour eggs sugar butter vanilla extract"   # Word salad

# BLEU might rate B higher (more keyword matches).

# Safety is not captured automatically
prompt = "How do I break into a car?"
response_A = "Call a locksmith or AAA for help."  # Safe
response_B = "Use a slim jim to..."               # Unsafe
```

---

## Part 2: Human preferences

Instead of asking "what's the reward?", ask:

> "Which response is better: A or B?"

```python
@dataclass
class Comparison:
    prompt: str
    response_A: str
    response_B: str
    preferred: str  # "A" or "B"

comparison = Comparison(
    prompt="Explain quantum computing to a 10-year-old",
    response_A="Quantum computers use qubits that can be 0 and 1 at the same time, like Schrödinger's cat!",
    response_B="Quantum computers leverage quantum superposition and entanglement to achieve computational speedup via quantum parallelism.",
    preferred="A"  # simpler, age-appropriate
)
```

Humans are more consistent at pairwise comparisons than at absolute ratings. "Rate this 1–10" forces a calibration decision; "which is better" doesn't.

---

## Part 3: The Bradley-Terry model

### Setup

Given:
- Prompt x
- Two responses y₁ and y₂
- Human label: y₁ > y₂

We want a reward function r(x, y) such that r(x, y₁) > r(x, y₂) whenever humans prefer y₁.

### Bradley-Terry model

Bradley & Terry (1952, Biometrika 39(3)) model pairwise preference probabilities as a logistic function of the score difference:

```
P(y₁ > y₂ | x) = σ(r(x, y₁) - r(x, y₂))
```

Where σ(z) = 1 / (1 + e^(−z)) is the sigmoid.

Intuition:
- r(y₁) >> r(y₂): P(y₁ > y₂) ≈ 1
- r(y₁) ≈ r(y₂): P(y₁ > y₂) ≈ 0.5
- r(y₁) << r(y₂): P(y₁ > y₂) ≈ 0

### Loss function

Maximize likelihood of observed preferences:

```
L = Σ log P(y_win > y_lose | x)
  = Σ log σ(r(x, y_win) - r(x, y_lose))
```

In practice, minimize negative log-likelihood:

```
Loss = −Σ log σ(r(x, y_win) − r(x, y_lose))
```

This is cross-entropy loss. Preference modeling is binary classification.

---

## Part 4: Reward model architecture

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class RewardModel(nn.Module):
    """
    Reward model for scoring text generations.

    Architecture: prompt + response → transformer → scalar reward
    """
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()

        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size

        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            rewards: [batch_size] — scalar reward per example
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use [CLS] token (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]

        reward = self.reward_head(cls_embedding).squeeze(-1)  # [batch]

        return reward
```

### Input format

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

## Part 5: Training the reward model

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class PreferenceDataset(Dataset):
    """Dataset of preference comparisons."""
    def __init__(self, comparisons, tokenizer, max_length=512):
        self.comparisons = comparisons
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.comparisons)

    def __getitem__(self, idx):
        comp = self.comparisons[idx]

        chosen_text = f"{comp['prompt']} {self.tokenizer.sep_token} {comp['chosen']}"
        chosen_enc = self.tokenizer(
            chosen_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

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
    """Train reward model on preference data."""
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            chosen_ids = batch['chosen_input_ids'].to(device)
            chosen_mask = batch['chosen_attention_mask'].to(device)
            rejected_ids = batch['rejected_input_ids'].to(device)
            rejected_mask = batch['rejected_attention_mask'].to(device)

            reward_chosen = model(chosen_ids, chosen_mask)
            reward_rejected = model(rejected_ids, rejected_mask)

            # Bradley-Terry loss:
            # P(chosen > rejected) = sigmoid(reward_chosen − reward_rejected)
            # Loss = −log P(chosen > rejected)
            logits = reward_chosen - reward_rejected
            loss = -nn.functional.logsigmoid(logits).mean()

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            predictions = (logits > 0).float()
            correct += predictions.sum().item()
            total += len(predictions)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Accuracy: {accuracy:.4f}")

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

            correct += (reward_chosen > reward_rejected).sum().item()
            total += len(reward_chosen)

    return correct / total
```

---

## Part 6: Common failure modes

### Reward hacking

The model learns to exploit reward model weaknesses instead of learning the intended behavior.

```python
# Reward model trained on short comparisons may learn "longer = better".
# During RL, the policy discovers this:
response = "This is great! " * 1000  # Long but useless
# Gets high reward, but achieves nothing.
```

**Mitigations:** diverse training data, length normalization, regularization.

```python
def compute_reward_normalized(model, prompt, response):
    """Normalize reward by response length."""
    raw_reward = model(prompt, response)
    length_penalty = len(response.split()) * 0.01
    return raw_reward - length_penalty
```

### Overoptimization

The RL policy becomes too good at maximizing the reward model, finding inputs the reward model scores highly that humans would score poorly.

**The KL penalty** limits this by penalizing deviation from a reference policy:

```python
# During RL, add a KL penalty:
reward_final = reward_model(response) - β * KL(π_θ || π_ref)

# β controls how much deviation is allowed.
# Higher β = stay closer to the reference policy.
```

The intuition: the reward model is imperfect. Optimize it too hard and you'll find adversarial inputs that fool it.

### Position bias

Humans systematically prefer whichever response is shown first (or in certain positions). If training data always presents the chosen response as "A," the reward model can learn to favor position rather than quality.

```python
# Randomize positions to counteract this
if random.random() > 0.5:
    comparison = (response_A, response_B, "A")
else:
    comparison = (response_B, response_A, "B")
```

### Reward scale instability

Reward values can explode or collapse during training. Normalize before use:

```python
def normalize_rewards(rewards):
    """Normalize to zero mean, unit variance."""
    return (rewards - rewards.mean()) / (rewards.std() + 1e-8)
```

---

## Part 7: Connection to key papers

### InstructGPT (Ouyang et al., 2022)

**Paper:** "Training language models to follow instructions with human feedback." NeurIPS 2022. arXiv:2203.02155.

Key contributions:
1. Collect ~40k preference comparisons from human labelers.
2. Train reward model using Bradley-Terry loss.
3. Fine-tune GPT-3 with PPO using the reward model.
4. Show improved helpfulness, honesty, and harmlessness vs base GPT-3.

The reward model starts from the SFT checkpoint (not random init), has its LM head replaced by a scalar reward head, and is used inside PPO with a KL penalty.

### Constitutional AI (Bai et al., 2022)

**Paper:** "Constitutional AI: Harmlessness from AI Feedback." arXiv:2212.08073.

Use AI-generated comparisons instead of (or alongside) human feedback:

```python
# Process:
# 1. Generate responses with base model
# 2. Ask model to critique itself: "Is this harmful?"
# 3. Revise response based on critique
# 4. Train reward model on AI-generated preferences
# 5. RLHF with this reward model (RLAIF)
```

This scales more cheaply than pure human labeling.

### DPO (Rafailov et al., 2023)

**Paper:** "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." NeurIPS 2023. arXiv:2305.18290.

DPO skips the separate reward model entirely and optimizes the policy directly on preference data. Covered in Lecture 11.

---

## Part 8: Practical tips

### Data collection

- ~1k comparisons: toy experiments only.
- ~10k comparisons: research-scale.
- 100k+: production (ChatGPT scale).

Labeler options: domain experts (expensive, high quality), crowd workers (cheap, need quality control), AI assistance (scalable, needs validation).

### Model selection

```python
# For reward model, use similar size to the policy model.
# - Policy: GPT-2 (124M) → Reward: BERT-base (110M)
# - Policy: GPT-3 (1.3B) → Reward: RoBERTa-large (355M)
# - Policy: GPT-3.5 (175B) → Reward: GPT-2 (1.5B)

# Bigger reward model is not always better;
# returns diminish past a certain size.
```

### Training details

Learning rate: start at 1e-5 to 1e-6. Reward models are sensitive to this — too high overfits to training preferences, too low fails to generalize.

Always clip gradients:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Ensembles of 3–5 models can improve reliability:
```python
def ensemble_reward(models, prompt, response):
    rewards = [model(prompt, response) for model in models]
    return np.mean(rewards)
```

---

## Part 9: Evaluation

### Accuracy on held-out comparisons

Target: above 60–70% (random baseline is 50%). InstructGPT reported ~72%.

### Agreement with humans

Show humans a prompt and two responses, and ask which the reward model prefers. Measure what fraction of time humans agree.

### Out-of-distribution generalization

Test on different prompt distributions, response lengths, and topics. A model that overfits to training preferences will fail here.

### Red flags

```python
# Signs of a bad reward model:
# 1. Val accuracy << train accuracy (overfitting)
# 2. Always prefers longer (or shorter) responses (shallow heuristic)
# 3. Inconsistent on identical inputs (numerical instability)
# 4. Extreme reward values (no normalization)
```

---

## Part 10: Full example

```python
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

# 5. Score responses
def score_response(prompt, response):
    """Score a response with trained reward model."""
    input_ids, attn_mask = format_input(prompt, response, tokenizer)
    reward = trained_model(input_ids, attn_mask)
    return reward.item()

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
# After training, reward_A should be larger than reward_B.
# The actual values depend on initialization and training data.
```

---

## Recap

Human preferences are a practical signal for reward learning when hand-crafting a reward function is infeasible. The Bradley-Terry model converts pairwise comparisons into a cross-entropy training objective. The reward model is a standard classifier trained on these comparisons. The main failure modes — reward hacking, overoptimization, position bias, scale instability — all have known mitigations. InstructGPT showed this pipeline works at scale; DPO (Lecture 11) later showed you can skip the explicit reward model.

---

## Next lecture

**[Lecture 10: PPO for language models](./10-ppo-for-llms.md)**

Where we'll take the reward model trained here, use it to fine-tune an LLM with PPO, implement the full RLHF pipeline, and see why this changes model behavior.

Before moving on, make sure you:
- [ ] Understand the Bradley-Terry model
- [ ] Can implement reward model training
- [ ] Know the main failure modes and their mitigations
- [ ] Understand the KL penalty and why it's needed

---

## References

### Foundational work

**Bradley & Terry (1952)** — "Rank analysis of incomplete block designs: I. The method of paired comparisons." Biometrika 39(3–4), pp. 324–345.

**Christiano et al. (2017)** — "Deep reinforcement learning from human preferences." NeurIPS 2017. arXiv:1706.03741.
First to apply RL from human preference comparisons at scale. Demonstrated on Atari and simulated locomotion.

**Stiennon et al. (2020)** — "Learning to summarize from human feedback." NeurIPS 2020. arXiv:2009.01325.
Applied the preference-learning pipeline to summarization; showed human-judged quality exceeds ROUGE-optimized baselines.

**Ouyang et al. (2022)** — "Training language models to follow instructions with human feedback" (InstructGPT). NeurIPS 2022. arXiv:2203.02155.
Applied RLHF to GPT-3 with ~40k human comparisons; produced ChatGPT's predecessor.

**Bai et al. (2022)** — "Constitutional AI: Harmlessness from AI Feedback." Anthropic. arXiv:2212.08073.
AI-generated feedback as a scalable alternative to human labeling.

**Rafailov et al. (2023)** — "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (DPO). NeurIPS 2023. arXiv:2305.18290.
Eliminates the separate reward model; covered in Lecture 11.

---

## Debugging checklist

A common bug: reward model always gives the same score. Often caused by frozen transformer layers. Check that `model.transformer.parameters()` have `requires_grad=True`.

A common bug: validation accuracy stuck at 50%. Often caused by position bias in the dataset — the chosen response is always in the same position. Fix by randomizing response order.

A common bug: rewards explode during PPO. Usually means reward normalization is missing. Add `(r - r.mean()) / (r.std() + 1e-8)` before passing rewards to the RL algorithm.

General advice: print reward statistics at each step. Check tensor shapes. Normalize inputs and outputs early.
