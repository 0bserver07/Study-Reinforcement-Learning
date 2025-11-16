# Lecture 10: PPO for Language Models - Complete RLHF Pipeline

> **The Full Picture**: "This is exactly how ChatGPT was trained. End to end."

**Time**: 6-8 hours | **Prerequisites**: Lectures 06, 09 | **Difficulty**: ⭐⭐⭐⭐⭐

---

## 🎯 Why This Matters

This is it. **The complete RLHF pipeline**:
- How ChatGPT learned to be helpful
- How Claude learned to be harmless
- How InstructGPT learned to follow instructions

**You're about to implement the EXACT technique** that transformed GPT-3 from "autocomplete on steroids" to "intelligent assistant."

**By the end**: You'll have working code for full RLHF training of language models.

---

## Part 1: The Complete Pipeline Overview

### The Three Stages

```
Stage 1: Supervised Fine-Tuning (SFT)
├─ Input: Base LLM + human demonstrations
├─ Process: Standard supervised learning
└─ Output: SFT model (can follow instructions somewhat)

Stage 2: Reward Model Training
├─ Input: SFT model + human preferences
├─ Process: Train RM to predict human preferences
└─ Output: Reward model r_φ(prompt, response)

Stage 3: RL Fine-Tuning (PPO)
├─ Input: SFT model + reward model
├─ Process: PPO to maximize reward
└─ Output: RLHF model (aligned with human preferences!)
```

**Key insight**: Each stage builds on the previous. Can't skip stages!

---

## Part 2: Stage 1 - Supervised Fine-Tuning

### Why SFT First?

**Base model** (GPT-3):
```
Prompt: "How do I bake a cake?"
Base model: "How do I bake a cake? How do I bake a pie? How..."
# Just continues the pattern, doesn't answer!
```

**After SFT**:
```
Prompt: "How do I bake a cake?"
SFT model: "To bake a cake, you'll need flour, eggs, sugar..."
# Actually answers! But quality varies
```

### SFT Implementation

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

def prepare_sft_data(examples):
    """
    Format data as prompt + response pairs.

    Input format:
    {
        "prompt": "Explain quantum computing",
        "response": "Quantum computers use qubits..."
    }

    Output format:
    "Prompt: Explain quantum computing\nResponse: Quantum computers use qubits..."
    """
    texts = []
    for prompt, response in zip(examples['prompt'], examples['response']):
        text = f"Prompt: {prompt}\nResponse: {response}"
        texts.append(text)
    return {"text": texts}


class SFTTrainer:
    """Supervised Fine-Tuning for instruction following."""

    def __init__(self, model_name="gpt2"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def train(self, dataset, output_dir="./sft_model", epochs=3):
        """Train SFT model."""

        # Tokenize data
        def tokenize(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=512
            )

        tokenized_data = dataset.map(tokenize, batched=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            fp16=True,  # Mixed precision
        )

        # Train
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_data,
        )

        trainer.train()
        self.model.save_pretrained(output_dir)
        return self.model


# Example usage
if __name__ == "__main__":
    # Load demonstration data
    # In practice, collect high-quality human demonstrations
    demo_data = [
        {
            "prompt": "Explain photosynthesis",
            "response": "Photosynthesis is the process by which plants..."
        },
        # ... 10k-100k examples
    ]

    sft_trainer = SFTTrainer("gpt2")
    sft_model = sft_trainer.train(demo_data)
```

**Personal note**: SFT is critical! I tried skipping straight to RL once. Model generated complete gibberish. Always start with SFT.

---

## Part 3: Stage 2 - Reward Model (Covered in Lecture 09)

Quick recap:

```python
# Train reward model on preference pairs
reward_model = RewardModel.from_pretrained("sft_model")

# Input: (prompt, response)
# Output: scalar reward
reward = reward_model("How do I code?", "Here's a Python example...")

# See Lecture 09 for full implementation
```

---

## Part 4: Stage 3 - PPO for LLMs (The Main Event!)

### The RL Problem Formulation

**State (s)**: Current prompt + partial generation
```python
s = "Explain quantum computing: Quantum computers use"
```

**Action (a)**: Next token
```python
a = "qubits"  # token ID: 47834
```

**Reward (r)**: From reward model + KL penalty
```python
# Terminal reward (after full generation)
completion = "Quantum computers use qubits that can be 0 and 1..."
r_RM = reward_model(prompt, completion)

# KL penalty (stay close to SFT model)
r_KL = -β * KL(π_RL || π_SFT)

# Total reward
r = r_RM + r_KL
```

**Policy (π)**: The language model itself
```python
π_θ(a|s) = P(next_token | prompt + partial_generation)
```

### The Complete PPO-LLM Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import copy

class PPOLLMTrainer:
    """
    PPO trainer for language models with RLHF.

    This is what ChatGPT uses!
    """

    def __init__(
        self,
        policy_model,          # The model we're training
        ref_model,             # Reference model (frozen SFT)
        reward_model,          # Trained reward model
        tokenizer,
        beta=0.1,              # KL penalty coefficient
        eps_clip=0.2,          # PPO clip parameter
        vf_coef=0.1,           # Value function coefficient
        gamma=1.0,             # Discount (usually 1 for LLMs)
        gae_lambda=0.95,       # GAE parameter
    ):
        self.policy = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.beta = beta
        self.eps_clip = eps_clip
        self.vf_coef = vf_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Value head (separate from policy)
        self.value_head = nn.Linear(
            policy_model.config.hidden_size, 1
        ).to(policy_model.device)

        # Optimizer for both policy and value
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.parameters()},
            {'params': self.value_head.parameters()}
        ], lr=1e-6)

    def generate_responses(
        self,
        prompts: List[str],
        max_length: int = 256,
        temperature: float = 0.7,
    ) -> Dict:
        """
        Generate responses and collect everything needed for PPO.

        Returns:
            - response_texts: Generated text
            - response_ids: Token IDs
            - log_probs: Log probabilities of each token
            - values: Value estimates
            - ref_log_probs: Reference model log probs (for KL)
        """
        results = {
            'response_texts': [],
            'response_ids': [],
            'log_probs': [],
            'values': [],
            'ref_log_probs': [],
            'prompts': prompts
        }

        self.policy.eval()

        for prompt in prompts:
            # Tokenize prompt
            prompt_ids = self.tokenizer(
                prompt,
                return_tensors='pt'
            ).input_ids.to(self.policy.device)

            # Generate with policy
            response_log_probs = []
            response_values = []
            response_ref_log_probs = []

            current_ids = prompt_ids

            for _ in range(max_length):
                # Get policy logits and hidden states
                with torch.no_grad():
                    outputs = self.policy(current_ids, output_hidden_states=True)
                    logits = outputs.logits[:, -1, :]  # Last token logits
                    hidden = outputs.hidden_states[-1][:, -1, :]  # Last hidden state

                    # Get value estimate
                    value = self.value_head(hidden)
                    response_values.append(value.item())

                    # Sample next token
                    probs = F.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1)

                    # Compute log prob
                    log_prob = F.log_softmax(logits, dim=-1)
                    token_log_prob = log_prob[0, next_token.item()]
                    response_log_probs.append(token_log_prob.item())

                    # Get reference model log prob (for KL)
                    ref_outputs = self.ref_model(current_ids)
                    ref_logits = ref_outputs.logits[:, -1, :]
                    ref_log_prob = F.log_softmax(ref_logits, dim=-1)
                    ref_token_log_prob = ref_log_prob[0, next_token.item()]
                    response_ref_log_probs.append(ref_token_log_prob.item())

                    # Append token
                    current_ids = torch.cat([current_ids, next_token], dim=-1)

                    # Stop at EOS
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break

            # Decode response
            response_text = self.tokenizer.decode(
                current_ids[0, prompt_ids.shape[1]:],
                skip_special_tokens=True
            )

            results['response_texts'].append(response_text)
            results['response_ids'].append(current_ids[0].tolist())
            results['log_probs'].append(response_log_probs)
            results['values'].append(response_values)
            results['ref_log_probs'].append(response_ref_log_probs)

        return results

    def compute_rewards(
        self,
        prompts: List[str],
        responses: List[str],
        log_probs: List[List[float]],
        ref_log_probs: List[List[float]]
    ) -> List[List[float]]:
        """
        Compute rewards for each token.

        Reward = reward_model(prompt, response) - β * KL(π||π_ref)

        Applied at the last token (sparse reward).
        """
        all_rewards = []

        for prompt, response, lp, ref_lp in zip(
            prompts, responses, log_probs, ref_log_probs
        ):
            # Get reward from reward model (only at end)
            with torch.no_grad():
                rm_reward = self.reward_model(prompt, response)

            # Compute KL divergence at each token
            kl_per_token = [lp_t - ref_lp_t for lp_t, ref_lp_t in zip(lp, ref_lp)]
            kl_penalty = [-self.beta * kl for kl in kl_per_token]

            # Reward = RM reward (at end) + KL penalty (per token)
            rewards = kl_penalty[:]
            rewards[-1] += rm_reward.item()  # Add RM reward to last token

            all_rewards.append(rewards)

        return all_rewards

    def compute_advantages(
        self,
        rewards: List[List[float]],
        values: List[List[float]]
    ) -> tuple:
        """
        Compute advantages using GAE.

        Returns:
            advantages, returns
        """
        all_advantages = []
        all_returns = []

        for reward_seq, value_seq in zip(rewards, values):
            advantages = []
            returns = []

            # Add terminal value (0)
            values_with_terminal = value_seq + [0.0]

            # Compute GAE
            gae = 0
            for t in reversed(range(len(reward_seq))):
                delta = (
                    reward_seq[t] +
                    self.gamma * values_with_terminal[t + 1] -
                    values_with_terminal[t]
                )
                gae = delta + self.gamma * self.gae_lambda * gae
                advantages.insert(0, gae)

            # Returns = advantages + values
            returns = [a + v for a, v in zip(advantages, value_seq)]

            all_advantages.append(advantages)
            all_returns.append(returns)

        # Normalize advantages (across all sequences)
        flat_adv = [a for adv_seq in all_advantages for a in adv_seq]
        mean = sum(flat_adv) / len(flat_adv)
        std = (sum((a - mean)**2 for a in flat_adv) / len(flat_adv))**0.5

        all_advantages = [
            [(a - mean) / (std + 1e-8) for a in adv_seq]
            for adv_seq in all_advantages
        ]

        return all_advantages, all_returns

    def ppo_update(
        self,
        response_data: Dict,
        rewards: List[List[float]],
        advantages: List[List[float]],
        returns: List[List[float]],
        num_epochs: int = 4
    ):
        """
        PPO update step for language model.
        """
        self.policy.train()

        for epoch in range(num_epochs):
            total_policy_loss = 0
            total_value_loss = 0
            total_loss = 0

            for i, prompt in enumerate(response_data['prompts']):
                # Get response data
                response_ids = response_data['response_ids'][i]
                old_log_probs = response_data['log_probs'][i]
                old_values = response_data['values'][i]
                adv = advantages[i]
                ret = returns[i]

                # Convert to tensors
                input_ids = torch.tensor([response_ids]).to(self.policy.device)
                old_log_probs_t = torch.tensor(old_log_probs)
                advantages_t = torch.tensor(adv).to(self.policy.device)
                returns_t = torch.tensor(ret).to(self.policy.device)

                # Forward pass with current policy
                outputs = self.policy(input_ids, output_hidden_states=True)
                logits = outputs.logits[0]  # [seq_len, vocab_size]
                hidden_states = outputs.hidden_states[-1][0]  # [seq_len, hidden_dim]

                # Compute new log probs
                # (only for response tokens, not prompt)
                prompt_len = len(self.tokenizer(prompt).input_ids)
                response_logits = logits[prompt_len-1:-1]  # Shift by 1 for next token prediction
                response_tokens = input_ids[0, prompt_len:]

                log_probs = F.log_softmax(response_logits, dim=-1)
                new_log_probs = log_probs.gather(
                    1, response_tokens.unsqueeze(-1)
                ).squeeze(-1)

                # Compute new values
                response_hidden = hidden_states[prompt_len-1:-1]
                new_values = self.value_head(response_hidden).squeeze(-1)

                # PPO clipped objective
                ratio = torch.exp(new_log_probs - old_log_probs_t.to(self.policy.device))
                surr1 = ratio * advantages_t
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_t
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(new_values, returns_t)

                # Total loss
                loss = policy_loss + self.vf_coef * value_loss

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_loss += loss.item()

            # Log epoch stats
            n = len(response_data['prompts'])
            print(f"  Epoch {epoch+1}/{num_epochs}: "
                  f"Policy Loss={total_policy_loss/n:.4f}, "
                  f"Value Loss={total_value_loss/n:.4f}")

        return {
            'policy_loss': total_policy_loss / n,
            'value_loss': total_value_loss / n,
            'total_loss': total_loss / n
        }

    def train_step(self, prompts: List[str], num_epochs: int = 4):
        """
        One full PPO training step.

        1. Generate responses
        2. Compute rewards
        3. Compute advantages
        4. PPO update
        """
        print(f"\n{'='*60}")
        print(f"PPO Training Step - {len(prompts)} prompts")
        print(f"{'='*60}")

        # 1. Generate
        print("Generating responses...")
        response_data = self.generate_responses(prompts)

        # 2. Compute rewards
        print("Computing rewards...")
        rewards = self.compute_rewards(
            prompts,
            response_data['response_texts'],
            response_data['log_probs'],
            response_data['ref_log_probs']
        )

        # 3. Compute advantages
        print("Computing advantages...")
        advantages, returns = self.compute_advantages(
            rewards,
            response_data['values']
        )

        # 4. PPO update
        print("Running PPO update...")
        metrics = self.ppo_update(
            response_data,
            rewards,
            advantages,
            returns,
            num_epochs
        )

        # Log sample
        print(f"\nSample generation:")
        print(f"Prompt: {prompts[0]}")
        print(f"Response: {response_data['response_texts'][0]}")
        print(f"Reward: {sum(rewards[0]):.2f}")

        return metrics


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("PPO for Language Models - RLHF Training")
    print("="*60)

    # Load models
    model_name = "gpt2"  # In practice, use your SFT model

    policy_model = AutoModelForCausalLM.from_pretrained(model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)
    # reward_model = load_reward_model()  # From Lecture 09
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Create trainer
    trainer = PPOLLMTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        reward_model=None,  # Would use real RM here
        tokenizer=tokenizer,
        beta=0.1,
        eps_clip=0.2,
    )

    # Training loop
    prompts = [
        "Explain how photosynthesis works:",
        "Write a Python function to reverse a string:",
        "What is the capital of France?",
    ]

    for step in range(10):
        metrics = trainer.train_step(prompts, num_epochs=4)
        print(f"\nStep {step}: {metrics}\n")

    # Save aligned model
    policy_model.save_pretrained("./rlhf_model")
    print("\nRLHF training complete!")
```

---

## Part 5: The Critical Components

### 1. KL Penalty (Most Important!)

**Why KL penalty**:
```python
# Without KL penalty:
# Model learns to game the reward model
# Generates nonsense that scores high

# Example:
reward_model("Explain quantum", "QUANTUM QUANTUM QUANTUM...")
# Scores 10/10 (loves the word "quantum")!

# With KL penalty:
kl = KL(π_RL || π_SFT)
reward = reward_model_score - β * kl

# Model stays close to sensible SFT outputs
```

**β tuning**:
- β = 0: No constraint, reward hacking
- β = 0.01: Very loose, still hacks
- β = 0.1: Sweet spot (InstructGPT used this)
- β = 1.0: Too tight, doesn't learn

### 2. Value Function

**Why separate value head**:
```python
# Policy outputs: P(next_token | context)
# Value outputs: Expected total reward from this state

# Share most layers, separate heads
# More efficient than separate networks
```

### 3. Sparse Rewards

**LLM reward is at the end**:
```python
# Not like Atari (reward every frame)
# LLM: only reward after full generation

# Prompt: "Explain quantum"
# Generate: "Quantum computers use..." (no reward yet)
# ...keep generating... (still no reward)
# Finish: "...and entanglement." (REWARD!)

# This is why GAE is critical
# Credit assignment across 100+ tokens
```

---

## Part 6: The Gotchas (These Took Me MONTHS)

### Gotcha #1: OOM (Out of Memory)

```python
# Problem: Storing gradients for full generation
# GPT-2: 124M parameters × 256 tokens = HUGE

# Solutions:
# 1. Gradient checkpointing
model.gradient_checkpointing_enable()

# 2. Gradient accumulation
for i in range(accumulation_steps):
    loss.backward()
# optimizer.step()  # Only once every N steps

# 3. Smaller batch sizes
# Instead of 64 prompts, use 4-8
```

### Gotcha #2: Reward Hacking

```python
# Model learns to exploit RM weaknesses

# Example hacks I've seen:
# 1. Repetition: "Great! Great! Great!..."
# 2. Verbosity: 1000 word answers to simple questions
# 3. Unrelated high-reward phrases

# Solutions:
# - Strong KL penalty (β=0.1)
# - Length normalization
# - Ensemble reward models
# - Regular RM updates
```

### Gotcha #3: Mode Collapse

```python
# Policy becomes deterministic
# Always generates same response

# Cause: Entropy coefficient too low
# Fix: Add entropy bonus

entropy = -Σ π(a) log π(a)
loss = policy_loss - 0.01 * entropy

# Encourages diversity
```

### Gotcha #4: Training Time

```python
# RLHF is SLOW

# InstructGPT (GPT-3 175B):
# - SFT: 16 hours on 256 GPUs
# - RM: 8 hours on 128 GPUs
# - PPO: 8 hours on 256 GPUs
# Total: ~3 days on massive cluster

# For GPT-2 scale:
# - SFT: 2-4 hours on 1 GPU
# - RM: 1-2 hours on 1 GPU
# - PPO: 4-8 hours on 1 GPU
# Total: ~12 hours

# Patience is key!
```

---

## Part 7: Evaluation

### How to Know If It's Working

**During training, monitor**:
```python
# 1. Reward should increase
plot(training_step, avg_reward)
# Should go up!

# 2. KL should stay bounded
plot(training_step, kl_divergence)
# Should be < 10 (if β=0.1)

# 3. Response quality (manual check)
# Sample generations every N steps
# Read them yourself!

# 4. Reward model score
# Should correlate with actual quality
# If not, RM is broken
```

**Final evaluation**:
```python
# 1. Human evaluation
# Show humans responses from SFT vs RLHF
# Ask: which is better?
# RLHF should win 70-80% of time

# 2. GPT-4 as judge
# Ask GPT-4 to compare responses
# Cheaper than humans

# 3. Benchmarks
# HHH (Helpful, Honest, Harmless)
# TruthfulQA, etc.
```

---

## Part 8: Real-World Numbers

### InstructGPT (GPT-3)

**Data**:
- SFT: 13k demonstrations
- RM: 33k comparisons
- PPO: 31k prompts

**Results**:
- 1.3B InstructGPT > 175B base GPT-3
- Labelers prefer InstructGPT 85% of time
- More truthful, less toxic

**Cost**: ~$1M in compute (estimated)

### Anthropic's Claude

**Process**:
- Constitutional AI (AI feedback instead of human)
- Multiple rounds of RLHF
- Iterative improvement

**Results**:
- Safer than competitors
- Follows nuanced instructions
- Better at refusing harmful requests

---

## Key Takeaways

1. **RLHF = SFT → RM → PPO** - three-stage pipeline
2. **KL penalty is critical** - prevents reward hacking
3. **Sparse rewards are hard** - need GAE for credit assignment
4. **Scale matters** - works better with more data
5. **This is ChatGPT** - exact technique, proven at scale
6. **Slow but worth it** - training takes days but results are dramatic
7. **Monitor everything** - reward, KL, quality samples

---

## Next Lecture

**[Lecture 13: RLHF for Code Generation](./13-rlhf-code.md)**

Apply RLHF to code specifically - execution feedback, test suites, AlphaCode!

Before moving on:
- [ ] Understand all three RLHF stages
- [ ] Know why KL penalty matters
- [ ] Can implement basic PPO-LLM
- [ ] Appreciate the engineering complexity

---

## References

### The Papers

**Ouyang et al. (2022)** - "Training language models to follow instructions with human feedback"
- **arXiv**: [2203.02155](https://arxiv.org/abs/2203.02155)
- InstructGPT paper
- First large-scale RLHF for LLMs
- Blueprint for ChatGPT

**Bai et al. (2022)** - "Training a Helpful and Harmless Assistant with RLHF"
- Anthropic's approach
- Constitutional AI
- Claude's foundation

**Stiennon et al. (2020)** - "Learning to summarize from human feedback"
- Summarization with RLHF
- Showed it scales to complex tasks

### Code Resources

**HuggingFace TRL** - https://github.com/huggingface/trl
- Production RLHF library
- PPO for transformers
- Used by many companies

**OpenAI Baselines** - https://github.com/openai/baselines
- Reference PPO implementation
- Well-tested, documented

---

## My Implementation War Stories

**Month 1**: OOM errors constantly. Tried gradient checkpointing, reduced batch size to 1. Still OOM. Finally: gradient accumulation + mixed precision. Success!

**Month 2**: Reward always negative, never improved. Bug: Forgot to add RM reward to final token. Only had KL penalty. Added RM reward, immediately started learning.

**Month 3**: Generated responses were gibberish. High reward but nonsense. Reward hacking! Increased β from 0.01 to 0.1. Problem solved.

**Month 4**: Training stalled after 1000 steps. KL divergence hit 100+. Policy diverged too far. Needed more frequent RM updates and higher β.

**Lessons**:
- Print sample generations EVERY step
- Monitor KL religiously
- Start with β=0.1, tune later
- Gradient accumulation is your friend
- Patience! RLHF is slow

---

*Last Updated: 2025*

---

**This is the complete pipeline. You now understand how ChatGPT works.**
