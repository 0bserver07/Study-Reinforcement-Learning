# Lecture 06: PPO - Proximal Policy Optimization

> **The Industry Standard**: "This is what actually works in production. This is what trained ChatGPT."

**Time**: 5-6 hours | **Prerequisites**: Lectures 01-02, 04 | **Difficulty**: ⭐⭐⭐⭐⭐

---

## 🎯 Why This Matters

**PPO is THE algorithm** for modern RL:
- ChatGPT was trained with PPO
- OpenAI uses PPO for everything
- DeepMind uses PPO variants
- Every major RL success story uses PPO

**Why?** Because it's the **first algorithm that actually works reliably**:
- Stable (doesn't explode or collapse)
- Sample efficient (learns from less data)
- Easy to tune (robust to hyperparameters)
- Scalable (works from Atari to LLMs)

**If you only learn ONE RL algorithm, make it PPO.**

---

## Part 1: The Problem PPO Solves

### Why Previous Methods Failed

**REINFORCE** (Lecture 02):
```python
# High variance → slow learning
# Every trajectory is independent → wasteful
# No constraint on update size → can break policy
```

**TRPO** (Trust Region Policy Optimization):
```python
# Constraint: KL(π_new || π_old) ≤ δ
# Requires second-order optimization (Fisher matrix)
# Computationally expensive
# Hard to implement correctly
```

**What we want**:
- ✅ Large updates when safe (learn fast)
- ✅ Small updates when risky (stay stable)
- ✅ First-order optimization (just gradients)
- ✅ Easy to implement

**PPO gives us all of this.**

---

## Part 2: The Core Idea (Intuition First!)

### The Trust Region Concept

Imagine you're hiking in fog:
- You can see 10 meters ahead
- Want to walk toward higher ground
- But can't see far enough to run

**Policy optimization is the same**:
- Current policy π_old is your current position
- Want to improve (go uphill)
- But can only trust local information

**TRPO's approach**: "Don't move more than δ distance"
**PPO's approach**: "Clip your step size if it's too large"

### The Clipping Mechanism

```python
# Standard policy gradient:
L = probability_ratio * advantage
#   = (π_new / π_old) * A

# PPO clips the ratio:
clipped_ratio = clip(ratio, 1-ε, 1+ε)
L_PPO = min(ratio * A, clipped_ratio * A)
```

**Intuition**:
- If ratio = 1.5 (new policy 50% more likely): clip to 1.2 (if ε=0.2)
- If ratio = 0.7 (new policy 30% less likely): clip to 0.8
- **Prevents large policy changes**

**Visual**:
```
Advantage > 0 (good action):
  ratio:  0.8  0.9  1.0  1.1  1.2  1.3  1.4
  used:   0.8  0.9  1.0  1.1  1.2  1.2  1.2  ← clipped
                                  ↑ε=0.2

Advantage < 0 (bad action):
  ratio:  0.6  0.7  0.8  0.9  1.0  1.1  1.2
  used:   0.8  0.8  0.8  0.9  1.0  1.1  1.2  ← clipped
          ↑ε=0.2
```

---

## Part 3: The Complete PPO Algorithm

### PPO-Clip (Most Common Variant)

```
For iteration = 1, 2, 3, ...:

  1. Collect trajectories using π_θ_old:
     - Run policy for N timesteps
     - Store: states, actions, rewards, log_probs

  2. Compute advantages:
     - Estimate value function V(s)
     - A(s,a) = Q(s,a) - V(s)
     - Usually use GAE (Generalized Advantage Estimation)

  3. Update policy for K epochs:
     - Compute ratio = π_θ(a|s) / π_θ_old(a|s)
     - Compute clipped objective:
       L_CLIP = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
     - Also update value function:
       L_VF = (V_θ(s) - V_target)²
     - Add entropy bonus:
       L_S = -entropy(π_θ)
     - Total loss: L = L_CLIP + c1*L_VF + c2*L_S

  4. θ_old ← θ
```

### Key Components Explained

**1. Advantage Estimation (GAE)**

Instead of full returns, use bootstrapping:
```python
# Monte Carlo (REINFORCE):
A_t = R_t - V(s_t)  # High variance

# TD(0):
A_t = r_t + γV(s_{t+1}) - V(s_t)  # Low variance, high bias

# GAE (best of both):
A_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
where δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

**λ = 0**: Pure TD (low variance)
**λ = 1**: Pure Monte Carlo (high variance)
**λ = 0.95**: Sweet spot (most common)

**2. Value Function**

PPO learns a value function alongside the policy (actor-critic):
```python
# Use same network or separate
V_θ(s) ≈ expected return from state s
```

**3. Entropy Bonus**

Encourages exploration:
```python
H(π) = -Σ π(a|s) log π(a|s)
# Higher entropy = more random = more exploration
```

---

## Part 4: Complete Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import gym

class ActorCritic(nn.Module):
    """
    Shared network for policy and value function.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # Policy head
        self.policy = nn.Linear(hidden_dim, action_dim)

        # Value head
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        """
        Returns both policy logits and value.
        """
        shared = self.shared(state)
        policy_logits = self.policy(shared)
        value = self.value(shared)
        return policy_logits, value

    def get_action(self, state):
        """
        Sample action from policy and return log probability.
        """
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action.item(), log_prob, entropy, value


class PPO:
    """
    Proximal Policy Optimization (PPO-Clip).
    """
    def __init__(
        self,
        env,
        hidden_dim=64,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        K_epochs=4,
        gae_lambda=0.95,
        c1=0.5,  # value loss coefficient
        c2=0.01, # entropy coefficient
    ):
        self.env = env
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.gae_lambda = gae_lambda
        self.c1 = c1
        self.c2 = c2

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # Policy network
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # Storage for rollouts
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def select_action(self, state):
        """Select action and store for training."""
        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action, log_prob, entropy, value = self.policy.get_action(state)

        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)

        return action

    def store_transition(self, reward, done):
        """Store reward and done flag."""
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_gae(self, next_value):
        """
        Compute Generalized Advantage Estimation.

        GAE(γ,λ) = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
        where δ_t = r_t + γV(s_{t+1}) - V(s_t)
        """
        advantages = []
        gae = 0

        # Convert to tensors
        rewards = torch.tensor(self.rewards)
        dones = torch.tensor(self.dones, dtype=torch.float)
        values = torch.cat(self.values).squeeze()

        # Add next value for bootstrapping
        values = torch.cat([values, next_value])

        # Compute advantages backward
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = values[t + 1]
            else:
                next_value = values[t + 1] * (1 - dones[t])

            # TD error
            delta = rewards[t] + self.gamma * next_value - values[t]

            # GAE
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages)

        # Returns = advantages + values
        returns = advantages + values[:-1]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update(self, next_state):
        """
        PPO update using collected rollouts.
        """
        # Get next value for GAE
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        with torch.no_grad():
            _, next_value = self.policy(next_state)

        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value)

        # Convert lists to tensors
        old_states = torch.cat(self.states)
        old_actions = torch.tensor(self.actions)
        old_log_probs = torch.stack(self.log_probs).detach()

        # PPO update for K epochs
        for epoch in range(self.K_epochs):
            # Evaluate actions under current policy
            logits, values = self.policy(old_states)
            dist = Categorical(logits=logits)

            log_probs = dist.log_prob(old_actions)
            entropy = dist.entropy()
            values = values.squeeze()

            # Ratio: π_θ(a|s) / π_θ_old(a|s)
            ratios = torch.exp(log_probs - old_log_probs)

            # Clipped surrogate objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value function loss
            value_loss = F.mse_loss(values, returns)

            # Entropy bonus (encourage exploration)
            entropy_loss = -entropy.mean()

            # Total loss
            loss = policy_loss + self.c1 * value_loss + self.c2 * entropy_loss

            # Update
            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping (important!)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        # Clear storage
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': -entropy_loss.item(),
            'total_loss': loss.item()
        }

    def train(self, num_episodes=1000, max_steps=500, update_every=2048):
        """
        Train PPO agent.
        """
        episode_rewards = []
        steps = 0

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False

            while not done and steps < max_steps:
                # Select action
                action = self.select_action(state)

                # Environment step
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Store
                self.store_transition(reward, done)
                episode_reward += reward
                steps += 1
                state = next_state

                # Update policy
                if steps % update_every == 0:
                    metrics = self.update(next_state)
                    print(f"Step {steps} | {metrics}")

            episode_rewards.append(episode_reward)

            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode + 1} | Avg Reward: {avg_reward:.2f}")

        return episode_rewards


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("PPO on CartPole-v1")
    print("="*60 + "\n")

    env = gym.make('CartPole-v1')

    agent = PPO(
        env,
        hidden_dim=64,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        K_epochs=4,
        gae_lambda=0.95,
    )

    rewards = agent.train(num_episodes=300, update_every=2048)

    print("\n" + "="*60)
    print(f"Training complete!")
    print(f"Final 50-episode average: {np.mean(rewards[-50:]):.2f}")
    print("="*60)
```

---

## Part 5: Why PPO Works (The Theory)

### The Objective Function

PPO maximizes:
```
L^CLIP(θ) = 𝔼[min(r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t)]
```

Where `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)`

**Case analysis**:

**Case 1: A_t > 0 (good action)**
- Want to increase π_θ(a|s)
- ratio > 1 means we're already increasing it
- Clip at 1+ε to prevent too large increase
- Result: **Conservative improvement**

**Case 2: A_t < 0 (bad action)**
- Want to decrease π_θ(a|s)
- ratio < 1 means we're already decreasing it
- Clip at 1-ε to prevent too large decrease
- Result: **Conservative discouragement**

**Personal Note**: This clicked when I realized: PPO is **pessimistic**. It takes the minimum of clipped and unclipped. Always chooses the safer option.

### Why Clipping Works

**Without clipping** (standard policy gradient):
```python
# One bad estimate can destroy policy
if advantage_estimate is very_wrong:
    policy_update is catastrophic
    policy is ruined
```

**With clipping**:
```python
# Bad estimates are bounded
if advantage_estimate is very_wrong:
    ratio gets clipped
    policy_update is limited
    policy stays reasonable
```

---

## Part 6: The Gotchas (I Spent Weeks on These)

### Gotcha #1: Advantage Normalization

```python
# Without normalization:
advantages = compute_advantages()  # Range: [-1000, +1000]
# Huge gradients, training explodes

# With normalization:
advantages = (advantages - mean) / (std + 1e-8)  # Range: ~[-3, +3]
# Stable gradients, reliable training

# ALWAYS NORMALIZE ADVANTAGES!
```

### Gotcha #2: Value Function Scale

```python
# Problem: Returns can be huge
returns = [500, 1000, 750, ...]  # CartPole

# Value network outputs tiny numbers at start
values = [0.1, 0.2, 0.15, ...]

# MSE loss is HUGE
loss = (returns - values)² # = (500 - 0.1)² = 250,000!

# Solution: Normalize returns OR clip value loss
returns = (returns - mean) / (std + 1e-8)
# OR
value_loss = torch.clamp(value_loss, -10, 10)
```

### Gotcha #3: Learning Rate

```python
# PPO is sensitive to learning rate

# Too high (1e-2):
#   - Policy changes too fast
#   - Violates trust region even with clipping
#   - Unstable

# Too low (1e-6):
#   - Learning is too slow
#   - Takes forever

# Sweet spot: 3e-4 (default)
# Can tune 1e-4 to 1e-3, but 3e-4 works 90% of time
```

### Gotcha #4: Number of Epochs (K)

```python
# K too small (K=1):
#   - Not enough updates per batch
#   - Sample inefficient

# K too large (K=20):
#   - Overfit to current batch
#   - Policy diverges from π_old
#   - Defeats purpose of clipping

# Sweet spot: K=4 or K=10
# More is NOT better!
```

### Gotcha #5: Batch Size

```python
# Too small (128 steps):
#   - High variance advantage estimates
#   - Noisy gradients

# Too large (100,000 steps):
#   - Sample inefficient
#   - Slow iteration

# Sweet spot: 2048-4096 for simple tasks
#             32768-65536 for complex tasks (LLMs)
```

---

## Part 7: PPO Variants

### PPO-Clip (What We Implemented)

```python
L = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
```

**Pros**: Simple, effective, most common
**Cons**: ε is a hyperparameter

### PPO-Penalty (Alternative)

```python
L = ratio * A - β * KL(π_θ || π_old)
```

**Pros**: Adaptive KL penalty
**Cons**: More hyperparameters (β, KL target)

### PPO-Adaptive (Research)

Adapts ε based on KL divergence:
```python
if KL > KL_target * 1.5:
    ε *= 0.5  # Decrease (more conservative)
elif KL < KL_target / 1.5:
    ε *= 2.0  # Increase (more aggressive)
```

---

## Part 8: When to Use PPO

### Use PPO When:

✅ **Need reliability** - PPO almost always works
✅ **Have on-policy data** - Collecting new data each update
✅ **Want good performance** - SOTA on many tasks
✅ **Continuous or discrete** - Works for both
✅ **Have compute** - Requires multiple epochs per batch

### Don't Use PPO When:

❌ **Off-policy learning** - Use SAC or TD3 instead
❌ **Extreme sample efficiency** - Model-based might be better
❌ **Simple problem** - DQN might suffice
❌ **Need simplicity** - For alignment, DPO is simpler

### Real-World Usage:

**OpenAI**: PPO for everything (Dota 2, robotics, ChatGPT)
**DeepMind**: PPO + variants (AlphaStar, Agent57)
**Anthropic**: PPO for RLHF (Claude)
**You**: Should probably start with PPO!

---

## Part 9: Hyperparameter Guide

### Default Values (Start Here)

```python
PPO(
    lr=3e-4,           # Learning rate
    gamma=0.99,        # Discount factor
    eps_clip=0.2,      # Clipping parameter
    K_epochs=4,        # Optimization epochs
    gae_lambda=0.95,   # GAE parameter
    c1=0.5,            # Value loss coefficient
    c2=0.01,           # Entropy coefficient
)
```

### When to Tune

**If learning is unstable**:
- Decrease lr (try 1e-4)
- Decrease eps_clip (try 0.1)
- Increase gradient clip (try 1.0)

**If learning is too slow**:
- Increase lr (try 1e-3)
- Increase K_epochs (try 10)
- Decrease c2 (less exploration)

**If policy is too deterministic**:
- Increase c2 (more entropy)
- Decrease eps_clip (allow more change)

**If sample inefficient**:
- Increase K_epochs (reuse data more)
- Tune gae_lambda (try 0.9-0.99)

---

## Part 10: Connection to RLHF

### PPO for Language Models

**Same algorithm, different domain**:

```python
# RL environment → LLM generation
state = prompt
action = next_token
reward = reward_model(prompt, completion)

# Key differences:
# 1. Huge action space (50k tokens vs 4 discrete actions)
# 2. Sequential generation (autoregressive)
# 3. Reward at end (sparse)
# 4. KL penalty critical (prevent nonsense)
```

**Why PPO for LLMs**:
- Handles high-dimensional action spaces
- Stable with large networks (billions of parameters)
- Can incorporate KL penalty (stay close to SFT model)
- Works with sparse rewards (only at end of sequence)

**We'll implement this in Lecture 10!**

---

## Key Takeaways

1. **PPO is the industry standard** - used everywhere
2. **Clipping prevents catastrophic updates** - trust region via simple mechanism
3. **Multiple epochs per batch** - sample efficient
4. **Actor-Critic architecture** - policy + value function
5. **GAE for variance reduction** - better than pure MC or TD
6. **Hyperparameters are robust** - defaults work 90% of time
7. **This is what trained ChatGPT** - foundation of RLHF

---

## Next Steps

Before moving on:
- [ ] Run the CartPole implementation
- [ ] Experiment with hyperparameters
- [ ] Understand why clipping helps
- [ ] Compare to REINFORCE (Lecture 02)
- [ ] Visualize advantage computation

**Next Lecture**: [Lecture 10: PPO for Language Models](./10-ppo-for-llms.md)

Where we'll apply PPO to LLMs for complete RLHF!

---

## References

### The Paper

**Schulman et al. (2017)** - "Proximal Policy Optimization Algorithms"
- **arXiv**: [1707.06347](https://arxiv.org/abs/1707.06347)
- Introduced PPO-Clip and PPO-Penalty
- Became instant standard
- OpenAI's workhorse algorithm

### Applications

**OpenAI Five** (Dota 2, 2019)
- Trained with PPO
- Beat world champions
- 10 months of training

**Ouyang et al. (2022)** - "Training language models to follow instructions" (InstructGPT)
- Used PPO for RLHF
- Created ChatGPT's predecessor
- Showed PPO scales to LLMs

**Christiano et al. (2017)** - "Deep reinforcement learning from human preferences"
- First RLHF with PPO
- Atari from preferences
- Foundation for modern alignment

---

## My Debugging Log

**Week 1**: Gradients exploded. Bug: Forgot to normalize advantages. Added normalization, fixed instantly.

**Week 2**: Policy collapsed (all actions same). Bug: Entropy coefficient too low (0.0001). Increased to 0.01, recovered.

**Week 3**: Value loss dominated. Bug: Value targets too large, MSE huge. Normalized returns, much better.

**Week 4**: Policy update too conservative. Bug: K_epochs=1 not enough. Increased to 4, faster learning.

**Lesson**: Print everything. Monitor ratio clipping percentage. If >80% of ratios clipped, ε too small.

```python
# Add this to debug:
clipped = (ratio < 1-eps) | (ratio > 1+eps)
print(f"Clipped: {clipped.float().mean():.2%}")

# Good: 10-40% clipped
# Too low (<5%): ε too large, not constraining
# Too high (>60%): ε too small, over-constraining
```

---

*Last Updated: 2025*

---

**Tomorrow's TODO**:
- [ ] Implement continuous action PPO (for robotics)
- [ ] Add recurrent policy (LSTM)
- [ ] Try on harder environment (MuJoCo)
- [ ] Compare to DPO (simpler but less flexible)
