# Deep RL Quick Reference & Cheat Sheet

A comprehensive quick reference for deep reinforcement learning algorithms, hyperparameters, and best practices.

---

## Table of Contents

1. [Algorithm Comparison Table](#algorithm-comparison-table)
2. [When to Use Which Algorithm](#when-to-use-which-algorithm)
3. [Hyperparameter Guide](#hyperparameter-guide)
4. [Key Formulas](#key-formulas)
5. [Common Bugs & Fixes](#common-bugs--fixes)
6. [Implementation Checklist](#implementation-checklist)

---

## Algorithm Comparison Table

| Algorithm | Type | Actions | Policy | Sample Efficiency | Stability | Complexity | Use Case |
|-----------|------|---------|--------|-------------------|-----------|------------|----------|
| **REINFORCE** | PG | Discrete/Continuous | Stochastic | Very Low | Medium | Low | Learning, simple tasks |
| **DQN** | Value | Discrete only | Deterministic | Medium | Medium | Medium | Atari, discrete control |
| **A2C** | Actor-Critic | Discrete/Continuous | Stochastic | Medium | Medium | Medium | General purpose |
| **A3C** | Actor-Critic | Discrete/Continuous | Stochastic | Medium | Medium | Medium | Parallel training |
| **TRPO** | PG + Trust Region | Discrete/Continuous | Stochastic | Medium | High | Very High | Research, safety-critical |
| **PPO** | PG + Clipping | Discrete/Continuous | Stochastic | Medium | High | Medium | **Best general purpose** |
| **DDPG** | Off-Policy AC | Continuous only | Deterministic | High | Low | Medium | Continuous control |
| **TD3** | Off-Policy AC | Continuous only | Deterministic | High | High | Medium | Robotics, continuous control |
| **SAC** | Off-Policy AC | Continuous only | Stochastic | Very High | Very High | Medium | **Best for continuous** |
| **Dyna-Q** | Model-Based | Discrete | Deterministic | High | Medium | Medium | Sample-limited environments |
| **MBPO** | Model-Based + SAC | Continuous | Stochastic | Very High | High | High | Expensive real-world data |
| **Dreamer** | Model-Based | Discrete/Continuous | Stochastic | Very High | Medium | Very High | High-dim observations |

### Legend
- **PG** = Policy Gradient
- **AC** = Actor-Critic
- **Sample Efficiency**: Low = millions of steps, High = tens of thousands
- **Stability**: How robust to hyperparameters
- **Complexity**: Implementation difficulty

---

## When to Use Which Algorithm

### Decision Tree

```
Are actions continuous or discrete?
│
├─ Discrete
│  │
│  ├─ Do you need off-policy (data reuse)?
│  │  ├─ Yes → DQN
│  │  └─ No → PPO
│  │
│  └─ Is sample efficiency critical?
│     ├─ Yes → DQN or Model-Based (Dyna-Q)
│     └─ No → PPO
│
└─ Continuous
   │
   ├─ Do you need maximum sample efficiency?
   │  ├─ Yes → SAC or MBPO
   │  └─ No → PPO
   │
   ├─ Is safety/stability critical?
   │  ├─ Yes → TRPO or PPO
   │  └─ No → SAC or TD3
   │
   └─ Do you have parallel environments?
      ├─ Yes → PPO
      └─ No → SAC
```

### By Domain

**Game Playing (Atari, Board Games)**
- Discrete actions → DQN
- Lots of compute → AlphaZero / MuZero (model-based)
- Limited compute → PPO

**Robotics**
- Continuous actions → SAC or TD3
- Expensive real data → MBPO
- Safety critical → TRPO (but slow)
- **Recommended: SAC**

**LLM Alignment (RLHF)**
- Always use PPO
- Why: On-policy needed for KL penalty, handles sparse rewards well
- See Lecture 10 for details

**Code Generation**
- Use PPO with execution-based rewards
- See Lecture 13 for details

**Recommender Systems**
- Discrete actions → DQN
- Continuous → SAC
- Need interpretability → Model-based

---

## Hyperparameter Guide

### Learning Rates

| Algorithm | Actor LR | Critic LR | Notes |
|-----------|----------|-----------|-------|
| REINFORCE | 1e-3 to 1e-4 | N/A | Start high, decay |
| DQN | N/A | 1e-4 to 1e-3 | Lower for stability |
| A2C/A3C | 3e-4 | Same | Shared optimizer |
| PPO | 3e-4 | 3e-4 | Very robust |
| TD3 | 3e-4 | 3e-4 | Same for both |
| SAC | 3e-4 | 3e-4 | Can use 1e-3 for faster learning |

**Rule of thumb**: Start with 3e-4, decrease if unstable, increase if too slow.

### Discount Factor (γ)

| Episode Length | Recommended γ | Reasoning |
|----------------|---------------|-----------|
| Short (<100 steps) | 0.99 | Standard |
| Medium (100-1000) | 0.99 | Standard |
| Long (>1000 steps) | 0.995 - 0.999 | Need longer horizon |
| Infinite horizon | 0.99 | Prevent vanishing |
| Sparse rewards | 0.99 - 0.999 | Higher for delayed rewards |

**Default: γ = 0.99** works for most cases.

### Buffer Sizes (Off-Policy Methods)

| Environment | Replay Buffer Size | Reasoning |
|-------------|-------------------|-----------|
| Simple (CartPole) | 10K - 100K | Small state space |
| Complex (Atari) | 1M | Need diverse experiences |
| Continuous Control | 1M | Standard |
| Very complex | 10M | If you have memory |

**Default: 1M** for most off-policy methods.

### Batch Sizes

| Algorithm | Batch Size | Notes |
|-----------|-----------|-------|
| REINFORCE | Full episode | By definition |
| DQN | 32 - 128 | Smaller for stability |
| A2C | Full episode or 5-20 steps | N-step returns |
| PPO | 64 - 4096 | Depends on parallel envs |
| TD3/SAC | 256 - 1024 | Larger is more stable |

**Default: 256** for off-policy, **64-256** for on-policy.

### PPO Specific

| Parameter | Recommended | Reasoning |
|-----------|-------------|-----------|
| ε (clip) | 0.2 | Standard, very robust |
| GAE λ | 0.95 | Balances bias-variance |
| Epochs | 3-10 | More for large batches |
| Minibatch size | 64 | For large batch training |
| Entropy coefficient | 0.01 | Lower for deterministic tasks |
| Value coefficient | 0.5 | Standard |

### SAC Specific

| Parameter | Recommended | Reasoning |
|-----------|-------------|-----------|
| τ (soft update) | 0.005 | Slow target updates |
| Initial α (entropy) | 0.2 | Auto-tuned in practice |
| Target entropy | -dim(action) | Heuristic |
| Updates per step | 1 | Can use 2-4 for sample efficiency |

### Exploration Parameters

| Method | Parameter | Recommended | Notes |
|--------|-----------|-------------|-------|
| ε-greedy (DQN) | ε_start | 1.0 | Full exploration initially |
| | ε_end | 0.01 | Minimum exploration |
| | ε_decay | 0.995 | Per episode |
| Gaussian noise (DDPG/TD3) | σ (std) | 0.1 - 0.2 | As fraction of action range |
| Entropy (SAC) | α | Auto-tuned | Let it learn |

---

## Key Formulas

### Value Functions

**State Value**
```
V^π(s) = E_π[R_t | S_t = s] = E_π[Σ_{k=0}^∞ γ^k r_{t+k} | S_t = s]
```

**Action Value (Q-function)**
```
Q^π(s,a) = E_π[R_t | S_t = s, A_t = a] = E_π[Σ_{k=0}^∞ γ^k r_{t+k} | S_t = s, A_t = a]
```

**Advantage**
```
A^π(s,a) = Q^π(s,a) - V^π(s)
```

### Bellman Equations

**Bellman Expectation (Value)**
```
V^π(s) = Σ_a π(a|s) [r(s,a) + γ Σ_{s'} p(s'|s,a) V^π(s')]
```

**Bellman Optimality (Value)**
```
V*(s) = max_a [r(s,a) + γ Σ_{s'} p(s'|s,a) V*(s')]
```

**Bellman Optimality (Q-function)**
```
Q*(s,a) = r(s,a) + γ Σ_{s'} p(s'|s,a) max_{a'} Q*(s',a')
```

### Policy Gradient

**REINFORCE**
```
∇_θ J(θ) = E_τ~π_θ [Σ_t ∇_θ log π_θ(a_t|s_t) R_t]
```

**With Baseline**
```
∇_θ J(θ) = E_τ~π_θ [Σ_t ∇_θ log π_θ(a_t|s_t) (R_t - b(s_t))]
```

**Actor-Critic**
```
∇_θ J(θ) = E_τ~π_θ [Σ_t ∇_θ log π_θ(a_t|s_t) A^π(s_t, a_t)]
```

### TD Learning

**TD Error**
```
δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

**Q-Learning Update**
```
Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s',a') - Q(s,a)]
```

**SARSA Update**
```
Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
```

### GAE (Generalized Advantage Estimation)

```
A_t^GAE = Σ_{l=0}^∞ (γλ)^l δ_{t+l}

where δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

### PPO

**Clipped Objective**
```
L^CLIP(θ) = E_t[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]

where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
```

### SAC (Maximum Entropy)

**Objective**
```
J(π) = E_τ~π[Σ_t r(s_t, a_t) + α H(π(·|s_t))]

where H(π(·|s)) = -Σ_a π(a|s) log π(a|s)
```

**Soft Bellman Backup**
```
Q*(s,a) = r(s,a) + γ E_{s'}[V*(s')]
V*(s) = E_{a~π}[Q*(s,a) - α log π(a|s)]
```

---

## Common Bugs & Fixes

### Bug 1: Training Doesn't Start / Stuck at Random Policy

**Symptoms:**
- Reward stays at random baseline
- Policy doesn't improve after many episodes

**Possible Causes:**
1. Learning rate too low
2. Rewards not normalized
3. Gradient vanishing
4. Network architecture too complex

**Fixes:**
```python
# Fix 1: Increase learning rate
optimizer = Adam(params, lr=1e-3)  # Try 10x higher

# Fix 2: Normalize rewards
rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

# Fix 3: Check gradient norms
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
print(f"Grad norm: {grad_norm}")  # Should be 0.1-10

# Fix 4: Simplify network
# Use 2 layers with 64-128 hidden units instead of deeper
```

### Bug 2: Training Collapses / Performance Drops

**Symptoms:**
- Was learning, suddenly drops to random
- Can't recover

**Possible Causes:**
1. Learning rate too high
2. No gradient clipping
3. PPO: too many epochs or no early stopping
4. Off-policy: target network not updating

**Fixes:**
```python
# Fix 1: Lower learning rate or use scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

# Fix 2: Clip gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

# Fix 3: PPO early stopping on KL divergence
if kl > 1.5 * target_kl:
    break  # Stop epoch early

# Fix 4: Check target network updates
if step % target_update_freq == 0:
    target_net.load_state_dict(policy_net.state_dict())
```

### Bug 3: High Variance / Unstable Learning

**Symptoms:**
- Reward has huge swings
- Can't tell if learning

**Possible Causes:**
1. No advantage normalization
2. Batch size too small
3. No entropy regularization
4. γ too high for sparse rewards

**Fixes:**
```python
# Fix 1: Always normalize advantages
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

# Fix 2: Increase batch size
batch_size = 256  # Instead of 32

# Fix 3: Add entropy bonus
loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

# Fix 4: Lower gamma for sparse rewards
gamma = 0.95  # Instead of 0.99
```

### Bug 4: Off-Policy Methods Not Learning

**Symptoms:**
- DQN/SAC/TD3 stuck at random

**Possible Causes:**
1. Replay buffer not full enough
2. Updating too early
3. Target network issues
4. Action scaling wrong

**Fixes:**
```python
# Fix 1: Warmup period
if len(replay_buffer) < 10000:
    continue  # Don't train yet

# Fix 2: Update frequency
if step % 4 == 0:  # Update every 4 steps, not every step
    agent.update()

# Fix 3: Soft update target networks
for target_param, param in zip(target_net.parameters(), net.parameters()):
    target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)

# Fix 4: Check action range
# Network outputs in [-1, 1], scale to actual range
action = network(state) * max_action
```

### Bug 5: Works in Training, Fails in Eval

**Symptoms:**
- Good reward during training
- Terrible at test time

**Possible Causes:**
1. Using training mode at test time
2. Exploration noise at test time
3. Overfitting to specific states

**Fixes:**
```python
# Fix 1: Set eval mode
model.eval()  # Disables dropout, batch norm

# Fix 2: No noise during evaluation
if training:
    action = policy(state) + noise
else:
    action = policy(state)  # Deterministic

# Fix 3: More diverse training data
# Increase exploration, vary initial states
```

---

## Implementation Checklist

### Before Training

- [ ] **Environment wrapper**: Normalized observations? Normalized rewards?
- [ ] **Action space**: Continuous in [-1, 1]? Discrete properly handled?
- [ ] **Network architecture**: Not too deep (2-3 layers usually enough)
- [ ] **Initialization**: Orthogonal or Xavier initialization
- [ ] **Optimizer**: Adam with lr=3e-4 is good default
- [ ] **Gradient clipping**: Max norm = 0.5-10
- [ ] **Hyperparameters**: Copied from paper or baseline?

### During Training

- [ ] **Logging**: Track episode reward, loss, entropy, grad norms
- [ ] **Visualization**: Plot rewards every 10-50 episodes
- [ ] **Checkpointing**: Save model every N episodes
- [ ] **Early stopping**: If training diverges, stop and debug
- [ ] **Ablation**: Test each component separately

### Debugging

- [ ] **Sanity checks**: Can it learn a trivial task? (e.g., CartPole)
- [ ] **Gradient check**: Are gradients flowing? Not exploding?
- [ ] **Overfit single episode**: Can it memorize one trajectory?
- [ ] **Hyperparameter sweep**: Try 3-5 learning rates
- [ ] **Compare to baseline**: Does it match published results?

### Code Quality

- [ ] **Vectorized operations**: Use batch processing, not loops
- [ ] **Reproducibility**: Set random seeds
- [ ] **Documentation**: Comment complex sections
- [ ] **Tests**: Unit tests for network forward pass, advantage computation
- [ ] **Profiling**: Is training time reasonable? Any bottlenecks?

---

## Quick Troubleshooting

**My algorithm is slower than paper says:**
- Check if you're using GPU (`model.to('cuda')`)
- Vectorize environment (multiple parallel envs)
- Profile code to find bottlenecks
- Use mixed precision training

**My results don't match the paper:**
- Verify exact same environment version
- Check if paper uses pre-processing (frame stacking, normalization)
- Try their exact hyperparameters
- Check if paper does hyperparameter search (published = best of many runs)

**It works on CartPole but not my task:**
- Your task might need more capacity (bigger network)
- Or domain-specific tricks (frame stacking for vision)
- Or longer training (Atari = 10M steps, not 1M)
- Or different algorithm (continuous control needs SAC/TD3, not DQN)

**Training takes forever:**
- Use vectorized environments (10-100x speedup)
- Use GPU if available
- Try more sample-efficient algorithm (off-policy > on-policy)
- Consider distributed training (Ray RLlib, Stable-Baselines3)

---

## Recommended Baselines

Don't implement from scratch unless learning! Use these:

**Stable-Baselines3** (Python/PyTorch)
```bash
pip install stable-baselines3
```
- Best for: Getting results quickly
- Algorithms: PPO, A2C, DQN, SAC, TD3
- Quality: Production-ready, well-tested

**CleanRL** (Python/PyTorch)
```bash
pip install cleanrl
```
- Best for: Learning implementations
- Algorithms: Most major algorithms
- Quality: Single-file implementations, very readable

**Ray RLlib** (Python/PyTorch)
```bash
pip install ray[rllib]
```
- Best for: Distributed training, scaling up
- Algorithms: Everything
- Quality: Production-ready, used in industry

**Spinning Up** (Python/PyTorch+TF)
- Best for: Educational implementations
- Algorithms: VPG, PPO, TRPO, DDPG, TD3, SAC
- Quality: Clean, well-documented

---

## Further Learning

### Must-Read Papers (in order)

1. **DQN (2015)**: Nature paper that started deep RL
2. **A3C (2016)**: Parallel actor-critic
3. **TRPO (2015)**: Trust regions
4. **PPO (2017)**: Practical policy gradients
5. **TD3 (2018)**: Addressing function approximation errors
6. **SAC (2018)**: Maximum entropy RL

### Best Resources

**Courses:**
- CS285 (Berkeley): deeplearningg.berkeley.edu/cs285
- Spinning Up: spinningup.openai.com

**Books:**
- Sutton & Barto: "Reinforcement Learning: An Introduction" (Free online)
- Sergey Levine's lecture notes (CS285)

**Blogs:**
- Lilian Weng: lilianweng.github.io
- OpenAI Spinning Up: spinningup.openai.com/en/latest/

**Communities:**
- r/reinforcementlearning
- RL Discord servers
- Papers With Code (RL section)

---

## Quick Reference Card (Print This!)

```
┌─────────────────────────────────────────────────────────┐
│                    DEEP RL CHEAT SHEET                  │
├─────────────────────────────────────────────────────────┤
│ ALGORITHM SELECTION                                     │
│                                                         │
│ Discrete actions:                                       │
│   • Sample efficient? → DQN                            │
│   • General purpose? → PPO                             │
│                                                         │
│ Continuous actions:                                     │
│   • Best performance? → SAC                            │
│   • General purpose? → PPO                             │
│   • Need deterministic? → TD3                          │
│                                                         │
│ Limited real data? → Model-Based (MBPO)                │
│ LLM training? → PPO                                     │
├─────────────────────────────────────────────────────────┤
│ HYPERPARAMETERS (defaults)                              │
│                                                         │
│ Learning rate:    3e-4                                  │
│ Batch size:       256 (off-policy), 64 (on-policy)    │
│ Gamma (γ):        0.99                                 │
│ Replay buffer:    1M                                    │
│ PPO clip (ε):     0.2                                  │
│ GAE lambda (λ):   0.95                                 │
│ Gradient clip:    0.5-10                               │
├─────────────────────────────────────────────────────────┤
│ DEBUGGING CHECKLIST                                     │
│                                                         │
│ ☐ Normalize observations & rewards                     │
│ ☐ Clip gradients                                       │
│ ☐ Normalize advantages                                 │
│ ☐ Add entropy bonus (0.01)                            │
│ ☐ Use orthogonal init                                  │
│ ☐ Check action scaling                                 │
│ ☐ Replay buffer warmup (off-policy)                   │
│ ☐ Test on CartPole first                              │
├─────────────────────────────────────────────────────────┤
│ COMMON FORMULAS                                         │
│                                                         │
│ Advantage: A(s,a) = Q(s,a) - V(s)                     │
│ TD error: δ = r + γV(s') - V(s)                       │
│ PPO ratio: r = π_new(a|s) / π_old(a|s)               │
│ Clipped obj: min(r·A, clip(r,1-ε,1+ε)·A)            │
└─────────────────────────────────────────────────────────┘
```

---

**This cheat sheet is part of the Deep RL Lecture Series.**
**See `/self-study-lectures/lectures/` for full tutorials.**

**Good luck with your RL journey! 🚀**
