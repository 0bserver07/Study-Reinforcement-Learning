# Lecture 05: Trust Region Policy Optimization (TRPO)

**Duration:** ~90 minutes
**Prerequisites:** Lecture 04 (Actor-Critic), basic calculus
**Goal:** Understand why policy updates fail, learn TRPO's solution, see why PPO was needed

---

## Why This Matters

In Lecture 04 (A2C), we updated policies with gradient descent:
```python
θ_new = θ_old + α * ∇J(θ)
```

**Problem:** One bad update can **destroy** your policy.

### The Catastrophic Failure Story

I was training a robot arm to grasp objects. After 1000 episodes of learning, one bad gradient step made it:
- Forget everything
- Start flailing randomly
- Never recover

**This happens all the time in policy gradient methods.**

TRPO (Trust Region Policy Optimization) solves this by asking:
> "What's the largest policy update I can make safely?"

This lecture is about understanding that question and TRPO's answer. Then in Lecture 06 (PPO), we'll learn a simpler approximation that everyone actually uses.

---

## Part 1: The Problem with Policy Gradients

### Small Parameter Change = Big Performance Change

Policies are fragile:

```python
# Policy before: π_old(a|s)
# After small gradient step: π_new(a|s)

# In some states, π_new might be completely different!
# Example: π_old(action_1|state) = 0.51, π_new(action_1|state) = 0.49
# Action flips from likely to unlikely!
```

### The Root Cause

Policy gradient uses first-order approximation:
```
J(θ_new) ≈ J(θ_old) + ∇J(θ_old)^T (θ_new - θ_old)
```

This is only valid for **tiny** changes in θ. But we don't know how tiny!

### Concrete Example: CartPole

```python
# State: Cart at position 0.1, velocity 0.5, pole angle 10°

# Old policy: π_old(left|s) = 0.7, π_old(right|s) = 0.3
# Policy says: strongly prefer left

# After one gradient update with learning rate 0.01:
# New policy: π_new(left|s) = 0.2, π_new(right|s) = 0.8
# Policy flipped to strongly prefer right!

# Performance collapsed: from 200 steps to 20 steps
```

**We need to constrain how much the policy can change.**

---

## Part 2: Trust Regions

### The Idea

Instead of:
```
maximize J(θ)
```

Do:
```
maximize J(θ_new)
subject to: KL(π_old || π_new) ≤ δ
```

Where KL is **KL divergence**, measuring how different the policies are.

### What is KL Divergence?

For discrete actions:
```
KL(π_old || π_new) = Σ_a π_old(a|s) log(π_old(a|s) / π_new(a|s))
```

**Intuition:** How surprised would π_old be to see actions from π_new?

- KL = 0: Policies are identical
- KL > 0: Policies are different
- KL = ∞: Policies assign probability 0 vs >0 to some action

### Example: CartPole

```python
# Old policy:
π_old(left) = 0.7, π_old(right) = 0.3

# New policy 1 (small change):
π_new1(left) = 0.65, π_new1(right) = 0.35
KL = 0.7*log(0.7/0.65) + 0.3*log(0.3/0.35) = 0.0043

# New policy 2 (big change):
π_new2(left) = 0.2, π_new2(right) = 0.8
KL = 0.7*log(0.7/0.2) + 0.3*log(0.3/0.8) = 0.892

# TRPO would allow new policy 1, reject new policy 2
```

---

## Part 3: TRPO Algorithm

### The Optimization Problem

```
maximize_θ: E_s[E_a~π_new[A(s,a)]]
subject to: E_s[KL(π_old(·|s) || π_new(·|s))] ≤ δ
```

Where:
- A(s,a) is advantage function
- δ is trust region size (typically 0.01)
- Expectation over states in trajectory

### Solving the Constrained Optimization

This is hard! TRPO uses:
1. **Linear approximation** of objective
2. **Quadratic approximation** of constraint
3. **Conjugate gradient** to find search direction
4. **Line search** to satisfy constraint

### Step-by-Step TRPO

**Step 1: Collect trajectories** using current policy π_old

**Step 2: Compute advantages** A(s,a) using GAE (from Lecture 04)

**Step 3: Approximate objective**
```
L(θ) = E[π_new(a|s)/π_old(a|s) * A(s,a)]

Gradient: g = ∇_θ L(θ) |_θ=θ_old
```

**Step 4: Approximate KL constraint**
```
KL ≈ (θ - θ_old)^T H (θ - θ_old) / 2

Where H = Hessian of KL divergence (Fisher information matrix)
```

**Step 5: Solve for search direction**
```
Find direction s that maximizes: g^T s
Subject to: s^T H s ≤ δ

Solution (Lagrangian): s = √(2δ / (g^T H^{-1} g)) * H^{-1} g
```

**Step 6: Line search**
```
Try: θ_new = θ_old + α^j s  for j=0,1,2,...
Accept first θ_new that:
  1. Improves objective: L(θ_new) > L(θ_old)
  2. Satisfies constraint: KL(θ_new, θ_old) ≤ δ
```

---

## Part 4: Implementation (Simplified TRPO)

This is complex, so I'll show the key parts:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TRPOAgent:
    """
    Trust Region Policy Optimization agent.

    Simplified implementation focusing on key concepts.
    """

    def __init__(
        self,
        policy,
        value_net,
        max_kl=0.01,
        damping=0.1,
        value_lr=1e-3
    ):
        self.policy = policy
        self.value_net = value_net
        self.max_kl = max_kl  # Trust region size
        self.damping = damping

        # Value function optimizer (policy updated via TRPO)
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=value_lr
        )

    def select_action(self, state):
        """Sample action from current policy."""
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits = self.policy(state_tensor)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

        return action.item()

    def compute_advantages(self, states, rewards, dones, gamma=0.99, lam=0.95):
        """Compute GAE advantages."""
        values = self.value_net(states).squeeze().detach().numpy()

        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t+1]

            if dones[t]:
                next_value = 0

            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + torch.tensor(values, dtype=torch.float32)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def get_log_probs(self, states, actions):
        """Get log probabilities of actions under current policy."""
        logits = self.policy(states)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        return log_probs

    def compute_policy_loss(self, states, actions, advantages, old_log_probs):
        """
        Compute surrogate loss: L(θ) = E[π_new/π_old * A]
        """
        new_log_probs = self.get_log_probs(states, actions)

        # Importance sampling ratio: π_new(a|s) / π_old(a|s)
        ratio = torch.exp(new_log_probs - old_log_probs)

        # Surrogate loss
        loss = (ratio * advantages).mean()

        return loss

    def compute_kl(self, states, old_logits):
        """
        Compute KL divergence between old and new policies.
        KL(π_old || π_new) = Σ π_old(a) log(π_old(a) / π_new(a))
        """
        new_logits = self.policy(states)

        old_probs = F.softmax(old_logits, dim=-1)
        new_probs = F.softmax(new_logits, dim=-1)

        # KL divergence
        kl = (old_probs * (torch.log(old_probs) - torch.log(new_probs))).sum(dim=-1).mean()

        return kl

    def fisher_vector_product(self, states, old_logits, v):
        """
        Compute Fisher-vector product: H * v
        where H is the Hessian of KL divergence (Fisher information matrix)

        Uses automatic differentiation instead of computing H explicitly.
        """
        kl = self.compute_kl(states, old_logits)

        # Gradient of KL
        grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
        flat_grad = torch.cat([grad.view(-1) for grad in grads])

        # Gradient-vector product
        gv_product = (flat_grad * v).sum()

        # Hessian-vector product (second derivative)
        hvp = torch.autograd.grad(gv_product, self.policy.parameters())
        hvp_flat = torch.cat([grad.contiguous().view(-1) for grad in hvp])

        # Add damping for numerical stability
        return hvp_flat + self.damping * v

    def conjugate_gradient(self, states, old_logits, b, n_steps=10):
        """
        Solve H * x = b using conjugate gradient.

        Returns x ≈ H^{-1} * b
        """
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rs_old = (r * r).sum()

        for _ in range(n_steps):
            Ap = self.fisher_vector_product(states, old_logits, p)
            alpha = rs_old / ((p * Ap).sum() + 1e-8)
            x += alpha * p
            r -= alpha * Ap
            rs_new = (r * r).sum()

            if rs_new < 1e-10:
                break

            p = r + (rs_new / rs_old) * p
            rs_old = rs_new

        return x

    def line_search(
        self,
        states,
        actions,
        advantages,
        old_log_probs,
        old_logits,
        fullstep,
        expected_improve
    ):
        """
        Backtracking line search to find step size that satisfies:
        1. Improves objective
        2. Satisfies KL constraint
        """
        old_params = self.get_flat_params()
        old_loss = self.compute_policy_loss(states, actions, advantages, old_log_probs)

        # Backtracking coefficients
        alphas = [0.5**i for i in range(10)]

        for alpha in alphas:
            # Try new parameters
            new_params = old_params + alpha * fullstep
            self.set_flat_params(new_params)

            # Check improvement
            new_loss = self.compute_policy_loss(states, actions, advantages, old_log_probs)
            actual_improve = new_loss - old_loss

            # Check KL constraint
            kl = self.compute_kl(states, old_logits)

            # Accept step if improves and satisfies constraint
            if actual_improve > 0 and kl <= self.max_kl:
                return True

        # No acceptable step found, revert to old parameters
        self.set_flat_params(old_params)
        return False

    def get_flat_params(self):
        """Get flattened policy parameters."""
        return torch.cat([param.view(-1) for param in self.policy.parameters()])

    def set_flat_params(self, flat_params):
        """Set policy parameters from flattened vector."""
        offset = 0
        for param in self.policy.parameters():
            param_size = param.numel()
            param.data.copy_(flat_params[offset:offset+param_size].view_as(param))
            offset += param_size

    def update(self, states, actions, rewards, dones):
        """
        TRPO update.

        Steps:
        1. Compute advantages
        2. Compute policy gradient
        3. Compute search direction using conjugate gradient
        4. Line search for step size
        5. Update value function
        """
        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = rewards

        # Compute advantages and returns
        advantages, returns = self.compute_advantages(states, rewards, dones)

        # Get old policy log probs and logits (before update)
        with torch.no_grad():
            old_log_probs = self.get_log_probs(states, actions)
            old_logits = self.policy(states)

        # Compute policy loss
        loss = self.compute_policy_loss(states, actions, advantages, old_log_probs)

        # Compute policy gradient
        grads = torch.autograd.grad(loss, self.policy.parameters())
        policy_grad = torch.cat([grad.view(-1) for grad in grads])

        # Compute search direction using conjugate gradient
        # Solve: H * s = g (where H is Fisher information matrix)
        search_direction = self.conjugate_gradient(states, old_logits, policy_grad)

        # Compute maximum step size
        # s^T H s ≤ 2δ → α_max = √(2δ / (s^T H s))
        sHs = (search_direction * self.fisher_vector_product(
            states, old_logits, search_direction
        )).sum()
        max_step_size = torch.sqrt(2 * self.max_kl / (sHs + 1e-8))
        fullstep = max_step_size * search_direction

        # Line search
        expected_improve = (policy_grad * fullstep).sum()
        success = self.line_search(
            states, actions, advantages, old_log_probs,
            old_logits, fullstep, expected_improve
        )

        # Update value function (standard supervised learning)
        for _ in range(5):  # Multiple epochs
            value_loss = F.mse_loss(self.value_net(states).squeeze(), returns)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        return {
            'policy_loss': loss.item(),
            'value_loss': value_loss.item(),
            'kl': self.compute_kl(states, old_logits).item(),
            'line_search_success': success
        }


def train_trpo(env_name='CartPole-v1', n_episodes=500):
    """Train TRPO agent."""
    import gymnasium as gym

    env = gym.make(env_name)

    # Policy and value networks
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = nn.Sequential(
        nn.Linear(state_dim, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, action_dim)
    )

    value_net = nn.Sequential(
        nn.Linear(state_dim, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1)
    )

    agent = TRPOAgent(policy, value_net, max_kl=0.01)

    rewards_per_episode = []

    for episode in range(n_episodes):
        # Collect trajectory
        states, actions, rewards, dones = [], [], [], []

        state, _ = env.reset()
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            state = next_state

        # Update agent
        metrics = agent.update(states, actions, rewards, dones)

        # Logging
        episode_reward = sum(rewards)
        rewards_per_episode.append(episode_reward)

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_per_episode[-50:])
            print(f"Episode {episode+1}: Avg Reward = {avg_reward:.2f}")
            print(f"  KL = {metrics['kl']:.4f}, "
                  f"Line Search Success = {metrics['line_search_success']}")

    return agent, rewards_per_episode
```

---

## Part 5: Why TRPO is Complicated

### Computational Cost

1. **Conjugate gradient:** Requires multiple Hessian-vector products (~10-20)
2. **Line search:** May need multiple forward passes (~10)
3. **Per update:** ~30x more computation than vanilla policy gradient

### Implementation Difficulty

1. Fisher information matrix computation (needs second derivatives)
2. Conjugate gradient solver
3. Line search with KL checking
4. Numerical stability issues

**This is why TRPO is rarely used in practice.**

---

## Part 6: TRPO's Contributions

Despite complexity, TRPO was hugely influential:

### 1. Monotonic Improvement Guarantee

TRPO (approximately) guarantees:
```
J(π_new) ≥ J(π_old)
```

Policy never gets worse! (In practice, line search ensures this)

### 2. Safe Exploration

Robot learning: One bad policy can damage hardware. TRPO prevents catastrophic failures.

### 3. Inspired PPO

PPO (Lecture 06) achieves similar results with **much simpler** implementation:
- No conjugate gradient
- No line search
- No second derivatives
- Just clipping!

---

## Part 7: Gotchas from Real Implementation

### Gotcha 1: Fisher Information Matrix

Computing Fisher exactly is expensive. Use **empirical Fisher**:
```python
# True Fisher: E_s,a[∇ log π(a|s) ∇ log π(a|s)^T]
# Empirical Fisher: Average over sampled trajectories
```

### Gotcha 2: Conjugate Gradient Steps

Too few steps: Poor solution
Too many steps: Numerical instability

Good default: 10-20 steps

### Gotcha 3: Damping

Fisher matrix can be singular. Add damping:
```python
H_damped = H + λI  # λ = 0.1 typical
```

### Gotcha 4: Line Search Failures

If line search fails (no step satisfies constraint), keep old policy:
```python
if not success:
    params = old_params  # Revert
```

Happens ~5-10% of updates.

### Gotcha 5: Computational Cost

TRPO is **slow**. On Atari:
- DQN: ~1 day
- A3C: ~4 hours
- TRPO: ~3 days

Use PPO instead!

---

## Part 8: When to Use TRPO

**Use TRPO when:**
1. Safety-critical applications (robotics)
2. Need monotonic improvement guarantees
3. Willing to pay computational cost
4. Research purposes (understanding trust regions)

**Don't use TRPO when:**
1. Want fast training (use PPO)
2. Need simple implementation (use PPO)
3. Limited compute (use PPO)
4. **Just use PPO** (seriously, everyone does)

---

## Summary

1. **Policy gradients can fail catastrophically** with one bad update
2. **Trust regions constrain updates** using KL divergence
3. **TRPO solves constrained optimization** using conjugate gradient and line search
4. **TRPO guarantees improvement** (approximately)
5. **TRPO is too complicated** for practical use
6. **PPO simplifies TRPO** with clipping (next lecture!)

---

## What's Next?

**Next lecture (06):** PPO - Proximal Policy Optimization

**The connection:** PPO asks: "Can we get TRPO's benefits with a simpler implementation?" Answer: **Yes!** PPO is now the most popular deep RL algorithm.

---

## Paper Trail

1. **TRPO (2015):** "Trust Region Policy Optimization" - Schulman et al., UC Berkeley
2. **Natural Policy Gradient (2002):** "A Natural Policy Gradient" - Kakade (theoretical foundation)
3. **PPO (2017):** "Proximal Policy Optimization" - Schulman et al., OpenAI (simplifies TRPO)

All in `/Modern-RL-Research/RLHF-and-Alignment/PAPERS.md`

---

## Exercise for Yourself

**Don't implement TRPO from scratch** (it's painful).

Instead:
1. Read the TRPO paper to understand trust regions
2. Compare to PPO (next lecture)
3. Appreciate why everyone uses PPO

**The key takeaway:** TRPO taught us *why* we need constrained updates. PPO taught us *how* to do it simply.

---

## Personal Note

I spent a week implementing TRPO for my PhD. It worked, but was fragile and slow. Then I discovered PPO and rewrote everything in a day. PPO matched TRPO's performance with 1/10th the code.

**Learn TRPO for understanding, use PPO for everything else.**
