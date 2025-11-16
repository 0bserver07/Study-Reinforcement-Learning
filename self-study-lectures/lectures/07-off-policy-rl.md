# Lecture 07: Off-Policy Learning - SAC and TD3

**Duration:** ~90 minutes
**Prerequisites:** Lecture 03 (DQN), Lecture 04 (Actor-Critic), Lecture 06 (PPO)
**Goal:** Learn off-policy methods for continuous control, implement SAC and TD3

---

## Why This Matters

All the methods we've learned so far have a problem:

- **REINFORCE, A2C, PPO:** On-policy (must throw away data after one update)
- **DQN:** Off-policy but only works for discrete actions

**What if we need:**
- Continuous actions (robot joint angles, steering wheel, throttle)
- Sample efficiency (robotics is expensive!)
- Off-policy learning (reuse data)

**This lecture is about methods that do all three: SAC and TD3.**

When Boston Dynamics trains their robots, when self-driving cars learn to drive, when robotic arms learn manipulation – they use these methods.

---

## Part 1: On-Policy vs Off-Policy

### On-Policy (PPO, A2C)

```python
# Collect data with current policy π_θ
for episode in range(n_episodes):
    trajectory = collect_data(policy=π_θ)
    update_policy(π_θ, trajectory)
    # MUST throw away trajectory, can never reuse!
```

**Limitation:** Every policy update requires fresh data. Extremely sample inefficient.

### Off-Policy (DQN, SAC, TD3)

```python
# Collect data with ANY policy (even random!)
replay_buffer = ReplayBuffer()

for episode in range(n_episodes):
    # Explore with behavior policy (e.g., π_θ + noise)
    trajectory = collect_data(policy=π_θ + noise)
    replay_buffer.add(trajectory)

    # Learn from ALL past data
    batch = replay_buffer.sample()
    update_policy(π_θ, batch)  # Can reuse old data!
```

**Advantage:** Reuse data many times. ~10-100x more sample efficient!

### Why PPO Can't Do This

PPO uses policy gradient:
```
∇J(θ) = E_{s,a~π_θ}[∇ log π_θ(a|s) * A(s,a)]
```

Expectation is **under current policy π_θ**. If data comes from old policy, this is wrong.

---

## Part 2: The Continuous Action Problem

### Why DQN Doesn't Work

DQN picks best action:
```python
Q_values = model(state)  # [Q(s,a1), Q(s,a2), ..., Q(s,an)]
action = argmax(Q_values)  # Pick best discrete action
```

**Problem:** If actions are continuous (e.g., joint angle in [0°, 360°]), we'd need infinite Q-values!

### The Solution: Policy Gradient for Continuous Actions

Instead of Q(s, a_1), Q(s, a_2), ..., learn:
1. **Actor:** Policy μ(s) that directly outputs action
2. **Critic:** Q(s, a) that evaluates any action

---

## Part 3: DDPG (Foundation)

DDPG (Deep Deterministic Policy Gradient, 2015) combines:
- Deterministic policy: a = μ(s)
- Q-learning: Q(s, a)
- Actor-critic architecture

### DDPG Architecture

```python
import torch
import torch.nn as nn

class Actor(nn.Module):
    """
    Deterministic policy: a = μ(s)
    Outputs continuous action directly.
    """
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
        self.max_action = max_action

    def forward(self, state):
        # Scale to action range
        return self.max_action * self.net(state)


class Critic(nn.Module):
    """
    Q-function: Q(s, a)
    Takes state AND action as input.
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Concatenate state and action
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Output single Q-value
        )

    def forward(self, state, action):
        # Concatenate and evaluate
        sa = torch.cat([state, action], dim=1)
        return self.net(sa)
```

### DDPG Algorithm

```python
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        # Actor and critic networks
        self.actor = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim, action_dim)

        # Target networks (like DQN)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.critic_target = Critic(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        # Replay buffer
        self.replay_buffer = ReplayBuffer()

    def select_action(self, state, noise=0.1):
        """Select action with exploration noise."""
        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]

        # Add Gaussian noise for exploration
        action += noise * np.random.randn(len(action))
        return action.clip(-self.max_action, self.max_action)

    def update(self, batch_size=256):
        """Update actor and critic."""
        # Sample from replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        # Update Critic
        with torch.no_grad():
            # Compute target Q-value
            next_action = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + (1 - done) * 0.99 * target_Q

        # Current Q-value
        current_Q = self.critic(state, action)

        # Critic loss (MSE)
        critic_loss = F.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        # Maximize Q(s, μ(s)) = minimize -Q(s, μ(s))
        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks (soft update)
        self.soft_update(self.actor, self.actor_target, tau=0.005)
        self.soft_update(self.critic, self.critic_target, tau=0.005)

    def soft_update(self, source, target, tau):
        """Soft update: θ_target = τ*θ_source + (1-τ)*θ_target"""
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                tau * source_param.data + (1.0 - tau) * target_param.data
            )
```

### DDPG Problems

DDPG is unstable and sensitive to hyperparameters:
1. **Overestimation:** Critic overestimates Q-values
2. **High variance:** Updates are noisy
3. **Hyperparameter sensitivity:** Small changes break training

**TD3 and SAC solve these problems.**

---

## Part 4: TD3 (Twin Delayed DDPG)

TD3 (2018) fixes DDPG with three tricks:

### Trick 1: Twin Critics (Clipped Double Q-Learning)

**Problem:** Single critic overestimates Q-values.

**Solution:** Use two critics, take minimum:

```python
class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)

        # TWO critics instead of one
        self.critic_1 = Critic(state_dim, action_dim)
        self.critic_2 = Critic(state_dim, action_dim)

        # TWO target critics
        self.critic_1_target = Critic(state_dim, action_dim)
        self.critic_2_target = Critic(state_dim, action_dim)

        # Copy weights
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

    def compute_target_q(self, next_state, reward, done):
        """Compute target Q using minimum of two critics."""
        with torch.no_grad():
            next_action = self.actor_target(next_state)

            # Get Q-values from both target critics
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)

            # Take minimum (reduces overestimation)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * 0.99 * target_Q

        return target_Q
```

### Trick 2: Delayed Policy Updates

**Problem:** Updating actor every step causes instability.

**Solution:** Update actor less frequently than critic:

```python
def update(self, batch_size=256, policy_delay=2):
    # Sample batch
    state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

    # ALWAYS update critics
    target_Q = self.compute_target_q(next_state, reward, done)

    # Update critic 1
    current_Q1 = self.critic_1(state, action)
    critic_1_loss = F.mse_loss(current_Q1, target_Q)
    self.critic_1_optimizer.zero_grad()
    critic_1_loss.backward()
    self.critic_1_optimizer.step()

    # Update critic 2
    current_Q2 = self.critic_2(state, action)
    critic_2_loss = F.mse_loss(current_Q2, target_Q)
    self.critic_2_optimizer.zero_grad()
    critic_2_loss.backward()
    self.critic_2_optimizer.step()

    # DELAYED actor update (only every policy_delay steps)
    if self.total_steps % policy_delay == 0:
        actor_loss = -self.critic_1(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self.soft_update_all()

    self.total_steps += 1
```

### Trick 3: Target Policy Smoothing

**Problem:** Deterministic policy can exploit Q-function errors.

**Solution:** Add noise to target actions:

```python
def compute_target_q(self, next_state, reward, done, noise_clip=0.5, policy_noise=0.2):
    with torch.no_grad():
        # Target action with smoothing noise
        next_action = self.actor_target(next_state)

        # Add clipped noise
        noise = torch.randn_like(next_action) * policy_noise
        noise = noise.clamp(-noise_clip, noise_clip)
        next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

        # Compute target Q
        target_Q1 = self.critic_1_target(next_state, next_action)
        target_Q2 = self.critic_2_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + (1 - done) * 0.99 * target_Q

    return target_Q
```

### Complete TD3 Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TD3:
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3)
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2
    ):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = Critic(state_dim, action_dim).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=3e-4)

        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = Critic(state_dim, action_dim).to(device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Compute the target Q value
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.discount * target_Q

        # Get current Q estimates
        current_Q1 = self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            actor_loss = -self.critic_1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
```

**TD3 is simple and works well!** Used heavily in robotics.

---

## Part 5: SAC (Soft Actor-Critic)

SAC (2018) takes a different approach: **Maximum Entropy RL**

### The Entropy Idea

Instead of just maximizing reward:
```
maximize: E[Σ r_t]
```

Also maximize entropy (randomness):
```
maximize: E[Σ r_t + α * H(π(·|s_t))]

where H(π) = -Σ π(a|s) log π(a|s)  # Entropy
```

**Intuition:** Prefer policies that:
1. Get high reward (obviously)
2. Are stochastic/random (explore automatically)

### Why This Helps

1. **Automatic exploration:** Entropy bonus encourages trying different actions
2. **Robustness:** Stochastic policies are more robust to perturbations
3. **Captures multiple solutions:** Can represent multimodal optimal policies

### SAC Architecture

Key difference from TD3: **Stochastic policy** instead of deterministic.

```python
class StochasticActor(nn.Module):
    """
    Stochastic policy: outputs mean and log_std, samples from Gaussian.
    """

    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # Output mean and log_std
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        features = self.net(state)

        mean = self.mean(features)
        log_std = self.log_std(features)
        log_std = torch.clamp(log_std, -20, 2)  # Limit range for stability

        return mean, log_std

    def sample(self, state):
        """Sample action from Gaussian policy."""
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Sample from Gaussian
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()  # Reparameterization trick

        # Apply tanh to bound actions
        action = torch.tanh(x) * self.max_action

        # Compute log probability (with tanh correction)
        log_prob = normal.log_prob(x)
        log_prob -= torch.log(self.max_action * (1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob


class SAC:
    """
    Soft Actor-Critic with automatic entropy tuning.
    """

    def __init__(self, state_dim, action_dim, max_action):
        self.actor = StochasticActor(state_dim, action_dim, max_action)

        # Twin critics (like TD3)
        self.critic_1 = Critic(state_dim, action_dim)
        self.critic_2 = Critic(state_dim, action_dim)
        self.critic_1_target = Critic(state_dim, action_dim)
        self.critic_2_target = Critic(state_dim, action_dim)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=3e-4)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=3e-4)

        # Entropy coefficient (learned automatically)
        self.target_entropy = -action_dim  # Heuristic: -dim(A)
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)

    def select_action(self, state, evaluate=False):
        """Select action (stochastic during training, mean during eval)."""
        state = torch.FloatTensor(state).unsqueeze(0)

        if evaluate:
            # Deterministic (mean) for evaluation
            with torch.no_grad():
                mean, _ = self.actor(state)
                action = torch.tanh(mean) * self.actor.max_action
            return action.cpu().numpy()[0]
        else:
            # Stochastic for exploration
            with torch.no_grad():
                action, _ = self.actor.sample(state)
            return action.cpu().numpy()[0]

    def train(self, replay_buffer, batch_size=256):
        # Sample batch
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        # Current alpha
        alpha = self.log_alpha.exp()

        # Update Critic
        with torch.no_grad():
            # Sample next action from current policy
            next_action, next_log_prob = self.actor.sample(next_state)

            # Compute target Q
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)

            # Soft Bellman backup: r + γ(Q - α*log π)
            target_Q = reward + (1 - done) * 0.99 * (target_Q - alpha * next_log_prob)

        # Current Q estimates
        current_Q1 = self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)

        # Critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Update Actor
        new_action, log_prob = self.actor.sample(state)
        Q1 = self.critic_1(state, new_action)
        Q2 = self.critic_2(state, new_action)
        Q = torch.min(Q1, Q2)

        # Actor loss: maximize Q - α*log π = minimize -(Q - α*log π)
        actor_loss = (alpha.detach() * log_prob - Q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update temperature α
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Soft update target networks
        for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
            target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)

        for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
            target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)
```

**SAC is the state-of-the-art for continuous control!**

---

## Part 6: TD3 vs SAC

| Feature | TD3 | SAC |
|---------|-----|-----|
| Policy | Deterministic | Stochastic |
| Exploration | Add noise to actions | Entropy maximization |
| Temperature α | Fixed (exploration noise) | Learned automatically |
| Complexity | Simpler | Slightly more complex |
| Performance | Excellent | Excellent (slightly better) |
| Use case | When you prefer deterministic | When you want automatic exploration |

**Both are great! SAC is more popular in research, TD3 is simpler.**

---

## Part 7: Gotchas from Real Implementation

### Gotcha 1: Hyperparameter Sensitivity

Off-policy methods are sensitive. Good defaults:
- Learning rate: 3e-4
- Batch size: 256
- Replay buffer: 1M
- γ: 0.99
- τ (soft update): 0.005

### Gotcha 2: Warmup Period

Fill replay buffer before training:
```python
for _ in range(10000):
    action = env.action_space.sample()  # Random
    next_state, reward, done, _ = env.step(action)
    replay_buffer.add(state, action, reward, next_state, done)
```

### Gotcha 3: Action Scaling

Environments have different action ranges:
```python
# MuJoCo: actions in [-1, 1]
# Custom env: actions in [0, 10]

# Always normalize to [-1, 1] for network, then scale back
normalized_action = actor(state)  # [-1, 1]
real_action = normalized_action * max_action
```

### Gotcha 4: Evaluation vs Training

```python
# Training: stochastic/noisy
action = agent.select_action(state, evaluate=False)

# Evaluation: deterministic
action = agent.select_action(state, evaluate=True)
```

### Gotcha 5: Updates per Step

Update multiple times per environment step:
```python
for _ in range(gradient_steps):  # gradient_steps = 1 or 2
    agent.train(replay_buffer)
```

---

## Part 8: When to Use What

**Use TD3 when:**
- Need simple, reliable off-policy method
- Deterministic policy is fine
- Don't want to tune entropy

**Use SAC when:**
- Want state-of-the-art performance
- Need stochastic policy
- Want automatic exploration tuning

**Use PPO when:**
- On-policy is okay
- Want simplicity
- Have lots of parallel environments

**Use DQN when:**
- Discrete actions only

---

## Summary

1. **Off-policy methods reuse data** (~10-100x more sample efficient)
2. **Continuous actions need actor-critic** (DQN doesn't work)
3. **TD3: Deterministic policy** with twin critics, delayed updates, target smoothing
4. **SAC: Stochastic policy** with entropy maximization
5. **Both use experience replay and target networks**
6. **SAC is state-of-the-art for continuous control**

---

## What's Next?

**Next lecture (08):** Model-Based RL - Learn a model of the environment

**The connection:** Model-free (TD3/SAC) is sample inefficient. Model-based learns environment dynamics to be even more efficient!

---

## Paper Trail

1. **DDPG (2016):** "Continuous Control with Deep Reinforcement Learning" - DeepMind
2. **TD3 (2018):** "Addressing Function Approximation Error in Actor-Critic Methods" - Fujimoto et al.
3. **SAC (2018):** "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning" - Haarnoja et al., UC Berkeley
4. **SAC with Auto-tuning (2019):** "Soft Actor-Critic Algorithms and Applications" - Haarnoja et al.

All in `/Modern-RL-Research/RLHF-and-Alignment/PAPERS.md`

---

## Exercise for Yourself

Implement TD3 and train on a MuJoCo task (e.g., HalfCheetah):

1. Start with the TD3 code above
2. Train for 1M steps
3. Compare to DDPG (remove tricks one by one)
4. See which tricks matter most

Then try SAC:
1. Implement stochastic policy
2. Add entropy maximization
3. Compare sample efficiency to TD3

**You'll see SAC reaches high performance faster!**
