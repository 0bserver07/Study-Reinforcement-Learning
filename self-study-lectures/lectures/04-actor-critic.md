# Lecture 04: Actor-Critic Methods

**Duration:** ~90 minutes
**Prerequisites:** Lecture 02 (Policy Gradients), Lecture 03 (Q-Learning)
**Goal:** Combine policy gradients with value functions, implement A2C/A3C from scratch

---

## Why This Matters

We've learned two approaches:
- **Lecture 02 (REINFORCE):** Policy gradient, high variance, needs full episodes
- **Lecture 03 (DQN):** Value-based, sample efficient, but only discrete actions

**Actor-Critic combines the best of both:**
- **Actor:** Policy network π(a|s; θ) (like REINFORCE)
- **Critic:** Value network V(s; φ) (like DQN)
- **Result:** Lower variance than REINFORCE, works for continuous actions

This architecture powers:
- A2C/A3C (OpenAI's baselines)
- PPO (Lecture 06)
- SAC (Lecture 07)
- Basically all modern deep RL

When OpenAI trained their Dota 2 bot, they used PPO (an actor-critic method). When DeepMind trained AlphaGo, they used policy gradients with value baselines (actor-critic). **This is the foundation of modern deep RL**.

---

## Part 1: The Problem with REINFORCE

Recall REINFORCE from Lecture 02:

```python
# Policy gradient
∇J(θ) = E[∇ log π(a|s) * R]

# Where R is return: R_t = r_t + γr_{t+1} + γ²r_{t+2} + ...
```

**Problem: High variance**. Even with same state/action, returns vary wildly based on random future events.

### Example: CartPole

```python
# Episode 1: Take action "move left" at state s
# Happened to balance well → R = 200

# Episode 2: Same state s, same action "move left"
# Unlucky physics → R = 50

# Gradient updates pull in opposite directions!
```

**Solution:** Use a baseline to reduce variance.

---

## Part 2: Value Function Baselines

### The Advantage Function

Instead of using return R, use **advantage** A(s,a):

```
A(s,a) = Q(s,a) - V(s)
```

**Intuition:** How much better is action a than the average action in state s?

- A(s,a) > 0: Action is better than average → increase probability
- A(s,a) < 0: Action is worse than average → decrease probability
- A(s,a) = 0: Action is average → no change

### Why This Reduces Variance

```
# Without baseline: R_t = 200 or 50 (high variance)
# With baseline: A_t = R_t - V(s_t)

# If V(s_t) = 150 (learned average return from state s_t):
# Episode 1: A_t = 200 - 150 = +50
# Episode 2: A_t = 50 - 150 = -100

# Now we see that episode 2 was actually below average!
```

### Policy Gradient with Baseline

```
∇J(θ) = E[∇ log π(a|s) * (R - V(s))]
      = E[∇ log π(a|s) * A(s,a)]
```

This is **variance reduction without bias** (proven mathematically).

---

## Part 3: Actor-Critic Architecture

### The Two Networks

**Actor:** Policy network π(a|s; θ)
```python
# Maps state → action probabilities (discrete)
# Or state → action distribution parameters (continuous)
```

**Critic:** Value network V(s; φ)
```python
# Maps state → expected return
```

### How They Work Together

1. **Actor** selects action based on policy
2. **Critic** evaluates how good the state is
3. **Advantage** = reward + γV(s') - V(s)
4. **Update actor** to increase probability of positive advantage actions
5. **Update critic** to better predict returns

---

## Part 4: A2C Implementation (Advantage Actor-Critic)

### Complete A2C from Scratch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import numpy as np

class ActorCritic(nn.Module):
    """
    Combined actor-critic network.

    Shared feature extractor → separate heads for policy and value.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head (policy)
        self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        """
        Returns both policy logits and value estimate.
        """
        features = self.shared(state)

        # Policy logits (unnormalized probabilities)
        logits = self.actor(features)

        # Value estimate
        value = self.critic(features)

        return logits, value

    def select_action(self, state):
        """
        Sample action from policy, return action and log probability.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        logits, value = self.forward(state_tensor)

        # Create categorical distribution from logits
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        # Sample action
        action = dist.sample()

        # Log probability of action
        log_prob = dist.log_prob(action)

        return action.item(), log_prob, value


class A2CAgent:
    """
    Advantage Actor-Critic agent.

    Updates policy and value function after each episode.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate=3e-4,
        gamma=0.99,
        value_coef=0.5,
        entropy_coef=0.01
    ):
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        # Actor-critic network
        self.model = ActorCritic(state_dim, action_dim)

        # Single optimizer for both actor and critic
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Storage for episode data
        self.reset_storage()

    def reset_storage(self):
        """Clear episode storage."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def select_action(self, state):
        """Select action and store data."""
        action, log_prob, value = self.model.select_action(state)

        # Store for training
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)

        return action

    def store_transition(self, reward, done):
        """Store reward and done flag."""
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_returns(self, next_state=None):
        """
        Compute returns (discounted sum of rewards).

        If episode is not done, bootstrap from value of next state.
        """
        returns = []
        R = 0

        # If episode truncated (not done), bootstrap from value estimate
        if next_state is not None and not self.dones[-1]:
            with torch.no_grad():
                state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                _, value = self.model(state_tensor)
                R = value.item()

        # Compute returns backwards
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                R = 0  # Reset at episode boundary
            R = reward + self.gamma * R
            returns.insert(0, R)

        return returns

    def update(self, next_state=None):
        """
        Update actor and critic after episode.
        """
        if len(self.rewards) == 0:
            return None

        # Compute returns
        returns = self.compute_returns(next_state)

        # Convert to tensors
        states = torch.tensor(np.array(self.states), dtype=torch.float32)
        actions = torch.tensor(self.actions, dtype=torch.long)
        log_probs = torch.stack(self.log_probs)
        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.cat(self.values)

        # Compute advantages
        advantages = returns - values.detach()

        # Normalize advantages (stabilizes training)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Re-evaluate actions to get current log probs and entropy
        logits, values = self.model(states)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        # New log probs and entropy
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # Actor loss (policy gradient with advantage)
        actor_loss = -(new_log_probs * advantages).mean()

        # Critic loss (MSE between predicted values and returns)
        critic_loss = F.mse_loss(values.squeeze(), returns)

        # Total loss
        loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

        # Gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        # Reset storage
        self.reset_storage()

        return {
            'loss': loss.item(),
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item()
        }


def train_a2c(env_name='CartPole-v1', n_episodes=1000):
    """Train A2C agent."""
    env = gym.make(env_name)

    agent = A2CAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        learning_rate=3e-4,
        gamma=0.99
    )

    rewards_per_episode = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Select action
            action = agent.select_action(state)

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition
            agent.store_transition(reward, done)

            state = next_state
            total_reward += reward

        # Update agent after episode
        metrics = agent.update()

        rewards_per_episode.append(total_reward)

        # Logging
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_per_episode[-50:])
            print(f"Episode {episode+1}: Avg Reward = {avg_reward:.2f}")
            if metrics:
                print(f"  Loss = {metrics['loss']:.3f}, "
                      f"Entropy = {metrics['entropy']:.3f}")

    return agent, rewards_per_episode


# Train
if __name__ == "__main__":
    agent, rewards = train_a2c()
```

---

## Part 5: A3C (Asynchronous Actor-Critic)

### The Parallelization Idea

Instead of one agent learning sequentially, run **multiple agents in parallel**:

```
Agent 1 → Environment 1 → experiences → shared network
Agent 2 → Environment 2 → experiences → shared network
Agent 3 → Environment 3 → experiences → shared network
...
Agent N → Environment N → experiences → shared network
```

**Benefits:**
1. Faster training (parallel collection)
2. More diverse experiences (different agents explore differently)
3. Breaks correlation in data (agents are independent)

### A3C Implementation

```python
import torch.multiprocessing as mp

class A3CWorker(mp.Process):
    """
    Worker process for A3C.

    Each worker has its own environment and local network.
    Updates are sent to shared global network.
    """

    def __init__(self, worker_id, global_model, optimizer, env_name, gamma=0.99):
        super().__init__()
        self.worker_id = worker_id
        self.global_model = global_model
        self.optimizer = optimizer
        self.env_name = env_name
        self.gamma = gamma

    def run(self):
        """Worker's main loop."""
        # Create local environment
        env = gym.make(self.env_name)

        # Create local model (same architecture as global)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        local_model = ActorCritic(state_dim, action_dim)

        while True:  # Continuous learning
            # Sync local model with global model
            local_model.load_state_dict(self.global_model.state_dict())

            # Collect trajectory
            states, actions, rewards, log_probs, values = self.collect_trajectory(
                env, local_model, max_steps=20
            )

            # Compute returns and advantages
            returns = self.compute_returns(rewards)
            advantages = returns - values

            # Compute loss
            loss = self.compute_loss(
                local_model, states, actions, returns, advantages
            )

            # Update global model
            self.optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), 40)

            # Copy gradients to global model
            for local_param, global_param in zip(
                local_model.parameters(), self.global_model.parameters()
            ):
                global_param._grad = local_param.grad

            # Update global model
            self.optimizer.step()

    def collect_trajectory(self, env, model, max_steps):
        """Collect trajectory for max_steps or until episode ends."""
        states, actions, rewards, log_probs, values = [], [], [], [], []

        state, _ = env.reset()

        for _ in range(max_steps):
            # Select action
            action, log_prob, value = model.select_action(state)

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)

            state = next_state

            if done:
                break

        return states, actions, rewards, log_probs, values

    def compute_returns(self, rewards):
        """Compute discounted returns."""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32)

    def compute_loss(self, model, states, actions, returns, advantages):
        """Compute actor-critic loss."""
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)

        # Forward pass
        logits, values = model(states)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        # Log probs and entropy
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # Actor loss
        actor_loss = -(log_probs * advantages.detach()).mean()

        # Critic loss
        critic_loss = F.mse_loss(values.squeeze(), returns)

        # Total loss
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        return loss


def train_a3c(env_name='CartPole-v1', n_workers=4):
    """Train A3C with multiple workers."""
    # Shared global model
    env = gym.make(env_name)
    global_model = ActorCritic(
        env.observation_space.shape[0],
        env.action_space.n
    )
    global_model.share_memory()  # Share across processes

    # Shared optimizer
    optimizer = optim.Adam(global_model.parameters(), lr=1e-3)

    # Create workers
    workers = []
    for worker_id in range(n_workers):
        worker = A3CWorker(worker_id, global_model, optimizer, env_name)
        workers.append(worker)

    # Start all workers
    for worker in workers:
        worker.start()

    # Wait for all workers
    for worker in workers:
        worker.join()


if __name__ == "__main__":
    # Note: A3C runs indefinitely, you'll need to add stopping condition
    train_a3c(n_workers=4)
```

### Why A3C Was Important (2016)

Before A3C:
- DQN took ~1 day to train on Atari (single GPU)
- Needed experience replay (large memory)

After A3C:
- Trains in ~4 hours (multi-core CPU)
- No experience replay needed (parallel exploration breaks correlation)
- Simpler and faster

---

## Part 6: Generalized Advantage Estimation (GAE)

### The Bias-Variance Trade-off

We can estimate advantages different ways:

**1-step TD (low variance, high bias):**
```
A_t = r_t + γV(s_{t+1}) - V(s_t)
```

**Monte Carlo (high variance, low bias):**
```
A_t = R_t - V(s_t)  # where R_t = sum of all future rewards
```

**GAE: Exponential average of all n-step returns**

### GAE Formula

```
A_t^GAE = Σ_{l=0}^∞ (γλ)^l δ_{t+l}

where δ_t = r_t + γV(s_{t+1}) - V(s_t)  # TD error
```

λ ∈ [0,1] controls trade-off:
- λ=0: 1-step TD (low variance, high bias)
- λ=1: Monte Carlo (high variance, low bias)
- λ=0.95: Good default (used in PPO)

### Implementation

```python
def compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95):
    """
    Compute Generalized Advantage Estimation.

    Args:
        rewards: List of rewards
        values: List of value estimates V(s_t)
        next_value: Value estimate V(s_T) for last state
        dones: List of done flags
        gamma: Discount factor
        lam: GAE parameter (λ)

    Returns:
        advantages: GAE advantages
        returns: TD(λ) returns
    """
    advantages = []
    gae = 0

    # Append next_value for bootstrapping
    values = values + [next_value]

    # Compute GAE backwards
    for t in reversed(range(len(rewards))):
        # TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
        if dones[t]:
            # Episode ended, no bootstrap
            delta = rewards[t] - values[t]
            gae = delta
        else:
            delta = rewards[t] + gamma * values[t+1] - values[t]
            # GAE: A_t = δ_t + γλδ_{t+1} + (γλ)²δ_{t+2} + ...
            gae = delta + gamma * lam * gae

        advantages.insert(0, gae)

    # Returns = advantages + values
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = advantages + torch.tensor(values[:-1], dtype=torch.float32)

    return advantages, returns


# Usage in A2C
def update_with_gae(self, next_state=None):
    """Update using GAE instead of simple returns."""
    # Get next value for bootstrapping
    if next_state is not None and not self.dones[-1]:
        with torch.no_grad():
            state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            _, next_value = self.model(state_tensor)
            next_value = next_value.item()
    else:
        next_value = 0

    # Extract values as list
    values_list = [v.item() for v in self.values]

    # Compute GAE
    advantages, returns = compute_gae(
        self.rewards,
        values_list,
        next_value,
        self.dones,
        gamma=self.gamma,
        lam=0.95
    )

    # Continue with normal A2C update...
```

**GAE is used in PPO (Lecture 06)** to get better advantage estimates.

---

## Part 7: Gotchas from Real Implementation

### Gotcha 1: Advantage Normalization

Always normalize advantages:

```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

Without this, training is unstable (some advantages dominate).

### Gotcha 2: Value Loss Coefficient

Critic loss often has larger magnitude than actor loss. Scale it:

```python
loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
```

Typical value_coef: 0.5

### Gotcha 3: Entropy Bonus

Without entropy bonus, policy collapses to deterministic too early:

```python
entropy = dist.entropy().mean()
loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy
```

Typical entropy_coef: 0.01

### Gotcha 4: Shared vs Separate Networks

**Shared (recommended):**
```python
shared_features → actor_head
                → critic_head
```
Faster, forces features useful for both tasks.

**Separate:**
```python
state → actor_network → policy
state → critic_network → value
```
More parameters, slower, but sometimes better.

### Gotcha 5: Gradient Clipping

Actor-critic can have exploding gradients:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
```

---

## Part 8: A2C vs A3C vs PPO

| Method | Update Frequency | Parallelization | Sample Efficiency | Stability |
|--------|------------------|-----------------|-------------------|-----------|
| A2C | After episode | Optional (vectorized envs) | Medium | Medium |
| A3C | Asynchronous | Yes (multi-process) | Medium | Medium |
| PPO | After batch | Optional | High | High |

**Modern practice:** Use PPO (Lecture 06) instead of A2C/A3C.

But A2C/A3C are simpler to understand and implement!

---

## Part 9: When to Use Actor-Critic

**Use actor-critic when:**
1. Both continuous and discrete actions
2. Need online learning
3. Baseline for research (A2C is simple)
4. Prototyping new ideas

**Don't use actor-critic when:**
1. Very sample inefficient (use off-policy methods like SAC)
2. Need stability guarantees (use PPO)
3. Discrete actions only (DQN is simpler)

---

## Summary

1. **Actor-critic combines policy gradients with value functions**
2. **Advantage A(s,a) = Q(s,a) - V(s)** reduces variance
3. **A2C:** Single-agent actor-critic with advantage
4. **A3C:** Parallel workers, faster training
5. **GAE:** Better advantage estimation (prepares for PPO)
6. **Key tricks:** Advantage normalization, entropy bonus, gradient clipping

---

## What's Next?

**Next lecture (05):** Trust Regions & TRPO - How to make safe policy updates

**The connection:** A2C/A3C can take bad policy steps and fail catastrophically. TRPO (and later PPO) solve this with constrained updates.

---

## Paper Trail

1. **Policy Gradient with Baselines (2000):** "Policy Gradient Methods for Reinforcement Learning with Function Approximation" - Sutton et al.
2. **A3C (2016):** "Asynchronous Methods for Deep Reinforcement Learning" - DeepMind
3. **GAE (2016):** "High-Dimensional Continuous Control Using Generalized Advantage Estimation" - Schulman et al.
4. **A2C Baselines (2017):** OpenAI Baselines implementation

All in `/Modern-RL-Research/RLHF-and-Alignment/PAPERS.md`

---

## Exercise for Yourself

Implement A2C and train on CartPole:

1. Start with basic actor-critic network
2. Add advantage estimation
3. Train for 500 episodes
4. Compare to REINFORCE (Lecture 02) and DQN (Lecture 03)

Then add GAE:
- Compare λ=0 vs λ=0.95 vs λ=1
- See how variance changes

**You'll see A2C is much more stable than REINFORCE and solves CartPole faster!**
