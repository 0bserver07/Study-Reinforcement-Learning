<!-- status: unreviewed | last-reviewed: never -->

# Lecture 02: Policy Gradients from Scratch

_Unreviewed — no one has checked this end to end. Treat the math, code, and citations as unverified._

**Time**: ~4 hours, with the [exercise](../../exercises/02-policy-gradients/) · **Prerequisites**: Lecture 01, basic calculus

---

## Why policy gradients

Policy iteration works on a 5×5 gridworld. It doesn't scale: an LLM's "state" is the whole token sequence so far, so there's no value table big enough, and the action space (the vocabulary) is huge. Policy gradients drop the table — you parameterize the policy with a network and move its parameters by gradient ascent. PPO, which is how RLHF trains LLMs, is a policy-gradient method, so this is the thing it's built on.

---

## Part 1: The Problem with Value-Based Methods

### Why We Need Something New

```python
# Imagine trying policy iteration on an LLM:
n_vocab = 50000
max_sequence_length = 2048
n_states = n_vocab ** max_sequence_length  # = 50000^2048

# That's a number with ~9,500 digits.
# The universe has ~10^80 atoms.
# We can't enumerate these states.
```

**Problems with value-based methods**:
1. **State space explosion**: Too many states
2. **Continuous actions**: What if actions are continuous (e.g., robot joint angles)?
3. **Stochastic policies**: Sometimes randomness is optimal (rock-paper-scissors)

### The Key Insight

> "What if we just parameterize the policy with a neural network and use gradient descent?"

```python
# Instead of:
#   policy_table[state] = action  # Needs |S| entries

# Do this:
#   policy_network(state) → action  # Fixed size network
#   θ = network parameters
#   π_θ(a|s) = probability of action a in state s
```

**This is policy gradients.**

---

## Part 2: Intuition First (Before the Math)

### The Goal

We have a policy π_θ parameterized by θ (neural network weights).

We want to maximize expected return:
```
J(θ) = 𝔼_{τ~π_θ}[R(τ)]
```

Where τ is a trajectory: (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T, a_T, r_T)

**In English**: "Find θ that makes the policy collect high rewards on average."

### The Gradient

We need ∇_θ J(θ) to do gradient ascent:
```
θ ← θ + α ∇_θ J(θ)
```

**Question**: How do we compute ∇_θ J(θ)?

**Answer**: The policy gradient theorem (coming up).

---

## Part 3: The Policy Gradient Theorem

### The Theorem

```
∇_θ J(θ) = 𝔼_{τ~π_θ}[Σ_t ∇_θ log π_θ(a_t|s_t) · R(τ)]
```

**Read this as**:
"The gradient is the expected sum of (log probability gradient) × (return)"

### Why this form is useful

1. **No model needed**: you never need P(s'|s,a) — the dynamics drop out (see Part 5)
2. **Differentiable**: it's just `log π_θ(a|s)`, so autodiff handles it
3. **Works for any action space**: discrete, continuous, structured (tokens)

### The Intuition

```python
# Imagine rolling out trajectory τ
τ = (s_0, a_0, r_0, s_1, a_1, r_1, ...)
R(τ) = Σ γ^t r_t  # Total return

# If R(τ) is high (good trajectory):
#   → increase probability of actions taken
#   → ∇_θ log π_θ(a_t|s_t) points toward "make a_t more likely"

# If R(τ) is low (bad trajectory):
#   → decrease probability of actions taken
#   → negative R(τ) reverses the gradient
```

**Note**: It helps to read this as supervised learning where the "label" is "do more of this if it worked, less if it didn't" — and the strength of that label is the return.

---

## Part 4: REINFORCE Algorithm (Policy Gradients in Practice)

### The Algorithm

```
1. Initialize policy π_θ (neural network with random weights θ)
2. For each episode:
   a. Roll out trajectory τ using π_θ
   b. Compute return R(τ) = Σ_t γ^t r_t
   c. Compute gradient: ∇_θ J ≈ Σ_t ∇_θ log π_θ(a_t|s_t) · R(τ)
   d. Update: θ ← θ + α · ∇_θ J
3. Repeat until converged
```

### Complete Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym  # `pip install gymnasium`; the old `gym` package has a different reset/step API

class PolicyNetwork(nn.Module):
    """
    Simple policy network for discrete actions.

    input: state (vector)
    output: action probabilities
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)  # Output probabilities
        )

    def forward(self, state):
        """
        Args:
            state: [batch_size, state_dim] or [state_dim]
        Returns:
            action_probs: [batch_size, action_dim] or [action_dim]
        """
        return self.network(state)

    def select_action(self, state):
        """
        Sample action from policy.

        Returns:
            action: int
            log_prob: log π_θ(a|s)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.forward(state_tensor)

        # Sample from categorical distribution
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()

        # Compute log probability
        log_prob = dist.log_prob(action)

        return action.item(), log_prob


class REINFORCE:
    """
    REINFORCE (Monte Carlo Policy Gradient) agent.
    """
    def __init__(self, env, hidden_dim=128, lr=1e-3, gamma=0.99):
        self.env = env
        self.gamma = gamma

        # Initialize policy network
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim)

        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Episode storage
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        """Select action and store log probability."""
        action, log_prob = self.policy.select_action(state)
        self.log_probs.append(log_prob)
        return action

    def store_reward(self, reward):
        """Store reward for current timestep."""
        self.rewards.append(reward)

    def compute_returns(self):
        """
        Compute discounted returns for each timestep.

        R_t = r_t + γr_{t+1} + γ²r_{t+2} + ...
        """
        returns = []
        R = 0

        # Compute returns backwards from terminal state
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        # Normalize returns (stabilizes training)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns

    def update_policy(self):
        """
        Update policy using policy gradient.

        ∇J = Σ_t ∇log π(a_t|s_t) · R_t
        """
        # Compute returns
        returns = self.compute_returns()

        # Compute policy gradient
        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            # Negative because we're doing gradient *ascent* but optimizer does descent
            policy_loss.append(-log_prob * R)

        # Sum losses
        policy_loss = torch.stack(policy_loss).sum()

        # Backprop and update
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # Clear episode storage
        self.log_probs = []
        self.rewards = []

        return policy_loss.item()

    def train(self, num_episodes=1000, print_every=100):
        """
        Train the policy with REINFORCE.
        """
        episode_rewards = []

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False

            # Roll out one episode
            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.store_reward(reward)
                episode_reward += reward
                state = next_state

            # Update policy after episode
            loss = self.update_policy()
            episode_rewards.append(episode_reward)

            # Logging
            if (episode + 1) % print_every == 0:
                avg_reward = np.mean(episode_rewards[-print_every:])
                print(f"Episode {episode + 1:4d} | "
                      f"Avg Reward: {avg_reward:7.2f} | "
                      f"Loss: {loss:8.4f}")

        return episode_rewards


# Run training
if __name__ == "__main__":
    print("="*60)
    print("REINFORCE on CartPole-v1")
    print("="*60 + "\n")

    # Create environment
    env = gym.make('CartPole-v1')

    # Create agent
    agent = REINFORCE(
        env,
        hidden_dim=128,
        lr=1e-2,
        gamma=0.99
    )

    # Train
    rewards = agent.train(num_episodes=1000, print_every=50)

    print("\n" + "="*60)
    print(f"Training complete!")
    print(f"Final 100-episode average: {np.mean(rewards[-100:]):.2f}")
    print("="*60)
```

**Roughly what you'll see** (exact numbers depend on the seed):
```
Episode   50 | Avg Reward:  ~20-30
Episode  200 | Avg Reward:  ~100-200   (climbing)
Episode  500 | Avg Reward:  ~300-475   (usually solved around here, sometimes a dip first)
```
REINFORCE on CartPole is noisy: it can climb and then collapse for a stretch before recovering. That's the variance problem this lecture is about — a value-function baseline (Lecture 04) is the usual fix.

---

## Part 5: Why REINFORCE Works (The Math)

### Deriving the Policy Gradient

Starting from the objective:
```
J(θ) = 𝔼_{τ~π_θ}[R(τ)]
     = ∫ P(τ|θ) R(τ) dτ
```

Taking the gradient:
```
∇_θ J(θ) = ∇_θ ∫ P(τ|θ) R(τ) dτ
         = ∫ ∇_θ P(τ|θ) R(τ) dτ
```

**The log-derivative trick**:
```
∇_θ P(τ|θ) = P(τ|θ) ∇_θ log P(τ|θ)
```

So:
```
∇_θ J(θ) = ∫ P(τ|θ) ∇_θ log P(τ|θ) R(τ) dτ
         = 𝔼_{τ~π_θ}[∇_θ log P(τ|θ) R(τ)]
```

**Now expand log P(τ|θ)**:
```
P(τ|θ) = p(s_0) Π_t P(s_{t+1}|s_t,a_t) π_θ(a_t|s_t)

log P(τ|θ) = log p(s_0) + Σ_t [log P(s_{t+1}|s_t,a_t) + log π_θ(a_t|s_t)]
```

**Take gradient** (only π_θ depends on θ):
```
∇_θ log P(τ|θ) = Σ_t ∇_θ log π_θ(a_t|s_t)
```

**Final result**:
```
∇_θ J(θ) = 𝔼_{τ~π_θ}[Σ_t ∇_θ log π_θ(a_t|s_t) · R(τ)]
```

**Note**: The environment dynamics P(s'|s,a) drop out of the gradient — you only differentiate the policy. That's the whole reason this works without a model of the environment.

---

## Part 6: The Gotchas

### Gotcha #1: High Variance

**Problem**: REINFORCE has **massive variance**.

```python
# Example: Two trajectories with same actions
τ1: R(τ1) = 100  # Got lucky
τ2: R(τ2) = -50  # Got unlucky

# Gradient estimates from these two will be wildly different!
# Leads to unstable training
```

**Solution**: Baseline subtraction (next section)

### Gotcha #2: Vanishing Gradients

**Problem**: log π_θ(a|s) can get very negative.

```python
# If π_θ(a|s) = 0.001 (rare action)
log_prob = np.log(0.001)  # = -6.9
# Gradient: ∇ log π ≈ very large
# But if π_θ(a|s) = 0.0001:
log_prob = np.log(0.0001)  # = -9.2
# Exploding gradients!
```

**Solution**: Gradient clipping

```python
# In PyTorch:
torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
```

### Gotcha #3: Credit Assignment

**Problem**: All actions get same return R(τ).

```python
# Trajectory: [a0, a1, a2, ..., a99]
#   a0-a98: random actions
#   a99: perfect action → +1000 reward

# REINFORCE gives same gradient to ALL actions!
# a0 gets credit for a99's success
```

**Solution**: Use returns-to-go (next section)

---

## Part 7: Improvements (Making It Actually Work)

### Improvement 1: Baseline Subtraction

**Idea**: Subtract average return to reduce variance.

```python
# Original REINFORCE:
loss = -log_prob * R

# With baseline:
loss = -log_prob * (R - baseline)
```

**Implementation**:
```python
def compute_returns_with_baseline(self):
    """Compute returns and subtract baseline."""
    returns = []
    R = 0

    for r in reversed(self.rewards):
        R = r + self.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)

    # Baseline = mean return
    baseline = returns.mean()

    # Center returns
    advantages = returns - baseline

    # Optional: normalize
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages
```

**Why this works**: Doesn't change expected gradient, but reduces variance.

**Proof**:
```
𝔼[∇ log π · (R - b)] = 𝔼[∇ log π · R] - b · 𝔼[∇ log π]
                       = 𝔼[∇ log π · R] - b · 0
                       = 𝔼[∇ log π · R]
```

(Because 𝔼[∇ log π] = 0 by properties of score function)

### Improvement 2: Returns-to-Go

**Idea**: Action at time t only affects rewards after t.

```python
# Instead of using total return R(τ) for all actions:
for t, log_prob in enumerate(log_probs):
    loss += -log_prob * R_total  # Bad

# Use return-to-go from time t:
for t, log_prob in enumerate(log_probs):
    R_to_go = sum(gamma**(k-t) * rewards[k] for k in range(t, T))
    loss += -log_prob * R_to_go  # Better!
```

**Implementation**:
```python
def compute_returns_to_go(self):
    """
    Compute returns-to-go for each timestep.

    G_t = r_t + γr_{t+1} + γ²r_{t+2} + ...
    """
    returns = []
    R = 0

    # Compute backwards
    for r in reversed(self.rewards):
        R = r + self.gamma * R
        returns.insert(0, R)

    return torch.tensor(returns)

# Usage in update:
def update_policy(self):
    returns = self.compute_returns_to_go()

    # Now each action gets its own return-to-go
    policy_loss = []
    for log_prob, R_to_go in zip(self.log_probs, returns):
        policy_loss.append(-log_prob * R_to_go)

    # Rest same...
```

### Improvement 3: Entropy Regularization

**Problem**: Policy can become deterministic too quickly (no exploration).

**Solution**: Add entropy bonus to encourage exploration.

```python
def compute_entropy(action_probs):
    """H(π) = -Σ π(a) log π(a)"""
    return -(action_probs * action_probs.log()).sum()

# In update:
entropy = compute_entropy(action_probs)
loss = policy_loss - 0.01 * entropy  # 0.01 is entropy coefficient
```

---

## Part 8: Connection to Modern Methods

### REINFORCE → Actor-Critic (Lecture 04)

```python
# REINFORCE uses Monte Carlo return:
R_t = r_t + γr_{t+1} + γ²r_{t+2} + ...  # High variance

# Actor-Critic uses value function bootstrap:
A_t = r_t + γV(s_{t+1}) - V(s_t)  # Lower variance

# Same policy gradient, better estimate
```

### Actor-Critic → PPO (Lecture 06)

```python
# AC can have large policy updates → instability

# PPO clips updates to trust region:
ratio = π_new / π_old
clipped_ratio = clip(ratio, 1-ε, 1+ε)
loss = min(ratio * A, clipped_ratio * A)

# Stable updates → state-of-the-art
```

### PPO → RLHF for LLMs (Lecture 10)

```python
# Apply PPO to language models:
# - States: prompt + partial generation
# - Actions: next token
# - Reward: from reward model trained on human preferences

# This is how ChatGPT is trained!
```

---

## Part 9: When to Use Policy Gradients

### Use Policy Gradients When:

1. **Continuous actions** (robot control, etc.)
2. **Stochastic policies needed** (rock-paper-scissors, exploration)
3. **High-dimensional actions** (generating text, images)
4. **Don't have environment model**

### Don't Use When:

1. **Can solve exactly** (small MDPs → use policy iteration)
2. **Discrete actions + model available** (use planning)
3. **Need sample efficiency** (policy gradients are sample-hungry)

**For LLMs**: Policy gradients (PPO specifically) are the standard. They handle the massive action space (vocabulary) and stochastic generation well.

---

## Part 10: Exercise

The graded version of this is [`exercises/02-policy-gradients/`](../../exercises/02-policy-gradients/): implement REINFORCE from a skeleton, get CartPole solved, make the tests pass. Hints there if you're stuck. After that, the variations worth trying:

- **Returns-to-go**: the code above already uses returns-to-go (each action gets the return *from its timestep on*, not the whole-episode return). Try switching to whole-episode return and watch it get worse.
- **Learned baseline**: replace the normalized-returns baseline with a value network `b(s) = V(s)` trained on the returns. That's the step toward actor-critic (Lecture 04).
- **Entropy bonus**: add `- β · H(π)` to the loss and see how exploration changes.
- **Harder env**: `Acrobot-v1`, or `LunarLander-v3` (`pip install "gymnasium[box2d]"`). REINFORCE's variance starts to bite.
- **Hyperparameters**: sweep learning rate over `[1e-4, 1e-3, 1e-2, 1e-1]` and γ over `[0.9, 0.95, 0.99]`. Which combinations diverge, which are just slow?

---

## Recap

Policy gradients optimize the policy directly, no value table needed: `∇J = 𝔼[∑_t ∇ log π_θ(a_t|s_t) · A_t]`. REINFORCE is the Monte Carlo version — roll out an episode, use the (discounted, return-to-go) returns as `A_t`, take the step. The catch is variance; baselines and returns-to-go knock it down, and everything later (actor-critic, PPO, RLHF) is this update with a better `A_t` and guardrails on the step size.

---

## Next Lecture

**[Lecture 03: Value Functions & Q-Learning](./03-value-functions-q-learning.md)**

But before that, make sure you:
- [ ] Ran the REINFORCE code
- [ ] Implemented baseline subtraction
- [ ] Can explain why environment dynamics drop out of gradient
- [ ] Understand variance problem and solutions

---

## References

- **Williams (1992)**, *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning* — introduces REINFORCE. (*Machine Learning* 8, 229–256.)
- **Sutton, McAllester, Singh, Mansour (2000)**, *Policy Gradient Methods for Reinforcement Learning with Function Approximation* — the policy gradient theorem and the actor-critic framing. (NeurIPS 1999.)
- **Schulman, Moritz, Levine, Jordan, Abbeel (2015)**, *High-Dimensional Continuous Control Using Generalized Advantage Estimation* — GAE, the advantage estimator usually paired with policy-gradient methods. [arXiv:1506.02438](https://arxiv.org/abs/1506.02438)
- **Schulman, Wolski, Dhariwal, Radford, Klimov (2017)**, *Proximal Policy Optimization Algorithms* — PPO (Lecture 06). [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- **Ouyang et al. (2022)**, *Training language models to follow instructions with human feedback* (InstructGPT) — PPO applied to RLHF (Lecture 10). [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- **Silver et al. (2017)**, *Mastering the game of Go without human knowledge* (AlphaGo Zero) — policy gradients with self-play. (*Nature* 550, 354–359.)

---

## Debugging checklist

When the policy isn't learning:

- **Print shapes.** Most policy-gradient bugs are shape or sign errors. `print(f"{returns.shape=} {log_probs[0].shape=}")` catches a lot.
- **Check the sign of the loss.** It should be a *minus*: `-(log_prob * advantage)`. The policy gradient is an ascent direction; the optimizer descends.
- **Make sure the log-prob carries gradient.** If you `.detach()`ed it, or stored `log_prob.item()` instead of the tensor, `loss.backward()` does nothing to the policy.
- **Detach anything that shouldn't have a gradient** — e.g. a learned baseline `V(s)` when you use it inside the policy loss; a gradient leaking through there biases the estimate.
- **Stack lists of tensors with `torch.stack`, not `torch.tensor`.** `torch.tensor([t1, t2, ...])` on a list of tensors does the wrong thing.
