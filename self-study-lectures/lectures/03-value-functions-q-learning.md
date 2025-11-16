# Lecture 03: Value Functions & Q-Learning

**Duration:** ~90 minutes
**Prerequisites:** Lecture 01 (MDPs and Bellman Equations)
**Goal:** Understand value-based RL, implement Q-Learning and DQN from scratch

---

## Why This Matters

In Lecture 02, we learned policy gradients: directly optimize the policy π(a|s). But there's another way: **learn how good each action is, then pick the best one**.

This is huge because:
1. **Sample efficiency:** Policy gradients need lots of data. Q-Learning can reuse old experiences.
2. **Off-policy learning:** Can learn from any data, even random exploration.
3. **Deterministic policies:** Great for discrete actions (Atari games, robotics).

When DeepMind beat humans at Atari in 2013 (the paper that started deep RL), they used **DQN (Deep Q-Networks)**. This lecture is about understanding that breakthrough.

---

## Part 1: Value Functions (The Foundation)

### Two Types of Value Functions

Remember from Lecture 01:

**State-value function** V^π(s): Expected return starting from state s, following policy π
```
V^π(s) = E_π[R_t + γR_{t+1} + γ²R_{t+2} + ... | S_t = s]
```

**Action-value function** Q^π(s,a): Expected return starting from state s, taking action a, then following π
```
Q^π(s,a) = E_π[R_t + γR_{t+1} + γ²R_{t+2} + ... | S_t = s, A_t = a]
```

### The Key Insight

If you know Q^*(s,a) for the optimal policy, you can act optimally:
```
π*(s) = argmax_a Q*(s,a)
```

**You don't need to store a policy at all!** Just store Q-values and pick the max.

---

## Part 2: Q-Learning (Tabular)

### The Q-Learning Update Rule

Start with random Q(s,a) for all states and actions. Then update:

```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
```

Where:
- α = learning rate
- r = immediate reward
- γ = discount factor
- s' = next state
- The term `r + γ max_a' Q(s',a')` is called the **TD target**
- The term `r + γ max_a' Q(s',a') - Q(s,a)` is called the **TD error**

### Why This Works

This is just the Bellman equation rewritten as an update:
```
Q*(s,a) = E[r + γ max_a' Q*(s',a')]
```

We're iteratively solving this equation!

### Complete Implementation

```python
import numpy as np
import gymnasium as gym

class QLearningAgent:
    """
    Tabular Q-Learning agent.
    Works for discrete state and action spaces.
    """

    def __init__(self, n_states, n_actions, learning_rate=0.1,
                 gamma=0.99, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

        # Initialize Q-table to zeros
        self.q_table = np.zeros((n_states, n_actions))

    def select_action(self, state, training=True):
        """
        Epsilon-greedy action selection.

        With probability epsilon: explore (random action)
        With probability 1-epsilon: exploit (best action)
        """
        if training and np.random.random() < self.epsilon:
            # Explore
            return np.random.randint(self.n_actions)
        else:
            # Exploit
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        """
        Q-Learning update: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        """
        # Current Q-value
        current_q = self.q_table[state, action]

        # TD target
        if done:
            # No future rewards
            td_target = reward
        else:
            # r + γ max_a' Q(s',a')
            max_next_q = np.max(self.q_table[next_state])
            td_target = reward + self.gamma * max_next_q

        # TD error
        td_error = td_target - current_q

        # Update Q-value
        self.q_table[state, action] += self.lr * td_error

        return td_error


def train_q_learning(env_name='FrozenLake-v1', n_episodes=10000):
    """
    Train Q-Learning on FrozenLake environment.

    FrozenLake: 4x4 grid, goal is to reach the goal without falling in holes.
    States: 16 (one for each grid cell)
    Actions: 4 (up, down, left, right)
    """
    env = gym.make(env_name)
    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=0.1
    )

    rewards_per_episode = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Select action
            action = agent.select_action(state, training=True)

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Update Q-table
            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)

        # Print progress
        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode+1}: Avg Reward (last 100) = {avg_reward:.2f}")

    return agent, rewards_per_episode


# Run training
if __name__ == "__main__":
    agent, rewards = train_q_learning()

    # Test the learned policy
    env = gym.make('FrozenLake-v1', render_mode='human')
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state, training=False)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f"Test episode reward: {total_reward}")
```

### Key Points

1. **Off-policy:** Q-Learning learns Q*(s,a), the optimal Q-function, even though it explores randomly. This is different from policy gradients which are on-policy.

2. **No policy storage:** We just store Q(s,a) and pick argmax at test time.

3. **Exploration:** Epsilon-greedy is simple but effective. Start with high ε (explore), decay over time (exploit).

---

## Part 3: Deep Q-Networks (DQN)

### The Problem with Tabular Q-Learning

FrozenLake has 16 states. Atari games have ~10^18 states (84×84 pixels, 3 frames stacked).

**We can't store a table!**

### The Solution: Function Approximation

Instead of storing Q(s,a) in a table, learn a neural network:
```
Q(s,a; θ) ≈ Q*(s,a)
```

Where θ are the network parameters.

### DQN Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    Deep Q-Network for Atari games.

    Input: 84x84x4 (4 stacked grayscale frames)
    Output: Q-values for each action
    """

    def __init__(self, n_actions):
        super().__init__()

        # Convolutional layers (for visual input)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        """
        x: (batch, 4, 84, 84) tensor of stacked frames
        Returns: (batch, n_actions) tensor of Q-values
        """
        # Normalize pixel values
        x = x / 255.0

        # Convolutional layers with ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)

        return q_values
```

### DQN for Simple Environments

For non-visual environments (like CartPole), use a simpler network:

```python
class SimpleDQN(nn.Module):
    """
    DQN for low-dimensional state spaces (e.g., CartPole).

    Input: state vector
    Output: Q-values for each action
    """

    def __init__(self, state_dim, n_actions, hidden_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, state):
        """
        state: (batch, state_dim) tensor
        Returns: (batch, n_actions) tensor of Q-values
        """
        return self.net(state)
```

---

## Part 4: DQN Training Tricks

Naive DQN doesn't work. You need these tricks:

### Trick 1: Experience Replay

**Problem:** RL data is sequential and correlated. SGD assumes i.i.d. data.

**Solution:** Store experiences in a replay buffer, sample random batches.

```python
import random
from collections import deque, namedtuple

# Experience tuple
Experience = namedtuple('Experience',
                        ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """
    Stores experiences and samples random batches for training.
    """

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        exp = Experience(state, action, reward, next_state, done)
        self.buffer.append(exp)

    def sample(self, batch_size):
        """Sample random batch of experiences."""
        batch = random.sample(self.buffer, batch_size)

        # Unzip batch
        states = torch.tensor([e.state for e in batch], dtype=torch.float32)
        actions = torch.tensor([e.action for e in batch], dtype=torch.long)
        rewards = torch.tensor([e.reward for e in batch], dtype=torch.float32)
        next_states = torch.tensor([e.next_state for e in batch], dtype=torch.float32)
        dones = torch.tensor([e.done for e in batch], dtype=torch.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
```

### Trick 2: Target Network

**Problem:** TD target `r + γ max_a' Q(s',a'; θ)` uses the same network we're updating. This causes instability.

**Solution:** Use a separate target network Q(s,a; θ^-) that updates slowly.

```python
# Update rule becomes:
Q(s,a; θ) ← Q(s,a; θ) + α[r + γ max_a' Q(s',a'; θ^-) - Q(s,a; θ)]

# θ^- is updated every C steps:
θ^- ← θ
```

---

## Part 5: Complete DQN Implementation

Putting it all together:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np

class DQNAgent:
    """
    Complete DQN agent with experience replay and target network.
    """

    def __init__(
        self,
        state_dim,
        n_actions,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_capacity=100000,
        batch_size=64,
        target_update_freq=1000
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Q-network and target network
        self.q_network = SimpleDQN(state_dim, n_actions)
        self.target_network = SimpleDQN(state_dim, n_actions)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Training step counter
        self.steps = 0

    def select_action(self, state, training=True):
        """Epsilon-greedy action selection."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        # Convert to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # Get Q-values
        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        # Return action with highest Q-value
        return q_values.argmax().item()

    def update(self):
        """Sample batch from replay buffer and perform one gradient step."""
        if len(self.replay_buffer) < self.batch_size:
            return None  # Not enough samples yet

        # Sample batch
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        # Compute current Q-values
        # Q(s,a) for the actions that were actually taken
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Compute target Q-values
        with torch.no_grad():
            # max_a' Q(s', a'; θ^-)
            next_q_values = self.target_network(next_states).max(1)[0]

            # r + γ max_a' Q(s', a'; θ^-) if not done, else just r
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss (MSE between current and target Q-values)
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        # Gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients (stabilizes training)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()

        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


def train_dqn(env_name='CartPole-v1', n_episodes=500):
    """Train DQN agent."""
    env = gym.make(env_name)

    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=64
    )

    rewards_per_episode = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Select action
            action = agent.select_action(state, training=True)

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store experience
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # Train agent
            loss = agent.update()

            state = next_state
            total_reward += reward

        # Decay epsilon
        agent.decay_epsilon()

        rewards_per_episode.append(total_reward)

        # Logging
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_per_episode[-50:])
            print(f"Episode {episode+1}: Avg Reward = {avg_reward:.2f}, "
                  f"Epsilon = {agent.epsilon:.3f}")

    return agent, rewards_per_episode


# Train
if __name__ == "__main__":
    agent, rewards = train_dqn()

    # Visualize learned policy
    env = gym.make('CartPole-v1', render_mode='human')
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state, training=False)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f"Test episode reward: {total_reward}")
```

---

## Part 6: DQN Improvements

### Double DQN

**Problem:** DQN overestimates Q-values due to max operation.

**Solution:** Use Q-network to select action, target network to evaluate:

```python
# Standard DQN target:
target = r + γ max_a' Q(s', a'; θ^-)

# Double DQN target:
a* = argmax_a' Q(s', a'; θ)  # Select with Q-network
target = r + γ Q(s', a*; θ^-)  # Evaluate with target network
```

Implementation change:

```python
# In update() method, replace:
next_q_values = self.target_network(next_states).max(1)[0]

# With:
# Select actions using Q-network
best_actions = self.q_network(next_states).argmax(1)
# Evaluate actions using target network
next_q_values = self.target_network(next_states).gather(1, best_actions.unsqueeze(1)).squeeze()
```

### Dueling DQN

**Idea:** Decompose Q(s,a) into state value V(s) and advantage A(s,a):

```
Q(s,a) = V(s) + A(s,a)
```

Architecture:

```python
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_dim=128):
        super().__init__()

        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, state):
        features = self.features(state)

        # Compute V(s) and A(s,a)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Combine: Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
        # Subtracting mean makes it identifiable
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values
```

**Why this helps:** Learning V(s) and A(s,a) separately is easier than learning Q(s,a) directly.

### Prioritized Experience Replay

**Idea:** Sample important transitions more frequently.

Importance = |TD error| = |r + γ max_a' Q(s',a') - Q(s,a)|

Big TD error → we were surprised → important transition.

```python
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.buffer = []
        self.priorities = []
        self.capacity = capacity
        self.alpha = alpha  # How much to prioritize
        self.pos = 0

    def push(self, experience, td_error):
        priority = (abs(td_error) + 1e-5) ** self.alpha

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.pos] = experience
            self.priorities[self.pos] = priority

        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        # Sample proportional to priority
        probs = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        # Compute importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize

        batch = [self.buffer[i] for i in indices]
        return batch, weights, indices

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha
```

---

## Part 7: Gotchas from Real Implementation

### Gotcha 1: Reward Clipping

Atari games have different reward scales. DQN clips rewards to [-1, 1]:

```python
reward = np.clip(reward, -1, 1)
```

Without this, some games dominate training.

### Gotcha 2: Frame Stacking

Single frame doesn't show velocity. Stack 4 frames:

```python
# Stack last 4 frames
frame_stack = np.stack([frame_t, frame_t-1, frame_t-2, frame_t-3])
```

### Gotcha 3: Gradient Clipping

DQN training is unstable. Always clip gradients:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
```

### Gotcha 4: Burning In Replay Buffer

Don't train until buffer has enough samples:

```python
if len(replay_buffer) < 10000:
    continue  # Just collect experiences
```

### Gotcha 5: Epsilon Scheduling

Start with high exploration (ε=1.0), slowly decay:

```python
epsilon = max(0.01, epsilon * 0.995)  # After each episode
```

---

## Part 8: When to Use DQN

**Use DQN when:**
1. Discrete action space (DQN doesn't work for continuous actions)
2. Off-policy learning needed (reuse old data)
3. Sample efficiency matters
4. Deterministic policy is okay

**Don't use DQN when:**
1. Continuous actions (use DDPG/SAC instead)
2. Stochastic policy needed (use policy gradients)
3. Very high-dimensional action space (exponential in number of actions)

---

## Summary

1. **Q-Learning:** Learn Q(s,a), pick argmax. Off-policy, sample efficient.
2. **DQN:** Use neural network for Q-function approximation.
3. **Key tricks:**
   - Experience replay (break correlations)
   - Target network (stabilize training)
   - Double DQN (reduce overestimation)
   - Dueling DQN (separate V and A)
   - Prioritized replay (sample important transitions)
4. **Trade-offs:** DQN is sample efficient but only works for discrete actions.

---

## What's Next?

**Next lecture (04):** Actor-Critic Methods - Best of both worlds (policy gradients + value functions)

**The connection:** DQN learns Q(s,a) and implicitly extracts policy. Actor-Critic explicitly learns both!

---

## Paper Trail

1. **Q-Learning (1989):** "Learning from Delayed Rewards" - Chris Watkins (PhD thesis)
2. **DQN (2013):** "Playing Atari with Deep Reinforcement Learning" - DeepMind
3. **DQN Nature (2015):** "Human-level control through deep reinforcement learning" - DeepMind
4. **Double DQN (2015):** "Deep Reinforcement Learning with Double Q-learning" - DeepMind
5. **Dueling DQN (2016):** "Dueling Network Architectures for Deep Reinforcement Learning" - DeepMind
6. **Prioritized Replay (2015):** "Prioritized Experience Replay" - DeepMind

All in `/Modern-RL-Research/RLHF-and-Alignment/PAPERS.md` (foundational papers)

---

## Exercise for Yourself

Implement DQN and train on CartPole:

1. Start with SimpleDQN architecture
2. Add experience replay
3. Add target network
4. Train for 500 episodes
5. Should solve CartPole (avg reward >195) in <200 episodes

Then try improvements:
- Double DQN
- Dueling DQN
- Compare sample efficiency

**You'll see DQN is way more sample efficient than REINFORCE from Lecture 02!**
