# Lecture 08: Model-Based Reinforcement Learning

**Duration:** ~90 minutes
**Prerequisites:** Lecture 03 (DQN), Lecture 06 (PPO), Lecture 07 (SAC)
**Goal:** Learn world models, implement Dyna and MBPO, understand when to use model-based RL

---

## Why This Matters

All previous lectures used **model-free RL**:
- Learn policy/value directly from environment interaction
- Sample inefficient (need millions of steps)
- Don't build understanding of environment

**Model-based RL is different:**
- Learn a model of environment dynamics
- Use model for planning or generating synthetic data
- Can be 10-100x more sample efficient!

When a self-driving car trains, it can't crash millions of times. When a robot learns to walk, hardware is expensive. **Model-based RL enables learning with limited real-world data.**

Recent breakthrough: MuZero (DeepMind, 2020) masters Atari, Chess, Go, and Shogi with **less data** than AlphaZero. This lecture teaches you how.

---

## Part 1: Model-Free vs Model-Based

### Model-Free (What We've Learned So Far)

```python
# Direct interaction with environment
for episode in range(n_episodes):
    state = env.reset()
    while not done:
        action = policy(state)  # Learn policy directly
        next_state, reward, done = env.step(action)
        update_policy(state, action, reward, next_state)
```

**Pros:** Simple, works for complex dynamics
**Cons:** Sample inefficient, no understanding of environment

### Model-Based

```python
# Learn a model of environment
model = WorldModel()  # Predicts next state and reward

# Collect data
real_data = collect_from_environment(random_policy)

# Learn model
model.fit(real_data)  # s_t, a_t → s_{t+1}, r_t

# Use model for planning or data generation
for episode in range(n_episodes):
    # Option 1: Pure planning (no policy learning)
    action = plan_with_model(model, state)

    # Option 2: Generate synthetic data
    synthetic_data = model.generate_rollouts()
    update_policy(synthetic_data)  # Learn from imagined experience!
```

**Pros:** Sample efficient, transferable, interpretable
**Cons:** Model errors compound, harder to implement

---

## Part 2: World Models (Learning Dynamics)

### The Dynamics Model

Learn:
```
s_{t+1} = f(s_t, a_t) + ε  # Deterministic + noise
r_t = g(s_t, a_t)           # Reward function
```

Or probabilistically:
```
p(s_{t+1} | s_t, a_t)  # Transition distribution
p(r_t | s_t, a_t)       # Reward distribution
```

### Implementation: Deterministic World Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class WorldModel(nn.Module):
    """
    Learn deterministic dynamics: (s,a) → (s', r)

    In practice, model delta: s' = s + Δs
    This is easier to learn than s' directly.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        # Dynamics network: (s, a) → Δs
        self.dynamics = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)  # Predict delta
        )

        # Reward network: (s, a) → r
        self.reward = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Predict reward
        )

    def forward(self, state, action):
        """
        Predict next state and reward.

        Args:
            state: (batch, state_dim)
            action: (batch, action_dim)

        Returns:
            next_state: (batch, state_dim)
            reward: (batch, 1)
        """
        sa = torch.cat([state, action], dim=1)

        # Predict delta and add to current state
        delta = self.dynamics(sa)
        next_state = state + delta

        # Predict reward
        reward = self.reward(sa)

        return next_state, reward

    def train_model(self, replay_buffer, batch_size=256, epochs=100):
        """
        Train world model on real data from replay buffer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        for epoch in range(epochs):
            # Sample batch of transitions
            states, actions, rewards, next_states, dones = \
                replay_buffer.sample(batch_size)

            # Predict next state and reward
            pred_next_states, pred_rewards = self.forward(states, actions)

            # Compute loss
            dynamics_loss = F.mse_loss(pred_next_states, next_states)
            reward_loss = F.mse_loss(pred_rewards, rewards)

            total_loss = dynamics_loss + reward_loss

            # Update model
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: "
                      f"Dynamics Loss = {dynamics_loss.item():.4f}, "
                      f"Reward Loss = {reward_loss.item():.4f}")

    def rollout(self, initial_state, policy, horizon=10):
        """
        Generate imagined trajectory using learned model.

        Args:
            initial_state: Starting state
            policy: Policy to execute
            horizon: Number of steps to simulate

        Returns:
            states, actions, rewards: Imagined trajectory
        """
        states = [initial_state]
        actions = []
        rewards = []

        state = initial_state

        for _ in range(horizon):
            # Select action from policy
            action = policy(state)

            # Predict next state and reward using model
            with torch.no_grad():
                next_state, reward = self.forward(state, action)

            states.append(next_state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        return states, actions, rewards
```

---

## Part 3: Dyna Architecture

Dyna (Sutton, 1990) combines:
1. **Direct RL:** Learn from real environment
2. **Model learning:** Learn world model from real data
3. **Planning:** Generate synthetic experience from model

### Dyna-Q Algorithm

```python
class DynaQAgent:
    """
    Dyna-Q: Combines Q-Learning with world model planning.

    Algorithm:
    1. Take real action, observe transition
    2. Update Q-function (like normal Q-learning)
    3. Update world model
    4. Generate n synthetic experiences from model
    5. Update Q-function on synthetic data
    """

    def __init__(self, state_dim, action_dim, n_planning_steps=10):
        # Q-network (like DQN)
        self.q_network = QNetwork(state_dim, action_dim)
        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=1e-3)

        # World model
        self.world_model = WorldModel(state_dim, action_dim)
        self.model_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=1e-3)

        # Replay buffer (stores real transitions)
        self.replay_buffer = ReplayBuffer()

        self.n_planning_steps = n_planning_steps
        self.action_dim = action_dim

    def update(self, state, action, reward, next_state, done):
        """
        Update agent with one real transition.
        """
        # Store in replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)

        # 1. Direct RL: Update Q-function on real data
        self.update_q_function(state, action, reward, next_state, done)

        # 2. Model Learning: Update world model on real data
        self.update_world_model(state, action, reward, next_state)

        # 3. Planning: Generate synthetic experiences and update Q-function
        for _ in range(self.n_planning_steps):
            self.planning_step()

    def update_q_function(self, state, action, reward, next_state, done):
        """Standard Q-learning update."""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
        action_t = torch.LongTensor([action])
        reward_t = torch.FloatTensor([reward])
        done_t = torch.FloatTensor([done])

        # Current Q-value
        q_values = self.q_network(state_t)
        q_value = q_values.gather(1, action_t.unsqueeze(1))

        # Target Q-value
        with torch.no_grad():
            next_q_values = self.q_network(next_state_t)
            max_next_q = next_q_values.max(1)[0]
            target_q = reward_t + (1 - done_t) * 0.99 * max_next_q

        # Loss and update
        loss = F.mse_loss(q_value.squeeze(), target_q)
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

    def update_world_model(self, state, action, reward, next_state):
        """Update world model on single transition."""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        action_t = F.one_hot(torch.LongTensor([action]), self.action_dim).float()
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
        reward_t = torch.FloatTensor([reward]).unsqueeze(0)

        # Predict next state and reward
        pred_next_state, pred_reward = self.world_model(state_t, action_t)

        # Loss and update
        dynamics_loss = F.mse_loss(pred_next_state, next_state_t)
        reward_loss = F.mse_loss(pred_reward, reward_t)
        loss = dynamics_loss + reward_loss

        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()

    def planning_step(self):
        """
        Generate one synthetic experience and update Q-function.
        """
        # Sample random real transition
        if len(self.replay_buffer) < 1:
            return

        state, _, _, _, _ = self.replay_buffer.sample(1)
        state = state[0]  # Single state

        # Sample random action
        action = np.random.randint(self.action_dim)
        action_onehot = F.one_hot(torch.LongTensor([action]), self.action_dim).float()

        # Predict next state and reward using model
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            next_state, reward = self.world_model(state_t, action_onehot)
            next_state = next_state.numpy()[0]
            reward = reward.item()

        # Update Q-function on synthetic data
        self.update_q_function(state, action, reward, next_state, done=False)
```

**Key idea:** Every real transition generates n synthetic transitions for learning!

---

## Part 4: MBPO (Model-Based Policy Optimization)

MBPO (2019) combines model-based with state-of-the-art model-free (SAC):

1. Train world model on real data
2. Generate short synthetic rollouts from model
3. Train SAC on mixture of real + synthetic data

### Why Short Rollouts?

**Problem:** Model errors compound exponentially:
```
1 step error: ε
2 step error: ε + ε²
10 step error: ε^10  # Explosion!
```

**Solution:** Only simulate short horizons (k=1-5 steps)

### MBPO Implementation

```python
class MBPOAgent:
    """
    Model-Based Policy Optimization

    Combines world model with SAC for sample-efficient learning.
    """

    def __init__(self, state_dim, action_dim, max_action):
        # World model (ensemble for uncertainty)
        self.ensemble_size = 5
        self.world_models = [
            WorldModel(state_dim, action_dim)
            for _ in range(self.ensemble_size)
        ]
        self.model_optimizers = [
            torch.optim.Adam(model.parameters(), lr=1e-3)
            for model in self.world_models
        ]

        # SAC agent (from Lecture 07)
        self.sac = SAC(state_dim, action_dim, max_action)

        # Replay buffers
        self.real_buffer = ReplayBuffer(capacity=1000000)
        self.model_buffer = ReplayBuffer(capacity=1000000)

        self.rollout_length = 1  # Start with 1-step rollouts
        self.max_rollout_length = 5

    def train_world_model(self, batch_size=256, epochs=100):
        """Train ensemble of world models on real data."""
        for model, optimizer in zip(self.world_models, self.model_optimizers):
            for epoch in range(epochs):
                # Sample batch
                states, actions, rewards, next_states, _ = \
                    self.real_buffer.sample(batch_size)

                # Predict
                pred_next_states, pred_rewards = model(states, actions)

                # Loss
                dynamics_loss = F.mse_loss(pred_next_states, next_states)
                reward_loss = F.mse_loss(pred_rewards, rewards)
                loss = dynamics_loss + reward_loss

                # Update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def generate_synthetic_data(self, n_samples=10000, rollout_length=5):
        """
        Generate synthetic rollouts from learned model.

        Branched rollouts: Start from real states, simulate forward.
        """
        # Sample initial states from real buffer
        initial_states, _, _, _, _ = self.real_buffer.sample(n_samples)

        for i in range(n_samples):
            state = initial_states[i]

            # Simulate rollout of length k
            for step in range(rollout_length):
                # Select action from current policy
                action = self.sac.select_action(state.cpu().numpy(), evaluate=False)
                action_t = torch.FloatTensor(action).unsqueeze(0)
                state_t = state.unsqueeze(0)

                # Predict next state using random ensemble member
                model = np.random.choice(self.world_models)
                with torch.no_grad():
                    next_state, reward = model(state_t, action_t)

                # Store synthetic transition
                self.model_buffer.add(
                    state.cpu().numpy(),
                    action,
                    reward.item(),
                    next_state[0].cpu().numpy(),
                    False  # Assume not done (model doesn't predict termination well)
                )

                state = next_state[0]

    def train_policy(self, batch_size=256, gradient_steps=1):
        """
        Train SAC on mixture of real and synthetic data.
        """
        for _ in range(gradient_steps):
            # Sample half real, half synthetic
            real_batch = self.real_buffer.sample(batch_size // 2)
            model_batch = self.model_buffer.sample(batch_size // 2)

            # Combine batches
            combined_batch = self.combine_batches(real_batch, model_batch)

            # Update SAC
            self.sac.train_on_batch(combined_batch)

    def train_step(self):
        """
        One MBPO training iteration:
        1. Collect real data
        2. Train world model
        3. Generate synthetic data
        4. Train policy on mixed data
        """
        # 1. Collect real data (e.g., 1000 steps)
        self.collect_real_data(n_steps=1000)

        # 2. Train world model
        if len(self.real_buffer) > 5000:  # Wait for enough data
            self.train_world_model(epochs=100)

        # 3. Generate synthetic data
        if len(self.real_buffer) > 5000:
            self.generate_synthetic_data(
                n_samples=10000,
                rollout_length=self.rollout_length
            )

            # Gradually increase rollout length
            self.rollout_length = min(
                self.rollout_length + 1,
                self.max_rollout_length
            )

        # 4. Train policy
        self.train_policy(batch_size=256, gradient_steps=50)

    def collect_real_data(self, n_steps):
        """Collect data from real environment."""
        for _ in range(n_steps):
            action = self.sac.select_action(self.state, evaluate=False)
            next_state, reward, done, _ = self.env.step(action)

            self.real_buffer.add(self.state, action, reward, next_state, done)

            self.state = next_state
            if done:
                self.state = self.env.reset()
```

**MBPO achieves SAC-level performance with 10-20x less real data!**

---

## Part 5: Dreamer (World Models for Atari)

Dreamer (Hafner et al., 2020) learns latent world models:

### The Idea

Instead of predicting pixels:
```
s_{t+1} = f(s_t, a_t)  # s is 84x84x3 pixels, huge!
```

Learn latent representation:
```
z_{t+1} = f(z_t, a_t)  # z is compact latent state (256-dim)
```

### Dreamer Architecture

```python
class Dreamer:
    """
    Simplified Dreamer architecture.

    Components:
    1. Encoder: pixels → latent state
    2. Dynamics: latent transition model
    3. Decoder: latent → reconstructed pixels
    4. Reward predictor
    5. Actor-critic (learned in latent space)
    """

    def __init__(self, obs_shape, action_dim, latent_dim=256):
        # Encoder: pixels → latent
        self.encoder = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, latent_dim)
        )

        # Latent dynamics: (z_t, a_t) → z_{t+1}
        self.dynamics = nn.GRUCell(latent_dim + action_dim, latent_dim)

        # Decoder: latent → pixels
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, obs_shape[0], kernel_size=4, stride=2),
            nn.Sigmoid()
        )

        # Reward predictor
        self.reward_predictor = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Actor-critic (operates in latent space)
        self.actor = Actor(latent_dim, action_dim)
        self.critic = Critic(latent_dim)

    def imagine_trajectory(self, initial_latent, horizon=15):
        """
        Imagine trajectory in latent space using learned dynamics.
        """
        latents = [initial_latent]
        actions = []
        rewards = []

        latent = initial_latent

        for _ in range(horizon):
            # Select action from actor
            action = self.actor(latent)

            # Predict next latent using dynamics model
            latent_action = torch.cat([latent, action], dim=-1)
            next_latent = self.dynamics(latent_action, latent)

            # Predict reward
            reward = self.reward_predictor(next_latent)

            latents.append(next_latent)
            actions.append(action)
            rewards.append(reward)

            latent = next_latent

        return latents, actions, rewards

    def train_world_model(self, trajectory_batch):
        """Train encoder, dynamics, decoder, reward predictor."""
        # Encode observations
        latents = [self.encoder(obs) for obs in trajectory_batch.observations]

        # Compute losses
        reconstruction_loss = 0
        dynamics_loss = 0
        reward_loss = 0

        for t in range(len(latents) - 1):
            # Dynamics: predict next latent
            pred_next_latent = self.dynamics(
                torch.cat([latents[t], trajectory_batch.actions[t]], dim=-1),
                latents[t]
            )
            dynamics_loss += F.mse_loss(pred_next_latent, latents[t+1].detach())

            # Reconstruction: decode latent back to pixels
            recon_obs = self.decoder(latents[t])
            reconstruction_loss += F.mse_loss(recon_obs, trajectory_batch.observations[t])

            # Reward prediction
            pred_reward = self.reward_predictor(latents[t])
            reward_loss += F.mse_loss(pred_reward, trajectory_batch.rewards[t])

        total_loss = reconstruction_loss + dynamics_loss + reward_loss
        # ... optimizer update ...

    def train_policy(self, initial_latents, horizon=15):
        """
        Train actor-critic by imagining trajectories.
        """
        # Imagine trajectories from initial latents
        latents, actions, rewards = self.imagine_trajectory(initial_latents, horizon)

        # Compute returns
        returns = compute_lambda_returns(rewards, values, gamma=0.99, lambda_=0.95)

        # Update actor (policy gradient)
        # Update critic (value learning)
        # ... (similar to A2C/PPO) ...
```

**Dreamer can match or exceed model-free methods on Atari with 10-50x less data!**

---

## Part 6: Gotchas from Real Implementation

### Gotcha 1: Model Errors Compound

**Problem:** Small prediction errors grow exponentially over long rollouts.

**Solution:** Use short rollouts (k=1-5) or model uncertainty.

### Gotcha 2: Model Overfitting

**Problem:** Model memorizes training data, fails on new states.

**Solution:** Use ensemble, add regularization, collect diverse data.

### Gotcha 3: Termination Prediction

**Problem:** Hard to predict when episode ends.

**Solution:** Don't predict termination, or use separate classifier.

### Gotcha 4: Stochastic Environments

**Problem:** Deterministic model can't capture randomness.

**Solution:** Use probabilistic models (predict distribution, not point estimate).

### Gotcha 5: Computational Cost

**Problem:** Training world model + policy is expensive.

**Solution:** Train model less frequently, use simpler architectures.

---

## Part 7: When to Use Model-Based RL

**Use model-based when:**
1. Limited real-world data (robotics, autonomous vehicles)
2. Environment is expensive (hardware, time)
3. Transfer to new tasks
4. Need interpretability (inspect learned model)

**Don't use model-based when:**
1. Environment is cheap (simulators like Atari)
2. Dynamics are very complex (hard to model)
3. Want simplicity (model-free is simpler)
4. Have unlimited data

---

## Summary

1. **Model-based RL learns environment dynamics** to generate synthetic data or plan
2. **Dyna: Combines direct RL + model learning + planning**
3. **MBPO: Short rollouts + SAC** for sample-efficient continuous control
4. **Dreamer: Latent world models** for high-dimensional observations
5. **Key challenge: Model errors compound**
6. **Solution: Short rollouts, ensembles, uncertainty quantification**
7. **Trade-off: Sample efficiency vs computational cost**

---

## What's Next?

You've now learned the full RL spectrum:
- **Lectures 01-02:** Foundations (MDPs, Policy Gradients)
- **Lecture 03:** Value-based (DQN)
- **Lectures 04-06:** Actor-Critic (A2C, TRPO, PPO)
- **Lecture 07:** Off-policy (SAC, TD3)
- **Lecture 08:** Model-based (Dyna, MBPO, Dreamer)
- **Lectures 09-13:** Applications to LLMs and Code

**Next steps:** Dive deeper into specific applications, implement projects, read latest papers!

---

## Paper Trail

1. **Dyna (1990):** "Integrated Architectures for Learning, Planning, and Reacting" - Sutton
2. **World Models (2018):** "World Models" - Ha & Schmidhuber
3. **MBPO (2019):** "When to Trust Your Model: Model-Based Policy Optimization" - Janner et al., UC Berkeley
4. **Dreamer (2020):** "Dream to Control: Learning Behaviors by Latent Imagination" - Hafner et al., DeepMind
5. **DreamerV2 (2021):** "Mastering Atari with Discrete World Models" - Hafner et al.
6. **MuZero (2020):** "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" - DeepMind

All in `/Modern-RL-Research/RLHF-and-Alignment/PAPERS.md`

---

## Exercise for Yourself

Start with Dyna-Q:
1. Implement tabular Dyna-Q on GridWorld
2. Compare to vanilla Q-learning
3. Vary n_planning_steps (0, 5, 10, 50)
4. See how sample efficiency improves

Then try MBPO:
1. Implement simple world model on CartPole
2. Generate synthetic rollouts
3. Train SAC on mixed data
4. Measure sample efficiency vs pure SAC

**You'll see dramatic sample efficiency gains with model-based methods!**

---

## Personal Note

I spent my PhD on model-based RL. The promise is exciting: 10-100x sample efficiency! But the reality is complex: model errors, overfitting, computational cost.

**My advice:**
- For research: Explore model-based methods
- For practical applications: Start with model-free (PPO/SAC), try model-based if you hit sample efficiency limits
- For robotics: Model-based is often necessary

The field is rapidly evolving. Recent work (Dreamer, MuZero) shows model-based can match or exceed model-free. **This is an exciting area to watch!**
