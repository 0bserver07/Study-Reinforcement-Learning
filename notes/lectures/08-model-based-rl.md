<!-- status: unreviewed | last-reviewed: never -->

# Lecture 08: Model-based reinforcement learning

_Unreviewed: no one has checked this end to end. Treat the math, code, and citations as unverified._

**Duration:** ~90 minutes
**Prerequisites:** Lecture 03 (DQN), Lecture 06 (PPO), Lecture 07 (SAC)
**Goal:** Learn world models, implement Dyna and MBPO, understand when to use model-based RL

---

## Why model-based RL

All previous lectures used **model-free RL**: learn a policy or value function directly from environment interaction. This is sample-inefficient (it can require millions of steps) and the agent builds no internal understanding of the environment.

**Model-based RL** takes a different approach: learn a model of the environment dynamics, then use that model for planning or for generating synthetic training data. This can be 10–100x more sample-efficient.

When a self-driving car trains, it can't crash millions of times. When a robot learns to walk, hardware is expensive. Model-based RL enables learning with limited real-world data.

Recent example: MuZero (Schrittwieser et al., Nature 2020) masters Atari, Chess, Go, and Shogi with less data than AlphaZero by planning with a learned model.

---

## Part 1: Model-free vs model-based

### Model-free

```python
# Direct interaction with environment
for episode in range(n_episodes):
    obs, info = env.reset()          # Gymnasium API: returns (obs, info)
    done = False
    while not done:
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)  # 5-tuple
        done = terminated or truncated
        update_policy(obs, action, reward, obs)
```

**Pros:** Simple, works for complex dynamics.
**Cons:** Sample-inefficient, no internal model of environment.

### Model-based

```python
# Learn a model of environment
model = WorldModel()  # Predicts next state and reward

# Collect data
real_data = collect_from_environment(random_policy)

# Learn model
model.fit(real_data)  # (s_t, a_t) → (s_{t+1}, r_t)

# Use model for planning or data generation
for episode in range(n_episodes):
    # Option 1: Pure planning (no policy learning)
    action = plan_with_model(model, state)

    # Option 2: Generate synthetic data
    synthetic_data = model.generate_rollouts()
    update_policy(synthetic_data)  # Learn from imagined experience
```

**Pros:** Sample-efficient, potentially transferable, more interpretable.
**Cons:** Model errors compound; harder to implement correctly.

---

## Part 2: World models (learning dynamics)

### The dynamics model

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

### Deterministic world model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class WorldModel(nn.Module):
    """
    Learn deterministic dynamics: (s, a) → (s', r)

    Predicts delta: s' = s + Δs
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
            nn.Linear(hidden_dim, 1)
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

        reward = self.reward(sa)

        return next_state, reward

    def train_model(self, replay_buffer, batch_size=256, epochs=100):
        """Train world model on real data from replay buffer."""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        for epoch in range(epochs):
            states, actions, rewards, next_states, dones = \
                replay_buffer.sample(batch_size)

            pred_next_states, pred_rewards = self.forward(states, actions)

            dynamics_loss = F.mse_loss(pred_next_states, next_states)
            reward_loss = F.mse_loss(pred_rewards, rewards)

            total_loss = dynamics_loss + reward_loss

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
            action = policy(state)

            with torch.no_grad():
                next_state, reward = self.forward(state, action)

            states.append(next_state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        return states, actions, rewards
```

---

## Part 3: Dyna architecture

Dyna (Sutton, 1990) combines:
1. **Direct RL:** learn from real environment interaction
2. **Model learning:** fit a world model to real data
3. **Planning:** generate synthetic experience from the model

### Dyna-Q

```python
import numpy as np

# Note: QNetwork and ReplayBuffer are assumed to be defined elsewhere
# (e.g., from Lecture 03).

class DynaQAgent:
    """
    Dyna-Q: combines Q-learning with world model planning.

    Algorithm:
    1. Take real action, observe transition
    2. Update Q-function (like normal Q-learning)
    3. Update world model
    4. Generate n synthetic experiences from model
    5. Update Q-function on synthetic data
    """

    def __init__(self, state_dim, action_dim, n_planning_steps=10):
        # Q-network (like DQN, see Lecture 03)
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
        """Update agent with one real transition."""
        self.replay_buffer.add(state, action, reward, next_state, done)

        # 1. Direct RL: update Q-function on real data
        self.update_q_function(state, action, reward, next_state, done)

        # 2. Model learning: update world model on real data
        self.update_world_model(state, action, reward, next_state)

        # 3. Planning: generate synthetic experiences and update Q-function
        for _ in range(self.n_planning_steps):
            self.planning_step()

    def update_q_function(self, state, action, reward, next_state, done):
        """Standard Q-learning update."""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
        action_t = torch.LongTensor([action])
        reward_t = torch.FloatTensor([reward])
        done_t = torch.FloatTensor([done])

        q_values = self.q_network(state_t)
        q_value = q_values.gather(1, action_t.unsqueeze(1))

        with torch.no_grad():
            next_q_values = self.q_network(next_state_t)
            max_next_q = next_q_values.max(1)[0]
            target_q = reward_t + (1 - done_t) * 0.99 * max_next_q

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

        pred_next_state, pred_reward = self.world_model(state_t, action_t)

        dynamics_loss = F.mse_loss(pred_next_state, next_state_t)
        reward_loss = F.mse_loss(pred_reward, reward_t)
        loss = dynamics_loss + reward_loss

        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()

    def planning_step(self):
        """Generate one synthetic experience and update Q-function."""
        if len(self.replay_buffer) < 1:
            return

        state, _, _, _, _ = self.replay_buffer.sample(1)
        state = state[0]

        action = np.random.randint(self.action_dim)
        action_onehot = F.one_hot(torch.LongTensor([action]), self.action_dim).float()

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            next_state, reward = self.world_model(state_t, action_onehot)
            next_state = next_state.numpy()[0]
            reward = reward.item()

        self.update_q_function(state, action, reward, next_state, done=False)
```

Every real transition generates `n_planning_steps` synthetic transitions for the Q-function update. This is the core Dyna idea.

---

## Part 4: MBPO (model-based policy optimization)

MBPO (Janner et al., 2019) pairs a world model with SAC:

1. Train world model on real data
2. Generate short synthetic rollouts from the model
3. Train SAC on a mixture of real and synthetic data

### Why short rollouts?

Model errors compound over time:
```
1-step error:   ε
2-step error:   ε + ε²
10-step error:  ≈ ε^10  # can explode
```

Keeping rollout length to k=1–5 steps limits error accumulation.

### MBPO implementation

```python
import numpy as np

# Note: SAC is defined in Lecture 07. ReplayBuffer is defined in Lecture 03.

class MBPOAgent:
    """
    Model-Based Policy Optimization.

    Combines world model with SAC for sample-efficient continuous control.
    """

    def __init__(self, state_dim, action_dim, max_action):
        # Ensemble of world models for uncertainty estimation
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

        self.real_buffer = ReplayBuffer(capacity=1000000)
        self.model_buffer = ReplayBuffer(capacity=1000000)

        self.rollout_length = 1
        self.max_rollout_length = 5

    def train_world_model(self, batch_size=256, epochs=100):
        """Train ensemble of world models on real data."""
        for model, optimizer in zip(self.world_models, self.model_optimizers):
            for epoch in range(epochs):
                states, actions, rewards, next_states, _ = \
                    self.real_buffer.sample(batch_size)

                pred_next_states, pred_rewards = model(states, actions)

                dynamics_loss = F.mse_loss(pred_next_states, next_states)
                reward_loss = F.mse_loss(pred_rewards, rewards)
                loss = dynamics_loss + reward_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def generate_synthetic_data(self, n_samples=10000, rollout_length=5):
        """
        Generate synthetic rollouts from learned model.

        Branched rollouts: start from real states, simulate forward.
        """
        initial_states, _, _, _, _ = self.real_buffer.sample(n_samples)

        for i in range(n_samples):
            state = initial_states[i]

            for step in range(rollout_length):
                action = self.sac.select_action(state.cpu().numpy(), evaluate=False)
                action_t = torch.FloatTensor(action).unsqueeze(0)
                state_t = state.unsqueeze(0)

                # Pick a random ensemble member for diversity
                model = np.random.choice(self.world_models)
                with torch.no_grad():
                    next_state, reward = model(state_t, action_t)

                self.model_buffer.add(
                    state.cpu().numpy(),
                    action,
                    reward.item(),
                    next_state[0].cpu().numpy(),
                    False  # Termination prediction is unreliable; skip it
                )

                state = next_state[0]

    def train_policy(self, batch_size=256, gradient_steps=1):
        """Train SAC on a mixture of real and synthetic data."""
        for _ in range(gradient_steps):
            real_batch = self.real_buffer.sample(batch_size // 2)
            model_batch = self.model_buffer.sample(batch_size // 2)

            combined_batch = self.combine_batches(real_batch, model_batch)
            self.sac.train_on_batch(combined_batch)

    def train_step(self):
        """
        One MBPO training iteration:
        1. Collect real data
        2. Train world model
        3. Generate synthetic data
        4. Train policy on mixed data
        """
        self.collect_real_data(n_steps=1000)

        if len(self.real_buffer) > 5000:
            self.train_world_model(epochs=100)

        if len(self.real_buffer) > 5000:
            self.generate_synthetic_data(
                n_samples=10000,
                rollout_length=self.rollout_length
            )

            self.rollout_length = min(
                self.rollout_length + 1,
                self.max_rollout_length
            )

        self.train_policy(batch_size=256, gradient_steps=50)

    def collect_real_data(self, n_steps):
        """Collect data from real environment using Gymnasium API."""
        for _ in range(n_steps):
            action = self.sac.select_action(self.state, evaluate=False)
            # Gymnasium step() returns 5-tuple: (obs, reward, terminated, truncated, info)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            self.real_buffer.add(self.state, action, reward, next_state, done)

            self.state = next_state
            if done:
                self.state, _ = self.env.reset()  # Gymnasium reset() returns (obs, info)
```

MBPO reaches SAC-level performance with roughly 10–20x less real environment data.

---

## Part 5: Dreamer (latent world models)

Dreamer (Hafner et al., arXiv:1912.01603) learns a compact latent dynamics model instead of predicting raw pixels.

### The idea

Predicting next pixel observations is expensive:
```
s_{t+1} = f(s_t, a_t)  # s is 84×84×3 pixels
```

Instead, learn a latent representation and predict in that space:
```
z_{t+1} = f(z_t, a_t)  # z is a compact latent state (~256-dim)
```

### Dreamer architecture

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

        self.reward_predictor = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Actor and Critic are assumed defined elsewhere
        self.actor = Actor(latent_dim, action_dim)
        self.critic = Critic(latent_dim)

    def imagine_trajectory(self, initial_latent, horizon=15):
        """Imagine trajectory in latent space using learned dynamics."""
        latents = [initial_latent]
        actions = []
        rewards = []

        latent = initial_latent

        for _ in range(horizon):
            action = self.actor(latent)

            latent_action = torch.cat([latent, action], dim=-1)
            next_latent = self.dynamics(latent_action, latent)

            reward = self.reward_predictor(next_latent)

            latents.append(next_latent)
            actions.append(action)
            rewards.append(reward)

            latent = next_latent

        return latents, actions, rewards

    def train_world_model(self, trajectory_batch):
        """Train encoder, dynamics, decoder, reward predictor."""
        latents = [self.encoder(obs) for obs in trajectory_batch.observations]

        reconstruction_loss = 0
        dynamics_loss = 0
        reward_loss = 0

        for t in range(len(latents) - 1):
            pred_next_latent = self.dynamics(
                torch.cat([latents[t], trajectory_batch.actions[t]], dim=-1),
                latents[t]
            )
            dynamics_loss += F.mse_loss(pred_next_latent, latents[t+1].detach())

            recon_obs = self.decoder(latents[t])
            reconstruction_loss += F.mse_loss(recon_obs, trajectory_batch.observations[t])

            pred_reward = self.reward_predictor(latents[t])
            reward_loss += F.mse_loss(pred_reward, trajectory_batch.rewards[t])

        total_loss = reconstruction_loss + dynamics_loss + reward_loss
        # ... optimizer update ...

    def train_policy(self, initial_latents, horizon=15):
        """Train actor-critic by imagining trajectories."""
        # Note: `values` must be computed from self.critic before this call
        latents, actions, rewards = self.imagine_trajectory(initial_latents, horizon)

        returns = compute_lambda_returns(rewards, values, gamma=0.99, lambda_=0.95)

        # Update actor (policy gradient) and critic (value learning)
        # ... similar to A2C/PPO from Lecture 04/06 ...
```

Dreamer can match or exceed model-free baselines on visual control tasks with substantially less real environment data.

---

## Part 6: Common failure modes

**Model errors compound.** Small prediction errors grow over long rollouts. Use short rollouts (k=1–5) or ensemble-based uncertainty estimates.

**Model overfitting.** The model memorizes training transitions and fails on states not yet visited. Ensembles, regularization, and diverse data collection help.

**Termination prediction.** Episode termination is hard to model reliably. A common workaround: don't predict termination at all, or train a separate binary classifier.

**Stochastic environments.** A deterministic model can't capture randomness. Use probabilistic models that output a distribution rather than a point estimate.

**Computational cost.** Training a world model alongside a policy is expensive. Train the model less frequently, or use a simpler architecture if the dynamics are low-dimensional.

---

## Part 7: When to use model-based RL

**Good fit:**
- Limited real-world data (robotics, autonomous vehicles)
- Expensive environment interaction (hardware or time costs)
- Transfer to related tasks
- Need to inspect or explain learned dynamics

**Poor fit:**
- Cheap simulators where data is essentially free (classic Atari benchmarks)
- Highly complex or chaotic dynamics that are hard to model
- Simplicity is a priority (model-free is less moving parts)
- Unlimited environment interaction is available

---

## Recap

Model-based RL learns a dynamics model to generate synthetic data or plan. Dyna combines direct RL with model-based planning using interleaved real and imagined updates. MBPO uses short model rollouts with SAC for sample-efficient continuous control. Dreamer learns in latent space to handle high-dimensional observations. The key challenge throughout is compounding model error; the main mitigations are short rollouts, ensembles, and uncertainty-aware models. The tradeoff is sample efficiency against implementation complexity and computational cost.

---

## What's next

You've now covered the full RL spectrum:
- Lectures 01–02: Foundations (MDPs, policy gradients)
- Lecture 03: Value-based (DQN)
- Lectures 04–06: Actor-critic (A2C, TRPO, PPO)
- Lecture 07: Off-policy (SAC, TD3)
- Lecture 08: Model-based (Dyna, MBPO, Dreamer)
- Lectures 09–13: Applications to LLMs and code

---

## References

1. **Sutton (1990)**: "Dyna, an integrated architecture for learning, planning, and reacting." SIGART Bulletin 2(4), pp. 160–163. (Also: ICML 1990 proceedings as "Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming".)
2. **Ha & Schmidhuber (2018)**: "World Models." arXiv:1803.10122.
3. **Chua et al. (2018)**: "Deep reinforcement learning in a handful of trials using probabilistic dynamics models" (PETS). NeurIPS 2018. arXiv:1805.12114.
4. **Janner et al. (2019)**: "When to trust your model: model-based policy optimization" (MBPO). NeurIPS 2019. arXiv:1906.08253.
5. **Hafner et al. (2020)**: "Dream to control: learning behaviors by latent imagination" (Dreamer). ICLR 2020. arXiv:1912.01603.
6. **Hafner et al. (2021)**: "Mastering Atari with discrete world models" (DreamerV2). ICLR 2021. arXiv:2010.02193.
7. **Schrittwieser et al. (2020)**: "Mastering Atari, Go, chess and shogi by planning with a learned model" (MuZero). Nature 588, pp. 604–609. arXiv:1911.08265.

Relevant papers also collected in `../../reference/papers/RLHF-and-Alignment/PAPERS.md`.

---

## Exercises

Start with Dyna-Q:
1. Implement tabular Dyna-Q on GridWorld.
2. Compare to vanilla Q-learning.
3. Vary `n_planning_steps` (0, 5, 10, 50).
4. Measure how sample efficiency changes.

Then try MBPO:
1. Implement a simple world model on CartPole.
2. Generate synthetic rollouts.
3. Train SAC on mixed data.
4. Measure sample efficiency vs pure SAC.

---

## A note on model-based RL in practice

Model-based RL promises 10–100x sample efficiency, but delivering on that promise is harder than it looks. Model errors, overfitting, and computational overhead are real costs.

A reasonable heuristic: start with model-free (PPO or SAC). If you hit a hard sample efficiency wall, try model-based. For robotics with expensive hardware, model-based is often necessary from the start.

The field is moving quickly; Dreamer and MuZero show model-based can match or beat model-free on hard benchmarks. It's a productive area to watch.
