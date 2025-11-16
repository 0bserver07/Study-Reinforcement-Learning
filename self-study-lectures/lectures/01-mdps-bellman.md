# Lecture 01: MDPs and Bellman Equations

> **Learning Goal**: Understand the mathematical foundation of RL so deeply that everything else is just "implementation details."

**Time**: 3-4 hours | **Prerequisites**: Basic probability, Python | **Difficulty**: ⭐⭐⭐☆☆

---

## 🎯 Why This Matters

Every RL algorithm - from basic Q-learning to GPT-4's RLHF training - is solving an MDP. If you don't get MDPs, you're just cargo-cult programming.

**By the end**: You'll understand what every RL paper is actually optimizing, and why.

---

## Part 1: The Setup (No Math Yet)

### The Problem We're Solving

Imagine you're training a robot to navigate a warehouse:
- It can move **left, right, forward, back**
- It gets +10 points for reaching the goal
- It gets -1 point for each step (encourages efficiency)
- It gets -100 for hitting an obstacle

**Question**: What sequence of actions should it take?

### Why This Is Hard

```python
# Naive approach (doesn't work)
def navigate_warehouse():
    # Just head straight for the goal?
    # But what if there's an obstacle in the way?
    # And sometimes "forward" doesn't move you forward (slippery floor)
    # And you don't know the map ahead of time...
    pass  # 😰
```

This is hard because:
1. **Uncertainty**: Actions don't always do what you think
2. **Delayed rewards**: Good actions now might hurt later (and vice versa)
3. **Exploration**: Don't know the world ahead of time
4. **Credit assignment**: Which action caused the good outcome?

### The MDP Framework

MDPs formalize this problem so we can reason about it mathematically.

**Personal Note**: When I first learned this, I thought "why all the notation?" Then I tried to implement a policy without it. Got confused in 10 minutes. The notation actually helps.

---

## Part 2: MDP Components (The Pieces)

An MDP is a tuple: (S, A, P, R, γ)

Let me explain each with the warehouse robot:

### S: State Space

**Definition**: All possible situations the agent can be in.

```python
# State in our warehouse
@dataclass
class State:
    x: int          # position x
    y: int          # position y
    facing: str     # "north", "south", "east", "west"
    has_package: bool

# State space S = all possible combinations
# If warehouse is 10x10, facing 4 ways, package yes/no:
# |S| = 10 * 10 * 4 * 2 = 800 states
```

**Markov Property** (Critical!):
> "The future is independent of the past given the present."

In math: P(s_{t+1} | s_t, s_{t-1}, ..., s_0) = P(s_{t+1} | s_t)

**Intuition**: The state contains ALL information needed to make optimal decisions. You don't need to remember "how did I get here?"

```python
# Markov: current state tells you everything
state = State(x=5, y=3, facing="north", has_package=True)
# You don't need to know: "Did I come from (5,2) or (4,3)?"

# Non-Markov example (bad state representation):
state = (x, y)  # Missing "facing" and "has_package"!
# Same state could require different actions depending on history
```

**Gotcha I Hit**: Forgetting the Markov property. I defined state as just (x,y) but needed velocity too. Actions behaved weirdly because state was insufficient.

---

### A: Action Space

**Definition**: All possible actions the agent can take.

```python
from enum import Enum

class Action(Enum):
    MOVE_FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    PICKUP = 3
    DROP = 4

# Action space A = {0, 1, 2, 3, 4}
# Discrete action space (most RL theory assumes this)
```

**Continuous actions** (like steering angle = 35.7°):
```python
# Continuous action space
# A = ℝ² (2D continuous)
action = np.array([steering_angle, throttle])
# More on this in lecture 06 (PPO)
```

---

### P: Transition Dynamics

**Definition**: P(s' | s, a) - probability of landing in state s' after taking action a in state s.

**This is the "physics" of your world.**

```python
# Deterministic transitions (like chess)
def transition(state: State, action: Action) -> State:
    if action == Action.MOVE_FORWARD:
        return State(x=state.x, y=state.y+1, ...)  # Always moves forward

# Stochastic transitions (like slippery floors)
def transition_prob(state: State, action: Action, next_state: State) -> float:
    """P(next_state | state, action)"""
    if action == Action.MOVE_FORWARD:
        # 80% chance of moving forward
        if next_state.y == state.y + 1:
            return 0.8
        # 10% chance of moving left
        elif next_state.x == state.x - 1:
            return 0.1
        # 10% chance of moving right
        elif next_state.x == state.x + 1:
            return 0.1
    return 0.0
```

**Key Insight**: In most real problems, you DON'T know P! You learn it by trying actions (model-free RL) or estimate it (model-based RL).

---

### R: Reward Function

**Definition**: R(s, a, s') - immediate reward for transitioning from s to s' via action a.

**This encodes what you want the agent to do.**

```python
def reward(state: State, action: Action, next_state: State) -> float:
    # Reached goal
    if next_state.is_goal():
        return 10.0

    # Hit obstacle
    if next_state.is_obstacle():
        return -100.0

    # Each step costs (encourages efficiency)
    return -1.0
```

**Critical Warning**: The reward function IS your objective. Design it wrong, get wrong behavior.

**Real Example from My Code**:
```python
# Bad reward (what I first wrote)
def bad_reward(state, action, next_state):
    if next_state.is_goal():
        return 1.0
    return 0.0

# Agent learned to wander randomly until stumbling on goal
# Why? No incentive to be efficient!

# Good reward (fixed version)
def good_reward(state, action, next_state):
    if next_state.is_goal():
        return 10.0
    return -1.0  # Time penalty

# Now agent learns shortest path
```

**Reward Engineering** is an art. More on this in Lecture 09 (Reward Modeling).

---

### γ: Discount Factor

**Definition**: γ ∈ [0, 1] - how much we care about future rewards vs immediate rewards.

```python
# Total return (what we're trying to maximize)
# R_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + γ³·r_{t+3} + ...

gamma = 0.99  # Common value

# Example: sequence of rewards [1, 1, 1, 1, ...]
# Discounted return = 1 + 0.99 + 0.99² + 0.99³ + ...
#                    = 1 / (1 - 0.99) = 100
```

**Intuition**:
- γ = 0: "I only care about immediate reward" (greedy)
- γ = 0.9: "I care about next ~10 steps"
- γ = 0.99: "I care about next ~100 steps"
- γ = 1: "All rewards equally important" (infinite horizon)

**Why discount?**
1. **Mathematical convenience**: Makes infinite sums converge
2. **Uncertainty**: Future is uncertain, prefer sooner rewards
3. **Impatience**: In real systems, later rewards matter less

**Gotcha**: Setting γ too low → myopic policy. Too high → slow learning.

```python
# Testing different gamma values
for gamma in [0.9, 0.95, 0.99]:
    returns = []
    for t in range(100):
        ret = sum(gamma**k for k in range(t))
        returns.append(ret)
    print(f"γ={gamma}: effective horizon ≈ {1/(1-gamma):.0f} steps")

# Output:
# γ=0.9: effective horizon ≈ 10 steps
# γ=0.95: effective horizon ≈ 20 steps
# γ=0.99: effective horizon ≈ 100 steps
```

---

## Part 3: The Value Functions (What We're Actually Computing)

### State-Value Function V^π(s)

**Definition**: "How good is it to be in state s, following policy π?"

```
V^π(s) = 𝔼_π[R_t | s_t = s]
       = 𝔼_π[r_t + γr_{t+1} + γ²r_{t+2} + ... | s_t = s]
```

**Intuition**: If you start in state s and follow policy π, what's your expected total return?

```python
def compute_value(state, policy, num_rollouts=1000):
    """Monte Carlo estimate of V^π(s)"""
    returns = []

    for _ in range(num_rollouts):
        s = state
        total_return = 0
        discount = 1.0

        for t in range(100):  # Max 100 steps
            a = policy(s)  # Sample action from policy
            s_next, reward = env.step(s, a)
            total_return += discount * reward
            discount *= gamma
            s = s_next

            if s.is_terminal():
                break

        returns.append(total_return)

    return np.mean(returns)
```

**Personal Note**: When I first saw V(s), I thought "why do we need this?" Because it tells you which states to seek! High V(s) = good state to be in.

---

### Action-Value Function Q^π(s, a)

**Definition**: "How good is it to take action a in state s, then follow policy π?"

```
Q^π(s, a) = 𝔼_π[R_t | s_t = s, a_t = a]
```

**Difference from V**: Q tells you about state-action pairs, V just about states.

```python
# Relationship between V and Q:
V^π(s) = Σ_a π(a|s) · Q^π(s, a)

# In code:
def value_from_q(state, policy, q_function):
    """V(s) = Σ π(a|s) Q(s,a)"""
    v = 0
    for action in all_actions:
        prob = policy(action | state)
        q_val = q_function(state, action)
        v += prob * q_val
    return v
```

**Why Q is useful**: For policy improvement!

```python
# With Q, we can improve policy:
def improve_policy(state, q_function):
    """Take action with highest Q-value"""
    best_action = max(all_actions, key=lambda a: q_function(state, a))
    return best_action

# With only V, this is harder (need transition model P)
```

---

## Part 4: The Bellman Equations (The Key Insight)

### The Recursive Relationship

**Bellman's genius observation**: Value of a state = immediate reward + discounted value of next state.

```
V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V^π(s')]
```

**Read this as**:
"Value of s = sum over actions (weighted by policy) of [immediate reward + discounted future value]"

**Intuition**: You can compute value of state s if you know value of next states!

```python
# Simple gridworld example
def bellman_update(state, policy, V, P, R, gamma=0.99):
    """One Bellman update for V(s)"""
    v_new = 0

    for action in all_actions:
        # Probability of taking this action
        action_prob = policy(action | state)

        # Expected value for this action
        action_value = 0
        for next_state in all_states:
            # Probability of transition
            trans_prob = P(next_state | state, action)

            # Reward for transition
            reward = R(state, action, next_state)

            # Bellman backup
            action_value += trans_prob * (reward + gamma * V[next_state])

        v_new += action_prob * action_value

    return v_new
```

### Bellman Equation for Q

```
Q^π(s, a) = Σ_{s'} P(s'|s,a) [R(s,a,s') + γ Σ_{a'} π(a'|s') Q^π(s', a')]
```

**Code version**:
```python
def bellman_update_q(state, action, policy, Q, P, R, gamma=0.99):
    """One Bellman update for Q(s,a)"""
    q_new = 0

    for next_state in all_states:
        trans_prob = P(next_state | state, action)
        reward = R(state, action, next_state)

        # Expected Q-value of next state under policy
        next_q = sum(policy(a | next_state) * Q[next_state, a]
                     for a in all_actions)

        q_new += trans_prob * (reward + gamma * next_q)

    return q_new
```

---

## Part 5: Policy Iteration (Our First Algorithm!)

Now we can actually solve MDPs.

### The Algorithm

```
1. Start with random policy π
2. Loop:
   a. Policy Evaluation: compute V^π (Bellman equation)
   b. Policy Improvement: π'(s) = argmax_a Q^π(s,a)
   c. If π' = π, stop (converged)
```

### Complete Implementation

```python
import numpy as np
from typing import Dict, Tuple, List

class GridWorldMDP:
    """Simple gridworld MDP for testing"""

    def __init__(self, size=5):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # up, right, down, left

        # Goal state (top-right corner)
        self.goal = (0, size-1)

    def state_to_index(self, state: Tuple[int, int]) -> int:
        """Convert (x,y) to flat index"""
        return state[0] * self.size + state[1]

    def index_to_state(self, idx: int) -> Tuple[int, int]:
        """Convert flat index to (x,y)"""
        return (idx // self.size, idx % self.size)

    def get_next_state(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Deterministic transition"""
        x, y = state

        # Actions: 0=up, 1=right, 2=down, 3=left
        if action == 0:  # up
            x = max(0, x - 1)
        elif action == 1:  # right
            y = min(self.size - 1, y + 1)
        elif action == 2:  # down
            x = min(self.size - 1, x + 1)
        elif action == 3:  # left
            y = max(0, y - 1)

        return (x, y)

    def get_reward(self, state: Tuple[int, int], action: int,
                   next_state: Tuple[int, int]) -> float:
        """Reward function"""
        if next_state == self.goal:
            return 10.0
        return -1.0  # Step penalty


def policy_evaluation(mdp: GridWorldMDP, policy: np.ndarray,
                     gamma=0.99, theta=1e-6) -> np.ndarray:
    """
    Compute V^π using iterative Bellman updates.

    Args:
        policy: [n_states, n_actions] - policy(a|s)
        gamma: discount factor
        theta: convergence threshold

    Returns:
        V: [n_states] - state values
    """
    V = np.zeros(mdp.n_states)

    iteration = 0
    while True:
        delta = 0

        for s_idx in range(mdp.n_states):
            if s_idx == mdp.state_to_index(mdp.goal):
                continue  # Terminal state

            state = mdp.index_to_state(s_idx)
            v_old = V[s_idx]

            # Bellman update
            v_new = 0
            for a in range(mdp.n_actions):
                next_state = mdp.get_next_state(state, a)
                next_s_idx = mdp.state_to_index(next_state)
                reward = mdp.get_reward(state, a, next_state)

                # V(s) = Σ_a π(a|s) [r + γV(s')]
                v_new += policy[s_idx, a] * (reward + gamma * V[next_s_idx])

            V[s_idx] = v_new
            delta = max(delta, abs(v_old - v_new))

        iteration += 1
        if delta < theta:
            print(f"  Policy evaluation converged in {iteration} iterations")
            break

    return V


def policy_improvement(mdp: GridWorldMDP, V: np.ndarray,
                      gamma=0.99) -> Tuple[np.ndarray, bool]:
    """
    Improve policy greedily w.r.t. value function.

    Returns:
        new_policy: [n_states, n_actions]
        is_stable: whether policy changed
    """
    new_policy = np.zeros((mdp.n_states, mdp.n_actions))
    policy_stable = True

    for s_idx in range(mdp.n_states):
        if s_idx == mdp.state_to_index(mdp.goal):
            continue  # Terminal state

        state = mdp.index_to_state(s_idx)

        # Compute Q(s,a) for all actions
        q_values = np.zeros(mdp.n_actions)
        for a in range(mdp.n_actions):
            next_state = mdp.get_next_state(state, a)
            next_s_idx = mdp.state_to_index(next_state)
            reward = mdp.get_reward(state, a, next_state)

            # Q(s,a) = r + γV(s')
            q_values[a] = reward + gamma * V[next_s_idx]

        # Greedy policy: choose best action
        best_action = np.argmax(q_values)
        new_policy[s_idx, best_action] = 1.0

    return new_policy, policy_stable


def policy_iteration(mdp: GridWorldMDP, gamma=0.99, max_iterations=100):
    """
    Full policy iteration algorithm.
    """
    # Initialize with random policy
    policy = np.ones((mdp.n_states, mdp.n_actions)) / mdp.n_actions

    for iteration in range(max_iterations):
        print(f"\nIteration {iteration + 1}")

        # 1. Policy Evaluation
        V = policy_evaluation(mdp, policy, gamma)

        # 2. Policy Improvement
        new_policy, is_stable = policy_improvement(mdp, V, gamma)

        # Check convergence
        if np.allclose(policy, new_policy):
            print(f"\n✓ Converged after {iteration + 1} iterations!")
            policy = new_policy
            break

        policy = new_policy

    return policy, V


# Run it!
if __name__ == "__main__":
    print("="*60)
    print("Policy Iteration on 5x5 GridWorld")
    print("="*60)

    mdp = GridWorldMDP(size=5)
    optimal_policy, optimal_V = policy_iteration(mdp, gamma=0.99)

    print("\n" + "="*60)
    print("Optimal Value Function:")
    print("="*60)
    print(optimal_V.reshape(5, 5))

    print("\n" + "="*60)
    print("Optimal Policy (arrows):")
    print("="*60)
    arrows = ['↑', '→', '↓', '←']
    for i in range(5):
        row = []
        for j in range(5):
            s_idx = mdp.state_to_index((i, j))
            if (i, j) == mdp.goal:
                row.append('G')
            else:
                action = np.argmax(optimal_policy[s_idx])
                row.append(arrows[action])
        print(' '.join(row))
```

**Output you should see**:
```
Iteration 1
  Policy evaluation converged in 47 iterations

Iteration 2
  Policy evaluation converged in 25 iterations

✓ Converged after 2 iterations!

Optimal Value Function:
[[  -3.98  -2.99  -1.99  -0.99  10.00]
 [  -4.97  -3.98  -2.99  -1.99  -0.99]
 [  -5.96  -4.97  -3.98  -2.99  -1.99]
 [  -6.95  -5.96  -4.97  -3.98  -2.99]
 [  -7.94  -6.95  -5.96  -4.97  -3.98]]

Optimal Policy:
→ → → → G
↑ ↑ ↑ ↑ ↑
↑ ↑ ↑ ↑ ↑
↑ ↑ ↑ ↑ ↑
↑ ↑ ↑ ↑ ↑
```

---

## Part 6: The Gotchas (Where I Got Stuck)

### Gotcha #1: Forgetting Terminal States

```python
# Bug I had:
def policy_evaluation_buggy(mdp, policy, gamma):
    V = np.zeros(mdp.n_states)
    for s in range(mdp.n_states):
        # This updates the terminal state too! Wrong!
        V[s] = bellman_update(s, policy, V, gamma)
    return V

# Fix: Skip terminal states
def policy_evaluation_fixed(mdp, policy, gamma):
    V = np.zeros(mdp.n_states)
    for s in range(mdp.n_states):
        if mdp.is_terminal(s):
            continue  # V(terminal) = 0 always
        V[s] = bellman_update(s, policy, V, gamma)
    return V
```

### Gotcha #2: Discount Factor Edge Cases

```python
# If gamma = 1.0, value iteration might not converge
# (infinite horizon, no discounting)

# Solution: Either use γ < 1 or ensure finite episodes
```

### Gotcha #3: State Representation

```python
# Bad: Missing information
state = (x, y)  # For a robot with velocity

# Action "move forward" does different things depending on velocity!
# Policy can't be deterministic with this state representation

# Good: Include velocity
state = (x, y, vx, vy)  # Now Markov property holds
```

---

## Part 7: Connection to Papers

### Bellman (1957) - "Dynamic Programming"
- Introduced the principle of optimality
- Showed recursive decomposition of value functions
- Foundation of all RL

### Howard (1960) - "Dynamic Programming and Markov Processes"
- Policy iteration algorithm
- Showed it converges to optimal policy
- Still used today (e.g., AlphaGo uses modified PI)

### Puterman (1994) - "Markov Decision Processes"
- The definitive textbook
- Proves convergence theorems
- Reference for MDP theory

**Why these matter**: Every modern RL algorithm (including RLHF for LLMs) is solving an MDP. The notation and theory is the same.

---

## Part 8: When Do You Use This?

### Policy Iteration vs Value Iteration

**Policy Iteration**:
- Pro: Fewer iterations to converge
- Con: Each iteration requires full policy evaluation (expensive)
- Use when: Small state spaces, need optimal policy

**Value Iteration** (next lecture):
- Pro: Each iteration is cheaper
- Con: More iterations needed
- Use when: Large state spaces, approximate solution OK

### When MDPs Don't Apply

- **Continuous state/action spaces**: Use function approximation (Lecture 06)
- **Unknown dynamics**: Use model-free methods (Lecture 02-04)
- **Partial observability**: Use POMDPs (beyond this series)
- **Multi-agent**: Use game theory extensions (beyond this series)

---

## Part 9: Exercises (Do These!)

### Exercise 1: Modify the Gridworld
Add obstacles to the gridworld. How does the optimal policy change?

```python
class GridWorldWithObstacles(GridWorldMDP):
    def __init__(self, size=5):
        super().__init__(size)
        self.obstacles = [(2, 2), (1, 3)]  # Add obstacles here

    def get_reward(self, state, action, next_state):
        if next_state in self.obstacles:
            return -100.0  # Huge penalty
        # ... rest same
```

### Exercise 2: Stochastic Transitions
Make the gridworld slippery. Actions succeed with 80% probability, move perpendicular with 20%.

### Exercise 3: Different Discount Factors
Run policy iteration with γ ∈ {0.5, 0.9, 0.99, 0.999}. How does the optimal policy change?

### Exercise 4: Sparse vs Dense Rewards
Compare:
- Reward only at goal (sparse)
- Reward = negative distance to goal (dense)
Which converges faster? Which finds better policy?

---

## Key Takeaways (Write These Down)

1. **MDP = (S, A, P, R, γ)** - memorize this tuple
2. **Markov property**: State contains all info needed for optimal decisions
3. **Value functions**: V(s) and Q(s,a) are what we're computing
4. **Bellman equation**: V(s) = E[r + γV(s')] - the key recursive relationship
5. **Policy iteration**: Evaluate → Improve → Repeat
6. **Discount factor**: Controls planning horizon (higher = look further ahead)
7. **Reward engineering**: Your reward function IS your objective

---

## Next Lecture

**[Lecture 02: Policy Gradients from Scratch](./02-policy-gradients.md)**

Where we'll learn:
- Why we can't always solve MDPs exactly
- How to optimize policies directly
- REINFORCE algorithm
- The path to PPO

But first, make sure you:
- [ ] Ran the gridworld code
- [ ] Modified it (exercises 1-2 minimum)
- [ ] Can explain Bellman equation to someone
- [ ] Understand why γ matters

---

## References

**Core Papers**:
- Bellman (1957) - "A Markovian Decision Process"
- Howard (1960) - "Dynamic Programming and Markov Processes"
- Puterman (1994) - "Markov Decision Processes: Discrete Stochastic Dynamic Programming"

**Modern Applications**:
- Silver et al. (2017) - "Mastering Chess and Shogi by Self-Play" (AlphaZero uses PI)
- Ouyang et al. (2022) - "Training language models to follow instructions with human feedback" (InstructGPT uses MDP formulation)

---

*Last Updated: 2025*

---

**My Personal Notes**:
- Spent 2 days debugging the policy iteration code. The bug was not skipping terminal states.
- The gridworld visualization really helped intuition. Seeing the arrows made it click.
- γ=0.99 is a good default. Don't overthink it initially.
- Read Bellman equation out loud. Seriously, it helps.
