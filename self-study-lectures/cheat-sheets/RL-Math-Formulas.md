# Deep RL Mathematical Formulas - Quick Reference

A comprehensive mathematical reference for all key formulas in deep reinforcement learning.

---

## Table of Contents

1. [Fundamentals](#fundamentals)
2. [Value Functions](#value-functions)
3. [Bellman Equations](#bellman-equations)
4. [Policy Gradients](#policy-gradients)
5. [Temporal Difference Learning](#temporal-difference-learning)
6. [Actor-Critic Methods](#actor-critic-methods)
7. [Trust Region Methods](#trust-region-methods)
8. [Off-Policy Methods](#off-policy-methods)
9. [Model-Based RL](#model-based-rl)
10. [Information Theory](#information-theory)

---

## Fundamentals

### MDP Definition

A Markov Decision Process is defined by the tuple (S, A, P, R, γ):

- **S**: State space
- **A**: Action space
- **P**: Transition dynamics `P(s'|s,a)`
- **R**: Reward function `R(s,a)` or `R(s,a,s')`
- **γ**: Discount factor, γ ∈ [0,1]

### Markov Property

```
P(S_{t+1} | S_t, A_t, S_{t-1}, A_{t-1}, ..., S_0, A_0) = P(S_{t+1} | S_t, A_t)
```

The future depends only on the present, not the past.

### Return (Cumulative Reward)

**Finite Horizon**
```
G_t = R_{t+1} + R_{t+2} + ... + R_T
```

**Infinite Horizon (Discounted)**
```
G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... = Σ_{k=0}^∞ γ^k R_{t+k+1}
```

**Recursive Form**
```
G_t = R_{t+1} + γG_{t+1}
```

---

## Value Functions

### State-Value Function

Expected return starting from state s, following policy π:

```
V^π(s) = E_π[G_t | S_t = s]
       = E_π[Σ_{k=0}^∞ γ^k R_{t+k+1} | S_t = s]
```

**Expanded**
```
V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R(s,a,s') + γV^π(s')]
```

### Action-Value Function (Q-Function)

Expected return starting from state s, taking action a, then following π:

```
Q^π(s,a) = E_π[G_t | S_t = s, A_t = a]
         = E_π[Σ_{k=0}^∞ γ^k R_{t+k+1} | S_t = s, A_t = a]
```

**Expanded**
```
Q^π(s,a) = Σ_{s'} P(s'|s,a) [R(s,a,s') + γ Σ_{a'} π(a'|s') Q^π(s',a')]
```

### Advantage Function

Advantage of taking action a in state s:

```
A^π(s,a) = Q^π(s,a) - V^π(s)
```

**Intuition**: How much better is action a compared to the average action in state s?

- A(s,a) > 0: action is better than average
- A(s,a) < 0: action is worse than average
- A(s,a) = 0: action is average

### Relationship Between V and Q

```
V^π(s) = Σ_a π(a|s) Q^π(s,a)
       = E_{a~π}[Q^π(s,a)]
```

---

## Bellman Equations

### Bellman Expectation Equation (State-Value)

```
V^π(s) = Σ_a π(a|s) [R(s,a) + γ Σ_{s'} P(s'|s,a) V^π(s')]
```

**Compact Form**
```
V^π(s) = E_π[R_{t+1} + γV^π(S_{t+1}) | S_t = s]
```

### Bellman Expectation Equation (Action-Value)

```
Q^π(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) Σ_{a'} π(a'|s') Q^π(s',a')
```

**Compact Form**
```
Q^π(s,a) = E_π[R_{t+1} + γQ^π(S_{t+1}, A_{t+1}) | S_t = s, A_t = a]
```

### Bellman Optimality Equation (State-Value)

```
V*(s) = max_a [R(s,a) + γ Σ_{s'} P(s'|s,a) V*(s')]
```

**Compact Form**
```
V*(s) = max_a E[R_{t+1} + γV*(S_{t+1}) | S_t = s, A_t = a]
```

### Bellman Optimality Equation (Action-Value)

```
Q*(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) max_{a'} Q*(s',a')
```

**Compact Form**
```
Q*(s,a) = E[R_{t+1} + γ max_{a'} Q*(S_{t+1}, a') | S_t = s, A_t = a]
```

### Optimal Policy

```
π*(a|s) = 1 if a = argmax_a Q*(s,a)
          0 otherwise

Or simply: π*(s) = argmax_a Q*(s,a)
```

---

## Policy Gradients

### Policy Gradient Theorem

```
∇_θ J(θ) = E_τ~π_θ [Σ_t ∇_θ log π_θ(a_t|s_t) G_t]
```

Where:
- J(θ) = E_τ~π_θ[Σ_t r_t] is the expected return
- τ = (s_0, a_0, r_0, s_1, a_1, r_1, ...) is a trajectory

### REINFORCE (Monte Carlo Policy Gradient)

```
∇_θ J(θ) ≈ (1/N) Σ_i Σ_t ∇_θ log π_θ(a_t^i|s_t^i) G_t^i
```

**With Baseline** (variance reduction):
```
∇_θ J(θ) ≈ (1/N) Σ_i Σ_t ∇_θ log π_θ(a_t^i|s_t^i) (G_t^i - b(s_t^i))
```

Common baseline: b(s) = V^π(s)

### Log Probability Trick

For Gaussian policy:
```
π_θ(a|s) = N(μ_θ(s), σ²)

log π_θ(a|s) = -1/2 [(a - μ_θ(s))² / σ² + log(2πσ²)]

∇_θ log π_θ(a|s) = (a - μ_θ(s)) / σ² · ∇_θ μ_θ(s)
```

For Categorical policy:
```
π_θ(a|s) = softmax(f_θ(s))_a = exp(f_θ(s)_a) / Σ_a' exp(f_θ(s)_a')

log π_θ(a|s) = f_θ(s)_a - log Σ_a' exp(f_θ(s)_a')

∇_θ log π_θ(a|s) = ∇_θ f_θ(s)_a - Σ_a' π_θ(a'|s) ∇_θ f_θ(s)_a'
```

---

## Temporal Difference Learning

### TD Error

```
δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)
```

**For Q-functions**:
```
δ_t = R_{t+1} + γQ(S_{t+1}, A_{t+1}) - Q(S_t, A_t)
```

### TD(0) Update (State-Value)

```
V(S_t) ← V(S_t) + α δ_t
       = V(S_t) + α [R_{t+1} + γV(S_{t+1}) - V(S_t)]
```

### Q-Learning Update

```
Q(S_t, A_t) ← Q(S_t, A_t) + α [R_{t+1} + γ max_a Q(S_{t+1}, a) - Q(S_t, A_t)]
```

**Key**: Uses max instead of following policy (off-policy).

### SARSA Update

```
Q(S_t, A_t) ← Q(S_t, A_t) + α [R_{t+1} + γQ(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]
```

**Key**: Uses actual next action from policy (on-policy).

### Expected SARSA Update

```
Q(S_t, A_t) ← Q(S_t, A_t) + α [R_{t+1} + γ Σ_a π(a|S_{t+1})Q(S_{t+1}, a) - Q(S_t, A_t)]
```

### n-Step TD

```
G_t^{(n)} = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n} + γ^n V(S_{t+n})

V(S_t) ← V(S_t) + α [G_t^{(n)} - V(S_t)]
```

### TD(λ) (λ-Return)

```
G_t^λ = (1-λ) Σ_{n=1}^∞ λ^{n-1} G_t^{(n)}

V(S_t) ← V(S_t) + α [G_t^λ - V(S_t)]
```

---

## Actor-Critic Methods

### Advantage Actor-Critic

**Advantage Estimate** (1-step TD):
```
A_t = R_{t+1} + γV(S_{t+1}) - V(S_t) = δ_t
```

**Policy Gradient**:
```
∇_θ J(θ) ≈ E[∇_θ log π_θ(a_t|s_t) A_t]
```

**Value Update**:
```
L_V = E[(V_φ(s_t) - G_t)²]
```

### Generalized Advantage Estimation (GAE)

```
A_t^GAE(γ,λ) = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
```

Where δ_t = r_t + γV(s_{t+1}) - V(s_t)

**Recursive Form**:
```
A_t^GAE = δ_t + (γλ)A_{t+1}^GAE
```

**Special Cases**:
- λ=0: A_t = δ_t (1-step TD, high bias, low variance)
- λ=1: A_t = Σ_k γ^k δ_{t+k} (Monte Carlo, low bias, high variance)
- λ=0.95: Good balance (default in PPO)

### A2C Loss

```
L_total = L_policy + c_1 L_value - c_2 H(π)
```

Where:
- L_policy = -E[log π_θ(a_t|s_t) A_t]
- L_value = E[(V_φ(s_t) - G_t)²]
- H(π) = -Σ_a π(a|s) log π(a|s) (entropy bonus)
- c_1 ≈ 0.5, c_2 ≈ 0.01

---

## Trust Region Methods

### KL Divergence (Kullback-Leibler)

**For Discrete Distributions**:
```
KL(p||q) = Σ_x p(x) log(p(x)/q(x))
```

**For Policies**:
```
KL(π_old||π_new) = E_{s~ρ}[Σ_a π_old(a|s) log(π_old(a|s)/π_new(a|s))]
```

**For Gaussians**:
```
KL(N(μ₁,σ₁²)||N(μ₂,σ₂²)) = log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 1/2
```

### TRPO Objective

```
maximize_θ E_s,a~π_old [π_θ(a|s)/π_old(a|s) · A(s,a)]

subject to E_s[KL(π_old(·|s)||π_θ(·|s))] ≤ δ
```

Where δ is trust region size (typically 0.01).

### Fisher Information Matrix

```
F = E_s,a~π [∇_θ log π_θ(a|s) · ∇_θ log π_θ(a|s)^T]
```

**Natural Gradient**:
```
θ_{new} = θ_old + α F^{-1} ∇_θ J(θ)
```

### PPO Clipped Objective

```
L^CLIP(θ) = E_t [min(r_t(θ) Â_t, clip(r_t(θ), 1-ε, 1+ε) Â_t)]
```

Where:
- r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t) (importance ratio)
- ε = 0.2 (clip parameter)
- Â_t = normalized advantage

**With Value Function and Entropy**:
```
L^CLIP+VF+S(θ) = E_t [L^CLIP(θ) - c_1 L^VF(θ) + c_2 S[π_θ](s_t)]
```

Where:
- L^VF(θ) = (V_θ(s_t) - V_t^target)²
- S[π_θ](s_t) = -Σ_a π_θ(a|s_t) log π_θ(a|s_t) (entropy)

---

## Off-Policy Methods

### Importance Sampling

**For Expectations**:
```
E_{x~p}[f(x)] = E_{x~q}[p(x)/q(x) · f(x)]
```

**For Policy Gradients**:
```
∇_θ J(θ) = E_τ~π_old [Π_t π_θ(a_t|s_t)/π_old(a_t|s_t) · Σ_t ∇_θ log π_θ(a_t|s_t) G_t]
```

### DQN Loss

```
L(θ) = E_{(s,a,r,s')~D} [(r + γ max_{a'} Q(s',a'; θ^-) - Q(s,a; θ))²]
```

Where:
- D is replay buffer
- θ^- are target network parameters (updated slowly)

### Double DQN

**Target**:
```
y_t = r_t + γ Q(s_{t+1}, argmax_{a'} Q(s_{t+1}, a'; θ); θ^-)
```

**Key**: Use online network to select action, target network to evaluate.

### DDPG (Deterministic Policy Gradient)

**Critic Update**:
```
L = E[(r + γQ(s', μ(s'; θ^μ); θ^Q) - Q(s, a; θ^Q))²]
```

**Actor Update**:
```
∇_θ J ≈ E[∇_a Q(s, a; θ^Q)|_{a=μ(s)} · ∇_θ μ(s; θ^μ)]
```

### TD3 (Twin Delayed DDPG)

**Clipped Double Q-Learning**:
```
y_t = r_t + γ min_{i=1,2} Q_i(s_{t+1}, μ(s_{t+1}) + ε; θ_i^-)
```

Where ε ~ N(0, σ) is target policy smoothing noise.

**Critic Loss**:
```
L = Σ_{i=1,2} E[(y_t - Q_i(s_t, a_t; θ_i))²]
```

### SAC (Soft Actor-Critic)

**Maximum Entropy Objective**:
```
J(π) = E_τ~π [Σ_t (r(s_t, a_t) + α H(π(·|s_t)))]
```

Where H(π(·|s)) = -Σ_a π(a|s) log π(a|s) is entropy.

**Soft Bellman Equation**:
```
V(s_t) = E_{a_t~π}[Q(s_t, a_t) - α log π(a_t|s_t)]

Q(s_t, a_t) = r(s_t, a_t) + γ E_{s_{t+1}}[V(s_{t+1})]
```

**Soft Policy Iteration**:
```
Q^{k+1}(s,a) = r(s,a) + γ E_{s'}[V^k(s')]

π^{k+1} = argmax_π E_{a~π}[Q^{k+1}(s,a) - α log π(a|s)]
```

**Temperature Tuning**:
```
J(α) = E_{a_t~π}[-α log π(a_t|s_t) - α H̄]
```

Where H̄ is target entropy (typically -dim(A)).

---

## Model-Based RL

### Dynamics Model

**Deterministic**:
```
s_{t+1} = f_θ(s_t, a_t)
```

**Stochastic**:
```
s_{t+1} ~ P_θ(·|s_t, a_t)
```

**Delta Model** (easier to learn):
```
s_{t+1} = s_t + Δf_θ(s_t, a_t)
```

### Model Predictive Control (MPC)

```
a_t* = argmax_{a_t,...,a_{t+H}} Σ_{k=0}^H r(s_{t+k}, a_{t+k})

subject to s_{t+k+1} = f(s_{t+k}, a_{t+k})
```

Execute only a_t*, then replan.

### Dyna-Q

**Model Update**:
```
Model(s,a) ← (r, s')  # Store observed transition
```

**Planning Step** (use model):
```
s, a ← random previously observed state-action
r, s' ← Model(s, a)
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
```

### MBPO (Model-Based Policy Optimization)

**Branched Rollout**:
```
s_0 ~ D_env (real data)
for k = 0 to K:
    a_k ~ π(·|s_k)
    s_{k+1} ~ P_θ(·|s_k, a_k) (model prediction)
    r_k = R_θ(s_k, a_k)
    Add (s_k, a_k, r_k, s_{k+1}) to D_model
```

**Policy Update** (on mixed data):
```
θ_π ← θ_π + α ∇_θ J(θ_π)  using D_env ∪ D_model
```

---

## Information Theory

### Entropy

**Discrete**:
```
H(X) = -Σ_x p(x) log p(x) = E_X[-log p(X)]
```

**Continuous**:
```
H(X) = -∫ p(x) log p(x) dx
```

**For Policies**:
```
H(π(·|s)) = -Σ_a π(a|s) log π(a|s)
```

**Maximum Entropy**: H(uniform) = log |A|

### KL Divergence

```
KL(p||q) = E_p[log p(X) - log q(X)]
         = Σ_x p(x) log(p(x)/q(x))
```

**Properties**:
- KL(p||q) ≥ 0
- KL(p||q) = 0 iff p = q
- KL(p||q) ≠ KL(q||p) (not symmetric)

### Jensen-Shannon Divergence

Symmetric version of KL:
```
JSD(p||q) = 1/2 KL(p||M) + 1/2 KL(q||M)

where M = (p + q) / 2
```

### Mutual Information

```
I(X; Y) = H(X) - H(X|Y)
        = H(Y) - H(Y|X)
        = E_{x,y}[log p(x,y)/(p(x)p(y))]
        = KL(p(x,y)||p(x)p(y))
```

---

## Useful Inequalities

### Jensen's Inequality

For convex function f:
```
f(E[X]) ≤ E[f(X)]
```

For concave function f:
```
f(E[X]) ≥ E[f(X)]
```

### Cauchy-Schwarz Inequality

```
|E[XY]|² ≤ E[X²]E[Y²]
```

### Hoeffding's Inequality

For bounded random variables X_i ∈ [a, b]:
```
P(|X̄ - μ| ≥ ε) ≤ 2 exp(-2nε²/(b-a)²)
```

Where X̄ = (1/n)Σ_i X_i

---

## Common Distributions

### Gaussian (Normal)

**PDF**:
```
N(x; μ, σ²) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))
```

**Log PDF**:
```
log N(x; μ, σ²) = -1/2 [(x-μ)²/σ² + log(2πσ²)]
```

**Entropy**:
```
H(N(μ, σ²)) = 1/2 log(2πeσ²)
```

### Categorical (Discrete)

**PMF**:
```
Cat(x; p) = p_x
```

**Log PMF**:
```
log Cat(x; p) = log p_x
```

**Entropy**:
```
H(Cat(p)) = -Σ_i p_i log p_i
```

### Beta Distribution

**PDF**:
```
Beta(x; α, β) = x^{α-1}(1-x)^{β-1} / B(α,β)
```

**Mean**: μ = α/(α+β)
**Variance**: σ² = αβ/[(α+β)²(α+β+1)]

Used for: Modeling probabilities, Thompson sampling

---

## Gradient Identities

### Score Function

```
∇_θ log p_θ(x) = ∇_θ p_θ(x) / p_θ(x)
```

### Log Derivative Trick

```
∇_θ E_x~p_θ[f(x)] = ∇_θ ∫ p_θ(x)f(x) dx
                   = ∫ ∇_θ p_θ(x) f(x) dx
                   = ∫ p_θ(x) ∇_θ log p_θ(x) f(x) dx
                   = E_x~p_θ[f(x) ∇_θ log p_θ(x)]
```

**This is the foundation of policy gradients!**

### Reparameterization Trick

For z ~ p_θ(z), write z = g_θ(ε) where ε ~ p(ε):

```
∇_θ E_z~p_θ[f(z)] = ∇_θ E_ε~p[f(g_θ(ε))]
                   = E_ε~p[∇_θ f(g_θ(ε))]
```

**Example (Gaussian)**:
```
z ~ N(μ_θ, σ²_θ)
z = μ_θ + σ_θ · ε, where ε ~ N(0,1)

∇_θ E_z[f(z)] = E_ε[∇_θ f(μ_θ + σ_θ · ε)]
```

---

## Matrix Calculus

### Vector-by-Scalar

```
∂x/∂θ where x ∈ ℝⁿ, θ ∈ ℝ
```

Result: Column vector in ℝⁿ

### Scalar-by-Vector

```
∂L/∂θ where L ∈ ℝ, θ ∈ ℝⁿ
```

Result: Row vector (gradient) in ℝⁿ

### Chain Rule

```
∂L/∂x = (∂L/∂y)(∂y/∂x)
```

**For Deep Learning**:
```
∂L/∂W^{(1)} = (∂L/∂h^{(2)})(∂h^{(2)}/∂h^{(1)})(∂h^{(1)}/∂W^{(1)})
```

### Jacobian

```
J_{ij} = ∂y_i/∂x_j
```

For y = f(x) where x ∈ ℝⁿ, y ∈ ℝᵐ, Jacobian J ∈ ℝᵐˣⁿ

---

## Notation Summary

| Symbol | Meaning |
|--------|---------|
| s, S | State |
| a, A | Action |
| r, R | Reward |
| γ | Discount factor |
| π | Policy |
| V^π(s) | State-value function under π |
| Q^π(s,a) | Action-value function under π |
| A^π(s,a) | Advantage function |
| θ | Policy parameters |
| φ | Value function parameters |
| α | Learning rate or temperature |
| ε | Clipping parameter (PPO) or exploration (ε-greedy) |
| τ | Trajectory or soft update parameter |
| λ | GAE parameter |
| H(π) | Entropy of policy |
| KL(p‖q) | KL divergence from q to p |

---

**This formula sheet is part of the Deep RL Lecture Series.**
**See `/self-study-lectures/lectures/` for detailed derivations and intuitions.**
