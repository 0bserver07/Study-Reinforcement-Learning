<!-- status: unreviewed | last-reviewed: never -->

# RL loss functions — cheat sheet

_Unreviewed — no one has checked this end to end. Treat the math and citations as unverified._

One place to look up the loss for every major RL and RLHF algorithm in this repo. Each block uses the same template so you can scan side by side.

**Template** (per algorithm): loss/objective in symbols (sign noted); symbol glossary; gradient w.r.t. policy params; PyTorch snippet; one line on variance/bias/stability; one line on what to watch in training.

**Sign convention**: an *objective* `J` is maximized; the *loss* `L = -J` is minimized by the optimizer. `r_t(θ)` always means the importance ratio `π_θ/π_old`, not the reward.

Notation matches [`RL-Math-Formulas.md`](./RL-Math-Formulas.md): `s, a, r, γ, π_θ, V^π, Q^π, A^π, θ, φ`. New symbols are defined per block. The PyTorch snippets assume `import torch; import torch.nn.functional as F`.

## Contents

REINFORCE · REINFORCE+baseline · A2C · GAE · TRPO · PPO-clip · PPO-KL · DQN · Double DQN · Dueling DQN · C51 · DDPG · TD3 · SAC · MuZero · Bradley-Terry RM · PPO-RLHF · DPO · IPO · KTO · ORPO · SimPO · GRPO · References.

---

## REINFORCE

Monte Carlo policy gradient. Williams 1992.

**Loss** (minimize):
```
L(θ) = - E_{τ~π_θ} [ Σ_t log π_θ(a_t|s_t) · G_t ]
```
where `G_t = Σ_{k=t}^T γ^{k-t} r_k` is the return-to-go.

**Gradient**:
```
∇_θ J = E_{τ~π_θ} [ Σ_t ∇_θ log π_θ(a_t|s_t) · G_t ]
```
Environment dynamics drop out — only the log-policy carries gradient. Derivation: [Lecture 02 Part 5](../lectures/02-policy-gradients.md).

```python
def reinforce_loss(log_probs, rewards, gamma=0.99):
    # log_probs: list[T] of scalar tensors from dist.log_prob(a), requires_grad
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return -(torch.stack(log_probs) * returns).sum()
```

**Tradeoff**: unbiased, very high variance (one MC sample per gradient estimate).

**Watch**: the reward curve — REINFORCE on CartPole climbs and crashes mid-run. Loss going to ±∞ usually means missing gradient clipping or a tiny action prob.

---

## REINFORCE with baseline

Subtract a state-dependent baseline `b(s)` to cut variance without biasing the gradient. Standard choice: `b(s) = V_φ(s)`, learned via MSE on returns.

**Loss**:
```
L(θ) = - E [ Σ_t log π_θ(a_t|s_t) · (G_t - b(s_t)) ]
```
Unbiasedness: `E_a[∇_θ log π_θ(a|s) · b(s)] = b(s) · ∇_θ Σ_a π_θ(a|s) = 0`.

```python
def reinforce_baseline_loss(log_probs, rewards, values, gamma=0.99):
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G; returns.insert(0, G)
    returns = torch.tensor(returns)
    advantages = returns - values.detach()                  # detach! no grad through baseline
    policy = -(torch.stack(log_probs) * advantages).sum()
    value  = F.mse_loss(values, returns)
    return policy + 0.5 * value
```

**Tradeoff**: unbiased, lower variance. Detaching the baseline inside the policy term is essential — a gradient leak biases the estimate.

**Watch**: advantage mean (should be ≈ 0 after the baseline kicks in) and value loss (plateauing high means the critic isn't fitting).

---

## A2C — advantage actor-critic

Synchronous A3C (Mnih et al. 2016, arXiv:1602.01783). 1-step TD advantage.

**Loss** (total):
```
L = L_policy + c_v · L_value - c_e · H[π_θ]
L_policy = - E [ log π_θ(a_t|s_t) · A_t ],   A_t = r_t + γ V_φ(s_{t+1}) - V_φ(s_t)
L_value  = E [ ( V_φ(s_t) - G_t )² ]
H[π_θ]   = - Σ_a π_θ(a|s) log π_θ(a|s)
```
`c_v ≈ 0.5`, `c_e ≈ 0.01`. Subtracting `H` *rewards* exploration.

```python
def a2c_loss(log_probs, values, rewards, dones, logits, gamma=0.99,
             c_v=0.5, c_e=0.01, bootstrap=0.0):
    R, returns = bootstrap, []
    for r, d in zip(reversed(rewards), reversed(dones)):
        R = r + gamma * R * (1.0 - d); returns.insert(0, R)
    returns = torch.tensor(returns)
    advantages = returns - values.detach()
    policy = -(torch.stack(log_probs) * advantages).mean()
    value  = F.mse_loss(values, returns)
    entropy = -(F.softmax(logits, -1) * F.log_softmax(logits, -1)).sum(-1).mean()
    return policy + c_v * value - c_e * entropy
```

**Tradeoff**: lower variance than REINFORCE, biased by the critic's error (worst early in training).

**Watch**: entropy. If it collapses to ~0 in the first few thousand updates, the policy stopped exploring — raise `c_e` or lower the LR.

---

## GAE — generalized advantage estimation

Schulman et al. 2015 (arXiv:1506.02438). An advantage *estimator*, not an algorithm. Plugs into PPO, A2C, GRPO.

**Estimator** (recursive form, the one you implement):
```
A_t = δ_t + γ λ A_{t+1},     A_T = 0
δ_t = r_t + γ V_φ(s_{t+1}) - V_φ(s_t)
```
`λ ∈ [0,1]`: 0 = 1-step TD (low variance, high bias from critic); 1 = MC (low bias, high variance); 0.95 is the default sweet spot.

```python
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95, last_value=0.0):
    T = len(rewards)
    advantages = torch.zeros(T)
    gae, next_v = 0.0, last_value
    for t in reversed(range(T)):
        nd = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nd - values[t]
        gae = delta + gamma * lam * nd * gae
        advantages[t] = gae
        next_v = values[t]
    returns = advantages + values
    return advantages, returns
```

**Tradeoff**: `λ` is the knob between bias (low λ, trust the critic) and variance (high λ, trust MC returns).

**Watch**: per-batch advantage mean/std. Mean should be ≈ 0 after normalization; std ≈ 0 means there's no signal in this batch.

---

## TRPO — surrogate objective with KL constraint

Schulman et al. 2015 (arXiv:1502.05477). Constrained optimization instead of soft penalty.

**Objective**:
```
maximize_θ  E_{s,a~π_old} [ (π_θ(a|s)/π_old(a|s)) · A^{π_old}(s,a) ]
subject to  E_s [ KL(π_old(·|s) || π_θ(·|s)) ] ≤ δ
```
`δ ≈ 0.01`. Solved via Lagrangian + natural gradient: conjugate gradient with Fisher-vector products gives the step direction, then line-search to satisfy the KL constraint. Full derivation in [Lecture 05](../lectures/05-trpo.md).

```python
def trpo_surrogate(logp_new, logp_old, advantages):
    ratio = (logp_new - logp_old).exp()
    return (ratio * advantages).mean()                       # ascend this

def mean_kl(logits_old, logits_new):
    p_old = F.softmax(logits_old, -1)
    return (p_old * (F.log_softmax(logits_old, -1)
                     - F.log_softmax(logits_new, -1))).sum(-1).mean()
# Full TRPO: ∇ surrogate, solve Fx=g via CG + Fisher-vector products,
# then backtracking line search constrained by mean_kl ≤ δ.
```

**Tradeoff**: stable (hard KL cap), expensive (CG + FVPs dominate compute).

**Watch**: per-update mean KL (should sit near `δ`) and line-search backtrack count (many backtracks → surrogate is a bad local model of `J`).

---

## PPO — clipped surrogate

Schulman et al. 2017 (arXiv:1707.06347). Approximate the trust region with clipping. First-order.

**Loss** (full):
```
L = - E [ min( r_t(θ) Â_t, clip(r_t(θ), 1-ε, 1+ε) Â_t ) ]
    + c_v · E [ ( V_φ(s_t) - V_t^target )² ]
    - c_e · E [ H[π_θ(·|s_t)] ]
```
`r_t(θ) = π_θ(a_t|s_t)/π_old(a_t|s_t)`, `ε ≈ 0.2`, `c_v ≈ 0.5`, `c_e ≈ 0.01`. `Â_t` is GAE, normalized.

Where the clip is active the gradient w.r.t. θ is zero — by design the algorithm stops pushing.

```python
def ppo_loss(logp_new, logp_old, advantages, values, returns,
             entropy=None, clip_eps=0.2, c_v=0.5, c_e=0.01):
    ratio = (logp_new - logp_old).exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy = -torch.min(surr1, surr2).mean()
    value  = F.mse_loss(values, returns)
    loss   = policy + c_v * value
    if entropy is not None:
        loss = loss - c_e * entropy.mean()
    return loss
```

**Tradeoff**: stable and forgiving. Too many epochs over the same batch pushes `ratio` far from 1 and the clip dominates.

**Watch**: **clip fraction** (fraction of samples where the clip activates) — healthy is 10–30%; above ~50% means lower the LR or epochs/batch. Also approximate KL between `π_old` and `π_θ`.

---

## PPO — adaptive KL variant

Same paper, soft-penalty version. Often less robust in practice; clip is preferred.

**Loss**:
```
L = - E [ r_t(θ) Â_t ] + β · E_s [ KL(π_old(·|s) || π_θ(·|s)) ]
```
**Adaptation**: after each update, if mean KL `< d_target/1.5` halve `β`; if `> d_target·1.5` double it. `d_target ≈ 0.01`.

```python
def ppo_kl_loss(logp_new, logp_old, advantages, kl_per_step, beta):
    ratio = (logp_new - logp_old).exp()
    return -(ratio * advantages).mean() + beta * kl_per_step.mean()

def adapt_beta(beta, mean_kl, d_target=0.01):
    if mean_kl < d_target / 1.5: return beta / 2
    if mean_kl > d_target * 1.5: return beta * 2
    return beta
```

**Tradeoff**: same idea as TRPO with a soft penalty; more sensitive to `β` initialization than clip.

**Watch**: mean KL (should track `d_target`) and `β` (should stabilize).

---

## DQN

Mnih et al. 2013 (arXiv:1312.5602) / 2015 Nature. Squared TD error against a target network.

**Loss**:
```
L(θ) = E_{(s,a,r,s')~D} [ ( y - Q_θ(s,a) )² ]
y = r + γ · max_{a'} Q_{θ^-}(s', a')         (y = r if s' is terminal)
```
`D` = replay buffer; `θ^-` = target net, periodically synced or Polyak-averaged. `y` is detached.

**Gradient**: `∇_θ L = -2 · E[(y - Q_θ(s,a)) · ∇_θ Q_θ(s,a)]` — regression on a moving target.

```python
def dqn_loss(q_net, target_net, batch, gamma=0.99):
    s, a, r, s_next, done = batch
    q_sa = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        q_next = target_net(s_next).max(1).values
        y = r + gamma * q_next * (1.0 - done)
    return F.smooth_l1_loss(q_sa, y)                         # Huber; MSE also fine
```

**Tradeoff**: stable thanks to target net + replay. The `max` biases Q *upward* — overestimation, fixed by Double DQN.

**Watch**: mean Q on a fixed eval state set — if it climbs while reward doesn't, you're overestimating. TD error magnitude should track reward scale.

---

## Double DQN

van Hasselt, Guez, Silver 2016 (arXiv:1509.06461). Online net picks the action; target net evaluates it.

**Loss**: same as DQN, with target:
```
y = r + γ · Q_{θ^-}( s', argmax_{a'} Q_θ(s', a') )
```

```python
def double_dqn_loss(q_net, target_net, batch, gamma=0.99):
    s, a, r, s_next, done = batch
    q_sa = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        a_star = q_net(s_next).argmax(1, keepdim=True)       # online selects
        q_next = target_net(s_next).gather(1, a_star).squeeze(1)  # target evaluates
        y = r + gamma * q_next * (1.0 - done)
    return F.smooth_l1_loss(q_sa, y)
```

**Tradeoff**: cuts overestimation bias; same variance as DQN.

**Watch**: same as DQN. The "no win" mode is when overestimation wasn't your bottleneck.

---

## Dueling DQN (architectural)

Wang et al. 2016 (arXiv:1511.06581). Not a loss change — same DQN or Double-DQN loss. Architecture only.

```
Q_θ(s,a) = V_θ(s) + ( A_θ(s,a) - mean_{a'} A_θ(s,a') )
```
Two heads on a shared trunk: a scalar `V` and a per-action `A`. Subtracting the mean fixes identifiability (without it, you can add `c` to V and subtract `c` from A and get the same Q).

```python
class DuelingDQN(torch.nn.Module):
    def __init__(self, in_dim, n_actions, h=128):
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(in_dim, h), torch.nn.ReLU(),
            torch.nn.Linear(h, h), torch.nn.ReLU())
        self.v_head = torch.nn.Linear(h, 1)
        self.a_head = torch.nn.Linear(h, n_actions)
    def forward(self, s):
        h = self.trunk(s)
        v, a = self.v_head(h), self.a_head(h)
        return v + (a - a.mean(dim=1, keepdim=True))
```

**Tradeoff**: helps when many actions share similar value at a state. Worst case: no improvement, slightly more params.

**Watch**: compare to plain DQN. Typically faster early learning on Atari.

---

## C51 — categorical DQN

Bellemare, Dabney, Munos 2017 (arXiv:1707.06887). Learn the full *distribution* of returns; loss is cross-entropy on a projected distributional Bellman target.

**Setup**: fix a support `{z_i = V_min + i·Δz}` for `i=0..N-1` (typically N=51 → "C51"). Net outputs `p_θ(s,a) ∈ Δ^{N-1}` via softmax. `Q_θ(s,a) = Σ_i z_i · p_θ(s,a)_i`.

**Target** (before projection): `T̂Z(s,a) = r + γ · Z_{θ^-}(s', a*)` with `a* = argmax_{a'} Σ_i z_i · p_{θ^-}(s',a')_i`. This shifts atoms to `r + γ·z_i`, which don't line up with `{z_i}`. Project back by splitting each atom's mass linearly onto the two nearest support points → target distribution `m`.

**Loss**: `L(θ) = - Σ_i m_i · log p_θ(s,a)_i` — standard cross-entropy.

```python
def project_target(reward, gamma, done, p_next, atoms, v_min, v_max):
    # p_next: [B, N] target distribution at s'; atoms: [N]
    Tz = reward.unsqueeze(1) + gamma * atoms.unsqueeze(0) * (1.0 - done.unsqueeze(1))
    Tz = Tz.clamp(v_min, v_max)
    dz = (v_max - v_min) / (atoms.size(0) - 1)
    b = (Tz - v_min) / dz
    l, u = b.floor().long(), b.ceil().long()
    m = torch.zeros_like(p_next)
    m.scatter_add_(1, l, p_next * (u.float() - b))
    m.scatter_add_(1, u, p_next * (b - l.float()))
    return m

def c51_loss(p_logits_sa, target_p):
    # p_logits_sa: [B, N] logits for chosen action; target_p: [B, N] no grad
    return -(target_p * F.log_softmax(p_logits_sa, dim=-1)).sum(-1).mean()
```

**Tradeoff**: lower-variance value estimates than scalar Q-learning. Sensitive to `V_min, V_max` — set them wrong and the projection clips real mass.

**Watch**: the distribution itself on fixed eval states — piling up at boundary atoms means widen the support.

---

## DDPG

Lillicrap et al. 2016 (arXiv:1509.02971). Deterministic actor + Q-critic, off-policy, continuous actions.

**Critic loss**:
```
L_Q = E_{(s,a,r,s')~D} [ ( y - Q_{θ^Q}(s,a) )² ],   y = r + γ Q_{θ^Q^-}(s', μ_{θ^μ^-}(s'))
```
**Actor loss** (deterministic policy gradient, Silver et al. 2014):
```
L_μ = - E_{s~D} [ Q_{θ^Q}(s, μ_{θ^μ}(s)) ],   ∇_{θ^μ} J ≈ E[ ∇_a Q · ∇_{θ^μ} μ ]
```
Target nets Polyak-averaged: `θ^- ← τ θ + (1-τ) θ^-`, `τ ≈ 0.005`.

```python
def ddpg_losses(q, q_tar, mu, mu_tar, batch, gamma=0.99):
    s, a, r, s_next, done = batch
    with torch.no_grad():
        a_next = mu_tar(s_next)
        y = r + gamma * q_tar(s_next, a_next).squeeze(-1) * (1.0 - done)
    critic = F.mse_loss(q(s, a).squeeze(-1), y)
    actor  = -q(s, mu(s)).mean()
    return critic, actor
```

**Tradeoff**: sample-efficient, notoriously brittle. Critic overestimation feeds bad gradients to the actor — TD3 was invented to fix this.

**Watch**: critic value drift on a fixed eval batch (climbing without reward climbing → overestimation). Exploration noise schedule.

---

## TD3

Fujimoto, van Hoof, Meger 2018 (arXiv:1802.09477). Three fixes on DDPG: twin critics, delayed actor updates, target-policy smoothing.

**Critic loss** (clipped double-Q):
```
ã = clip( μ_{θ^μ^-}(s') + clip(ε, -c, c), a_low, a_high ),   ε ~ N(0, σ²)
y  = r + γ · min_{i=1,2} Q_{θ_i^-}(s', ã)
L_Q = Σ_{i=1,2} E [ ( y - Q_{θ_i}(s, a) )² ]
```
**Actor loss** (every `d` critic steps, `d ≈ 2`):
```
L_μ = - E [ Q_{θ_1}(s, μ_{θ^μ}(s)) ]
```
`σ ≈ 0.2`, noise clip `c ≈ 0.5`.

```python
def td3_critic_loss(q1, q2, q1t, q2t, mu_t, batch,
                    gamma=0.99, noise_std=0.2, noise_clip=0.5,
                    a_low=-1.0, a_high=1.0):
    s, a, r, s_next, done = batch
    with torch.no_grad():
        noise = (torch.randn_like(a) * noise_std).clamp(-noise_clip, noise_clip)
        a_next = (mu_t(s_next) + noise).clamp(a_low, a_high)
        y_q = torch.min(q1t(s_next, a_next), q2t(s_next, a_next)).squeeze(-1)
        y = r + gamma * y_q * (1.0 - done)
    return (F.mse_loss(q1(s, a).squeeze(-1), y)
          + F.mse_loss(q2(s, a).squeeze(-1), y))

def td3_actor_loss(q1, mu, s):
    return -q1(s, mu(s)).mean()
```

**Tradeoff**: much more stable than DDPG. Min-of-two breaks overestimation; delayed actor updates stop the actor chasing a noisy critic.

**Watch**: both critic losses (one collapsing to ~0 while the other doesn't means batches too small / critics correlated). Gap between `Q_1` and `Q_2` on eval states should stay small but nonzero.

---

## SAC — soft actor-critic

Haarnoja et al. 2018 (arXiv:1801.01290) + auto-temperature variant (arXiv:1812.05905). Maximum-entropy RL: per-step entropy bonus.

**Objective**: `J(π) = E_{τ~π} [ Σ_t r(s_t, a_t) + α · H(π(·|s_t)) ]`.

**Critic** (twin Q, like TD3):
```
ã ~ π_φ(·|s'),   y = r + γ ( min_{i=1,2} Q_{θ_i^-}(s', ã) - α · log π_φ(ã|s') )
L_Q = Σ_{i=1,2} E [ ( y - Q_{θ_i}(s, a) )² ]
```
**Actor** (reparameterized; `a_φ(ε; s) = tanh(μ_φ(s) + σ_φ(s)·ε)`):
```
L_π = E [ α · log π_φ(a_φ|s) - min_{i=1,2} Q_{θ_i}(s, a_φ) ]
```
**Temperature** (auto-tune):
```
L_α = E [ -α · ( log π_φ(a|s) + H̄ ) ],   H̄ = -|A| typically
```

```python
def sac_losses(pi, q1, q2, q1t, q2t, log_alpha, batch,
               gamma=0.99, target_entropy=-3.0):
    s, a, r, s_next, done = batch
    alpha = log_alpha.exp()
    with torch.no_grad():
        a_next, logp_next = pi.sample_with_logp(s_next)
        q_next = torch.min(q1t(s_next, a_next), q2t(s_next, a_next)).squeeze(-1)
        y = r + gamma * (q_next - alpha * logp_next) * (1.0 - done)
    q_loss = (F.mse_loss(q1(s, a).squeeze(-1), y)
            + F.mse_loss(q2(s, a).squeeze(-1), y))
    a_pi, logp_pi = pi.rsample_with_logp(s)                  # reparameterized
    q_pi = torch.min(q1(s, a_pi), q2(s, a_pi)).squeeze(-1)
    pi_loss    = (alpha.detach() * logp_pi - q_pi).mean()
    alpha_loss = -(log_alpha * (logp_pi.detach() + target_entropy)).mean()
    return q_loss, pi_loss, alpha_loss
```

**Tradeoff**: among the most stable continuous-control algorithms. Sensitive to action-space scale — normalize actions to `[-1, 1]`.

**Watch**: `α` itself — should settle to a value holding policy entropy near `H̄`. `α → 0` → deterministic policy; unbounded growth → unreachable entropy target.

---

## MuZero

Schrittwieser et al. 2020 (Nature 588, arXiv:1911.08265). Planning with a learned model in *latent* space. Three loss terms summed over K unroll steps.

**Networks**: `h(o) → s_0` (representation), `g(s, a) → (s', r̂)` (dynamics), `f(s) → (p, v)` (prediction).

**Loss** (per trajectory, sum over `k=0..K`):
```
L(θ) = Σ_k [ ℓ_r(u_{t+k}, r̂_{t+k}) + ℓ_v(z_{t+k}, v_{t+k}) + ℓ_p(π_{t+k}^MCTS, p_{t+k}) ] + c‖θ‖²
```
- `u_{t+k}` observed reward, `r̂` predicted from `g`.
- `z_{t+k}` n-step bootstrap return target, `v` predicted from `f`.
- `π_{t+k}^MCTS` is the visit-count distribution from the search tree, `p` predicted from `f`.
- `ℓ_r, ℓ_v` are cross-entropy on a categorical reward/value representation (or MSE in continuous settings); `ℓ_p` is cross-entropy on the policy distribution.

```python
def muzero_step_loss(pred_p, pred_v, pred_r, target_p, target_v, target_r):
    # all pred_* are logits; target_* are detached distributions
    loss_p = -(target_p * F.log_softmax(pred_p, dim=-1)).sum(-1)
    loss_v = -(target_v * F.log_softmax(pred_v, dim=-1)).sum(-1)
    loss_r = -(target_r * F.log_softmax(pred_r, dim=-1)).sum(-1)
    return (loss_p + loss_v + loss_r).mean()
# Total: sum over K unroll steps + L2 weight decay.
```

**Tradeoff**: categorical value head (not scalar regression) is what makes the loss stable across Atari + chess + Go. Sensitive to replay-buffer staleness.

**Watch**: reward-prediction loss drops fastest, policy next, value slowest. Value loss diverging → wrong bootstrap range for the reward scale.

---

## Bradley-Terry reward model

Bradley-Terry 1952. Standard preference loss for RM training in RLHF (Christiano et al. 2017, arXiv:1706.03741; Ouyang et al. 2022, arXiv:2203.02155).

**Loss**:
```
L(φ) = - E_{(x, y_w, y_l)~D_pref} [ log σ( r_φ(x, y_w) - r_φ(x, y_l) ) ]
```
`r_φ(x, y)` is the scalar reward (typically a linear head on the LM's last-token hidden state).

**Gradient** with `Δ = r_φ(x,y_w) - r_φ(x,y_l)`:
```
∇_φ L = - E [ σ(-Δ) · ( ∇_φ r_φ(x,y_w) - ∇_φ r_φ(x,y_l) ) ]
```
`σ(-Δ)` down-weights pairs the model already gets right and focuses gradient on the hard ones.

```python
def bt_reward_loss(r_chosen, r_rejected):
    # both: [B] scalar rewards from the RM
    return -F.logsigmoid(r_chosen - r_rejected).mean()
```

**Tradeoff**: well-behaved logistic regression on differences. Variance in practice comes from annotator noise and unidentified additive scale of `r_φ`.

**Watch**: pairwise accuracy on a held-out preference set (65–75% on noisy human data is normal, higher on synthetic). Also the average gap `Δ` — if it keeps growing, RM may be learning spurious features.

---

## PPO-RLHF combined loss

InstructGPT recipe (Ouyang et al. 2022, arXiv:2203.02155): PPO clip + per-token KL to the SFT reference, baked into the *reward* (not a separate loss term).

**Per-token reward** (the key trick):
```
r̃_t = r_φ(x, y) · 1[t = T]  -  β · ( log π_θ(y_t | x, y_<t) - log π_ref(y_t | x, y_<t) )
```
RM fires once at the end; per-token KL fires every step. Compute GAE on `r̃_t` to get advantages, then run the PPO loss as above.

```python
def rlhf_token_rewards(rm_scores, logp_policy, logp_ref, beta=0.05):
    # rm_scores: [B]; logp_policy, logp_ref: [B, T] per-token log-probs of sampled tokens
    kl = logp_policy - logp_ref                              # [B, T]
    r = -beta * kl
    r[:, -1] = r[:, -1] + rm_scores                          # terminal RM reward
    return r
# Then: advantages, returns = compute_gae(...); loss = ppo_loss(...)
```

**Tradeoff**: KL stops the policy drifting into regions where the RM is wrong. `β` too low → reward hacking; too high → policy can't move.

**Watch**: average per-token KL to `π_ref` (target 5–10 nats over a full response); RM score (climbing while sample quality drops = reward hacking); clip fraction. Always eval generations qualitatively.

---

## DPO

Rafailov et al. 2023 (arXiv:2305.18290). Skip the RM: the closed-form solution to the KL-constrained RLHF objective lets you train on preferences directly.

**Loss**:
```
L_DPO(θ) = - E [ log σ( β · ( log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x) ) ) ]
```
With `Δ_w = log π_θ(y_w|x) - log π_ref(y_w|x)` and `Δ_l` likewise: `L = - E[ log σ( β(Δ_w - Δ_l) ) ]`.

The "implicit reward" defined by the algorithm is `r̂_θ(x,y) = β · log(π_θ(y|x)/π_ref(y|x))` (the partition function cancels in the difference). `β` ≈ 0.01–0.5; larger → closer to `π_ref`.

**Gradient** (DPO paper §4):
```
∇_θ L_DPO = - β · E [ σ(-β(Δ_w - Δ_l)) · ( ∇_θ log π_θ(y_w|x) - ∇_θ log π_θ(y_l|x) ) ]
```
`σ(-β(Δ_w - Δ_l))` is the wrong-ness gating: easy pairs contribute almost no gradient.

```python
def dpo_loss(pol_lp_w, pol_lp_l, ref_lp_w, ref_lp_l, beta=0.1):
    # each: [B] sum of per-token log-probs over the response
    chosen_logratio   = pol_lp_w - ref_lp_w
    rejected_logratio = pol_lp_l - ref_lp_l
    logits = beta * (chosen_logratio - rejected_logratio)
    return -F.logsigmoid(logits).mean()
```

**Tradeoff**: lower variance than PPO-RLHF (no MC rollouts). Bias from the BT model being an imperfect preference model.

**Watch**:
- **Chosen vs rejected logit gap** `β·(Δ_w - Δ_l)` — should climb.
- `Δ_w` itself — the failure mode is `Δ_w < 0` (policy makes the *chosen* response less likely than the reference does); both `Δ_w, Δ_l` going strongly negative is a known DPO pathology.
- Pairwise accuracy on a held-out preference set.

---

## IPO

Azar et al. 2023 (arXiv:2310.12036). Squared-loss DPO variant; addresses DPO's tendency to overfit when preferences are near-deterministic.

**Loss**:
```
L_IPO(θ) = E [ ( log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x) - 1/(2β) )² ]
```
Pushes the policy toward a *fixed* log-ratio gap `1/(2β)` rather than maximizing it.

**Gradient** with `m = Δ_w - Δ_l - 1/(2β)`:
```
∇_θ L_IPO = 2 · E [ m · ( ∇_θ log π_θ(y_w|x) - ∇_θ log π_θ(y_l|x) ) ]
```
Once `m ≈ 0`, the gradient vanishes — built-in regularization DPO lacks.

```python
def ipo_loss(pol_lp_w, pol_lp_l, ref_lp_w, ref_lp_l, beta=0.1):
    chosen_logratio   = pol_lp_w - ref_lp_w
    rejected_logratio = pol_lp_l - ref_lp_l
    margin = chosen_logratio - rejected_logratio
    target = 1.0 / (2.0 * beta)
    return ((margin - target) ** 2).mean()
```

**Tradeoff**: more stable than DPO when preferences are noisy or near-deterministic. Need to pick the target margin sensibly.

**Watch**: margin distribution — should concentrate around `1/(2β)`. If many examples sit well above it, you're effectively running DPO.

---

## KTO

Ethayarajh et al. 2024 (arXiv:2402.01306, ICML 2024). Works on binary "this response is desirable / undesirable" labels — no pairs needed.

**Loss** (per example, with `ρ_θ(x,y) = β · (log π_θ(y|x) - log π_ref(y|x))` and a detached KL estimate `KL_est = E_{x'}[ KL(π_θ(·|x') || π_ref(·|x')) ]`):
```
L_KTO = E [ 1[y desirable]   · ( 1 - σ( ρ_θ - λ_D · KL_est ) )
          + 1[y undesirable] · ( 1 - σ( λ_U · KL_est - ρ_θ ) ) ]
```
`λ_D, λ_U` are gain/loss weights (loss aversion in the prospect-theory sense; defaults can be asymmetric for chat data).

```python
def kto_loss(pol_lp, ref_lp, labels, kl_estimate, beta=0.1,
             lam_d=1.0, lam_u=1.0):
    # pol_lp, ref_lp: [B] response log-probs; labels: [B] in {+1, -1}
    # kl_estimate: scalar, batch-level KL(pol||ref); detach it before passing
    rho = beta * (pol_lp - ref_lp)
    pos = 1 - torch.sigmoid(rho - lam_d * kl_estimate)
    neg = 1 - torch.sigmoid(lam_u * kl_estimate - rho)
    is_pos = (labels > 0).float()
    return (is_pos * pos + (1 - is_pos) * neg).mean()
```

**Tradeoff**: higher variance than DPO when pairs are available (KTO discards pair structure) but works on data DPO can't. The KL estimate is a moving target — re-estimate it periodically.

**Watch**: class balance (extreme imbalance → one sigmoid dominates). KL estimate drift between estimations means batch size too small.

---

## ORPO

Hong et al. 2024 (arXiv:2403.07691, EMNLP 2024). Combine SFT and preference optimization in one pass — no reference model.

**Loss**:
```
L_ORPO = L_SFT(θ; y_w) + λ · L_OR(θ; y_w, y_l)
L_SFT  = - log π_θ(y_w|x)                                    (NLL on chosen)
odds_θ(y|x) = π_θ(y|x) / (1 - π_θ(y|x))
L_OR   = - log σ( log( odds_θ(y_w|x) / odds_θ(y_l|x) ) )
```
`λ ≈ 0.1`. Odds-ratio is the natural reference-free comparison; the partition function cancels.

```python
def orpo_loss(pol_lp_w, pol_lp_l, lam=0.1):
    # log_odds(y) = log p - log(1 - p); use log1p(-exp) for stability
    sft = -pol_lp_w.mean()
    log_odds_w = pol_lp_w - torch.log1p(-torch.exp(pol_lp_w).clamp(max=1 - 1e-7))
    log_odds_l = pol_lp_l - torch.log1p(-torch.exp(pol_lp_l).clamp(max=1 - 1e-7))
    or_term = -F.logsigmoid(log_odds_w - log_odds_l).mean()
    return sft + lam * or_term
```

**Tradeoff**: one training stage, one model. OR term can be unstable when `π_θ(y)` is near 1 (odds blow up); clip the inputs to log1p.

**Watch**: both losses separately. Healthy: SFT drops first, then OR pulls chosen above rejected. Random OR spikes → numerical edge of odds; lower `λ` or improve stability.

---

## SimPO

Meng, Xia, Chen 2024 (arXiv:2405.14734). Reference-free, length-normalized DPO variant.

**Loss**:
```
L_SimPO = - E [ log σ( β · ( (1/|y_w|) log π_θ(y_w|x) - (1/|y_l|) log π_θ(y_l|x) ) - γ ) ]
```
`|y|` is response length in tokens; `γ` is a target reward margin (paper sweeps 0.5–2.0). No `π_ref` — the policy's own length-normalized log-prob is the reward.

```python
def simpo_loss(pol_lp_w, pol_lp_l, len_w, len_l, beta=2.0, gamma=1.0):
    avg_w = pol_lp_w / len_w
    avg_l = pol_lp_l / len_l
    logits = beta * (avg_w - avg_l) - gamma
    return -F.logsigmoid(logits).mean()
```

**Tradeoff**: length normalization counteracts DPO's bias toward longer responses (longer responses have larger absolute log-probs). Sensitive to `γ` — too large → loss saturates, no signal.

**Watch**: average response length on held-out prompts — growing means the length normalization isn't doing its job. Margin distribution should sit above `γ` for chosen responses.

---

## GRPO

Shao et al. 2024 (DeepSeekMath, arXiv:2402.03300). PPO without a critic: use the within-group mean reward as the baseline.

**Setup**: for each prompt `x`, sample K completions `{y_1..y_K}` from `π_old`; score each with reward `r_i` (often a verifiable checker — see [Lecture 15](../lectures/15-rl-verifiable-rewards.md)).

**Advantage** (group-relative):
```
A_i = ( r_i - mean({r_1..r_K}) ) / ( std({r_1..r_K}) + ε )
```
**Loss**:
```
ρ_i(θ) = exp( log π_θ(y_i|x) - log π_old(y_i|x) )
L_CLIP_i = min( ρ_i · A_i, clip(ρ_i, 1-ε_c, 1+ε_c) · A_i )
L_GRPO   = - (1/(B·K)) Σ_{b,i} L_CLIP_i  +  β · E [ KL(π_θ || π_ref) ]
```
`ρ_i` is a *sequence-level* ratio (sum of per-token log-prob differences inside an `exp`).

```python
def grpo_loss(logp_new, logp_old, rewards, kl_per_seq,
              eps_clip=0.2, beta=0.04, eps_std=1e-8):
    # logp_new, logp_old, rewards, kl_per_seq: all [B, K]
    mean = rewards.mean(dim=1, keepdim=True)
    std  = rewards.std(dim=1, keepdim=True) + eps_std
    advantages = ((rewards - mean) / std).detach()
    ratio = (logp_new - logp_old).exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages
    policy = -torch.min(surr1, surr2).mean()
    return policy + beta * kl_per_seq.mean()
```

**Tradeoff**: higher variance than PPO-with-critic (group mean from K samples is noisy) but no critic to train. Sensitive to within-group reward variance — all-zero or all-one groups give zero advantage and no signal.

**Watch**:
- Within-group reward variance — should be > 0 for most prompts; otherwise drop or rebalance the prompt mix.
- Clip fraction (same as PPO).
- KL to reference — runaway KL means `β` too low or the checker is gameable.

---

## References

Verified arXiv IDs (≤ 10 cited beyond the trivially-named papers above; full list below).

- Williams 1992. *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning.* Machine Learning 8. REINFORCE.
- Mnih, Kavukcuoglu, Silver et al. 2013. *Playing Atari with Deep Reinforcement Learning.* arXiv:1312.5602. DQN.
- Silver, Lever, Heess et al. 2014. *Deterministic Policy Gradient Algorithms.* ICML 2014. DPG theorem (background for DDPG).
- Schulman, Levine, Abbeel, Jordan, Moritz 2015. *Trust Region Policy Optimization.* arXiv:1502.05477. TRPO.
- Schulman, Moritz, Levine, Jordan, Abbeel 2015. *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* arXiv:1506.02438. GAE.
- Lillicrap, Hunt, Pritzel et al. 2016. *Continuous Control with Deep Reinforcement Learning.* ICLR 2016. arXiv:1509.02971. DDPG.
- van Hasselt, Guez, Silver 2016. *Deep Reinforcement Learning with Double Q-learning.* arXiv:1509.06461. Double DQN.
- Wang, Schaul, Hessel et al. 2016. *Dueling Network Architectures for Deep Reinforcement Learning.* arXiv:1511.06581. Dueling DQN.
- Mnih, Badia, Mirza et al. 2016. *Asynchronous Methods for Deep Reinforcement Learning.* arXiv:1602.01783. A3C/A2C.
- Bradley, Terry 1952. *Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons.* Biometrika 39, 324–345. Bradley-Terry model.
- Christiano, Leike, Brown et al. 2017. *Deep Reinforcement Learning from Human Preferences.* NeurIPS 2017. arXiv:1706.03741. Preference-based RL.
- Bellemare, Dabney, Munos 2017. *A Distributional Perspective on Reinforcement Learning.* ICML 2017. arXiv:1707.06887. C51.
- Schulman, Wolski, Dhariwal, Radford, Klimov 2017. *Proximal Policy Optimization Algorithms.* arXiv:1707.06347. PPO.
- Fujimoto, van Hoof, Meger 2018. *Addressing Function Approximation Error in Actor-Critic Methods.* ICML 2018. arXiv:1802.09477. TD3.
- Haarnoja, Zhou, Abbeel, Levine 2018. *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.* ICML 2018. arXiv:1801.01290. SAC.
- Haarnoja et al. 2018. *Soft Actor-Critic Algorithms and Applications.* arXiv:1812.05905. SAC with auto-tuned temperature.
- Schrittwieser, Antonoglou, Hubert et al. 2020. *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model.* Nature 588, 604–609. arXiv:1911.08265. MuZero.
- Ouyang, Wu, Jiang et al. 2022. *Training Language Models to Follow Instructions with Human Feedback.* NeurIPS 2022. arXiv:2203.02155. InstructGPT / PPO-RLHF.
- Rafailov, Sharma, Mitchell, Ermon, Manning, Finn 2023. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* arXiv:2305.18290. DPO.
- Azar, Rowland, Piot et al. 2023. *A General Theoretical Paradigm to Understand Learning from Human Preferences.* arXiv:2310.12036. IPO.
- Ethayarajh, Xu, Muennighoff, Jurafsky, Kiela 2024. *KTO: Model Alignment as Prospect Theoretic Optimization.* ICML 2024. arXiv:2402.01306.
- Shao, Wang, Zhu et al. 2024. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* arXiv:2402.03300. GRPO.
- Hong, Lee, Thorne 2024. *ORPO: Monolithic Preference Optimization without Reference Model.* EMNLP 2024. arXiv:2403.07691.
- Meng, Xia, Chen 2024. *SimPO: Simple Preference Optimization with a Reference-Free Reward.* arXiv:2405.14734.
