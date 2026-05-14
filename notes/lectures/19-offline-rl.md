<!-- status: unreviewed | last-reviewed: never -->

# Lecture 19: Offline RL

_Unreviewed — no one has checked this end to end. Treat the math, code, and citations as unverified._

**Time**: ~2-3 h · **Prerequisites**: Lectures 03, 04, 07; helpful if you've seen Lecture 11 (DPO)

---

## The offline setting

Lectures 02–08 all assume you can interact with an environment while training. The policy collects transitions, computes gradients, and updates — then does it again. The data distribution changes as the policy improves. That's online RL.

Offline RL removes that assumption. You have a fixed dataset D = {(s, a, r, s')} collected by some behavior policy μ — maybe an earlier trained agent, a scripted controller, human demonstrations, logged user sessions — and that dataset is all you get. No further interaction. No new trajectories. You need to find a good policy from whatever is there.

This setting comes up constantly in practice. Robotics systems often run expensive physical hardware. Healthcare applications cannot let a learning agent experiment freely. Recommendation systems have years of logged data but can't run A/B tests indefinitely. Anywhere safety or cost constrains live experimentation, offline RL is the relevant framework.

The entire LLM alignment pipeline — RLHF, DPO, and the iterative variants from [Lecture 11: DPO](./11-dpo.md) and [Lecture 17: Online & Iterative Preference Optimization](./17-online-iterative-preference.md) — is offline RL on preference data. The dataset of (prompt, chosen, rejected) triples was collected under some distribution; training does not add new human preferences in the loop. Offline RL is the principled study of what goes wrong here and what to do about it.

---

## Distributional shift and extrapolation error

The failure mode of naive offline RL is clean enough to state precisely.

In standard Q-learning, the Bellman backup is:

```
Q(s, a) ← r + γ max_{a'} Q(s', a')
```

The `max_{a'}` is the problem. During online training, the policy will eventually visit the states that Q says are good — so if Q is wrong about some action, the agent will try it, get the real reward, and correct itself. The data distribution tracks the policy.

In the offline setting, you can never try actions not in the dataset. But the `max_{a'}` still runs over the full action space, including actions that μ never took. Q is essentially unconstrained on those actions — it was never trained to be accurate there — so it can be wildly optimistic. The policy then chases those optimistic actions into states it has no data for, generating even more extrapolated Q-values. The error accumulates across Bellman backups.

Fujimoto, Meger, and Precup (2019, arXiv:1812.02900 — BCQ) demonstrated this empirically. They showed that standard off-policy algorithms like DDPG and DQN, applied to a static dataset, perform near-randomly or worse than the behavior policy, even when the dataset contains very good trajectories. The culprit is what they call **extrapolation error**: the Q-function assigns arbitrary values to out-of-distribution (OOD) state-action pairs, and the greedy policy exploits those values.

Call the behavior policy μ and the learned policy π. The trouble is that π may take actions a with μ(a|s) ≈ 0. There is no data to correct Q at those points. More precisely: the TD error is only well-defined where (s, a) appears in the dataset. At unseen (s, a) pairs, neural network Q-functions generalize — but without the Bellman fixed-point supervision that would normally pull Q toward the true value, that generalization is unconstrained. Gradient descent minimizes the loss on the data; off the data, the Q-surface can be anything.

This interacts badly with the max. When you take `max_{a'} Q(s', a')` across a continuous or large discrete action space, you're looking for the highest point on an underdetermined surface. That maximum tends to land on whichever OOD action the network happens to assign a high value to — and since Q-values there are untrained, they can be arbitrarily large. The policy then heads toward those actions, reaches states with even less data coverage, and the cycle continues.

---

## Policy constraints (BCQ, BEAR)

The first family of fixes keeps π close to μ, so the policy doesn't stray into regions where Q is inaccurate.

**BCQ (Batch-Constrained Q-learning)** — Fujimoto et al. 2019, arXiv:1812.02900. Rather than maximizing Q over all actions, BCQ learns a generative model of the behavior policy (a conditional VAE on (s, a)) and restricts the max to actions that the generative model considers plausible:

```
π(s) = argmax_{a: G(s) samples a} Q(s, a)
```

where G(s) generates K candidate actions from the behavior distribution and takes the best Q among them. The policy can only select actions that look like something μ might have done. If μ never took action a from state s, BCQ will not take it either.

The cost: if μ was suboptimal, π is constrained to be suboptimal too. BCQ can improve over the average behavior in the dataset, but it cannot easily stitch together good fragments from different trajectories to find something better than the best trajectory.

**BEAR (Bootstrapping Error Accumulation Reduction)** — Kumar et al. 2019, arXiv:1906.00949. The intuition in BCQ is to match the distribution of μ. BEAR argues this is too strict: you want actions on the *support* of μ (actions μ takes at all, even rarely), not necessarily actions μ takes frequently. A suboptimal dataset might have taken good actions occasionally; you want to be able to pick those.

BEAR enforces a maximum mean discrepancy (MMD) constraint between the policy and the behavior:

```
max_π E[Q(s, π(s))]  subject to  MMD(π(·|s), μ(·|s)) ≤ ε
```

The MMD is estimated from samples drawn from π and from the dataset. When MMD is small, π stays on the support of μ; as ε grows, the constraint relaxes. The practical implementation adds this as a Lagrangian penalty and jointly learns the policy and the Lagrange multiplier.

Both BCQ and BEAR work: they perform better than unconstrained offline Q-learning. The limitation is that matching the behavior distribution limits how much you can improve over it.

One practical difference between the two: BCQ constrains which actions the policy can select (it filters candidates through the generative model). BEAR constrains how far the policy's action distribution can drift from the data distribution in aggregate. BCQ is more conservative — it effectively does weighted behavioral cloning and can miss good actions that appear infrequently in the data. BEAR is looser, which lets it recover value from rare good actions, but requires tuning the MMD threshold ε and the kernel bandwidth. Neither is obviously better across all datasets; the right choice depends on how suboptimal the behavior policy was and how well-covered the good regions of the state-action space are.

---

## Value pessimism (CQL)

The second family takes a different approach: rather than constraining what actions the policy can take, constrain the Q-values themselves. Make Q pessimistic about OOD actions so that the greedy policy never wants to take them.

**CQL (Conservative Q-Learning)** — Kumar, Zhou, Tucker, and Levine 2020, arXiv:2006.04779.

The standard Bellman loss is:

```
L_Bellman = E_{(s,a,s') ~ D}[(Q(s,a) - (r + γ max_{a'} Q(s', a')))²]
```

CQL adds a regularizer that simultaneously pushes Q down on OOD actions and pulls it up on in-data actions:

```
L_CQL = α * (E_{s~D, a~π}[Q(s,a)] - E_{(s,a)~D}[Q(s,a)]) + L_Bellman
```

The first term takes the expectation of Q over actions drawn from the *current policy* (which may be OOD), and the second takes it over the dataset. The difference is minimized — meaning Q is discouraged from being high on policy actions unless those actions appear in the data.

In continuous action spaces this is approximated by sampling from the current policy and from a uniform distribution. In practice you can also treat it as minimizing Q over a softmax-weighted distribution of actions and maximizing over the data distribution.

The theoretical result (Kumar et al. 2020) is that CQL produces a lower bound on the true Q-function for the current policy — the policy never gets tricked into chasing inflated values, because the regularizer pushes those values down. Policy improvement still happens because the lower bound is tight for actions in the data, where Q is pulled up.

CQL is implemented as a simple additive term on top of any Q-learning update. Concretely, for discrete actions:

```python
import torch
import torch.nn.functional as F

def cql_loss(q_network, states, actions, rewards, next_states, dones,
             gamma=0.99, alpha=1.0):
    """
    CQL loss for discrete actions.

    alpha controls how conservative Q is — larger alpha pushes
    OOD Q-values down more aggressively.
    """
    batch_size = states.shape[0]

    # Standard Bellman target
    with torch.no_grad():
        next_q = q_network(next_states)          # (B, n_actions)
        target = rewards + (1 - dones) * gamma * next_q.max(dim=1).values

    # Current Q at the actions taken in the dataset
    q_all = q_network(states)                    # (B, n_actions)
    q_taken = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Standard Bellman loss
    bellman_loss = F.mse_loss(q_taken, target)

    # CQL regularizer:
    #   push down Q across all actions (logsumexp ≈ max under softmax),
    #   pull up Q on the in-data actions.
    # logsumexp(Q) is a soft upper bound on max(Q).
    logsumexp_q = torch.logsumexp(q_all, dim=1)   # (B,)
    cql_reg = (logsumexp_q - q_taken).mean()

    return bellman_loss + alpha * cql_reg
```

One hyperparameter (α) controls how conservative the Q-function is. Too small and OOD overestimation slips back in; too large and Q is suppressed everywhere, making the policy too pessimistic to improve meaningfully. CQL is sensitive to α in practice.

The CQL regularizer has a clean theoretical interpretation: with the right α, the expected value of the learned Q-function under the policy (what the policy thinks it will get) is a lower bound on the true policy value. This means the policy cannot overestimate its own return. Policy improvement steps on top of a lower-bounded Q are guaranteed not to regress: if CQL says policy A is better than policy B, A is actually better. That guarantee does not hold for unconstrained offline Q-learning or for BCQ, which lack the pessimism term.

---

## Implicit Q-learning (IQL)

BCQ and BEAR constrain what the policy can do. CQL constrains Q-values directly. Both still require querying Q(s, a) for out-of-distribution actions during the update — the policy or regularizer samples OOD actions and evaluates them.

IQL (Kostrikov, Nair, and Levine 2021, arXiv:2110.06169) avoids querying Q on OOD actions entirely, by construction.

The key observation is that the Bellman backup needs `max_{a'} Q(s', a')`, the value of the best action at the next state. In offline RL, computing that max by sampling from the learned policy risks querying OOD actions. IQL avoids the max altogether by separating the problem into two parts:

1. Learn a state-value function V(s) that approximates the value of the *best* actions available in the dataset at state s — without ever evaluating Q on unseen actions.
2. Learn Q(s, a) using V(s') as the backup target instead of `max_{a'} Q(s', a')`.
3. Extract the policy using advantage-weighted regression.

**Step 1: Expectile regression for V.** The key insight is that the standard L2 loss for fitting V would minimize the mean of Q over the behavior distribution — the average quality of actions in the dataset. What you want instead is the value of the *best* actions in the dataset, which corresponds to an upper quantile. Expectile regression generalizes quantile regression using an asymmetric squared loss instead of an asymmetric absolute loss. You can estimate an upper quantile of Q^μ(s, ·) over the behavior distribution by fitting V with:

```
L_V = E_{(s,a) ~ D}[L²_τ(Q(s,a) - V(s))]
```

where the expectile loss is:

```python
def expectile_loss(diff, tau):
    """
    Asymmetric L2 loss. For tau > 0.5, penalizes under-estimation
    more heavily than over-estimation, so V learns to track upper
    quantiles of Q(s,·) over the data distribution.
    """
    weight = torch.where(diff > 0, tau, 1 - tau)
    return (weight * diff.pow(2)).mean()
```

With τ = 0.5, this is standard MSE and V estimates the mean of Q under μ. With τ → 1, V approaches the max of Q under μ — the value of the best in-dataset actions. A typical value is τ = 0.7–0.9. No OOD action is ever queried: V is fit only against Q evaluated at (s, a) pairs that appear in the dataset.

**Step 2: Q update using V.** Because V(s') approximates the value of the best actions at the next state, you can use it in place of the Bellman max:

```
Q(s, a) ← r + γ V(s')
```

Again, only in-dataset (s, a) pairs appear on the left side. No OOD evaluation.

**Step 3: Policy extraction via advantage-weighted regression (AWR).** Once Q and V are trained, extract the policy by cloning actions weighted by their advantage:

```
π = argmax_π E_{(s,a) ~ D}[exp(β(Q(s,a) - V(s))) log π(a|s)]
```

High-advantage actions — those that Q says are better than what V expects at that state — get high weight and are cloned strongly. Low-advantage actions get low weight and are effectively ignored.

Putting these together:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class IQL:
    """
    Implicit Q-Learning (Kostrikov et al. 2021).

    Networks:
      - q1, q2: two Q-functions (take min for stability)
      - v: state value function
      - policy: actor for AWR extraction
    """

    def __init__(self, q1, q2, v, policy, tau=0.8, beta=3.0,
                 gamma=0.99, lr=3e-4):
        self.q1 = q1
        self.q2 = q2
        self.v = v
        self.policy = policy
        self.tau = tau          # expectile quantile for V
        self.beta = beta        # inverse temperature for AWR
        self.gamma = gamma

        self.q_optimizer = torch.optim.Adam(
            list(q1.parameters()) + list(q2.parameters()), lr=lr)
        self.v_optimizer = torch.optim.Adam(v.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    def update_v(self, states, actions):
        """Fit V using expectile regression against the current Q."""
        with torch.no_grad():
            q = torch.min(self.q1(states, actions),
                          self.q2(states, actions))   # use min for safety

        v = self.v(states)
        diff = q - v
        loss = expectile_loss(diff, self.tau)

        self.v_optimizer.zero_grad()
        loss.backward()
        self.v_optimizer.step()
        return loss.item()

    def update_q(self, states, actions, rewards, next_states, dones):
        """Update Q using V(s') as the Bellman target."""
        with torch.no_grad():
            target = rewards + self.gamma * (1 - dones) * self.v(next_states)

        q1 = self.q1(states, actions)
        q2 = self.q2(states, actions)
        loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()
        return loss.item()

    def update_policy(self, states, actions):
        """Advantage-weighted regression: clone actions weighted by exp(β * A)."""
        with torch.no_grad():
            q = torch.min(self.q1(states, actions),
                          self.q2(states, actions))
            v = self.v(states)
            adv = q - v
            # Clamp to avoid numerical blow-up; exp(β*A) can get huge.
            weights = (self.beta * adv).clamp(max=10.0).exp()

        log_probs = self.policy.log_prob(states, actions)
        loss = -(weights * log_probs).mean()

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        return loss.item()
```

IQL is widely used as a baseline in 2024–2025 offline RL work for a few reasons: it's simple, it's stable (no OOD queries means no runaway extrapolation), and it achieves strong scores on D4RL benchmarks (Fu et al. 2020, arXiv:2004.07219). It's also the natural offline RL counterpart to advantage-weighted regression, which predates it as an online method.

The τ hyperparameter matters. Low τ gives V closer to the mean of in-data returns, which is safer but leaves more performance on the table. High τ pushes V toward the best in-data trajectories, which can eke out more improvement but is noisier. In most D4RL results, τ = 0.7 or 0.8 works well.

A complete self-contained update loop, showing all three losses in sequence:

```python
import torch
import torch.nn.functional as F


def expectile_loss_fn(diff, tau):
    weight = torch.where(diff > 0,
                         torch.full_like(diff, tau),
                         torch.full_like(diff, 1 - tau))
    return (weight * diff.pow(2)).mean()


def iql_update(batch, q1, q2, v, policy, q_opt, v_opt, pi_opt,
               tau=0.8, beta=3.0, gamma=0.99):
    """
    One IQL update step.

    batch: dict with keys states, actions, rewards, next_states, dones.
    The order matters: V first, then Q (uses updated V), then policy.
    """
    s  = batch["states"]
    a  = batch["actions"]
    r  = batch["rewards"]
    ns = batch["next_states"]
    d  = batch["dones"]

    # V update: τ-expectile regression against current Q.
    # Only in-dataset (s, a) pairs appear; no OOD actions queried.
    with torch.no_grad():
        q_target = torch.min(q1(s, a), q2(s, a))
    v_pred = v(s)
    v_loss = expectile_loss_fn(q_target - v_pred, tau)
    v_opt.zero_grad(); v_loss.backward(); v_opt.step()

    # Q update: Bellman backup using V(s') instead of max_{a'} Q(s', a').
    with torch.no_grad():
        backup = r + gamma * (1.0 - d) * v(ns)
    q_loss = (F.mse_loss(q1(s, a), backup) +
              F.mse_loss(q2(s, a), backup))
    q_opt.zero_grad(); q_loss.backward(); q_opt.step()

    # Policy update: advantage-weighted behavioral cloning.
    with torch.no_grad():
        adv = torch.min(q1(s, a), q2(s, a)) - v(s)
        w   = (beta * adv).clamp(max=10.0).exp()
    pi_loss = -(w * policy.log_prob(s, a)).mean()
    pi_opt.zero_grad(); pi_loss.backward(); pi_opt.step()

    return {"v_loss": v_loss.item(), "q_loss": q_loss.item(),
            "pi_loss": pi_loss.item(), "mean_adv": adv.mean().item()}
```

The clamp on the advantage before `exp` is not optional — without it, a single large positive advantage can produce `exp(large)` that overflows to `inf` and NaN-poisons the whole batch. In practice, target Q-networks with soft updates stabilize training further; add them if training is unstable.

---

## Decision Transformer

BCQ, BEAR, CQL, and IQL are all value-based: they learn Q, optimize against it, and constrain the optimization to avoid OOD errors. There is a completely different approach that sidesteps value estimation entirely.

Decision Transformer (Chen, Lu, Rajeswaran, et al. 2021, arXiv:2106.01345) treats offline RL as supervised sequence modeling. Given a trajectory (R, s₁, a₁, R₂, s₂, a₂, ...) where R_t is the *return-to-go* from time t (sum of future rewards), a causal Transformer is trained to predict a_t given (R_t, s_t) and preceding context. At inference time, you specify a desired return and the model generates actions to achieve it.

The training objective is straightforward cross-entropy (for discrete) or MSE (for continuous):

```
L = E_{(τ) ~ D}[|| a_t - π(R_t, s_t, history) ||²]
```

No Bellman backup. No Q-function. No distributional shift correction. Just next-action prediction conditioned on desired return.

This is appealing for several reasons. It inherits whatever benefits large Transformers bring to sequence modeling — stable training, long-context memory, easy scaling. The return-conditioning lets you specify at inference time how well you want the policy to perform.

The limitation: Decision Transformer cannot stitch trajectories. If the dataset contains trajectory A that does steps 1–5 well and trajectory B that does steps 6–10 well, but no single trajectory does both, DT has no mechanism to combine them. It learns to imitate what it saw, conditioned on return. A value-based method can in principle learn Q(s, a) correctly for states that appear in good and bad contexts separately, then piece together a policy that picks the good action in each state even if no trajectory demonstrated the full sequence. DT cannot do this by design.

In practice, DT performs well when the dataset contains long, high-return trajectories (so the model has good demonstrations to condition on) and poorly on tasks requiring stitching of suboptimal sub-trajectories. On D4RL's "expert" datasets (which contain near-optimal rollouts), it's competitive. On "medium" datasets (suboptimal), CQL and IQL tend to beat it.

The Decision Transformer framing is worth understanding even if you use IQL in practice, because it's the direct ancestor of the way RLHF is often thought about in language modeling: an SFT model trained to imitate demonstrations (including return-like conditioning on quality) is conceptually a Decision Transformer. When a model is instruction-tuned on high-quality data and then RLHF'd, the SFT stage does something similar to what DT does — condition on good behavior — and the RLHF stage corresponds to the value-based offline RL step that the value-based methods add on top. The two-stage structure (imitation then preference optimization) mirrors the two-phase view that DT + fine-tuning represents.

---

## The bridge to DPO

The connection between offline RL and language model alignment is direct.

DPO ([Lecture 11: DPO](./11-dpo.md)) operates on a fixed dataset of (prompt, chosen, rejected) triples. The "actions" are responses; the "behavior policy" is whatever model generated the preference data. The training objective pushes log π_θ(y_w) up and log π_θ(y_l) down, weighted by β. No new preference labels are collected during training.

That is offline RL. The reward is implicit (the preference label), the behavior policy μ is the data-generating model, and the learned policy π_θ should improve over μ. The same distributional-shift concern applies: DPO's implicit reward is:

```
r_DPO(x, y) = β log(π_θ(y|x) / π_ref(y|x))
```

This is well-calibrated only where the data has coverage — where μ generated responses that were labeled. When π_θ drifts to generate responses in regions the preference dataset never covered, the implicit reward extrapolates without correction. The model can satisfy the DPO loss by exploiting coverage gaps rather than genuinely improving.

[Lecture 17: Online & Iterative Preference Optimization](./17-online-iterative-preference.md) documented the practical finding that iterating — regenerating preference data from the current policy — closes most of this gap. This is the language-model version of going from offline to online RL: replace the static dataset with freshly collected on-policy data. The iterative DPO recipes used at labs in 2024–2025 are the applied manifestation of the same lesson that BCQ and CQL researchers identified in 2018–2020 for robotics datasets.

The offline RL algorithms also anticipate specific practical choices in preference optimization. IQL's advantage-weighted regression shows up as a component in several alignment methods: rather than hard-constraining the policy to the behavior distribution (as in BCQ), or imposing a pessimistic penalty (as in CQL), you upweight in-data good actions proportional to their advantage and let the rest be forgotten. KTO (Kahneman-Tversky Optimization) and some RPO variants use conceptually similar weighting. The β parameter in DPO plays the role of α in CQL — it controls how tightly the policy is regularized toward the reference (behavior) distribution.

The connection is not just metaphorical. The DPO derivation (Rafailov et al. 2023) starts from the KL-constrained RLHF objective and derives a closed-form solution. That solution is a special case of constrained offline policy optimization under a particular reward structure. The optimality conditions that make DPO work are the same distributional-shift conditions that BCQ and CQL analyze, just in a continuous text space instead of a physical state space.

---

## When to use this

**You want offline RL when**: you have logged data, can't collect more (cost, safety, latency, deployment constraints), or want to use historical logs before committing to a live training loop.

A common pattern in robotics and recommendation is offline pretraining followed by limited online fine-tuning: train offline with IQL or CQL until the policy is reasonably good, then deploy it (or run it in a simulator) to collect a small amount of fresh data, and fine-tune on that. This is cheaper than training from scratch online and safer than random online exploration. IQL is particularly well-suited for this because the same advantage-weighted regression loss works for both offline and online phases — you don't need to switch algorithms when you go from static data to fresh rollouts.

**Common pitfalls**:

Dataset coverage is a ceiling. Offline RL can at best learn a policy as good as the best trajectories in the dataset (with stitching-capable methods like CQL/IQL) or as good as the dataset average (with strict imitation-based methods). No algorithm extracts a policy better than the best behavior the dataset represents — you cannot improve over data you don't have.

Evaluation is deceptive offline. Q-values and policy loss metrics computed on the dataset do not reliably predict environment performance. A Q-value that looks good offline may be wildly optimistic on the actions actually taken. To know whether an offline RL policy is any good, you need held-out interactive evaluation — run the policy in the actual environment (or a simulator) and measure real returns. Pure offline metrics are insufficient.

Hyperparameter tuning is harder without interaction. In online RL you can run a policy and see if it works. Offline, you either need a simulator or you commit to evaluation runs with the trained policy. CQL's α and IQL's τ both interact with dataset quality in ways that are hard to predict without environment feedback.

**Rough method comparison on D4RL**. To give a concrete sense of where each method fits:

- Behavioral cloning (just supervised learning on the data) works on expert datasets and fails on medium ones, because it imitates the average behavior rather than the best.
- BCQ improves over behavioral cloning on medium data but is constrained by how good μ was.
- CQL works well across dataset types, including medium-replay and mixed datasets, because it explicitly handles OOD actions. It's the most common strong baseline in the 2020–2022 literature.
- IQL matches or beats CQL with simpler training (no OOD sampling, no α tuning tradeoff) and is the more common starting point from 2022 onward.
- Decision Transformer is competitive on expert datasets, weaker on medium and medium-replay where stitching matters.

**Standard benchmarks**: D4RL (Fu et al. 2020, arXiv:2004.07219) provides standardized offline datasets across locomotion (HalfCheetah, Hopper, Walker), AntMaze navigation, Adroit hand manipulation, and kitchen tasks. Each task has multiple dataset types — "expert" (optimal demonstrations), "medium" (a partially trained policy), "medium-replay" (replay buffer of a medium policy), and "medium-expert" (a mix). The different dataset qualities test different regimes: can you imitate expert data (easy), improve over medium data (moderate), or stitch suboptimal fragments into something better (hard)?

---

## Exercises

1. Implement vanilla DQN on an offline dataset from D4RL (the "medium" HalfCheetah split) without any conservative regularization. Measure the episode return after training. Then add the CQL regularizer from the code snippet above with α = 1.0 and α = 5.0 and compare. The unconstrained version should underperform the behavior policy; CQL should recover some or all of it.

2. Implement IQL on the same dataset using the update loop above. Sweep τ ∈ {0.5, 0.7, 0.9} and plot the final returns. τ = 0.5 gives mean-regression for V; τ = 0.9 gives near-max regression. For suboptimal data, higher τ should win — but watch for instability at τ = 0.9.

3. Train Decision Transformer on D4RL "medium-expert" data (a mixture of expert and mediocre trajectories). Then train IQL on the same data. Compare their final return. For the "expert-only" split, DT should be competitive; for "medium-only," IQL should win. The difference illustrates stitching vs. imitation.

4. Offline evaluation is unreliable: plot Q-value estimates for the unconstrained DQN from exercise 1 versus the CQL version, alongside the actual ground-truth returns from rollouts. The DQN Q-values should be dramatically overestimated relative to ground truth; CQL's should be closer (possibly underestimates). This is the core reason pure offline metrics mislead.

5. DPO as offline RL: take a small preference dataset (100–200 pairs, any open-source dataset will do), train DPO to convergence, and then measure how the implicit reward `β log(π_θ(y|x)/π_ref(y|x))` distributes for chosen vs. rejected responses that were *not* in the training set. Then add new preference pairs generated by the trained policy (one round of iterative DPO), retrain, and re-measure. Does the implicit reward become better-calibrated on the new pairs? This exercise makes the distributional-shift argument concrete in the preference-optimization setting rather than the MuJoCo setting.

---

## References

**Fujimoto, Meger, Precup (2019)**. "Off-Policy Deep Reinforcement Learning without Exploration." ICML 2019. arXiv:1812.02900. Verified. Introduces BCQ; demonstrates that standard off-policy algorithms fail on fixed datasets; names and characterizes extrapolation error.

**Kumar, Fu, Tucker, Levine (2019)**. "Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction." NeurIPS 2019. arXiv:1906.00949. Verified. Introduces BEAR; relaxes BCQ's distribution-matching to support-matching via MMD; analyzes bootstrapping error accumulation.

**Kumar, Zhou, Tucker, Levine (2020)**. "Conservative Q-Learning for Offline Reinforcement Learning." NeurIPS 2020. arXiv:2006.04779. Verified. Introduces CQL; adds a regularizer that pushes Q down on OOD actions and up on in-data actions; proves the resulting Q is a policy-value lower bound.

**Kostrikov, Nair, Levine (2021)**. "Offline Reinforcement Learning with Implicit Q-Learning." arXiv:2110.06169. Verified. Introduces IQL; uses expectile regression for V so Q is never queried on OOD actions; extracts policy via advantage-weighted regression.

**Chen, Lu, Rajeswaran, Lee, Grover, Laskin, Abbeel, Srinivas, Mordatch (2021)**. "Decision Transformer: Reinforcement Learning via Sequence Modeling." NeurIPS 2021. arXiv:2106.01345. Verified. Treats offline RL as return-conditioned next-action prediction with a causal Transformer; competitive on expert datasets, weaker on suboptimal or stitching-required data.

**Fu, Kumar, Nachum, Tucker, Levine (2020)**. "D4RL: Datasets for Deep Data-Driven Reinforcement Learning." arXiv:2004.07219. Verified. Introduces the D4RL benchmark suite; standardizes offline RL evaluation across locomotion, navigation, manipulation tasks with multiple dataset quality levels.

**Rafailov, Sharma, Mitchell, Manning, Ermon, Finn (2023)**. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." arXiv:2305.18290. Cited in Lecture 11. The DPO derivation; the resulting objective is a special case of constrained offline policy optimization on preference data.

---

## Where this leaves the series

The curriculum now runs from foundations — MDPs, Bellman equations, policy gradients, Q-learning, actor-critic — through modern deep RL — TRPO, PPO, off-policy continuous control, model-based methods — into LLM alignment — reward modeling, PPO for language models, DPO, constitutional AI, verifiable rewards, agentic RL — and through the frontier of iterative preference optimization, reasoning with RL, and now offline RL as the principled foundation underneath the alignment work. The arc connects the earliest tabular Q-learning convergence theorems to the practical question of why DPO benefits from iterative on-policy data collection: both are instances of the same distributional shift problem, analyzed at different levels of abstraction.

Offline RL is where the series closes the loop. The foundations lectures (01–08) built tools for an agent that can interact freely. The alignment lectures (09–17) applied those tools under a new constraint — human preferences, not environment rewards — and found that the constraint forced the field to think carefully about data distribution. Offline RL is the general theory of that constraint. Understanding BCQ, CQL, and IQL makes the design decisions in DPO, iterative preference optimization, and the β parameter legible as engineering choices grounded in a body of theory, not just empirical tricks.

For the full index of lectures see [`../README.md`](../README.md). For the curriculum order and topic map see [`../../CURRICULUM.md`](../../CURRICULUM.md).

---

_End of Lecture 19._
