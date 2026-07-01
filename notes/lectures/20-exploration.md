<!-- status: unreviewed | last-reviewed: never -->

# Lecture 20: Exploration — from ε-greedy to intrinsic motivation

_Unreviewed — no one has checked this end to end. Treat the math, code, and citations as unverified._

**Time**: ~3 h · **Prerequisites**: Lectures 01, 03

---

## Where this fits

[Lecture 01](./01-mdps-bellman.md) set up the MDP and the Bellman equations under the assumption that you already had a way to estimate `Q(s, a)` for every state-action pair you cared about. [Lecture 03](./03-value-functions-q-learning.md) introduced Q-learning and gestured at ε-greedy as the way the agent collected data. This lecture is about that gesture.

Exploration is the part of RL that asks: how does the agent decide which state-action pairs to try? Every value-based method assumes the data distribution covers what matters. Every policy-gradient method assumes the policy will occasionally sample actions outside its current best estimate so the gradient has something to push against. When those assumptions hold by luck — small state space, dense reward, easy initialization — exploration looks like a non-problem. When they don't, no amount of clever updating fixes it: the agent never sees the reward at all, and there is nothing to update toward.

The classical case study is Montezuma's Revenge. A DQN agent with ε-greedy exploration scores zero on Montezuma's Revenge after 200 million Atari frames, while reaching superhuman performance on most other Atari games in the same training budget (Mnih et al. 2015, doi:10.1038/nature14236; the Atari results are summarized in their Extended Data Table 2). The reason is not that DQN can't learn the game — once you give it any reward signal it does fine. The reason is that ε-greedy never accidentally finds the first key. The first reward in Montezuma's Revenge requires a roughly 100-step sequence of specific actions, and uniform random sampling has effectively zero probability of producing it.

This lecture covers the standard toolbox for that problem, then asks what changes when you swap an Atari-style environment for an LLM that has to discover good reasoning traces rather than good control trajectories.

---

## The exploration-exploitation problem

The split is easy to state and hard to solve. At every decision point the agent has two options:

- **Exploit**: pick the action that looks best under its current estimates. This maximizes expected immediate return assuming the estimates are correct.
- **Explore**: pick an action whose value is uncertain. This may pay off later if the action turns out to be better than current estimates suggest, but it costs you the immediate exploitation reward.

Greedy-only strategies fail because the estimates are wrong, especially early. A bandit pull with `Q(a) = 0.6` from one sample looks better than an unpulled arm with `Q(a) = 0`, but the unpulled arm could be 0.9. Pure exploration fails because it never commits to anything good: the agent spends all its samples on actions it already knows are bad.

The classical formalization is the **multi-armed bandit** — one state, K actions, stochastic rewards drawn from unknown distributions. Even this stripped-down case is non-trivial. The full sequential decision problem (MDPs with state transitions) is strictly harder because the value of exploring an action depends on what state it puts you in, and that state-value depends on the policy from that state onward, which depends on its own exploration.

Two structural reasons exploration is hard in MDPs that don't exist in bandits:

- **Sparse reward**: the reward is zero almost everywhere and non-zero at a handful of states. Most random trajectories see only zeros, so the gradient has no preference among them. Montezuma's Revenge is the canonical case; most realistic robotics tasks have this shape too.
- **Deceptive reward**: the reward gives misleading short-term signal. The example from Ecoffet et al. is a level where a small reward is available near the start, but committing to that reward means you can't reach the much larger reward later. A greedy or near-greedy policy gets stuck on the local optimum and never explores past it. This is harder than sparse reward in a sense: random exploration would actually help, but the value function actively pushes the policy away from the productive directions.

Most of what follows is a catalog of strategies for one or both of these problems.

---

## ε-greedy and its decay

With probability `1 - ε`, take the greedy action `argmax_a Q(s, a)`. Otherwise take a uniformly random action. That's it.

```python
def epsilon_greedy(q_values, epsilon):
    if random.random() < epsilon:
        return random.randrange(len(q_values))
    return int(q_values.argmax())
```

It works when:

- the state space is small enough that uniform random eventually reaches every reachable state,
- rewards are dense enough that random trajectories see non-zero signal frequently,
- the optimal policy doesn't require long sequences of specific actions to reach reward.

It fails when any of those break. The Montezuma's Revenge case has all three pathologies at once: the screen is large, reward is sparse, and reaching reward requires precise sequences.

### Decay schedules

ε is usually decayed over training. Early on you need broad coverage; late in training you want to mostly exploit so reported performance reflects what the policy has actually learned. Common schedules:

- **Linear**: `ε_t = max(ε_min, ε_0 - (ε_0 - ε_min) * t / T_decay)`. The DQN Nature paper used `ε_0 = 1.0`, `ε_min = 0.1`, `T_decay = 1M frames` (Mnih et al. 2015).
- **Exponential**: `ε_t = ε_min + (ε_0 - ε_min) * exp(-t / τ)`. Smoother but more sensitive to `τ`.
- **Reciprocal**: `ε_t = c / (c + t)` for some constant `c`. Goes to zero in the limit, which gives nice asymptotic guarantees but bites you in finite time if `c` is small.

A separate ε is sometimes kept for evaluation runs (often `ε_eval = 0.05` for DQN) so the score doesn't reflect pure greedy behavior, which can be brittle.

### Why uniform random is the wrong primitive

When ε does fire, what does ε-greedy do? It samples uniformly over actions. That's the cheapest thing — but it ignores everything the agent has learned about which non-greedy actions are at least plausible. If the current Q-values say `Q(s) = [10.0, 9.9, -50.0, -100.0]`, ε-greedy with `ε=0.1` will sometimes try action 3 (Q = -100). There is no reason to. Action 1 (Q = 9.9) is the candidate worth checking against the current greedy action; actions 2 and 3 are exploratory tax.

This motivates **Boltzmann (softmax) exploration**:

```
π(a | s) = exp(Q(s, a) / T) / Σ_a' exp(Q(s, a') / T)
```

Temperature `T` controls how sharp the distribution is. `T → 0` recovers greedy; `T → ∞` recovers uniform. The benefit is that actions are sampled in proportion to estimated value, so obviously-bad actions get vanishing probability while the close-call actions get tried. The downside is that Boltzmann is sensitive to the scale of Q. If you double all Q-values, you've effectively halved the temperature. That makes hyperparameter tuning fiddly across environments with different reward scales.

Boltzmann is also undirected: it samples the same distribution at every visit to a state, regardless of how many times you've been there. After a million visits to `s`, an action whose Q-value is slightly below greedy still gets sampled at the same rate as the first visit. There is no notion of "I've already tried this enough."

---

## UCB

Upper Confidence Bound exploration replaces uniform-random tie-breaking with an explicit uncertainty bonus. The classical version, UCB1 (Auer, Cesa-Bianchi, Fischer 2002, doi:10.1023/A:1013689704352), picks the action that maximizes:

```
a_t = argmax_a [ Q_hat(a) + c * sqrt(ln(t) / N(a)) ]
```

where:

- `Q_hat(a)` is the empirical mean reward for action `a`,
- `N(a)` is the number of times `a` has been pulled,
- `t` is the total number of pulls so far,
- `c` is a tuning constant (often √2 in textbook treatments).

The intuition is that you pick the action whose plausible best value (mean plus a confidence bonus) is highest. Actions you've sampled a lot have small bonuses; actions you've barely tried have large bonuses; the bonus shrinks for everyone as `t` grows but shrinks faster for actions you keep pulling.

### Regret bound

UCB1's regret bound is the reason it's a textbook algorithm even though better methods exist. For a stochastic bandit with K arms, rewards bounded in [0, 1], and gaps Δ_a = μ* − μ_a between the optimal arm and each suboptimal arm:

```
R_T ≤ Σ_{a : Δ_a > 0} [ (8 ln T) / Δ_a + (1 + π²/3) * Δ_a ]
```

The leading term is **O((K ln T) / Δ)** where Δ is the smallest non-zero gap. This is logarithmic in horizon `T`, and a matching lower bound (Lai-Robbins 1985) says you can't do asymptotically better in a worst-case sense. The takeaway: with the right uncertainty bonus, exploration cost grows only as `ln T`, not as `T`. Cumulative regret is bounded by a constant times log of how many decisions you've made.

The proof idea (sketch): an action is suboptimal but gets pulled means either (a) its empirical mean is much higher than its true mean (probability decays exponentially in `N(a)`), or (b) the optimal action's empirical mean is much lower than its true mean (same bound), or (c) the confidence bonus is large enough that the suboptimal action still looks best, which forces `N(a) < (constant) * ln(t) / Δ_a²`. Summing over arms and rounds gives the result. Lattimore and Szepesvári's book *Bandit Algorithms* (Cambridge University Press, 2020, no DOI for the open PDF) has a clean version.

### UCB1 on a multi-armed bandit

```python
import math
import random


class UCB1Bandit:
    """UCB1 on a stationary K-armed Bernoulli bandit."""

    def __init__(self, true_means):
        self.true_means = list(true_means)
        self.K = len(true_means)
        self.counts = [0] * self.K
        self.values = [0.0] * self.K  # empirical mean reward
        self.t = 0

    def select(self, c=math.sqrt(2)):
        # Pull each arm once to bootstrap. This is the standard UCB1
        # initialization — without it the ln(t)/N(a) term is undefined.
        for a in range(self.K):
            if self.counts[a] == 0:
                return a
        ucb_scores = [
            self.values[a] + c * math.sqrt(math.log(self.t) / self.counts[a])
            for a in range(self.K)
        ]
        return max(range(self.K), key=lambda a: ucb_scores[a])

    def step(self):
        self.t += 1
        a = self.select()
        # Bernoulli reward draw
        r = 1.0 if random.random() < self.true_means[a] else 0.0
        # Incremental mean update
        self.counts[a] += 1
        n = self.counts[a]
        self.values[a] += (r - self.values[a]) / n
        return a, r


def run_demo(T=10000, seed=0):
    random.seed(seed)
    bandit = UCB1Bandit(true_means=[0.20, 0.25, 0.27, 0.30, 0.50])
    total_reward = 0.0
    for _ in range(T):
        _, r = bandit.step()
        total_reward += r
    optimal_mean = max(bandit.true_means)
    regret = T * optimal_mean - total_reward
    print(f"pulls per arm: {bandit.counts}")
    print(f"empirical means: {[round(v, 3) for v in bandit.values]}")
    print(f"cumulative regret: {regret:.1f}  (vs T*μ* = {T * optimal_mean:.1f})")


if __name__ == "__main__":
    run_demo()
```

The interesting behavior to watch is the `pulls per arm` count. UCB1 spends most of its budget on the best arm (~80–90% of pulls in this configuration after T=10000) but keeps occasionally checking the others. The check rate falls as `1 / Δ²` per the regret analysis, which is why arms with means 0.27 and 0.25 get checked more often than the arm at 0.20 — they're closer to the optimum and so harder to rule out.

### Why UCB doesn't transplant to deep RL directly

UCB needs `N(s, a)` — the number of times you've visited each state-action pair. In a tabular MDP that's a hash table. In Atari, the state is a stack of image frames, and `N(s, a) = 1` for almost every state you've ever seen — every frame is slightly different. The bonus term `sqrt(ln(t) / N(s, a))` would just be a constant, which isn't useful as a directional signal.

The deep RL exploration literature is largely about answering "what's the right generalization of `N(s, a)` when the state is a 210×160×3 image?" Three approaches: pseudo-counts from learned density models (Bellemare 2016), prediction error of a fixed random network (RND, Burda 2018), and prediction error of a learned forward model (ICM, Pathak 2017). All three replace explicit count with a proxy for novelty.

---

## Thompson sampling

A different family of methods avoids the explicit bonus by sampling from a posterior over reward parameters.

For each arm `a`, maintain a Bayesian posterior `p(μ_a | data)`. At each step, draw a sample `μ̃_a ~ p(μ_a | data)` for every arm and pull `argmax_a μ̃_a`. Update the posterior with the observed reward. Repeat.

For Bernoulli rewards with a Beta prior, the posterior after `S_a` successes and `F_a` failures is `Beta(α_0 + S_a, β_0 + F_a)`, which makes the sampling step a one-line call. For Gaussian rewards with a Normal-Normal model, the posterior is Gaussian and similarly tractable.

Thompson sampling has the same `O(log T)` regret rate as UCB on stochastic bandits (Agrawal & Goyal 2012, arXiv:1111.1797) and tends to be slightly better in practice. The randomization smooths out the brittleness of UCB's deterministic tie-breaking. Thompson sampling also generalizes more naturally to contextual bandits and to structured action spaces where maintaining explicit confidence intervals is awkward.

The downside in deep RL: maintaining a useful posterior over `Q(s, a)` parameterized by a neural network is hard. Approximate methods exist — Bayesian dropout, bootstrapped DQN (Osband et al. 2016, arXiv:1602.04621), ensemble methods — and they help, but none are as clean as the bandit case.

---

## Optimistic initialization

The cheapest exploration trick that sometimes works: initialize `Q(s, a)` to a value higher than any plausible true reward. Now greedy action selection automatically explores: every unvisited action looks better than every visited one (because visited actions have been corrected downward), so the agent systematically tries them.

For a tabular Q-table with rewards in `[0, 1]` and discount `γ = 0.99`, set `Q_init = 100` (well above the true max `1 / (1 - 0.99) = 100`, so all of them). The first action gets picked, returns some reward less than 100, and its `Q` value drops. The next greedy choice picks an as-yet-untried action. Eventually every action has been tried enough times for its `Q` to be roughly correct.

This is closely related to UCB — both inject an optimism bias to encourage exploration. The difference is that optimistic initialization is a one-shot bias built into the initial values rather than a recurring bonus, so it decays naturally as the agent collects data. The downside: it doesn't recover from a bad start in non-stationary environments, and it's hard to get the initialization right for function approximators (initializing a deep Q-network to "high values everywhere" is not as simple as `Q_init = 100`).

---

## The deep RL exploration problem

The classical methods above all assume you can either count visits or maintain a posterior. Deep RL with high-dimensional state breaks both. The Atari hard-exploration suite — Montezuma's Revenge, Pitfall, Private Eye, Venture, Solaris, Gravitar — became the standard benchmark for whether a method actually helps where ε-greedy fails. Most papers in this section evaluate primarily on these games, with Montezuma's Revenge as the headline benchmark and Pitfall as the "even harder" benchmark (Pitfall has both sparse reward and deceptive reward, and goes negative for many actions).

For reference points, raw ε-greedy DQN scores zero on Montezuma's Revenge across full training runs (Mnih et al. 2015). Human players get around 4,367 (the average of the human baselines used in the same Nature paper). The methods below all try to close some of that gap.

---

## Pseudo-counts from density models

Bellemare et al. 2016, *Unifying Count-Based Exploration and Intrinsic Motivation* (arXiv:1606.01868), proposed the first method that produced non-trivial scores on Montezuma's Revenge.

The setup is to define a "pseudo-count" `N̂(s)` for any state, including states never exactly seen before, using a density model. If `ρ_n(s)` is the model's probability for `s` after seeing `n` states, and `ρ'_n(s)` is the probability after also training on `s` one more time, the pseudo-count is:

```
N̂(s) = ρ_n(s) * (1 - ρ'_n(s)) / (ρ'_n(s) - ρ_n(s))
```

The intuition: if seeing `s` makes the model substantially more confident about predicting `s` next time, then the model's effective count for `s` was small. If seeing `s` barely changes the model, then the effective count was already large. The exact formula falls out of treating the density model as a recoding-rate estimator and identifying it with the count.

The exploration bonus then mirrors UCB:

```
r_intrinsic(s) = β / sqrt(N̂(s))
```

with `β` a tuning constant. The total reward used for training is the environment reward plus the intrinsic bonus, and the policy is trained on that combined signal.

The original paper used a CTS density model (a pixel-level autoregressive density model from earlier work). A follow-up (Ostrovski et al. 2017, *Count-Based Exploration with Neural Density Models*, arXiv:1703.01310) replaced the CTS model with PixelCNN and got substantially better results — they reported 3,705 on Montezuma's Revenge, which was state-of-the-art at the time.

What this buys you: a directional exploration signal in high-dimensional state space. On Montezuma's Revenge, the agent now seeks out frames it hasn't visited often, which leads it through the rooms to discover keys, doors, and eventually points. It does not solve Pitfall — that requires handling the deceptive reward, which a novelty bonus doesn't address.

What it costs: training a density model alongside the policy is expensive. The density model also has to be careful enough to actually distinguish "seen many times" from "seen once" in pixel space, which is hard when frames vary continuously. The pseudo-count gets noisy in regions of state space where the density model is poorly calibrated.

A related method (Tang et al. 2017, *#Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning*, arXiv:1611.04717) replaces the density model with a hash function: states get mapped to a discrete hash code via a learned encoder, and counts are kept in a hash table. This is cheaper but more sensitive to the encoder's choice of features.

---

## Intrinsic Curiosity Module (ICM)

Pathak et al. 2017, *Curiosity-driven Exploration by Self-supervised Prediction* (arXiv:1705.05363), defines intrinsic reward as the agent's own prediction error on a forward model.

The architecture:

1. An encoder `φ(s)` maps state to a feature vector.
2. A **forward model** predicts `φ̂(s_{t+1})` from `φ(s_t)` and action `a_t`.
3. An **inverse model** predicts `â_t` from `(φ(s_t), φ(s_{t+1}))`.

The encoder is trained by the inverse model's loss (predict which action you took, given two consecutive features). The forward model is trained to predict the next-feature given the current feature and action. The intrinsic reward at step `t` is the forward model's prediction error:

```
r_intrinsic_t = (η / 2) * || φ̂(s_{t+1}) - φ(s_{t+1}) ||²
```

The total reward used for training the policy is `r_intrinsic + r_extrinsic` (often weighted).

The trick is in the encoder. If `φ` were the identity (raw pixels), the agent would get high curiosity reward for any visual change, including TV-screen noise it can't predict but also can't influence. The inverse-model training forces `φ` to encode only the parts of the state that depend on the agent's actions — because that's what's needed to predict the action from two consecutive features. Effects of agent action are encoded; irrelevant background noise is mostly dropped. The forward-model error then reflects "things I did that I can't yet predict the consequences of," which is a useful curiosity signal.

The paper's notable empirical claims: ICM agents successfully explore VizDoom and Super Mario Bros. levels with sparse or no extrinsic reward, learning to navigate and reach goals from intrinsic reward alone. On Montezuma's Revenge they show improvement over ε-greedy but don't claim state-of-the-art.

### The noisy-TV problem

A robot in a room with a TV showing static. The TV is unpredictable — every frame is genuinely random — so a curiosity bonus based on prediction error rewards the agent for staring at the TV forever. The agent's actions don't influence the TV, but it can't predict it either. ICM's inverse-model encoder is designed to filter this out, but in practice the filter isn't perfect, and noisy-TV-style failure modes are a recurring problem for curiosity methods.

This is the motivation for the next method, which sidesteps the problem entirely by predicting a deterministic but arbitrary target.

---

## Random Network Distillation (RND)

Burda et al. 2018, *Exploration by Random Network Distillation* (arXiv:1810.12894), gets state-of-the-art results on Montezuma's Revenge without the architectural fuss of ICM.

The idea is to take a randomly-initialized neural network `f_target: S → R^d`, freeze it, and train a second network `f_predictor: S → R^d` to predict its output. The intrinsic reward at state `s` is:

```
r_intrinsic(s) = || f_predictor(s) - f_target(s) ||²
```

The predictor learns to match the target on states it sees often, so the error goes down. On novel states the error is high. The error tracks novelty without needing an explicit density model, an inverse model, or any architectural commitment beyond two MLPs.

The reason this avoids the noisy-TV problem: `f_target` is deterministic. Given the same input, it always returns the same output. So even if a state has stochastic visual content (a TV showing static), the target output for any specific frame is fixed. The predictor can learn it. There is no infinite source of irreducible error.

RND reported a score of 8,152 on Montezuma's Revenge in the original paper (averaging over runs; best run was ~10,000), substantially better than the prior count-based work. On Pitfall it did not solve the deceptive-reward problem either.

### RND pseudocode

This is a sketch of the predictor-update step plus the intrinsic-reward computation. A full RND implementation also normalizes the intrinsic reward by a running standard deviation and combines it with the extrinsic reward via two separate value heads, but the core mechanism is below.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNDNetwork(nn.Module):
    """A small CNN-or-MLP that maps state to a d-dim feature vector."""

    def __init__(self, state_dim, feature_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, feature_dim),
        )

    def forward(self, s):
        return self.net(s)


class RND:
    def __init__(self, state_dim, feature_dim=512, lr=1e-4, device="cpu"):
        self.target = RNDNetwork(state_dim, feature_dim).to(device)
        self.predictor = RNDNetwork(state_dim, feature_dim).to(device)
        # Freeze the target. It never updates.
        for p in self.target.parameters():
            p.requires_grad = False
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.device = device

    def intrinsic_reward(self, states):
        """No gradient — just the prediction error per state."""
        with torch.no_grad():
            target_out = self.target(states)
            pred_out = self.predictor(states)
            err = (pred_out - target_out).pow(2).mean(dim=-1)
        return err  # shape [batch]

    def update(self, states):
        """One step of predictor training on a batch of states."""
        target_out = self.target(states).detach()
        pred_out = self.predictor(states)
        loss = F.mse_loss(pred_out, target_out)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


# Usage inside a training loop:
#   states = batch of states from the replay buffer / rollouts
#   r_int = rnd.intrinsic_reward(states)   # add to extrinsic reward
#   r_combined = r_ext + beta * r_int
#   <do policy gradient update on r_combined>
#   rnd.update(states)                      # predictor learns the states it just saw
```

A few practical points the original paper emphasizes:

- **Normalize the observations** (running mean/std) before feeding into either network. Without this the prediction error is dominated by global brightness changes and other large-scale effects rather than meaningful novelty.
- **Normalize the intrinsic reward** by a running standard deviation. The raw prediction errors have arbitrary scale; the policy gradient needs a roughly stable reward magnitude.
- **Use two value heads** — one for extrinsic reward (the actual game score) and one for intrinsic reward — and combine them with separate discount factors. The extrinsic value head sees the real game reward and learns a long-horizon value; the intrinsic value head sees only the curiosity bonus and uses a shorter horizon since curiosity bonuses don't persist (once the predictor learns a state, the bonus goes to zero).
- **Use PPO** with many parallel environments. RND was reported with 128 parallel actors. The exploration benefit compounds with rollout diversity.

---

## Never Give Up (NGU)

Badia et al. 2020, *Never Give Up: Learning Directed Exploration Strategies* (arXiv:2002.06038), addresses two problems that RND doesn't.

First, RND's curiosity signal is **lifetime novelty** — once the predictor has seen a state, the bonus goes to zero forever, even if the agent visits a different part of the same episode that requires re-passing through that state. There's no within-episode notion of "I haven't seen this in *this* episode."

Second, RND's intrinsic reward decays as training progresses, since the predictor gradually learns every reachable state. Late in training the curiosity signal is small everywhere, and the agent loses its exploration drive even on tasks where it hasn't yet found all the reward.

NGU adds an **episodic memory** component. Within an episode, every visited state is stored in a memory buffer. The episodic intrinsic reward at state `s_t` is large when `s_t` is far from everything in the buffer (in some learned embedding) and small when `s_t` is close to recently-visited states. The episodic component re-resets at every episode boundary, so even states the agent has seen many times across training can contribute exploration signal if they're novel within the current episode.

This is combined with an RND-style lifetime bonus for the cross-episode component. The total intrinsic reward is a product:

```
r_int = α_episodic * (1 + r_lifetime)
```

The product structure means episodic novelty is gated by lifetime novelty — the agent doesn't waste effort on within-episode novelty if it's already a part of state space it's been to many times.

NGU also trains an explicit ensemble of policies with different exploration intensities (different mixtures of intrinsic-to-extrinsic reward weight) using a shared value function with a discrete "exploration policy index." This lets the agent be more exploitative in some rollouts and more exploratory in others without needing to manually retune `β`.

NGU was the first agent to achieve non-zero scores on **Pitfall** without demonstrations, with reported mean 8,400 on Pitfall in the paper's headline result. This was a notable jump: Pitfall combines sparse and deceptive reward, and earlier methods had been stuck at or near zero. The Agent57 paper (Badia et al. 2020, arXiv:2003.13350) is the follow-up that extended NGU to surpass human baselines on all 57 Atari games, including the remaining hard-exploration ones.

---

## Go-Explore

Ecoffet et al. 2019, *Go-Explore: a New Approach for Hard-Exploration Problems* (arXiv:1901.10995), takes a fundamentally different approach. Instead of injecting exploration via intrinsic reward, it separates exploration into two phases:

1. **Explore until solved**. Maintain an archive of "interesting" states (typically: states with high score, or new state representations). At each iteration, pick a state from the archive, deterministically reset the simulator to that state (Go-Explore requires this — the environment must be deterministic and resettable), and run a random rollout from there. Add any new interesting states to the archive. Repeat.

2. **Robustify**. Once you have a trajectory that achieves high reward, train a policy (via imitation learning + RL) to follow that trajectory robustly, so the resulting policy doesn't need the simulator-reset trick.

The first phase is essentially undirected exploration with checkpointing. The archive guarantees that exploration progress isn't lost: any productive trajectory becomes a starting point for further exploration. The second phase converts the trajectory-set into a policy that can run from the actual initial state.

Go-Explore reported scores of **>2,000,000** on Montezuma's Revenge and **400,000** on Pitfall in its first paper, vastly exceeding all prior methods. Both numbers are essentially "solved" — past any reasonable threshold of human performance.

The critical caveat: Go-Explore in the form described above requires environment-state save/restore. Standard Atari emulators allow this, but most real environments don't. A follow-up version of Go-Explore relaxes this requirement by using a goal-conditioned policy to "return" to an archived state instead of resetting, at the cost of more failed attempts at returning to deep parts of the state space.

What Go-Explore demonstrates is that the hard-exploration Atari problems are tractable if you can separate "find any trajectory that gets reward" from "learn a robust policy that does the same thing." The classical exploration methods conflate these two problems by trying to find reward and learn from it simultaneously. Go-Explore decoupling them makes the search problem dramatically easier — at the cost of requiring the simulator interface.

This is similar in spirit to how RLVR (Lecture 15) handles search for math: generate many candidate solutions, filter by verifier, then train on the filtered traces. The "explore broadly, then concentrate on what worked" structure is the same.

---

## What buys what

| Method | Helps on sparse reward | Helps on deceptive reward | Helps on noisy stochastic state | Notes |
|---|---|---|---|---|
| ε-greedy | No | No | N/A | Baseline. Fails on Montezuma's. |
| Boltzmann | No | No | N/A | Slightly better tie-breaking than ε-greedy. |
| UCB1 | N/A (bandit) | N/A | N/A | Optimal regret for bandits. Doesn't transplant to deep RL. |
| Thompson sampling | N/A (bandit) | N/A | N/A | Same regret rate as UCB, often better in practice. |
| Optimistic init | Sometimes | No | No | Tabular trick; degrades in non-stationary settings. |
| Count pseudo-counts | Yes | No | Sometimes (depends on density model) | Bellemare 2016; first non-zero Montezuma score. |
| ICM | Yes | No | Noisy-TV problem | Pathak 2017; good on Mario / VizDoom. |
| RND | Yes | No | Robust to noisy state | Burda 2018; state-of-the-art Montezuma at the time. |
| NGU | Yes | Sometimes | Yes | Badia 2020; first non-zero Pitfall. |
| Go-Explore | Yes | Yes | Yes (if simulator-resettable) | Ecoffet 2019; effectively solves both games but requires simulator state access. |

The honest summary: dedicated exploration bonuses (count-based, RND, ICM) are good at sparse-reward problems but don't really fix deceptive reward. Deceptive reward needs either a different optimization paradigm (Go-Explore's archive-and-rollout, NGU's episodic memory + intrinsic reward, evolution strategies) or explicit reward shaping.

A related and underrated baseline: noisy networks (Fortunato et al. 2017, arXiv:1706.10295), which add parametric noise to the network weights themselves, generating action diversity through perturbed forward passes rather than action-level epsilon-randomization. This often outperforms ε-greedy on Atari without any explicit intrinsic reward, simply because the action distribution is correlated across timesteps in a useful way rather than independently sampled.

---

## What changes for LLM RL

The exploration story for LLM RL — the GRPO / RLVR pipeline from [Lecture 15](./15-rl-verifiable-rewards.md) — looks almost nothing like the Atari story above. There are good reasons.

### The "state space" is reasoning traces, not screens

In Atari, the agent explores state space — different rooms, different positions, different game states. The set of states is vast and structured by the environment's dynamics, and exploration means traversing more of it.

In LLM RL, the state at each step is the (prompt, partial response) prefix. The action is the next token. From the perspective of the RL machinery this looks like any other sequential decision problem, but the qualitative shape is different. "Exploration" in an LLM doesn't mean visiting unfamiliar tokens — every token has been seen during pre-training. It means exploring different *reasoning traces*: different chains of intermediate steps that the model could write before arriving at an answer.

Two traces that start `Let me think about this problem...` could diverge in step 5 into a wrong calculation or step 5 into a correct one. From a state-space view, those are just two paths through the tree of token sequences. From a reasoning view, those are two strategies. Exploration in LLM RL is mostly about making sure the rollouts cover enough strategies for at least some of them to land on a correct answer that the verifier can score.

### Temperature and top-p as the primary exploration knobs

The standard sampling parameters double as the standard exploration controls.

- **Temperature**: divides logits before the softmax. Higher temperature flattens the distribution and increases diversity. GRPO training typically uses `temperature = 0.7` to `1.0` for rollouts (compared to `0.0` or `0.1` for deterministic deployment). The DeepSeekMath paper used `temperature = 0.6` for sampling; many open implementations use 0.7 or 1.0.
- **Top-p (nucleus) sampling**: at each step, keep only the smallest set of tokens whose cumulative probability exceeds `p`, renormalize, and sample. Common defaults are `p = 0.9` or `0.95`. This cuts the tail without flattening the head.
- **Top-k**: keep only the top-k tokens. Sharper than top-p and used less often in recent practice.

These are exploration in a structural sense — they directly determine the diversity of the rollouts you train on. If temperature is too low, every rollout for the same prompt is identical, the group standard deviation in GRPO is zero, and there is no gradient. If temperature is too high, rollouts include enough garbage that the verifier never gives correctness reward and the format reward dominates.

The DeepSeekMath paper reports that they tried different temperatures and found that around 0.6–0.7 worked well for math reasoning. Higher temperatures gave more diverse but lower-quality rollouts; lower temperatures gave less diversity but more reliable structure. The trade-off is direct: you want enough exploration to find correct solutions sometimes, but not so much that the rollouts are uninformative.

### Best-of-N as inference-time exploration

A complementary lever is best-of-N at inference: sample N completions, score each with a verifier (or a reward model), return the highest-scoring one. Performance on math and code typically improves substantially with N — pass@64 is much higher than pass@1 for the same model — and this is essentially exploration applied at inference time rather than training time. The exploration here doesn't update the policy; it just multiplies the chances that any one sample is correct.

This intersects with training-time exploration in a useful way: a model that produces diverse rollouts (high entropy) benefits more from best-of-N than a model that always produces the same answer. Low entropy means low pass@N relative to pass@1. The exploration-related failure mode in RLVR is **entropy collapse**, discussed in Lecture 15: the policy converges to a single reasoning style, group std collapses, and best-of-N gets you nothing.

### High-temperature rollouts in GRPO

The mechanic that connects exploration to training in GRPO is straightforward. For each prompt:

1. Sample K rollouts at temperature `T > 0` (typically 0.6–1.0).
2. Score each rollout with the verifier.
3. Compute group-relative advantages.
4. Apply the GRPO update, including a KL penalty to a reference policy.

The KL penalty is doing some of the work that intrinsic motivation does in Atari RL: it keeps the policy from collapsing into a narrow distribution. Lecture 15 covers this. Without the KL penalty, the policy entropy decays quickly because optimization with binary rewards has no incentive to maintain diversity once one good solution has been found.

A more direct exploration mechanism that has shown up in some RLVR work is an **entropy bonus**: add `λ * entropy(π)` to the policy loss to explicitly reward distribution spread. This is the classical SAC-style entropy regularization (Haarnoja et al. 2018, arXiv:1801.01290) applied to LLM policies. It helps but is sensitive to the coefficient; too high and the policy becomes incoherent, too low and entropy still collapses.

### What's mostly absent

You generally don't see count-based bonuses, RND-style novelty signals, or ICM-style curiosity in published LLM RL pipelines. There are conceptual reasons.

- **Counts don't make sense over text**. Two completions to the same prompt are almost never identical token-for-token, so the count-of-trajectories is always 1. You could try counting on the level of features (e.g., embeddings of the reasoning trace), but the engineering effort hasn't been justified by results.
- **The verifier already provides a strong signal**. In the RLVR setting, the verifier gives binary correctness on every rollout. That's a richer training signal than intrinsic curiosity, and adding more reward signals on top would dilute it without obvious gain.
- **The hard-exploration analog isn't really sparse-reward Atari**. It's more like: "the model never produces a correct reasoning trace for hard problems." The fix is not to add curiosity to the rollouts; it's to either (a) get the model to a higher base capability via SFT or pre-training so its rollouts sometimes succeed (the DeepSeek-R1 cold-start SFT does this), or (b) seed the policy with high-quality demonstrations (rejection-sampling fine-tuning, STaR, etc., from Lecture 15).

So the LLM RL story for exploration is "tune the sampling temperature, set the KL penalty, watch entropy, and hope your base model is good enough that some rollouts succeed." If they don't, the response is to improve the base model or do warm-start SFT — not to add an intrinsic-motivation term.

There is some early work on more elaborate exploration for reasoning RL — for instance, methods that explicitly maintain diversity across the K completions in a group, or that branch reasoning at intermediate steps using MCTS-style search. These haven't yet displaced the simple "temperature + KL + verifier" approach in published frontier-model training, and a person reading this lecture should treat claims about more sophisticated exploration as live research rather than settled practice.

### A note on coverage

A useful reframing: exploration in LLM RL is mostly about **coverage of reasoning strategies**, not coverage of state space. If your training set has 1000 math problems and the model can sometimes solve 600 of them, the exploration question is roughly "are the 400 failing problems failing because the model can't sample the right approach, or because the model can sample the right approach but rarely?" The answer changes what you should do:

- **Can sample correct approach but rarely**: more rollouts per prompt (larger K), higher temperature, more training steps. Standard GRPO exploration tuning.
- **Cannot sample correct approach at all**: more base capability is needed. SFT on demonstrations, distillation from a stronger model, or curriculum from easier problems.

This is closer in spirit to bandit exploration (which arm should I pull more?) than to MDP exploration (what region of state space haven't I visited?). The reasoning-trace analog of "have I tried this arm enough?" is "has this prompt been rolled out enough at high enough temperature to plausibly find a correct trace?"

---

## A small experiment to try

A self-contained experiment that doesn't need much compute, useful for building intuition for the trade-offs:

Take a 4-armed Bernoulli bandit with means `[0.20, 0.25, 0.27, 0.30]` — all close, no obvious winner. Run for `T = 5000`. Compare four strategies:

1. **Pure greedy** (after one pull per arm to initialize). Will often lock onto the wrong arm.
2. **ε-greedy with `ε = 0.1`** fixed.
3. **UCB1** with `c = √2`.
4. **Thompson sampling** with `Beta(1, 1)` prior.

Plot cumulative regret over time for each, averaged over 100 seeds. The classical result is that greedy has linear regret (often locks onto a sub-optimal arm and stays there), ε-greedy has linear regret with smaller slope (the ε * (mean gap) tax never goes away), and UCB1 and Thompson have logarithmic regret. UCB1's regret line is bumpy due to deterministic tie-breaking; Thompson is smoother.

The bandit case is where the regret theory is cleanest. The point of running this isn't to discover the result — it's to internalize the difference between linear and logarithmic regret. After you've seen ε-greedy never close the gap on this setup, the case for more careful exploration in deep RL is less abstract.

A modest extension: rerun the same comparison on a small **gridworld** with a single rewarded state two-thirds of the way across, walls that funnel the agent away, and `γ = 0.99`. Use tabular Q-learning under (a) ε-greedy with `ε = 0.1`, (b) optimistic init with `Q_init = 10`, (c) ε-greedy with count-bonus `r' = r + β / sqrt(N(s, a))` for `β = 0.1`. The count-bonus version typically explores faster than the other two and reaches the rewarded state in fewer episodes. The gridworld is where you can still see and reason about the count `N(s, a)` directly, before everything becomes a learned approximation in pixel space.

---

## Failure modes to recognize

A summary of where exploration breaks, regardless of method:

- **Sparse reward and small budget**: if the agent never sees a reward, no exploration method helps. Even Go-Explore needs the simulator to produce reward when reached; if reward only triggers after a 10,000-step sequence and you have a 1M-step budget, even unbiased random search has very low probability of finding it. The fix is usually a reward shaping function, demonstration data, or a curriculum of easier sub-problems.
- **Deceptive reward without dedicated handling**: novelty bonuses don't help when the agent has high intrinsic reward for finding the local optimum. Pitfall is the canonical example. Methods that explicitly handle this — Go-Explore, NGU — pay extra in either simulator requirements or architectural complexity.
- **Noisy-TV / stochastic environment**: curiosity methods rewarded for raw prediction error get trapped by genuinely random parts of the environment. ICM's inverse-model encoder is supposed to fix this but doesn't always. RND avoids it structurally because the prediction target is deterministic.
- **Reward scale changes**: if the intrinsic reward grows or shrinks relative to extrinsic reward as training progresses, the effective exploration weight is moving without you controlling it. Most modern implementations normalize the intrinsic reward by a running standard deviation. Without this, the relative weight of `r_int` and `r_ext` drifts.
- **LLM entropy collapse**: in RLVR, if the policy converges to one solution, group std goes to zero and gradients vanish. Watch entropy explicitly. Add an entropy bonus or raise temperature if it drops sharply.
- **Distribution shift between training and deployment**: a policy trained with `T = 1.0` rollouts may behave differently when deployed at `T = 0.0`. The greedy mode of a high-entropy policy can be a worse trajectory than any of the high-entropy samples it was trained on, because greedy is a different distribution. Evaluation at deployment temperature, not training temperature.

---

## When to reach for which method

A rough decision tree, for completeness:

```
Is this a bandit (single state)?
├── Yes → Thompson sampling (if Bayesian posterior is tractable),
│         otherwise UCB1. Either gets you log T regret.
│
└── No → Is the state space small enough for tabular methods?
    │
    ├── Yes → ε-greedy with decay is usually enough. Add optimistic
    │         initialization or count bonuses if you have sparse reward.
    │
    └── No → Is reward dense or sparse?
        │
        ├── Dense → ε-greedy with decay on epsilon. Noisy networks
        │            (Fortunato 2017) often beats ε-greedy without
        │            extra reward terms.
        │
        └── Sparse → Add an intrinsic motivation signal:
            │       - RND if the environment has stochastic visual
            │         elements (avoid noisy-TV via deterministic target);
            │       - ICM if you want the inverse-model interpretability
            │         and accept the noisy-TV risk;
            │       - Pseudo-counts (Bellemare 2016) for theoretical
            │         elegance but more engineering effort.
            │
            │       Is reward also deceptive?
            │
            ├── No  → RND or ICM is likely enough.
            └── Yes → Need episodic memory (NGU) or archive-based
                      search (Go-Explore, if simulator is resettable).
                      No magic; both add infrastructure complexity.

For LLM RL with verifiable rewards:
  Tune temperature and KL penalty. Watch entropy.
  Don't reach for intrinsic motivation — work on the base model and
  the verifier first, and adjust K (group size) and temperature.
```

---

## References

All arXiv IDs verified against arxiv.org.

**Classical**

- Auer, Cesa-Bianchi, Fischer. 2002. "Finite-time Analysis of the Multiarmed Bandit Problem." *Machine Learning*, 47(2-3), 235–256. doi:10.1023/A:1013689704352. — The UCB1 algorithm and the `O(K ln T / Δ)` regret bound.
- Agrawal, Goyal. 2012. "Analysis of Thompson Sampling for the Multi-armed Bandit Problem." arXiv:1111.1797. — First near-optimal regret bound for Thompson sampling.
- Sutton, Barto. 2018. *Reinforcement Learning: An Introduction*, 2nd edition. MIT Press. — Chapter 2 (multi-armed bandits) and §13.4 (exploration vs exploitation in policy gradient) are the textbook references for everything in the first half of this lecture.
- Lattimore, Szepesvári. 2020. *Bandit Algorithms*. Cambridge University Press. — The bandit reference if you want the full regret analysis for UCB and Thompson with clean proofs.

**Pseudo-counts and density-based exploration**

- Bellemare, Srinivasan, Ostrovski, Schaul, Saxton, Munos. 2016. "Unifying Count-Based Exploration and Intrinsic Motivation." arXiv:1606.01868. — Defines pseudo-counts from a CTS density model, first non-trivial Montezuma's Revenge score.
- Ostrovski, Bellemare, van den Oord, Munos. 2017. "Count-Based Exploration with Neural Density Models." arXiv:1703.01310. — Follows up with PixelCNN density model, ~3,700 on Montezuma.
- Tang, Houthooft, Foote, Stooke, Chen, Duan, Schulman, De Turck, Abbeel. 2017. "#Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning." arXiv:1611.04717. — Hash-based count proxy instead of a density model.

**Curiosity and prediction-error methods**

- Stadie, Levine, Abbeel. 2015. "Incentivizing Exploration In Reinforcement Learning With Deep Predictive Models." arXiv:1507.00814. — Earlier prediction-error-as-curiosity work.
- Pathak, Agrawal, Efros, Darrell. 2017. "Curiosity-driven Exploration by Self-supervised Prediction." arXiv:1705.05363. — ICM. Forward + inverse model for curiosity; noisy-TV problem is acknowledged here.
- Burda, Edwards, Storkey, Klimov. 2018. "Exploration by Random Network Distillation." arXiv:1810.12894. — RND. Cleanest avoidance of noisy-TV; best Montezuma's Revenge score at publication.

**Episodic memory and combined methods**

- Badia, Sprechmann, Vitvitskyi, Guo, Piot, Kapturowski, Tieleman, Arjovsky, Pritzel, Bolt, Blundell. 2020. "Never Give Up: Learning Directed Exploration Strategies." arXiv:2002.06038. — NGU. Episodic + lifetime intrinsic reward, first non-zero Pitfall without demonstrations.
- Badia, Piot, Kapturowski et al. 2020. "Agent57: Outperforming the Atari Human Benchmark." arXiv:2003.13350. — Builds on NGU to surpass human-baseline on all 57 Atari games.

**Archive-based methods**

- Ecoffet, Huizinga, Lehman, Stanley, Clune. 2019. "Go-Explore: a New Approach for Hard-Exploration Problems." arXiv:1901.10995. — Archive of interesting states + simulator reset; effectively solves Montezuma's and Pitfall.

**Other exploration mechanisms**

- Fortunato, Azar, Piot et al. 2017. "Noisy Networks for Exploration." arXiv:1706.10295. — Parameter-space noise as exploration. Often beats ε-greedy on Atari.
- Osband, Blundell, Pritzel, Van Roy. 2016. "Deep Exploration via Bootstrapped DQN." arXiv:1602.04621. — Ensemble of Q-networks; approximate Thompson sampling for deep RL.
- Haarnoja, Zhou, Abbeel, Levine. 2018. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor." arXiv:1801.01290. — Maximum-entropy framework; the principle behind entropy regularization as an exploration mechanism in modern actor-critic.

**Atari benchmark reference**

- Mnih, Kavukcuoglu, Silver, Rusu, Veness, Bellemare, Graves, Riedmiller, Fidjeland, Ostrovski, Petersen, Beattie, Sadik, Antonoglou, King, Kumaran, Wierstra, Legg, Hassabis. 2015. "Human-level control through deep reinforcement learning." *Nature*, 518(7540), 529–533. doi:10.1038/nature14236. — DQN on Atari, including the Montezuma's Revenge zero score under ε-greedy.

**For LLM RL exploration**

- Shao et al. 2024. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." arXiv:2402.03300. — GRPO, including sampling temperature settings used in practice. See also Lecture 15.
- DeepSeek-AI. 2025. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv:2501.12948. — R1-Zero and R1 training, where the cold-start SFT functions as an "exploration warm-up" that makes the rollouts useful enough for GRPO to find learning signal. Also covered in Lecture 15.

There is likely relevant recent work specifically on exploration for reasoning RL (entropy regularization, branched rollouts, MCTS-style search) that this lecture doesn't cover; you should not treat the LLM-side coverage as comprehensive.

---

## Next lecture

This is the final numbered lecture in the current series. The reading lists in [`reference/papers/`](../../reference/papers/) cover topics that don't have dedicated lectures yet, including more recent work on exploration for reasoning models.
