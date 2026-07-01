<!-- status: unreviewed | last-reviewed: never -->

# Lecture 22: World models

_Unreviewed: no one has checked this end to end. Treat the math, code, and citations as unverified._

**Time**: ~3 h · **Prerequisites**: Lecture 08

---

## Where this fits

[Lecture 08: Model-based RL](./08-model-based-rl.md) introduced the basic idea: learn a model of environment dynamics, then use it to plan or to generate synthetic training data. That lecture covered Dyna-Q, MBPO with short rollouts from an ensemble of forward models, and a sketch of Dreamer.

This lecture is the closer look. The world-model field has produced three substantial lines of work since 2018: the **Dreamer family** (latent dynamics with recurrent state-space models), the **MuZero family** (latent dynamics with Monte Carlo tree search), and the **transformer world-model line** (autoregressive token models, sometimes trained on raw video). Each made the model-based promise concrete in a different way. In parallel, the video-generation work after Sora opened a debate about whether large generative video models are already implicit world models, and whether that question has a useful answer.

What you should leave with: a clear definition of what a world model is and isn't, the architectural split between explicit and latent transition models, the three families and what each one does that the others don't, a code sketch you can run for an imagination rollout in latent space, and a defensible position on when model-based is worth the engineering cost in 2026.

---

## What a world model is

A world model is a learned function that predicts what happens next in some environment. The most general form is the joint:

```
p(s_{t+1}, r_t | s_t, a_t)
```

The model takes the current state and the action the agent will take, and returns a distribution over the next state and the reward. With this in hand, an agent can roll forward in imagination: sample an action, sample a next state, sample the reward, sample another action, and so on. The trajectory is fake (no environment was touched), but you can compute returns along it and use them to train a policy.

This is the same shape as the dynamics model from [Lecture 08](./08-model-based-rl.md). The label "world model" is used most often when the model also encodes high-dimensional observations (pixels, video) into a compact latent state, so that prediction happens in latent space rather than pixel space. That naming is fuzzy in practice: MuZero's dynamics network is a "world model" in this sense; PETS's probabilistic ensemble is usually called a "dynamics model." They are the same thing functionally; the term tracks the architecture, not the role.

There are two design axes worth pulling out before going further.

### Explicit vs latent dynamics

An **explicit** transition model predicts directly in the original state space. If states are 4-dimensional vectors (CartPole), the model maps 4D → 4D. If states are 84×84×3 pixel frames, the model maps 84×84×3 → 84×84×3. MBPO uses explicit models on proprioceptive control tasks where the state dimension is small.

A **latent** dynamics model first encodes observations into a compact representation `z`, then learns transitions in latent space:

```
z_t   = encoder(o_t)             # observation -> latent
z_{t+1} = dynamics(z_t, a_t)     # latent rollout
o_{t+1} ≈ decoder(z_{t+1})       # optional, only for training signal
r_t     ≈ reward_head(z_t, a_t)
```

The latent space is typically a few hundred dimensions, regardless of input modality. Three things change once you move to latent dynamics:

- Rollouts are cheap. You don't pay decoder cost during imagination, only when you need to reconstruct an observation for the training loss.
- The transition model has a simpler job. Predicting the next 256-dim latent is easier than predicting the next 21,168-dim pixel frame.
- The representation can be shaped by auxiliary objectives (reward prediction, contrastive losses, KL constraints) so the latent captures task-relevant structure rather than every pixel.

The cost is that the latent dynamics are only as good as the encoder. If the encoder discards information the agent needs, no amount of clever dynamics training will recover it. This shows up in Dreamer's reliance on a reconstruction loss: without forcing the latent to be sufficient for pixel reconstruction, it's tempting for the encoder to throw away useful structure.

### Model role: planning vs imagination training

A second axis is how you use the model once you have it.

- **Planning** uses the model at decision time. Given the current state, search over action sequences, evaluate each one by rolling forward through the model, and pick the action whose rollout looks best. Model-predictive control, cross-entropy method, MuZero's MCTS, all decision-time planning.
- **Imagination training** uses the model at training time. Generate synthetic trajectories from the model, treat them as if they were real experience, and train a value function or policy on them. Dreamer's actor-critic trains on imagined rollouts in latent space.

The two are not exclusive. MuZero does both: MCTS at decision time, replay on real trajectories at training time. Dreamer is mostly the imagination-training pattern. PETS is mostly the planning pattern. Whether to plan or to train policies in imagination shows up again later when you decide what the model is actually for.

---

## The premise: when is a model worth learning?

The honest version of the model-based pitch goes like this. If you have a perfect simulator (Atari, MuJoCo, any video game), model-free RL is fine: interactions are cheap, so sample efficiency stops being a hard constraint. The reason to learn a model is one of:

1. **Real-world data is expensive.** Robotics, drug discovery, real driving, recommendation systems with serving cost. Every interaction has a price, and you want as much learning per interaction as possible.
2. **Planning helps decision quality.** Some problems need lookahead at decision time (board games, scheduling, chess-like tactics), and even a perfect simulator is something you'd want to do tree search over.
3. **You need to inspect or transfer.** A learned model is something you can probe, perturb, condition on counterfactuals. A pure policy is opaque by comparison.

What does not motivate model-based RL by itself: "the model gives us a more general representation." It might, but representation learning is its own field and you can do contrastive or self-supervised pretraining without committing to a transition model.

The premise everywhere is some version of: cheaper data, better decisions, or a representation you can manipulate. If none of those apply, model-free is simpler and usually wins.

---

## Ha & Schmidhuber's World Models (2018)

The paper that popularised the term is Ha and Schmidhuber, "World Models" (arXiv:1803.10122). It's the clearest reference for the three-piece decomposition that later work follows.

The architecture has three components:

- **V (vision)**: a variational autoencoder that compresses each observation `o_t` into a latent `z_t`.
- **M (memory)**: a mixture-density recurrent network that predicts the distribution over `z_{t+1}` given `z_t` and `a_t`. The "memory" carries information across time so the model can be partially observable.
- **C (controller)**: a small linear controller (a few thousand parameters) that maps the concatenation of `z_t` and the MDN-RNN's hidden state `h_t` to an action.

V and M are trained on rollouts from a random policy. C is trained with CMA-ES (evolution strategies, not gradient-based RL) entirely inside the M-rollout. The agent never sees the real environment during C's training. Then C is evaluated in the real environment.

A few things made this paper influential beyond the specific architecture:

- The components are independently trainable. You can stare at V's reconstructions, sample from M to see if rollouts look like the game, then train C. This is debuggable in a way that end-to-end models are not.
- The controller is intentionally tiny. The world model carries the representation; the policy on top is small and fast. Later work (Dreamer, IRIS) drops this asymmetry (the policy gets bigger), but the layering idea persists.
- It works on visually rich tasks. The Car Racing and Doom (ViZDoom) experiments showed that imagination-only training could transfer to the real environment, at least on these specific games.

The limitations: the random-policy data collection bounds what the world model ever sees, so V and M never learn to model the parts of state space that an interesting policy would visit. Subsequent Dreamer work fixes this with a loop: refit the world model as the policy improves and collects new data.

---

## The Dreamer family

The Dreamer line (PlaNet, Dreamer (v1), DreamerV2, DreamerV3) is the most worked-out latent world model program. All four are by Danijar Hafner with various coauthors. They share an architecture pattern (the **recurrent state-space model**, RSSM) and a training pattern (encode → predict → reconstruct → train policy in imagination), and the differences across the four are largely about which losses and target scales to use.

### The recurrent state-space model

The latent state in Dreamer has two pieces:

```
h_t : deterministic recurrent hidden state  (GRU output)
z_t : stochastic latent                     (Gaussian or categorical)
s_t = (h_t, z_t)                            : full latent state
```

The deterministic part `h_t` carries history. The stochastic part `z_t` captures the randomness in transitions: without it, the model would collapse to a deterministic recurrent encoder.

Two distributions over `z_t` are learned:

- **Prior**: `p(z_t | h_t)`. Predicts the latent purely from history. Used at imagination time when there is no real observation.
- **Posterior**: `q(z_t | h_t, o_t)`. Predicts the latent from history *and* the current observation. Used at training time to reconstruct what actually happened.

Training pushes the prior toward the posterior with a KL term, so that at imagination time the prior is a decent stand-in for the posterior. A reconstruction loss on `o_t` from `s_t` keeps the latent informative. A reward prediction loss keeps it tied to the reward function.

The transition update is:

```
h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})        # advance the recurrent state
z_t ~ posterior(h_t, o_t)   or prior(h_t)    # sample the stochastic part
o_t ~ decoder(h_t, z_t)                      # reconstruct (training only)
r_t ~ reward(h_t, z_t)                       # predict reward
```

This is the RSSM. It first appears in PlaNet and is reused (with variations) in every Dreamer paper since.

### PlaNet (Hafner et al., 2018)

PlaNet ("Learning Latent Dynamics for Planning from Pixels," arXiv:1811.04551) introduced the RSSM and used it for **planning**, not imagination training. At decision time, PlaNet runs the cross-entropy method (CEM) over action sequences in latent space:

1. Sample a population of action sequences.
2. Roll each one forward through the RSSM, summing predicted rewards.
3. Keep the top fraction and refit a Gaussian over them.
4. Sample again. Repeat for a few iterations.
5. Execute the first action of the best sequence. Replan at the next step.

This is model-predictive control with a learned latent model. The agent has no policy network at all: every decision goes through CEM. PlaNet showed that this works on DeepMind Control Suite tasks from pixels, with much less environment data than model-free baselines of the time.

### Dreamer (v1) (Hafner et al., 2019)

Dreamer ("Dream to Control: Learning Behaviors by Latent Imagination," arXiv:1912.01603) swapped the planner for a policy. Instead of CEM at decision time, train an actor and a critic in latent imagination:

1. Sample a batch of real states from the replay buffer; encode them to `s_0`.
2. Roll the policy forward through the RSSM for `H` steps (typically H=15), producing imagined `s_1, s_2, ..., s_H` and rewards.
3. Compute λ-returns from the imagined rewards and the critic's value estimates.
4. Update the actor with the policy gradient through the differentiable rollout: the dynamics model is differentiable, so the policy gradient flows through the entire rollout rather than just sampling.
5. Update the critic on the same λ-returns.

The actor and critic are small MLPs on top of `s_t`. Imagination horizons of 15 work because the model is trained on the same buffer and never strays far from on-policy. Crucially, gradients can flow through the model: the policy gradient is "analytic" through the differentiable transition function, not just sampled.

The replay buffer is filled by the policy acting in the real environment, with exploration noise on actions. The world model is retrained on the growing buffer periodically. So the system is a loop: act → store → fit model → train policy in imagination → act.

### DreamerV2 (Hafner et al., 2020)

DreamerV2 ("Mastering Atari with Discrete World Models," arXiv:2010.02193) replaced the continuous Gaussian `z_t` with **categorical** latents. The stochastic part is now a vector of one-hot categoricals, with the straight-through estimator carrying gradients through the sample.

The motivation in the paper: discrete latents are better at capturing multimodal predictions. A car can turn left or right at an intersection; a Gaussian latent tends to average. Categorical latents can represent that bimodality directly.

DreamerV2 was the first model-based method to outperform a strong model-free baseline (Rainbow DQN) on the full Atari benchmark while using comparable data. That result mattered because Atari had been dominated by model-free methods since DQN: model-based was generally seen as a robotics technique, not a video-games-from-pixels one.

### DreamerV3 (Hafner et al., 2023)

DreamerV3 ("Mastering Diverse Domains through World Models," arXiv:2301.04104) is the version most people will use today. The headline result is that a **single fixed hyperparameter setting** works across more than 150 tasks spanning continuous control, Atari, DMLab, Crafter, and Minecraft. Earlier Dreamer versions needed retuning for each environment family; DreamerV3 doesn't.

The technical changes are largely about scale-invariant losses:

- **Symlog prediction.** The reward and value heads predict `symlog(x) = sign(x) * log(|x| + 1)` instead of `x`. The inverse `symexp(y) = sign(y) * (exp(|y|) - 1)` is applied at evaluation to recover the original scale. This compresses large rewards (a reward of 1000 maps to about 6.9) and expands small ones, so the loss surface looks similar whether the environment rewards are in the range of [-1, 1] or [-10000, 10000]. The same trick is applied to value targets.
- **Two-hot encoding for the value head.** Instead of regressing a scalar, the value function predicts a softmax distribution over a fixed grid of bins (e.g., 255 bins spanning a symlog-transformed range). The target is a two-hot vector: probability mass at the two bins straddling the true value, weighted by distance. This converts value regression into classification, which behaves better when targets vary by orders of magnitude.
- **KL balancing with free bits.** The KL between prior `p(z_t|h_t)` and posterior `q(z_t|h_t,o_t)` is split into two terms: `KL[stop_grad(q) || p]` (which trains the prior) and `KL[q || stop_grad(p)]` (which trains the encoder), and weighted separately (1.0 and 0.1 in DreamerV3). Each term is also clipped from below by a "free bits" floor (1 nat per dimension), so the loss is zero when the KL is already small. This stops the encoder from collapsing the posterior into the prior in the early training, which would make the latent uninformative.
- **Replay ratios and training step scheduling** that don't require per-task tuning. DreamerV3 uses a fixed ratio of gradient steps per environment step across all environments, with a small warmup. Earlier Dreamer versions had per-task knobs.

DreamerV3 also collected diamonds in Minecraft from scratch (no demonstrations, no curriculum, no human data), a long-standing milestone in sparse-reward sample-efficient RL.

On Atari100k (a benchmark where the agent gets only 100,000 environment frames, roughly two hours of real-time play), DreamerV3 reaches a median human-normalized score above 1.0, beating EfficientZero and IRIS on aggregate. The exact comparisons shift across paper versions and table conventions: don't quote a specific number without checking the table in the version of the paper you read.

### What you get from Dreamer

If you build a Dreamer-style system today, this is roughly what you sign up for:

- A latent world model with ~10–100M parameters that learns alongside the policy.
- An actor and a critic, both small (1–10M parameters), trained entirely in imagination.
- A training loop that interleaves real rollouts, world model updates, and imagination-based actor-critic updates.
- Sample efficiency comparable to or better than the best model-free methods on most benchmarks, achieved without per-task hyperparameter tuning.

The engineering cost is real. The world model is the dominant cost: it's a non-trivial RNN-plus-CNN with several loss terms, and getting the loss weights and KL schedule right took years of iteration to settle. The DreamerV3 paper's contribution is, in a sense, "we found the recipe that doesn't need to be retuned." Treat that recipe as load-bearing.

---

## The MuZero family

MuZero takes a different bet. Instead of training a policy in imagination, MuZero runs Monte Carlo tree search at decision time, using a learned model to expand the tree. The model doesn't have to reconstruct observations: it only has to predict rewards, values, and policy logits, all in latent space.

### MuZero (Schrittwieser et al., 2019)

MuZero ("Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model," arXiv:1911.08265) generalizes AlphaZero to environments where the rules are unknown. AlphaZero needed a perfect simulator (the rules of chess); MuZero learns one.

The model has three networks:

```
h(o)             -> s   # representation: observation -> latent state
g(s, a)          -> s', r  # dynamics: latent + action -> next latent + reward
f(s)             -> p, v   # prediction: latent -> policy logits + value
```

Note what's missing: there is no decoder. The latent `s` is only required to be useful for predicting reward, value, and policy. It does not have to reconstruct the observation. This is the key architectural departure from Dreamer.

At decision time, MCTS expands a search tree rooted at the current state, with each node holding a (latent, value, policy prior, visit count). The dynamics function expands a node by applying `g(s, a)` to get the next latent and the immediate reward. The value comes from `f`. The selection rule is PUCT, the same one AlphaZero uses, balancing the empirical Q-value with an exploration bonus weighted by the policy prior and visit counts:

```
a* = argmax_a [ Q(s, a) + c(s) * P(s, a) * sqrt(sum_b N(s, b)) / (1 + N(s, a)) ]

c(s) = log( (sum_b N(s, b) + c_base + 1) / c_base ) + c_init
```

`Q(s, a)` is the running mean of values backed up through edge `(s, a)`. `N(s, a)` is the visit count for that edge. `P(s, a)` is the policy prior from `f(s)`. The `c(s)` coefficient grows slowly with total visits, so exploration stays meaningful even when the tree gets deep. Typical values: `c_init ≈ 1.25`, `c_base ≈ 19652`.

A single simulation walks from the root to a leaf by repeatedly applying the PUCT argmax. At the leaf, the dynamics function expands a new node, the prediction function gives the leaf's prior and value, and the value is backed up along the trajectory by averaging into the `Q` values of every edge traversed. The "policy" the agent actually executes at the root is the normalized visit-count distribution `π(a|s) ∝ N(s, a)^(1/τ)` (with temperature `τ` annealed during training). The visit counts implicitly capture the search effort, which is what makes MCTS more conservative than just taking the argmax of `Q`.

Training uses real trajectories from self-play (in board games) or environment interaction (in Atari). The losses are:

- Policy loss: cross-entropy between `p` from `f` and the MCTS visit distribution at that state.
- Value loss: MSE between `v` from `f` and the bootstrapped n-step return from the real trajectory.
- Reward loss: MSE between `g`'s predicted reward and the observed reward.

All three losses are summed across the rollout horizon (typically 5 steps unrolled from each state).

MuZero matched or exceeded AlphaZero on Go, chess, and shogi *without access to the rules*, and matched state-of-the-art model-free agents on the Atari benchmark. The Nature 2020 version is the canonical reference for the full results.

### EfficientZero (Ye et al., 2021)

EfficientZero ("Mastering Atari Games with Limited Data," arXiv:2111.00210) adapted MuZero for the Atari100k regime (only 100k environment frames, roughly two hours of real-time gameplay). Three changes:

- **Self-supervised consistency loss.** The latent predicted by the dynamics network is pushed to match the latent of the actual next observation (encoded with the representation network) using a SimSiam-style objective. This gives a denser training signal for the dynamics function: without it, the only supervision is reward and value, which is sparse.
- **End-to-end prediction of value prefix.** Instead of predicting individual rewards along the unroll and summing them later, EfficientZero predicts the discounted sum of rewards directly. This avoids compounding small per-step prediction errors.
- **Model-based off-policy correction.** When sampling from the replay buffer, the value targets can be re-bootstrapped using the current model rather than the value at collection time.

EfficientZero was the first system to exceed median human performance on Atari100k. That benchmark has continued to move (DreamerV3, IRIS, and later transformer-based world models have all pushed numbers further), but EfficientZero remains the canonical reference for the MuZero-on-limited-data setting.

### Sampled MuZero (Hubert et al., 2021)

MuZero's MCTS enumerates actions at each node. For discrete action spaces with up to ~20 actions this is fine; for continuous or very large discrete spaces, full enumeration is impossible.

Sampled MuZero ("Learning and Planning in Complex Action Spaces," arXiv:2104.06303) replaces enumeration with sampling. At each node, draw `K` actions from the policy prior, and expand the tree only on those. The PUCT objective is reweighted to remain unbiased given the sampling distribution.

This makes MuZero applicable to continuous control (where the action space is uncountable) and to extremely large discrete spaces (such as combinatorial optimization). The cost is increased variance (with finite samples, the tree might miss important actions), but in practice the sampling is well-behaved as long as the policy prior is reasonably exploratory.

### What MuZero buys you (and what it costs)

- **No reconstruction loss.** The latent only needs to support reward, value, and policy. This avoids modeling irrelevant pixel detail.
- **Decision-time planning.** MCTS gives clear lookahead behavior, useful for games and any setting where the value of search-at-test-time is high.
- **Strong empirical results** on the hardest benchmarks (chess, Go, Atari).

The costs:

- The architecture is more involved than Dreamer's. Implementing PUCT, the search tree, the unroll loss, and the training loop correctly is significant work.
- MCTS at decision time costs compute. For 50-simulation search, every decision is ~50 forward passes through the dynamics and prediction networks.
- The latent is not interpretable. Without a decoder, you can't visualize what the model "thinks" is happening, only verify it through downstream value and policy quality.

A reasonable rule: reach for MuZero when planning at decision time is the value proposition (board games, tree-search-friendly tasks, problems with clear discrete action structure). Reach for Dreamer when you want imagination-based training of a fast reactive policy.

### Dreamer vs MuZero, side by side

| Axis | Dreamer (V3) | MuZero |
|---|---|---|
| Latent supervision | Reconstruction + reward + KL | Reward + value + policy only |
| Decoder required | Yes (during training) | No |
| Decision-time use | Reactive policy (forward pass) | MCTS (e.g., 50 simulations) |
| Policy training | Actor-critic in imagination | Cross-entropy on MCTS visit counts |
| Imagination horizon | ~15 steps | ~5 steps (unroll for training); MCTS at decision time |
| Action space | Continuous or discrete | Discrete (Sampled MuZero handles continuous) |
| Gradient through dynamics | Yes (differentiable rollout) | No (used only for prediction, not gradient flow into policy) |
| Decision compute | ~1 forward pass | ~50 forward passes per simulation |
| Sample efficiency on Atari100k | Strong (median HNS > 1.0) | Strong (EfficientZero variant > 1.0) |

The differences trace back to a single design decision: whether to train the latent against a generative reconstruction loss or only against decision-relevant heads. Everything else (the MCTS, the imagination rollouts, the choice of policy class) flows from that.

---

## Transformer world models

The third line replaces the RNN/CNN architecture with transformers operating on a discrete tokenization of observations. The motivation: transformers handle long-range dependencies better than GRUs, scale gracefully with parameter count, and benefit from the engineering pile that has accumulated around them.

### IRIS (Micheli et al., 2022)

IRIS ("Transformers are Sample-Efficient World Models," arXiv:2209.00588) has two components:

- A **discrete autoencoder** (VQ-VAE) that tokenizes each frame into a small grid of discrete tokens (e.g., 16 tokens per frame).
- An **autoregressive transformer** that predicts the next token given all previous tokens and actions. Tokens for one frame are predicted sequentially, then the next action is consumed, then the next frame's tokens, and so on.

The interleaved structure (action, frame tokens, action, frame tokens, ...) is essentially a flat sequence model. Imagination rollouts work by sampling tokens autoregressively until you have a full frame, then conditioning on an action, then sampling the next frame.

A policy is trained on imagined rollouts, similar to Dreamer's actor-critic. On Atari100k, IRIS reaches mean and median human-normalized scores above 1.0, competitive with EfficientZero at the time of publication.

IRIS demonstrated three things:

- A pure-transformer architecture can match RNN-based world models on sample efficiency.
- Discrete tokenization of frames is a viable representation for video.
- The autoregressive sampling cost is tolerable for short rollouts (15–20 steps) at small frame resolution.

The cost is sampling speed. Sampling frame-by-frame token-by-token is slow compared to a single RNN step. This becomes painful at higher resolutions and longer horizons.

### TWM (Robine et al., 2023)

TWM ("Transformer-based World Models Are Happy with 100K Interactions," Robine et al., arXiv:2303.07109) followed IRIS with a slightly different recipe: a transformer over discrete-VAE tokens, trained jointly with a Dyna-style policy update. Also reached strong results on Atari100k. The principal contribution was showing that transformer world models could be trained data-efficiently with a careful tokenizer and replay schedule.

The 2025 follow-up "Improving Transformer World Models for Data-Efficient RL" (arXiv:2502.01591) introduced three further refinements (nearest-neighbor tokenizer, Dyna warmup, block teacher forcing) and reported state-of-the-art on a 1M-step Atari benchmark. The relevant takeaway: the transformer-world-model line is active and the numbers keep moving.

### GAIA-1: world models for driving

GAIA-1 ("GAIA-1: A Generative World Model for Autonomous Driving," Hu et al., arXiv:2309.17080, from Wayve) pushed transformer world models toward a different goal: video generation for autonomous driving. The architecture has two parts:

- An autoregressive transformer over discrete tokens, conditioning on video tokens, text, and actions.
- A video diffusion model that decodes the latent token stream back into high-resolution video.

GAIA-1 is trained on logged driving data. Given a prompt (a few seconds of video, plus optional text and action conditioning), it generates a continuation: what the next several seconds of camera footage might look like, conditioned on a chosen action sequence ("turn left," "brake," "lane change"). The same prompt with a different action conditioning produces a different rollout.

Two things make GAIA-1 interesting as a world model rather than as a video generator:

- The action conditioning is non-cosmetic. Different actions produce trajectories that respect approximate vehicle dynamics: turning left really does move the world to the right in the next frame.
- The scale (9B parameters by the time of the scaled report) is the largest publicly described world model in 2023.

GAIA-1 is *not* used directly for policy training in the paper. It's pitched as a simulator: a way to generate adversarial driving scenarios for training and validation of downstream models. That positioning matters: it's a world model used the way the model-based community always wanted to use them (as a counterfactual generator), but the actual policy still trains in real or scripted environments.

### Genie (Bruce et al., 2024)

Genie ("Genie: Generative Interactive Environments," Bruce et al., ICML 2024, arXiv:2402.15391) pushed in a different direction. Genie is trained on a large corpus of unlabeled internet video. It has no access to actions during training. The architecture:

- A **video tokenizer** that maps frames to discrete tokens.
- A **latent action model** that, given two consecutive frames, infers a discrete latent action that explains the transition. This is a small set of "actions" (typically 8) inferred without supervision.
- A **dynamics model** that, given a sequence of frame tokens and latent actions, predicts the next frame's tokens.

At inference, a user provides an initial frame and a sequence of latent action codes; Genie generates the corresponding video. The latent action vocabulary is small enough that you can map it to controller inputs by hand (left/right/jump in a platformer).

Genie shows that you can extract action-controllable world models from passive video: no labels, no logged action streams, no reward. That's a different premise than the Dreamer/MuZero line, which assumed you had agent-environment interaction data to train on. It also opens the door to scaling laws: throw more internet video at the same architecture and see what happens.

This is one of the points where world models converge with general-purpose generative models. Genie isn't trained for any specific environment; it's trained on whatever video it sees, and the "world" it learns is whatever the videos depict.

---

## Video generation as implicit world modeling

The Sora release in February 2024 (OpenAI's text-to-video model) triggered a debate that's still ongoing: are large video generation models already world models in some functional sense?

OpenAI's framing in their technical writeup was: video models like Sora are "world simulators" in that they can generate physically plausible scenes with consistent object permanence over short timescales. The implicit argument is that the model has had to learn an internal model of physics, object identity, and scene structure to generate coherent video, even though it was trained with a pure generative objective (reconstruct video, then condition on text).

The pushback, most clearly assembled in the survey "Is Sora a World Simulator? A Comprehensive Survey on General World Models and Beyond" (Zhu et al., arXiv:2405.03520, May 2024), is that "world model" has a technical meaning in RL (`p(s_{t+1}, r_t | s_t, a_t)`, conditionable on actions, usable for planning or training), and most video generative models don't satisfy it. Specifically:

- Most video models are not action-conditioned. They generate plausible video continuations but you can't tell them what action to take and have the rollout reflect that.
- Most video models are not used for planning or policy training. Their outputs are evaluated for visual quality, not for downstream control utility.
- The "physics understanding" claim is hard to defend rigorously. There are visible failures (a glass on a table morphing as it falls, an object failing to remain itself across cuts) that suggest the model is doing texture-and-motion synthesis with weak causal structure underneath.

The honest summary is that there's a continuum. GAIA-1 and Genie are action-conditioned video models used as world models in the RL sense: they're on the world-model end. A pure text-to-video model with no action conditioning is on the video-generation end. Whether something in between counts depends on what you're trying to do with it.

For an RL practitioner, the relevant question is: does adding action conditioning to a video generation model produce something useful for control? Genie suggests yes, at least when "action" is a small discrete vocabulary the model can recover from unlabeled video. Whether this scales to dexterous manipulation, real driving, or general-purpose embodied agents is an open empirical question as of early 2026.

---

## The LLM-as-world-model thread

A separate line, mostly orthogonal to the visual world models above, asks whether an LLM can serve as a world model for reasoning and planning tasks.

The cleanest example is "Reasoning with Language Model is Planning with World Model" (Hao et al., EMNLP 2023, arXiv:2305.14992), introducing RAP (Reasoning via Planning). The setup: cast multi-step reasoning as MDP-style planning. The state is the current proof or reasoning trace. The action is the next reasoning step. The "world model" is the LLM, prompted to predict the resulting state. The "policy" is the LLM, prompted to suggest the next action. A reward function (also LLM-derived) scores trajectories.

With these pieces in place, MCTS can search over reasoning trees. RAP showed measurable gains on Blocksworld planning, GSM8K math, and PrOntoQA logical reasoning, by using the LLM in both the world-model and the policy role simultaneously.

The connection back to this lecture: in RAP, the LLM is doing exactly what Dreamer's latent transition model does: predicting the next state given the current state and action. It just happens that "state" is "a partial reasoning trace" and "action" is "a candidate next reasoning step." The MCTS over imagined rollouts is the same template as MuZero, with a frozen LLM in place of the learned dynamics network.

The limitations are the obvious ones. The LLM as a world model inherits all of the LLM's hallucination tendencies: it will confidently predict a next state that's inconsistent with the action it was given. There's no learning loop closing the gap. And the planning time is dominated by LLM forward passes, which are expensive enough that the search budget has to be tight.

Newer work in this thread has tried to fix the hallucination problem by adding verifiers (when the action is "run a code block," the world model can be replaced with actually running the code), and has pushed the planning structure further (depth-first search, beam search with critic models, learned process reward models). The basic shape (LLM doubling as policy and world model, with search on top) is the part that's worth knowing for this lecture.

---

## A worked sketch: imagination rollout against a learned latent model

The following code shows the imagination loop for a Dreamer-style latent world model. It's a faithful skeleton: the loss terms, training schedule, and replay buffer plumbing are omitted, but the rollout logic is what matters for the lecture.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class RSSM(nn.Module):
    """
    Recurrent state-space model (Dreamer-style).

    Latent state is (h, z):
      - h : deterministic recurrent state from a GRU
      - z : stochastic state, here a small Gaussian (DreamerV2/V3 use categoricals)
    """

    def __init__(self, action_dim, h_dim=200, z_dim=30, embed_dim=200):
        super().__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim

        # Recurrent update: GRU consumes previous (z, a) plus carries h
        self.gru = nn.GRUCell(z_dim + action_dim, h_dim)

        # Prior: p(z_t | h_t) -- used at imagination time, no observation
        self.prior_mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, 2 * z_dim),  # mean and log-std
        )

        # Posterior: q(z_t | h_t, e_t) -- e_t is the embedded observation
        self.post_mlp = nn.Sequential(
            nn.Linear(h_dim + embed_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, 2 * z_dim),
        )

    def initial_state(self, batch_size, device):
        h = torch.zeros(batch_size, self.h_dim, device=device)
        z = torch.zeros(batch_size, self.z_dim, device=device)
        return h, z

    def step_recurrent(self, h_prev, z_prev, action):
        x = torch.cat([z_prev, action], dim=-1)
        h = self.gru(x, h_prev)
        return h

    def prior(self, h):
        params = self.prior_mlp(h)
        mean, log_std = params.chunk(2, dim=-1)
        std = F.softplus(log_std) + 1e-3
        return torch.distributions.Normal(mean, std)

    def posterior(self, h, embed):
        x = torch.cat([h, embed], dim=-1)
        params = self.post_mlp(x)
        mean, log_std = params.chunk(2, dim=-1)
        std = F.softplus(log_std) + 1e-3
        return torch.distributions.Normal(mean, std)


class ImaginationRollout:
    """
    Run a policy in imagination using a trained world model.

    No environment is touched. Used for actor-critic training in Dreamer.
    """

    def __init__(self, world_model: RSSM, reward_head, value_head, policy):
        self.wm = world_model
        self.reward_head = reward_head
        self.value_head = value_head
        self.policy = policy

    def rollout(self, h0, z0, horizon: int):
        """
        h0, z0 : starting latent state, shape [batch, h_dim], [batch, z_dim]
        horizon: imagination steps

        Returns:
            states  : list of (h, z) tuples, length horizon+1
            actions : list of actions, length horizon
            rewards : tensor [batch, horizon]
            values  : tensor [batch, horizon+1]  (bootstrap target included)
        """
        states = [(h0, z0)]
        actions = []
        rewards = []
        values = [self.value_head(h0, z0)]

        h, z = h0, z0

        for _ in range(horizon):
            # Policy picks an action from the current latent
            action = self.policy(h, z)
            actions.append(action)

            # Advance the deterministic part
            h_next = self.wm.step_recurrent(h, z, action)

            # Sample the stochastic part from the prior (no observation available)
            z_next = self.wm.prior(h_next).rsample()  # rsample for gradient flow

            # Predict reward at the new latent
            r = self.reward_head(h_next, z_next)
            rewards.append(r)
            values.append(self.value_head(h_next, z_next))

            states.append((h_next, z_next))
            h, z = h_next, z_next

        rewards = torch.stack(rewards, dim=1)   # [batch, horizon]
        values = torch.stack(values, dim=1)     # [batch, horizon+1]
        return states, actions, rewards, values


def lambda_returns(rewards, values, gamma=0.99, lam=0.95):
    """
    Compute λ-returns from a sequence of imagined rewards and value estimates.

    rewards : [batch, T]
    values  : [batch, T+1]  -- value at each state, including bootstrap
    """
    batch, T = rewards.shape
    returns = torch.zeros_like(rewards)
    last = values[:, -1]
    for t in reversed(range(T)):
        # G_t = r_t + gamma * ((1 - lam) * V_{t+1} + lam * G_{t+1})
        returns[:, t] = rewards[:, t] + gamma * ((1 - lam) * values[:, t + 1] + lam * last)
        last = returns[:, t]
    return returns


def actor_critic_update(rollout: ImaginationRollout, h0, z0, horizon=15,
                        actor_optim=None, critic_optim=None):
    """
    Single update step using imagined rollouts.

    Notice: there is NO env.step() in this function.
    All trajectories come from the world model.
    """
    states, actions, rewards, values = rollout.rollout(h0, z0, horizon)

    # λ-returns as targets for the critic and as advantages for the actor.
    returns = lambda_returns(rewards, values)
    advantages = returns.detach() - values[:, :-1].detach()

    # Actor loss: maximize the differentiable return through the model.
    # Because the rollout is fully differentiable (rsample everywhere),
    # we can take a gradient through the entire imagined trajectory.
    actor_loss = -returns.mean()

    # Critic loss: regress the value head onto the returns.
    critic_loss = F.mse_loss(values[:, :-1], returns.detach())

    if actor_optim is not None:
        actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        actor_optim.step()

    if critic_optim is not None:
        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

    return {
        "actor_loss": actor_loss.item(),
        "critic_loss": critic_loss.item(),
        "mean_return": returns.mean().item(),
    }
```

The thing to notice in this sketch: the `actor_critic_update` function has no `env.step()` call. Every state, every reward, every value comes from the world model. The agent is training on imagined experience end to end. The real environment shows up earlier (when fitting the world model on the replay buffer) and later (when collecting new data with the policy), but not in this inner loop.

The differentiable rollout (`rsample` plus a GRU that propagates gradients through time) is what makes the actor loss work as-is. Without differentiable dynamics, you'd need REINFORCE-style policy gradients on top, which the Dreamer family uses for discrete action spaces.

---

## Where world models break

Three failure modes recur across every model-based system.

### Model exploitation

Once the policy is trained against the model, it will find and exploit anywhere the model is wrong. If the model predicts that taking a certain action sequence leads to high reward (but only because the model has never seen that region of state space and is hallucinating), the policy will happily go there.

This is the model-based analog of reward hacking. It's especially visible in the early days of training, when the world model is undertrained and the policy can easily find "cracks" in it. The mitigations are familiar from [Lecture 08](./08-model-based-rl.md):

- **Short rollouts.** Limit the imagination horizon so errors don't compound far enough to dominate the return.
- **Model ensembles with uncertainty.** Treat states where ensemble members disagree as out-of-distribution; penalize the imagined return there. PETS did this; some Dreamer variants don't, which makes them more brittle on hard exploration tasks.
- **Replay refresh.** Re-collect real data with the current policy and retrain the model frequently. As long as the model tracks the policy's distribution, exploitation is limited to that distribution, and the policy can't reach beyond it without the model adapting.

Dreamer keeps this in check by retraining the world model often and using a short imagination horizon (15 steps). MuZero relies on the value function: MCTS uses real-trajectory-bootstrapped value targets, so an over-optimistic dynamics expansion gets corrected by the value estimate at depth.

### Compounding error horizon

Even a well-trained model is wrong at every step. Small per-step errors accumulate. The 10-step prediction error is roughly the 1-step error compounded: exponentially in the worst case, polynomially in the typical case if errors are uncorrelated.

The practical consequence: imagination horizons are short. Dreamer uses H=15. MBPO uses H=1–5 and grows it slowly as the model improves. MuZero's MCTS only unrolls 5 steps in the dynamics-loss training (and during search itself, the bootstrap value handles longer horizons).

The compounding-error wall is the structural reason model-based RL hasn't replaced model-free on tasks where data is cheap. If you have 100M environment frames, model-free can use all of them; model-based has to throttle imagination to keep error bounded, which limits how much value it can extract from the model on top of the real data.

### Representation collapse

A specific failure mode of reconstruction-free models (MuZero, EfficientZero without the consistency loss) is **representation collapse**. The latent has no constraint to be informative beyond what reward and value prediction require. If the rewards are sparse and the value function is poorly learned early in training, the encoder gets almost no useful gradient signal, and the latent can collapse to a near-constant vector. Once that happens, the dynamics function has nothing to work with, the value function has nothing to predict from, and training stalls.

EfficientZero's self-supervised consistency loss is partly motivated by this. By forcing `g(h(o_t), a_t)` to match `h(o_{t+1})` (the next observation encoded through the same representation network), the encoder gets a dense supervisory signal that doesn't depend on the reward being informative. Dreamer side-steps the collapse problem with the reconstruction loss: the encoder has to preserve enough information to decode pixels, so it can't collapse.

The general pattern: latent dynamics models need at least one dense, action-or-observation-grounded loss to avoid degenerate latents. Reconstruction is one choice; self-supervised consistency is another; contrastive predictive coding is a third. A model trained only on reward and value (with sparse rewards and uninitialized value targets) is fragile.

### The simulation-vs-control mismatch

A model that's good at simulating the environment is not automatically good at supporting decisions. The two objectives diverge.

- A model trained to minimize reconstruction loss on observations spends its capacity on visually salient features (textures, lighting, identifiable objects).
- A model used for control needs to capture the features that change with actions and that affect reward.

These two sets of features overlap but are not identical. A leaf blowing in the wind matters for a video model trained on reconstruction but doesn't matter for a driving policy. A subtle change in friction matters for a control policy but doesn't show up in pixels.

This is one reason MuZero's reconstruction-free approach is principled: by training the latent only on reward, value, and policy targets, the model is forced to allocate capacity to control-relevant features. The cost (you can't debug the latent by looking at reconstructions) is real but bounded.

Dreamer accepts the tradeoff in the other direction. The reconstruction loss makes the latent debuggable and keeps it from collapsing, but it also pays attention to features that don't matter for control. DreamerV3's empirical generality suggests the tradeoff works out in practice, but the theoretical objection stands.

The Sora-as-world-simulator discussion is the same tension at a larger scale. A model that perfectly reconstructs video can still be useless for control if it doesn't capture how the world responds to agent actions in the regime that matters.

---

## When to reach for model-based in 2026

A practitioner's heuristic, after eight years of post-Dreamer literature.

**Reach for model-free first** if:
- The environment is a cheap simulator (Atari, MuJoCo, any video game, a deterministic logic puzzle).
- You have ≥1M cheap interactions per training run.
- You want a minimum-fuss implementation. PPO, SAC, or one of their well-tested off-the-shelf implementations will solve most problems faster than you can stand up a Dreamer-style system from scratch.

**Reach for Dreamer (DreamerV3)** if:
- Sample efficiency matters and your environment is image-based.
- You want a single recipe that works across tasks without per-task tuning.
- You're OK with the operational complexity of training a world model alongside a policy.

**Reach for MuZero (or EfficientZero on small data)** if:
- Decision-time planning has clear value (board games, combinatorial decisions, problems where lookahead beats a reactive policy).
- The action space is discrete and small enough for tree search, or you can use Sampled MuZero for larger spaces.
- You're willing to budget compute for MCTS at every decision.

**Reach for a transformer world model** if:
- You're already invested in the transformer toolchain and want to fold a world model into it.
- The environment has long-range temporal dependencies that RNNs handle poorly.
- You can tolerate the autoregressive sampling cost at imagination time.

**Reach for a video-generation-style world model** (Genie, GAIA-1 patterns) if:
- You have a lot of passive video and need to extract action-controllable rollouts from it.
- You're building a simulator for downstream training rather than a policy directly.
- You're prepared to do significant work on the interface between "video tokens" and "actions a real agent would take."

**Reach for an LLM-as-world-model** (RAP, similar) if:
- The task is symbolic reasoning, planning, or anything where the state can be described in language.
- The action space is "next reasoning step" rather than a low-level motor command.
- You have a verifier that can correct the LLM's hallucinations at planning time (code execution, a theorem prover, a search engine).

**Don't reach for a world model at all** if:
- The environment has dynamics that defeat current models: long-horizon stochasticity, complex multi-agent settings without good interaction data, adversarial agents.
- The model would need to be more complex than the policy you're trying to train. There are problems where the policy is simpler than the dynamics; in those cases, model-free wins by default.

The honest summary as of early 2026: model-based RL has earned its place in the sample-efficient regime (DreamerV3 on Crafter/Minecraft, EfficientZero on Atari100k), in the planning-heavy regime (AlphaZero/MuZero on board games), and in the simulator-building regime (GAIA-1, Genie). It has not displaced model-free in the unlimited-data regime, and it probably won't. The convergence with generative video remains an open question: the architectures are increasingly similar, but the evaluation criteria are not yet aligned across the two communities.

---

## Recap

A world model is a learned `p(s', r | s, a)`. It can be explicit (predict in the original state space) or latent (predict in a learned compact representation). It can be used for planning at decision time or for training a policy in imagination, or both.

Ha & Schmidhuber's 2018 paper crystallized the three-piece pattern (encoder, transition model, controller) and trained a controller entirely inside the model. The Dreamer line (PlaNet, Dreamer v1/v2/v3) built the recurrent state-space model and made imagination-based actor-critic training work end-to-end with a single recipe by DreamerV3. The MuZero line dropped the decoder, trained the latent only against reward, value, and policy targets, and added MCTS at decision time. The transformer line (IRIS, TWM, GAIA-1, Genie) replaced the RNN with autoregressive token models and started to converge with general-purpose video generation.

Where world models break is the familiar trio: model exploitation, compounding error, and the gap between simulating and controlling. The mitigations are short rollouts, ensembles, frequent retraining, and (in MuZero's case) skipping reconstruction entirely.

In 2026, the practical question is rarely "model-based or not?" in the abstract. It's "is sample efficiency or planning quality enough of a constraint here that the engineering cost of a world model pays off?" If yes, the recipes are mature enough to deploy. If no, model-free remains the simpler choice.

---

## References

1. **Ha & Schmidhuber (2018)**: "World Models." arXiv:1803.10122.
2. **Hafner et al. (2018)**: "Learning Latent Dynamics for Planning from Pixels" (PlaNet). arXiv:1811.04551.
3. **Hafner et al. (2019)**: "Dream to Control: Learning Behaviors by Latent Imagination" (Dreamer). arXiv:1912.01603.
4. **Hafner et al. (2020)**: "Mastering Atari with Discrete World Models" (DreamerV2). arXiv:2010.02193. ICLR 2021.
5. **Hafner et al. (2023)**: "Mastering Diverse Domains through World Models" (DreamerV3). arXiv:2301.04104.
6. **Schrittwieser et al. (2019)**: "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" (MuZero). arXiv:1911.08265. Published as Nature 588, 604–609 (2020).
7. **Ye et al. (2021)**: "Mastering Atari Games with Limited Data" (EfficientZero). arXiv:2111.00210.
8. **Hubert et al. (2021)**: "Learning and Planning in Complex Action Spaces" (Sampled MuZero). arXiv:2104.06303.
9. **Micheli, Alonso & Fleuret (2022)**: "Transformers are Sample-Efficient World Models" (IRIS). arXiv:2209.00588.
10. **Robine et al. (2023)**: "Transformer-based World Models Are Happy with 100K Interactions" (TWM). arXiv:2303.07109.
11. **Hu et al. (2023)**: "GAIA-1: A Generative World Model for Autonomous Driving." arXiv:2309.17080.
12. **Bruce et al. (2024)**: "Genie: Generative Interactive Environments." arXiv:2402.15391. ICML 2024.
13. **Zhu et al. (2024)**: "Is Sora a World Simulator? A Comprehensive Survey on General World Models and Beyond." arXiv:2405.03520.
14. **Hao et al. (2023)**: "Reasoning with Language Model is Planning with World Model" (RAP). arXiv:2305.14992. EMNLP 2023.

The OpenAI Sora technical report ("Video generation models as world simulators," February 2024) is a web post, not a peer-reviewed paper. Cite the URL if you reference it directly; don't claim it's an arXiv paper.

---

## Exercises (planned, not yet implemented)

1. Implement the RSSM forward pass and verify it can fit a simple environment (e.g., a 2D point-mass with action-conditioned dynamics). Measure how prediction quality degrades over horizon.
2. Take the imagination-rollout sketch above and wire it into a Dreamer-style training loop on `CartPole` or `Pendulum`. Compare sample efficiency to SAC.
3. Implement a small MuZero-style training step: representation, dynamics, prediction networks; one-step PUCT search; cross-entropy on visit counts. Train on a tiny gridworld and check that search depth improves the policy.
4. Replace the RNN dynamics in your Dreamer setup with a small transformer. Compare the two on a partial-observability task (e.g., a delayed-reward gridworld). Measure imagination-time cost.

---

## What's next

Lecture 22 closes out the explicit model-based block. The lectures above this number (in the planned curriculum) move into open exploration, intrinsic motivation, and large-scale RL infrastructure. World models reappear as a building block whenever sample efficiency is the binding constraint: keep this lecture nearby when you hit one.
