<!-- status: unreviewed | last-reviewed: never -->

# Lecture 32: Meta-RL and in-context RL

_Unreviewed — no one has checked this end to end. Treat the math, code, and citations as unverified._

**Time**: ~2-3 h · **Prerequisites**: Lectures 02, 04, 19

---

## The setup

Every algorithm in lectures 02 through 08 trains on one task. Pick CartPole, pick HalfCheetah, pick Atari Breakout — you put a policy through millions of environment steps, and the thing it ends up doing is "solve this task." If you swap the cart pole for a cart-and-three-poles, the policy starts from scratch.

Meta-RL changes the question. Suppose you have a distribution of tasks p(T) — different cart-pole masses, different maze layouts, different MuJoCo bodies, different reward locations in a gridworld. Each task is its own MDP with its own dynamics and rewards. The goal is to train an agent that, when handed a *new* task drawn from p(T), reaches good performance quickly — in a few episodes, or in some setups a few thousand environment steps. The training signal is performance after adaptation, not performance on any fixed task.

The framing borrows directly from few-shot supervised learning. In few-shot classification you don't train a classifier for a single label set; you train a meta-learner that takes K labeled examples of new classes and produces a classifier for those classes. Meta-RL is the same shape: K episodes (or K transitions, depending on the algorithm) on a new task, then a policy that does well on that task.

The reason to care: tasks share structure. If you have a hundred maze layouts and they all use the same physics, the same controls, and the same general goal type, an agent that has solved 99 of them shouldn't need to relearn how walls work for the 100th. Meta-RL formalizes that intuition and gives you algorithms that exploit the shared structure.

There's a second reason that's become more interesting recently: meta-RL is one of the cleanest bridges to what large language models do at inference time. An LLM given a few-shot prompt — "input → output, input → output, input → ?" — is conditioning on a small dataset to behave differently. That is in-context learning, and from one angle it's a form of meta-learning. The connection lets us read in-context RL with transformers as a continuation of the meta-RL story rather than a separate phenomenon, and it lets us reuse intuition from one to think about the other.

This lecture covers two main families of meta-RL (gradient-based and recurrence-based), the transformer-meta-RL bridge that runs through Decision Transformer and Algorithm Distillation, and how all of this sits next to LLM in-context learning. Then the failure modes, because they're significant.

---

## What the meta-objective looks like

Before the algorithms, the objective. A standard RL objective is:

```
J(θ) = E_{τ ~ π_θ} [ Σ_t γ^t r_t ]
```

You maximize expected return under the policy you're training, on the single MDP you're training on. Meta-RL replaces this with a *meta-objective* over a distribution of MDPs:

```
J_meta(θ) = E_{T ~ p(T)} [ J_T(θ_T') ]
```

Where:
- T is a task drawn from a task distribution p(T) — each T defines its own MDP.
- θ_T' is the parameters after adapting to T from some procedure starting at θ (a few gradient steps, or rolling out a recurrent net, or sliding a context window).
- J_T(·) is the standard RL return on task T.

The thing you optimize over is θ, the *pre-adaptation* parameters, but you only get credit for what happens *after* adaptation. This is the inner-outer structure that shows up in every meta-RL method, and it's the source of most of the difficulty.

A concrete sketch of the procedure:

```
for meta-iteration in 1..N:
    sample a minibatch of tasks {T_1, ..., T_B} ~ p(T)
    for each T_i:
        do an "inner-loop" adaptation: collect data on T_i, produce adapted policy θ_i'
        evaluate adapted policy on T_i (more data, more rollouts), get J_T_i(θ_i')
    "outer-loop" update: ∇_θ (1/B) Σ_i J_T_i(θ_i')
```

The split is sometimes called "meta-train" (the outer optimization) and "meta-test" or "few-shot adaptation" (the inner loop applied to a held-out task at evaluation time). When papers report a meta-RL number like "78% success after one gradient step on a held-out maze," they mean: meta-trained on the maze distribution, evaluated on a maze p(T) hasn't shown the model before, with one gradient step using K rollouts of that specific maze.

How the inner loop produces θ' is the main design choice, and it splits the field.

### The exploration question

Before the algorithms, one thing is worth flagging because every meta-RL method has to handle it implicitly and most papers don't make it explicit.

In single-task RL, exploration is about how to *find* a good policy on the MDP you're training on. In meta-RL, exploration also has to happen *during* the inner loop on each new task. When the agent is handed task T at meta-test time, it doesn't know which task it is yet — it has to take some actions, see some rewards, and *infer* the task from observations before exploitation is even possible.

So a meta-RL agent has two related exploration jobs:

- Meta-exploration: outer-loop exploration of the task distribution during meta-training. This is what the outer optimizer handles.
- Task-inference exploration: inner-loop, in-task exploration to figure out *which* task you're on. This is what the inner-loop adaptation procedure needs to be good at.

Gradient-based methods (MAML) handle this implicitly: the K inner-loop rollouts on the new task supply some exploration just by being rollouts. If the policy you adapted from is reasonably wide (high-entropy), the K rollouts cover enough of the action space to give a useful inner gradient. If the meta-initialization has collapsed to a near-deterministic policy, K rollouts will all be near-identical and the inner gradient is uninformative. Some MAML variants add an explicit entropy bonus to the meta-objective to keep θ from over-committing.

Recurrence-based methods (RL²) handle exploration through the recurrent net's learned dynamics — the network learns *when* in the meta-episode to take exploratory actions and when to commit, because that's what maximizes cumulative meta-episode return. This works in principle but is hard in practice: the outer optimizer has to discover the explore-then-exploit pattern on its own, and short meta-episodes don't give enough budget for the policy to learn that it should explore early.

This is worth knowing because "my meta-RL isn't working" sometimes means "the inner loop isn't actually doing useful exploration on the new task, so adaptation can't possibly work." Diagnostic check: at meta-test on a held-out task, look at the action entropy and state coverage during the first K rollouts. If those rollouts all do the same thing, no inner-loop algorithm can help.

---

## Family 1: gradient-based meta-RL (MAML and friends)

### The MAML idea

MAML — Model-Agnostic Meta-Learning, Finn, Abbeel, Levine 2017, arXiv:1703.03400 — picks the simplest possible inner loop: a few gradient steps. The meta-objective is:

```
J_MAML(θ) = E_{T ~ p(T)} [ J_T( θ - α ∇_θ L_T(θ) ) ]
```

For one inner step with learning rate α. The thing being optimized is θ, but the return is measured after one (or a few) gradient steps starting from θ. You're learning an *initialization* such that one step of normal gradient descent on a new task lands you in a good place for that task.

In the RL version, "L_T(θ)" is something like the negative policy-gradient return on a few sampled trajectories from task T under the current policy. You run K rollouts, compute the policy gradient on those, take an inner step to get θ_T' = θ - α ∇L_T(θ), then collect *more* rollouts under π_θ' and compute the meta-gradient ∇_θ J_T(θ_T'). That meta-gradient threads back through the inner update, so you need to differentiate through the inner gradient — second derivatives.

For supervised meta-learning this works fine. For RL the inner gradient is itself a policy gradient — a high-variance Monte Carlo estimator. Differentiating through it adds another layer of variance and another set of moving parts. The original MAML-RL experiments used vanilla policy gradient as the inner loss, kept K small (around 20 rollouts), and used TRPO for the outer step.

### A skeleton implementation

The minimal MAML-RL outer-loop step, in pseudo-PyTorch, one inner gradient step:

```python
import torch
import torch.nn.functional as F


def policy_gradient_loss(log_probs, returns):
    """REINFORCE loss: -E[ log pi(a|s) * R ] over a batch of rollouts."""
    return -(log_probs * returns.detach()).mean()


def maml_rl_step(meta_policy, tasks, alpha=0.1, n_rollouts_inner=20,
                 n_rollouts_outer=20, env_factory=None):
    """
    One outer-loop meta-update with one inner gradient step.

    Args:
        meta_policy: the policy network whose params we are meta-training.
                     params live in meta_policy.parameters().
        tasks: a batch of tasks (each defining an env). Each `env_factory(T)`
               returns a fresh env for task T.
        alpha: inner-loop learning rate.

    The outer optimizer (not shown) takes meta_policy.parameters() as targets
    and steps on the returned meta_loss.
    """
    meta_loss = 0.0

    for T in tasks:
        env = env_factory(T)

        # --- INNER LOOP: one gradient step on K rollouts from theta. ---
        # Collect inner-loop rollouts with the meta-policy.
        rollouts_inner = collect_rollouts(meta_policy, env, n=n_rollouts_inner)
        log_probs_inner, returns_inner = rollouts_inner  # both [n_steps]

        inner_loss = policy_gradient_loss(log_probs_inner, returns_inner)

        # Compute the inner gradient w.r.t. meta_policy params, but DO NOT
        # apply it. We need it to be a differentiable function of theta so
        # the outer gradient can flow through it.
        grads = torch.autograd.grad(
            inner_loss,
            list(meta_policy.parameters()),
            create_graph=True,        # keep the graph so we can second-diff
        )

        # The adapted parameters theta' = theta - alpha * grad.
        # We need to use these as the params of the *same* network for the
        # outer rollouts. The cleanest way: a functional forward pass that
        # takes a parameter dict explicitly.
        adapted_params = [
            p - alpha * g for p, g in zip(meta_policy.parameters(), grads)
        ]

        # --- OUTER LOOP: collect new rollouts under theta' and measure
        #     the post-adaptation return.
        rollouts_outer = collect_rollouts_with_params(
            meta_policy, env, adapted_params, n=n_rollouts_outer,
        )
        log_probs_outer, returns_outer = rollouts_outer

        # Outer loss for this task. The grad of this w.r.t. theta will, by
        # the chain rule, pull through the inner step's grads (because
        # adapted_params depend differentiably on theta).
        outer_loss_T = policy_gradient_loss(log_probs_outer, returns_outer)
        meta_loss = meta_loss + outer_loss_T

    meta_loss = meta_loss / len(tasks)
    return meta_loss
```

A few notes about why this is harder than it looks.

`create_graph=True` keeps the inner gradient inside the autograd graph so the outer call to `meta_loss.backward()` can take a second derivative of the inner loss with respect to θ. That second derivative is a Hessian-vector product in disguise. Memory scales with the size of the inner-loop computation graph — every inner rollout's forward pass has to be retained.

`collect_rollouts_with_params` is the awkward part. PyTorch modules normally close over their own parameters; to forward-pass with `adapted_params` instead, you need either a functional API like `torch.func` (formerly `functorch`), a manual reparameterization, or a separate snapshotted module. The original MAML implementations used a custom forward pass that takes a parameter dict; modern code uses `torch.func.functional_call`.

The "inner rollouts" and "outer rollouts" must be different samples. Re-using the same rollouts to compute both the inner gradient and the outer return is a known bug — it makes the meta-objective biased and overfits the inner step to the specific batch.

### Reptile and first-order MAML

The second-derivative computation is expensive and noisy. Two simpler approximations:

**First-order MAML (FOMAML)** drops the second derivative entirely. After computing the inner step, it pretends the inner step was a constant and computes the outer gradient as if the adapted policy's parameters were independent from θ. Empirically this works much better than it should: on many problems FOMAML is within a few percent of full MAML at a fraction of the compute.

**Reptile** — Nichol, Achiam, Schulman 2018, arXiv:1803.02999 — goes further. You don't compute an outer gradient at all. Instead, for each task, run several SGD steps from θ to reach some θ_T', and then move θ a small distance toward each θ_T':

```
θ ← θ + ε ( (1/B) Σ_i θ_T_i' - θ )
```

It is essentially "average where multi-task SGD lands you." The theoretical justification (from the same paper) is that to first order the Reptile update direction agrees with the MAML gradient. The big practical advantage is that it works with any inner optimizer — Adam, gradient clipping, learning-rate schedules — without having to make the optimizer differentiable.

For RL specifically, Reptile is easier to implement and debug. FOMAML is easier than MAML. Full MAML has slightly cleaner theory and sometimes squeezes out a percent or two more performance, but in practice most published meta-RL work that uses a MAML-style approach uses one of the approximations.

### When gradient-based meta-RL helps

The implicit assumption is that good policies for different tasks in p(T) live close to each other in parameter space, in the sense that a few gradient steps span the gap. If task A's optimal policy and task B's optimal policy require very different network behaviors — different gating, different feature use — one or two gradient steps won't get from one to the other. The meta-initialization then has to be a compromise that's bad on both.

Where this lands: gradient-based meta-RL tends to work on task distributions where the tasks are *parametrically* related — varying goal positions in the same maze layout, varying friction coefficients on the same robot, varying reward magnitudes in the same MDP topology. It tends to struggle when the task distribution covers structurally different problems (different reward functions in MDPs with different state semantics).

A useful sanity check: if you can describe what changes across tasks as a small continuous parameter (mass, goal location, target velocity), MAML-style methods have a chance. If the task changes are more like "different game with different rules," gradient-based meta-RL is the wrong tool — you want something that can condition on a longer task description, which is what the recurrence-based and transformer-based methods give you.

### A note on the "MAML = initialization vs. MAML = learned learning rule" distinction

The original MAML paper presents the method as learning a good initialization. There's a second reading that became more prominent later: MAML can also be seen as learning a *learning rule*. The fixed-α inner step is the "rule," and the meta-trained θ is what makes that rule work.

Subsequent work made the learning rule itself learnable. Meta-SGD (Li et al. 2017) learns per-parameter inner-loop learning rates. ALPaCA and similar methods (across years 2018-2020) learn task-specific adaptation rules on top of features that the meta-training process selects. The pattern is "expand what's meta-learned" — initialization first, then learning rate, then optimizer, then more of the learning algorithm itself. The recurrent and transformer methods in the next sections are the limit of this expansion: the entire inner loop is implicit in the network's forward dynamics.

---

## Family 2: recurrence-based meta-RL (RL² and "Learning to RL")

### The idea

Instead of a few gradient steps as the inner loop, use a recurrent neural network that consumes experience from the new task and updates its hidden state. The hidden state plays the role of the adapted policy: at the start of a meta-episode on a new task the hidden state is empty, and as the policy collects more transitions, the hidden state accumulates task-relevant statistics. The recurrent net is, in effect, an RL algorithm — but a learned one, not a hand-designed one.

Two papers introduced this almost simultaneously in late 2016:

- **RL²** — Duan, Schulman, Chen, Bartlett, Sutskever, Abbeel 2016, arXiv:1611.02779. Title: "Fast Reinforcement Learning via Slow Reinforcement Learning." The "slow RL" is the outer-loop meta-training; the "fast RL" is what the recurrent net learns to do at inference time.
- **Learning to Reinforcement Learn** — Wang, Kurth-Nelson, Tirumala, Soyer, Leibo, Munos, Blundell, Kumaran, Botvinick 2016, arXiv:1611.05763. Same core idea, framed from a neuroscience perspective.

The setup. A meta-episode consists of several full episodes on a single task T:

```
t=0:  s_0, take a_0, see r_0, s_1   # episode 1, step 1
t=1:  s_1, take a_1, see r_1, s_2   # episode 1, step 2
...
t=H:  episode 1 ends; reset env, but DO NOT reset RNN hidden state.
t=H+1: s_0, take a_0, see r_0, s_1  # episode 2, step 1 — RNN has memory of ep 1
...
```

The recurrent policy takes as input at each step not just the current observation s_t but also the previous action and previous reward: input_t = (s_t, a_{t-1}, r_{t-1}, done_{t-1}). The hidden state carries information across timesteps and across episodes within the meta-episode. The outer loop optimizes total return over the *whole* meta-episode using a standard RL algorithm (the original papers used TRPO and A2C respectively).

What this does, when it works: across meta-training, the recurrent network learns that the way to maximize cumulative reward on the *meta*-episode is to behave like an RL algorithm — explore for the first few episodes to figure out the task, then exploit what it has learned for the remaining ones. The exploration policy and the exploitation strategy are both emergent properties of the recurrent net's learned dynamics. Nobody hand-coded an explore-exploit balance.

### Why the network ends up being an RL algorithm

Consider a meta-task distribution where each task is a 2-armed bandit with a different reward distribution. The outer RL algorithm trains a recurrent policy to maximize total reward across, say, 100 pulls of the bandit. To do well in expectation across many such bandits, the policy needs to:

1. Pull each arm at least once to learn its reward (exploration).
2. Pull whichever arm seems best for the rest of the episode (exploitation).
3. Maybe re-check the worse-looking arm once or twice in case the first pull was unlucky (handle observation noise).

If the policy doesn't do something like this, its total reward will be lower on average than a policy that does. Outer-loop training pushes it toward this kind of behavior because that behavior wins more return. The result is a network whose forward dynamics implement (an approximation of) an exploration-exploitation algorithm.

The point — and this is the part that prefigures everything that comes later in this lecture — is that the network has learned an *algorithm*, not a policy. It does not have a fixed action distribution for a fixed input; its behavior changes as it sees more data, in roughly the way you'd want a learning algorithm's behavior to change.

### Practical considerations

A few things from the experimental literature:

- The recurrent network has to be reset (hidden state cleared) at the boundary of each meta-episode, but *not* between episodes within a meta-episode. The whole point is that information carries across.
- The signal that the policy is "learning during the episode" is that performance improves within a meta-episode — return on episode 3 is higher than episode 1. If that doesn't happen, the recurrent net hasn't actually learned to adapt.
- Long context is a real problem. RNNs have well-known difficulty with horizons beyond ~50–100 steps. RL² and similar methods are largely limited by what an LSTM or GRU can remember about its task. Replacing the RNN with a transformer addresses this directly, which is what the later section is about.
- Meta-training is expensive. Each meta-episode is several full episodes; the outer-loop update is on a batch of meta-episodes; many outer updates are needed. Total environment steps for meta-training are typically several orders of magnitude more than for single-task RL of comparable complexity.

### Recurrence vs. gradient: a comparison

| | Gradient-based (MAML) | Recurrence-based (RL²) |
|---|---|---|
| Inner loop | A few gradient steps | RNN consumes transitions |
| What "adaptation" is | Updated weights θ' | Updated hidden state h |
| Inference cost | Forward pass + grad steps | Forward pass only |
| Outer training cost | Need second derivatives (or approximations) | Standard RL on the meta-episode |
| Generalization | Tied to "θ' close to θ in param space" | Tied to what the RNN can encode in its state |
| Easy to debug? | No — inner-outer loop is fragile | No — but for different reasons (long-horizon RL) |

Neither is the obvious winner. Recurrence-based methods have aged into the modern in-context RL story because transformers are good at the kind of sequence modeling the inner loop wants. Gradient-based methods have aged into the "fine-tuning is essentially a meta-learned thing" intuition that shows up in some LLM work. Both are still in use.

### A worked example: the 2-armed bandit, again

To make the recurrence-based story concrete, consider the simplest meta-RL setting that's worth running: a 2-armed bandit meta-distribution. Each task T is a 2-armed bandit with arm reward means (μ_0(T), μ_1(T)) sampled from some prior — for instance, both uniform in [0, 1]. A meta-episode is 100 pulls of one fixed bandit.

A non-meta-trained policy that pulls arm 0 always gets E[μ_0] = 0.5 reward per pull in expectation, so total return is 50 — and a policy that pulls arm 1 always also gets 50. The optimal single-task strategy is "figure out which arm is better and exploit it," which a Bayes-optimal Thompson sampling agent can do in ~100 pulls fairly well, achieving total return close to E[max(μ_0, μ_1)] · 100 ≈ 67.

An RL² agent on this meta-distribution, trained with A2C on the cumulative meta-episode return, learns something close to Thompson sampling — not literally, but functionally. Within-meta-episode return per pull starts near 0.5 (random exploration), rises over the first ~10 pulls (the RNN's hidden state starts to encode which arm is better), and asymptotes near max(μ_0, μ_1) for the remaining 90 pulls. Average meta-episode return ends up close to the Bayes-optimal value.

The interesting part is what the policy does internally: nobody handed it a Thompson sampling algorithm, but its outputs trace out the same kind of behavior. It pulls each arm a few times, then commits to the better one. The "algorithm" is whatever the LSTM hidden-state dynamics learned to do as a consequence of being optimized on the meta-objective.

This is the kind of result the next section's transformer-based methods scale up. Replace "2-armed bandit, 100 pulls, LSTM" with "more complex task family, longer horizon, transformer," and you get the in-context RL story.

---

## The transformer-meta-RL bridge

This is where the field starts converging with what large language models do.

### Decision Transformer as a meta-conditioning device

Decision Transformer — Chen, Lu, Rajeswaran, Lee, Grover, Laskin, Abbeel, Srinivas, Mordatch 2021, arXiv:2106.01345 — was covered in [Lecture 19: Offline RL](./19-offline-rl.md) as a way to do offline RL without a value function: treat trajectories as sequences of (return-to-go, state, action) tuples and train a causal transformer to predict the next action.

The relevant property for this lecture is what happens at inference time. You feed the model a sequence that begins with a *target* return-to-go and the current state, and it produces an action. By changing the target return-to-go at inference, you change the policy's behavior:

```
[R = 10.0, s_0]  → a_0   # policy that's trying to get a return of 10
[R = 100.0, s_0] → a_0'  # potentially very different action
```

This is in-context policy specification. The transformer is not retrained between the two queries; the only thing that changed is the context. From a meta-learning angle, "the return-to-go target" is a one-token specification of the task ("get me this much reward"). The model has been trained on many trajectories with many different return targets, and at inference it conditions on one.

It is not literally meta-RL — the task distribution is "the same MDP with different desired return levels," not different MDPs. But it is in-context control: the inference-time behavior is specified by the prefix, not by retraining.

**Trajectory Transformer** — Janner, Li, Levine 2021, arXiv:2106.02039 — does something similar but trains a transformer over (state, action, reward, return) sequences and uses beam search at inference time to plan trajectories that maximize return. The two papers came out within a day of each other on arXiv and are often discussed together. Trajectory Transformer leans more on planning; Decision Transformer is purely conditional generation.

A code sketch of a Decision Transformer forward pass, on a small offline dataset, to make the input format concrete:

```python
import torch
import torch.nn as nn


class DecisionTransformer(nn.Module):
    """
    Minimal Decision Transformer skeleton for a small offline dataset.

    Sequence format (interleaved):
        [R_1, s_1, a_1, R_2, s_2, a_2, ..., R_K, s_K, a_K]

    where R_t is the return-to-go from timestep t. Each modality gets its
    own embedding layer; the three streams are interleaved along the
    sequence dimension before being fed to a causal transformer.

    Loss: predict a_t given everything up to and including s_t. So at the
    action positions in the output, we compute a cross-entropy (or MSE,
    for continuous actions) against the dataset action.
    """

    def __init__(self, state_dim, action_dim, max_ep_len=1000,
                 hidden_size=128, n_heads=4, n_layers=3,
                 max_seq_len=20, action_is_discrete=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_is_discrete = action_is_discrete
        self.max_seq_len = max_seq_len

        # Per-modality embeddings — return, state, action all go to hidden_size.
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state  = nn.Linear(state_dim, hidden_size)
        if action_is_discrete:
            self.embed_action = nn.Embedding(action_dim, hidden_size)
        else:
            self.embed_action = nn.Linear(action_dim, hidden_size)

        # Timestep embedding (positional, but per-timestep across the 3 streams).
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)

        # Layernorm before the transformer (DT does this).
        self.embed_ln = nn.LayerNorm(hidden_size)

        # The transformer itself — a standard causal stack.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=n_heads, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=n_layers)

        # Action head — predicts a_t from the state-position hidden output.
        if action_is_discrete:
            self.predict_action = nn.Linear(hidden_size, action_dim)
        else:
            self.predict_action = nn.Linear(hidden_size, action_dim)

    def forward(self, returns_to_go, states, actions, timesteps):
        """
        Shapes (B = batch size, K = sequence length):
            returns_to_go: [B, K, 1]
            states:        [B, K, state_dim]
            actions:       [B, K]              (discrete) or [B, K, action_dim]
            timesteps:     [B, K]              (long)

        Returns: action_logits at the state positions, shape [B, K, action_dim].
        """
        B, K = states.shape[0], states.shape[1]

        time_emb = self.embed_timestep(timesteps)                  # [B, K, H]

        ret_emb = self.embed_return(returns_to_go) + time_emb       # [B, K, H]
        state_emb = self.embed_state(states) + time_emb             # [B, K, H]
        action_emb = self.embed_action(actions) + time_emb          # [B, K, H]

        # Interleave: [R_1, s_1, a_1, R_2, s_2, a_2, ...]  →  [B, 3K, H]
        # stack to [B, K, 3, H] then reshape to [B, 3K, H].
        stacked = torch.stack(
            (ret_emb, state_emb, action_emb), dim=2,
        ).reshape(B, 3 * K, self.hidden_size)

        stacked = self.embed_ln(stacked)

        # Causal mask so position i can only attend to positions <= i.
        seq_len = stacked.shape[1]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=stacked.device), diagonal=1
        ).bool()

        out = self.transformer(stacked, mask=causal_mask)           # [B, 3K, H]

        # The output at the state position (index 3t + 1) is what predicts a_t.
        # Pull out those positions: shape [B, K, H].
        out_at_state = out[:, 1::3, :]

        action_logits = self.predict_action(out_at_state)           # [B, K, A]
        return action_logits


def dt_loss(model, batch, action_is_discrete=True):
    """
    Cross-entropy or MSE loss for predicting a_t at the state position.

    batch fields:
        returns_to_go: [B, K, 1]
        states:        [B, K, state_dim]
        actions:       [B, K]              (discrete)
        timesteps:     [B, K]
    """
    logits = model(
        batch["returns_to_go"], batch["states"],
        batch["actions"], batch["timesteps"],
    )
    if action_is_discrete:
        # Standard cross-entropy at every position, averaged.
        B, K, A = logits.shape
        return nn.functional.cross_entropy(
            logits.reshape(B * K, A),
            batch["actions"].reshape(B * K),
        )
    else:
        return nn.functional.mse_loss(logits, batch["actions"])


# --- Inference loop sketch ---
#
# At rollout time, you supply a target return-to-go and run autoregressively,
# updating the return-to-go after each observed reward:
#
#   R_t+1 = R_t - r_t
#
# The model consumes the growing context and produces the next action.
# Crucially, no gradient descent or weight update happens at inference —
# the policy is specified entirely by the prefix.
```

The action lives at every third token in the interleaved sequence; the conditioning quantity (return-to-go) lives at every (3k)th. The architecture is conventional transformer; the meta-RL flavor is entirely in the input format and the inference protocol.

### Algorithm Distillation: the strongest in-context RL claim

Algorithm Distillation — Laskin, Wang, Oh, Parisotto, Spencer, Steigerwald, Strouse, Hansen, Filos, Brooks, Gazeau, Sahni, Singh, Mnih 2022, arXiv:2210.14215 — takes the in-context idea further.

The setup: pick an RL algorithm (any one — they use DQN-like algorithms). Pick a distribution of tasks p(T). For each task in p(T), run the RL algorithm from scratch and record its *entire training history* — every observation, every action, every reward, every episode, from random initialization through final convergence. Concatenate these histories into very long sequences.

Train a causal transformer to do next-action prediction on these histories.

At inference time, drop the transformer into a new task T' ~ p(T). It starts producing actions; you give it the rewards; it produces more actions; over the course of many episodes, its in-context performance *improves*. The transformer's behavior is replaying the learning algorithm it was trained on — and it converges on the new task without any weight updates, by virtue of conditioning on the growing history.

This is the strong in-context RL claim. The network is not storing a policy for each task; it has learned *how the source RL algorithm behaves over time given experience*. When you give it new experience, it produces actions that the source algorithm would have produced at the same point in its learning trajectory. The "in-context" learning is the network mimicking the source algorithm's inner loop.

Algorithmically, the headline result is that the distilled transformer can be *more sample-efficient* than the source algorithm in some settings. The argument is that the transformer has access to many tasks worth of demonstrations of what good learning trajectories look like, so it can compress out the wasted exploration of the source algorithm and reproduce the productive parts.

A few constraints worth knowing about:

- The transformer's context window must be long enough to span enough of a learning history that improvement is visible within context. Laskin et al. work with thousands of timesteps in context. For longer-horizon tasks this becomes a real constraint.
- "Cross-episode" credit assignment is mechanical: the transformer sees observation–action–reward–observation, repeated across episode boundaries, with no special handling of episode resets beyond a token.
- The source RL algorithm must itself be a good learning algorithm. The transformer can at best mimic what it was shown. If the source DQN had a bug or hyperparameter problem, the transformer reproduces it.

### Decision-Pretrained Transformer

A close cousin: Decision-Pretrained Transformer — Lee, Xie, Pacchiano, Chandak, Finn, Nachum, Brunskill 2023, arXiv:2306.14892. The paper "Supervised Pretraining Can Learn In-Context Reinforcement Learning" makes a related point: a transformer trained with supervised learning on (context, optimal action) pairs from many tasks performs in-context RL on new tasks. The supervised target is the action a Bayes-optimal policy would take given the same context. The result is similar in flavor to Algorithm Distillation — pretraining on a meta-task distribution produces a model whose in-context behavior is a learning algorithm — but the construction is supervised rather than distillation-from-RL.

The convergence of these results is the main takeaway: there are multiple ways to set up the training data and the architecture, and several of them produce networks whose forward pass implements something that behaves like an RL algorithm.

---

## The LLM in-context learning connection

GPT-3 — Brown et al. 2020, arXiv:2005.14165, "Language Models are Few-Shot Learners" — was the paper that made few-shot in-context learning a central capability of large language models. Give the model:

```
sea otter => loutre de mer
peppermint => menthe poivrée
cheese => ?
```

and it produces a reasonable French translation, without any gradient update. The paper framed this as in-context learning, sometimes meta-learning, and pointed out that scaling the model up made few-shot performance markedly better.

The connection back to this lecture: from a meta-learning angle, a pretrained LLM is the result of an enormous, implicit meta-training process. The training distribution is "all natural-language text," which contains a vast number of sub-tasks (summarize this, classify that, follow these few-shot examples). The model has been optimized to predict the next token across all of them, which is roughly equivalent — under reasonable assumptions — to performing well on any sub-task that the prompt specifies. The "in-context" capability is the inference-time consequence of that implicit meta-training.

This is the angle a few papers and reviews have argued. The frame is useful but should not be over-stated:

- LLM in-context learning is not literally MAML or RL². It does not have a clean inner-outer loop. The outer loop is "predict the next token of a giant text corpus" — a much messier objective than "maximize post-adaptation return." The fact that in-context learning emerges from it is empirically observed, not designed for.
- In-context learning in LLMs may not be doing the same thing as MAML or RL² internally. There is genuine uncertainty in the mechanistic interpretability literature about what in-context learning *is*. Some experiments suggest it's something like Bayesian update over implicit task priors; others suggest it's doing something more like gradient descent in activation space; others suggest a mix. None of these stories are settled.
- "In-context RL within LLMs" — meaning, an LLM that, given a description of an RL task and some trajectory data in its prompt, behaves like an RL learner on that task — is a real and active research direction. Specific recent papers on this come and go; rather than cite ones I haven't verified, I'll say only that there is published work in this thread and it draws on both the Algorithm Distillation line and the LLM in-context-learning line.

The intuition that does carry over cleanly: an LLM doing in-context learning on a few-shot task is *behaving as if* it has been meta-trained to learn from prompts. Whether the underlying mechanism is meta-learning in a precise sense is an open question. For practical purposes — designing prompts, debugging behavior, predicting what will and won't transfer — the meta-learning frame is one of the better intuition pumps available.

### Reading in-context RL through the meta-RL lens

Three concrete patterns from the meta-RL literature that translate directly to how an LLM does in-context tasks:

**More examples → better adaptation, with diminishing returns.** In recurrence-based meta-RL, performance on the new task improves over the meta-episode as the recurrent net consumes more transitions. The curve usually saturates: the first few transitions help a lot, the 50th transition helps a little, the 500th helps almost nothing. LLM few-shot learning shows the same shape — going from 0 to 4 examples is a big jump, going from 32 to 64 is often invisible.

**The training-task distribution sets the inference-task ceiling.** A meta-RL agent trained on mazes of size 5×5 generalizes to mazes of size 7×7 poorly and to mazes of size 20×20 not at all. An LLM trained on web text plus code generalizes to "translate this English sentence to French" because the training corpus contained translation, and fails to generalize to in-context tasks whose form is absent from training. When in-context learning fails on a particular task, the first hypothesis to check is that the task structure was rare or absent during training.

**Exploration in context.** In a meta-RL setting where the inner loop has to figure out the task from observations, the agent has to actually take exploratory actions early to learn what task it's on. In an LLM in-context setting where the prompt describes a novel task, the model has to "explore" the task — typically by trying a structure, getting (in the case of a multi-turn dialogue) feedback, and refining. The LLM analog of exploration is messier than the meta-RL version because there's no obvious reward signal mid-prompt, but the shape of the problem is the same: you can't exploit a task structure you haven't identified.

These translations are intuition pumps, not theorems. They're useful when reasoning about what kinds of in-context behavior should work, where they should break, and how a deliberate change to training data (more diverse prompts, more varied few-shot demonstrations) should change inference-time behavior. They are not substitutes for evaluation.

---

## When meta-RL is worth it

The cost of meta-RL is real. The inner-outer loop is harder to debug than a single-loop RL training run. Meta-training takes far more compute than training on any single task. The whole approach assumes you have a meaningful task distribution to train on, which is itself a nontrivial design problem.

So it's worth being honest about when this is the right tool.

**Meta-RL is worth it when:**

- Tasks in your distribution share enough structure that the meta-policy can capture the shared part. Different goal locations in the same maze: yes. Different friction coefficients on the same robot: yes. Different reward functions over completely different state spaces: probably not.
- New tasks arrive often enough that the amortized cost of per-task RL is the bottleneck. If you have to deploy on a new task every day, taking a month of single-task RL each time is unworkable; spending a year on meta-training so you can adapt in an hour might pay back.
- The "fast adaptation" budget at test time is small. If you have unlimited test-time compute on each new task, classical RL is simpler and gets you there. Meta-RL pays off when adaptation has to be cheap.

**Meta-RL is not worth it when:**

- Tasks differ in fundamental ways. The meta-policy will end up averaging over incompatible task distributions, and the resulting policy is worse on each individual task than per-task RL would be. There is no positive transfer because there is no shared structure to transfer.
- You have just one task and abundant compute. Just train on that task. The meta-RL overhead buys you nothing.
- Your task distribution is poorly specified. If you can't write down what makes two tasks "the same kind of task," you can't sample from p(T) coherently, and the meta-training will produce something with hard-to-predict behavior on what you actually care about.

A version of this last problem shows up in practice as **meta-overfitting**: the meta-policy is excellent on tasks drawn from p(T) but degrades fast as test tasks drift outside the training distribution. Just as a regular RL agent can overfit to its training environment, a meta-RL agent can overfit to its training *distribution* of environments. The cure is the usual one — broader, more diverse meta-training — but it's expensive.

---

## What goes wrong

A short list of failure modes you'll hit:

**Inner-outer loop debugging is fragile.** If meta-training isn't working, the diagnostic question is: is the inner loop broken (each individual task isn't being adapted to well), or is the outer loop broken (the meta-update isn't improving inner-loop performance)? You usually need to instrument both — log per-task post-adaptation return, and watch how it changes over meta-iterations. If post-adaptation return on the training tasks isn't going up, the outer loop is wrong. If post-adaptation return on training tasks is fine but test tasks are bad, you have meta-overfitting.

**Reward design must reflect adaptation cost, not just final performance.** A common bug: meta-train with the standard task reward and find that the meta-policy is great after 10 episodes of adaptation. But your deployment target was "good after 2 episodes." The meta-objective didn't penalize slow adaptation, only final return. If you care about how *fast* adaptation happens, the meta-objective needs to reflect that — for example by measuring total cumulative return over the meta-episode, which naturally rewards being good earlier rather than only at the end. The recurrence-based methods get this right by construction (the meta-episode return is cumulative). MAML-style methods often need you to think harder about it.

**The task distribution is doing more work than the algorithm.** Two meta-RL projects with the same algorithm and very different task distributions will report wildly different results. If you read a meta-RL paper that shows strong results on "MAML-RL benchmark," look at the actual benchmark first — what are the tasks, how varied are they, how aligned are they with what you care about. A lot of the variance in published meta-RL results is downstream of benchmark construction, not algorithmic choices.

**Second-order gradients are noisy.** For full MAML, the inner-loop gradient is a stochastic estimator (sampled rollouts), and the second derivative of a stochastic estimator is doubly noisy. This is the practical reason FOMAML and Reptile remain in use — they trade a theoretical gap for a large variance reduction in the outer-loop update.

**Long-context dependence in recurrence-based methods.** RL² with an LSTM hits the same long-horizon limits as any other LSTM-based RL agent. Replacing with a transformer (as in the Decision Transformer / Algorithm Distillation line) helps but introduces its own issues: positional encodings, context-window cost, and the offline-data dependency that comes with transformer training.

**Off-distribution test tasks degrade fast.** Meta-trained policies on a maze distribution will work well on mazes that look like the training mazes and fail abruptly on mazes that are slightly outside that distribution. The meta-policy has learned a strategy for a specific kind of structure; novel structure breaks it. Same kind of brittleness as in supervised meta-learning.

**Hyperparameter coupling.** In single-task RL you can tune learning rate, batch size, and discount independently and usually find a working setting. In meta-RL there are two sets of hyperparameters (inner and outer) that interact. Inner learning rate α and outer learning rate β are coupled — too-large α makes the inner step jump past the good region; too-small α means K rollouts can't actually adapt the policy; the outer step has to be tuned for whichever α you picked. A common debugging trap is to retune the outer optimizer when the actual problem is the inner step size.

**No clean test-set protocol.** In supervised meta-learning there's a standard split: meta-train tasks vs. meta-test tasks. In meta-RL the convention exists but isn't enforced — some papers report results on tasks drawn from p(T) without separating training and test instances of that distribution. Read meta-RL papers carefully: a "98% success rate" on training-instance tasks is much less interesting than the same number on held-out instances. The honest evaluation is held-out p(T) tasks the meta-policy has never seen.

**Compute cost is non-linear in task count.** Meta-training on B tasks per outer-loop step takes roughly B times the compute of single-task training, but the number of outer steps doesn't drop in proportion. Total compute for a meta-RL run is typically 10x to 100x more than training on the most expensive single task in the distribution. If you have a fixed compute budget and only a couple of related tasks, separate single-task runs may beat one meta-run.

---

## Where this sits in the curriculum

This lecture closes a loop that started in [Lecture 02: Policy Gradients](./02-policy-gradients.md), where a policy was trained on a single MDP, and ran through [Lecture 19: Offline RL](./19-offline-rl.md), where Decision Transformer reframed offline RL as sequence modeling. Meta-RL takes the position that the *thing you train* should not be a policy for one task but a policy-producer for many tasks. Decision Transformer and Algorithm Distillation turn the policy-producer into something whose inference-time forward pass *is* the inner loop, with no weight updates needed at test time.

That last move — putting the inner loop inside a transformer's forward pass on a long context — is the move that connects meta-RL to LLM in-context learning. Once both sides have a "the network does the learning at inference time, in its activations" framing, the algorithmic distinction between "few-shot RL adaptation" and "few-shot prompt conditioning" becomes mostly about what's in the prompt and what kind of data the model was trained on. The mechanism is the same shape.

The practical upshot for an LLM-oriented reader: when you reason about what an in-context LLM can do on a new task, the meta-RL literature gives you a vocabulary for it (inner loop, outer loop, meta-overfitting, exploration in context). When you reason about why an in-context strategy works or fails — for instance, why few-shot examples help on some tasks and not others — the meta-RL intuitions about shared task structure, distributional shift between meta-train and meta-test, and the inner-loop adaptation budget all transfer.

---

## Exercises

1. Implement first-order MAML on a simple task distribution: 1D MDPs with goal positions sampled uniformly on the line. The "task" is reaching the goal; observations are (current position, target). Train a meta-policy with FOMAML and compare its 1-step post-adaptation return to (a) a policy trained jointly on all tasks (no adaptation) and (b) a policy trained from scratch on a single held-out task. Plot post-adaptation return vs. number of inner-loop rollouts K ∈ {1, 5, 20}.

2. Implement RL² on a multi-armed bandit meta-distribution: each meta-episode is 100 pulls of a bandit whose arm rewards are sampled fresh per meta-episode. Train an LSTM policy with A2C on the cumulative meta-episode return. After training, freeze the network and analyze its behavior on held-out bandits: does it explore for the first few pulls, then exploit? Plot per-pull cumulative regret over the meta-episode and compare to an explicit Thompson sampling baseline.

3. Train a small Decision Transformer on a D4RL offline dataset and study return-conditioning. After training, query the model at inference with target returns at the 10th, 50th, and 90th percentile of dataset returns. Measure the actual achieved return for each target. Does it scale with the target, and at what target does it stop tracking?

4. Replicate the core Algorithm Distillation observation on a toy task family — for example, 10×10 gridworlds with the goal in different cells. (a) Train tabular Q-learning from scratch on each gridworld separately and record the full training history (state, action, reward, next state) for each. (b) Train a small causal transformer on next-action prediction over these histories. (c) At inference, drop the transformer into a held-out gridworld and feed it observations and rewards. Plot in-context return per episode and verify it improves over the course of many in-context episodes, without weight updates.

5. The LLM connection, qualitatively. Pick an in-context few-shot prompt for any task an LLM does well (e.g., translation, classification). Vary the number of examples (K = 1, 4, 16). Vary the consistency of the example structure (mix formats vs. keep one format). For each, record accuracy on a held-out test set. Then write up — in 2–3 paragraphs — how the resulting curves do or don't match what the meta-RL literature would predict (faster improvement with more examples; sensitivity to distribution shift between examples and test). This is not a code-heavy exercise; it's a calibration exercise on whether the meta-RL intuitions transfer.

---

## References

**Finn, Abbeel, Levine (2017)**. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." ICML 2017. arXiv:1703.03400. Verified. The original MAML paper; learns an initialization such that one (or a few) gradient steps on a new task reaches good performance.

**Nichol, Achiam, Schulman (2018)**. "On First-Order Meta-Learning Algorithms." arXiv:1803.02999. Verified. Introduces Reptile; analyzes first-order approximations to MAML; shows the Reptile update direction agrees with the MAML gradient to first order.

**Duan, Schulman, Chen, Bartlett, Sutskever, Abbeel (2016)**. "RL²: Fast Reinforcement Learning via Slow Reinforcement Learning." arXiv:1611.02779. Verified. Trains a recurrent policy on multi-episode meta-trajectories; the RNN's hidden state acts as the adapted policy; the network learns an RL algorithm by emergence.

**Wang, Kurth-Nelson, Tirumala, Soyer, Leibo, Munos, Blundell, Kumaran, Botvinick (2016)**. "Learning to Reinforcement Learn." arXiv:1611.05763. Verified. Independent contemporary of RL² with similar core idea; framed from a neuroscience perspective.

**Chen, Lu, Rajeswaran, Lee, Grover, Laskin, Abbeel, Srinivas, Mordatch (2021)**. "Decision Transformer: Reinforcement Learning via Sequence Modeling." NeurIPS 2021. arXiv:2106.01345. Verified. Treats offline RL as return-conditioned next-action prediction with a causal transformer; the inference-time return conditioning is a one-token policy specification.

**Janner, Li, Levine (2021)**. "Offline Reinforcement Learning as One Big Sequence Modeling Problem." NeurIPS 2021. arXiv:2106.02039. Verified. The Trajectory Transformer paper; trains a transformer on (s, a, r, R) sequences and uses beam search to plan trajectories at inference.

**Laskin, Wang, Oh, Parisotto, Spencer, Steigerwald, Strouse, Hansen, Filos, Brooks, Gazeau, Sahni, Singh, Mnih (2022)**. "In-context Reinforcement Learning with Algorithm Distillation." arXiv:2210.14215. Verified. Trains a causal transformer on full RL training histories across many tasks; at inference, the transformer's in-context behavior improves over the course of many episodes without weight updates.

**Lee, Xie, Pacchiano, Chandak, Finn, Nachum, Brunskill (2023)**. "Supervised Pretraining Can Learn In-Context Reinforcement Learning." arXiv:2306.14892. Verified. The Decision-Pretrained Transformer paper; shows that supervised pretraining on (context, optimal-action) pairs from many tasks produces a transformer with in-context RL ability on held-out tasks.

**Brown et al. (2020)**. "Language Models are Few-Shot Learners." NeurIPS 2020. arXiv:2005.14165. Verified. The GPT-3 paper; documents in-context few-shot learning and explicitly discusses the meta-learning frame for what large language models do at inference time.

---

_End of Lecture 32._
