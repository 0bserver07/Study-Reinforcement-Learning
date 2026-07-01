<!-- status: unreviewed | last-reviewed: never -->

# Lecture 25: Long-horizon credit assignment

_Unreviewed: no one has checked this end to end. Treat the math, code, and citations as unverified._

**Time**: ~3 h · **Prerequisites**: Lectures 02, 04, 16

---

## Where this fits

[Lecture 16](./16-agentic-rl.md) set up the agentic RL loop: a policy emits an action, an environment returns an observation, the history grows, and a terminal reward shows up at the end. The example episodes there were short (a 3-action code fix, a ReAct call-and-answer trace), and the GRPO group baseline was enough to make policy gradient work.

This lecture is about what breaks when the episode gets long. The horizon stops being 5 or 20 steps. It becomes 50, 500, or 5000. The model edits dozens of files, runs hundreds of shell commands, queries a search index a thousand times, negotiates back-and-forth across 200 turns, or runs a literature review that takes a research-grade afternoon. At the end, a single bit comes back: solved, or not.

How do you decide that the file rename on step 7 was the move that mattered? How do you assign credit to the action at step 312 in a 500-step trajectory when 311 of the other actions were also taken and the episode succeeded? This is the long-horizon credit assignment problem, and as of early 2026 it is the hardest open problem in agentic RL. There is no agreed answer.

The lecture is in four parts. First, set up the problem and the classical answers (TD(λ), GAE, baselines) and explain why they don't translate cleanly to the LLM-agent setting. Second, walk through the modern (2023-2025) heuristics: outcome reward + group baselines, process reward models, hindsight relabeling, self-critique, tree search at training time, curriculum decomposition, and what each one buys and costs. Third, two code sketches: a hindsight-relabel loop and a tree-search-then-train recipe. Fourth, the open questions.

---

## The problem, stated formally

A trajectory looks like τ = (s₁, a₁, s₂, a₂, ..., s_T, a_T), where each sₜ is the agent's full observable history at step t and each aₜ is the action it emitted. The episode terminates after T steps (T may be variable; we'll use T to mean the realized horizon). The reward signal is a single scalar at the end:

```
R(τ) = r_T ∈ {0, 1}   (in the verifiable-binary case)
```

or some bounded scalar. All intermediate rewards rₜ for t < T are zero unless the environment explicitly provides per-step signal (which it usually doesn't, in the cases we care about).

The policy gradient ([Lecture 02](./02-policy-gradients.md)) for this setting is:

```
∇_θ J(θ) = E_τ [ R(τ) · Σ_{t=1}^{T} ∇_θ log π_θ(aₜ | sₜ) ]
```

Every action's log-prob gets scaled by the same scalar: the terminal reward. The action at step 7 and the action at step 312 are treated as equally responsible for whether `R(τ) = 1` or `R(τ) = 0`. That is the credit assignment problem at its starkest: the gradient estimator has no way to tell them apart.

### Why the variance grows with the horizon

Take T independent actions, each of which contributes some randomness to the trajectory. The log-prob gradient ∇ log π_θ(aₜ | sₜ) is a vector whose magnitude is roughly constant across t (it doesn't shrink with horizon). The sum Σₜ ∇ log π_θ(aₜ | sₜ) is a sum of T noisy vectors, so its variance scales linearly with T. Multiply by the terminal reward R(τ), also a noisy scalar across the population of trajectories, and the variance of the gradient estimator is `O(T · Var(R))`.

In CartPole with T ≈ 200, that variance is already enough to make REINFORCE unstable without baselines and returns-to-go ([Lecture 02 Part 6](./02-policy-gradients.md#part-6-the-gotchas)). At T = 500, with a binary reward and a success rate of, say, 5%, the gradient estimator from a single batch is mostly noise. At T = 5000, you can run a 50-trajectory batch and still see no signal; the policy doesn't move in a recognizable direction. The gradient is unbiased; it just has so much variance that you can't detect the underlying mean from any tractable number of samples.

This is not a hyperparameter problem. No amount of learning-rate tuning fixes it. The variance is in the estimator.

### The discount factor stops meaning what you want

In tabular RL, the discount γ ∈ (0, 1) plays two roles. Mathematically, it keeps infinite-horizon returns finite. Modeling-wise, it expresses that rewards in the distant future are worth less than rewards now, partly because of uncertainty, partly because the agent might die before getting there.

For an agentic episode with sparse terminal reward, γᵀ⁻ᵗ becomes the credit assigned to the action at step t. If γ = 0.99 and T = 500, the action at step 1 gets γ⁴⁹⁹ ≈ 0.007 of the reward. The action at step 499 gets γ¹ = 0.99. Essentially all the credit lands on the last handful of actions, which are probably the cleanup actions (writing the answer, calling `finish()`), not the ones that mattered.

Setting γ = 1 removes that bias but reintroduces the variance problem in its full ugly form: every action gets the same credit, regardless of position. Setting γ close to 0 turns the problem into nearly-myopic optimization, which is fine if the reward is dense but useless when the reward only arrives at step T.

There is no single value of γ that handles a 500-step sparse-reward episode well. The discount-as-credit-decay trick was designed for an MDP where time itself implies uncertainty about whether you'll survive to see a later reward. In an agentic episode where the agent will, with probability 1, reach the end of the trajectory and get its reward, the discount is just an arbitrary decay you've imposed on the credit assignment, and it doesn't correspond to anything in the problem structure.

### Bias-variance tradeoff, named

The classical framing makes this concrete. There are two main families of return estimators:

- **Monte Carlo (MC).** Use the actual realized return Gₜ = Σ_{k≥t} γᵏ⁻ᵗ rₖ. Unbiased. Variance grows with horizon T.
- **Temporal Difference (TD).** Bootstrap from a value function: Gₜ ≈ rₜ + γ V̂(sₜ₊₁). Variance is low (you've replaced the future with a single-step lookup), but bias depends on how wrong V̂ is.

You can interpolate. TD(λ) and Generalized Advantage Estimation (GAE; Schulman, Moritz, Levine, Jordan, Abbeel 2015, [arXiv:1506.02438](https://arxiv.org/abs/1506.02438)) define a weighted sum of n-step returns:

```
A^GAE = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}     where δₜ = rₜ + γ V̂(sₜ₊₁) - V̂(sₜ)
```

λ = 1 recovers Monte Carlo (high variance, unbiased). λ = 0 recovers one-step TD (low variance, high bias if V̂ is wrong). Practitioners typically use λ ∈ [0.9, 0.99] for game-like environments where T is a few hundred and a value function with a few million parameters is enough to be useful.

For an LLM agent at T = 500, GAE doesn't translate cleanly. The reasons are interrelated and worth being explicit about.

### Why classical fixes break in the LLM-agent regime

**1. The value function is expensive.** V̂(sₜ) is supposed to estimate the expected return from state sₜ. In the agentic setting, sₜ is the entire conversation history plus environment state up to step t; for a 500-step trajectory in a SWE-bench-style task, that's tens of thousands of tokens of context. A value head bolted on top of the policy LM has to process that context, which means full forward-pass cost per step. If the policy is a 70B model, the critic is either also 70B (doubling memory and compute) or a separate, smaller model that doesn't have the same world-understanding as the policy and produces low-quality value estimates.

**2. The value function is untrained, especially at the start.** In CartPole, the value function gets supervised by the Monte Carlo returns of every episode, and a few thousand episodes is enough to fit a small network. In agentic RL, episodes are expensive (each step is a model call plus an environment call), success rates are low (5% on a hard task means most episodes return zero), and the value function has nothing to fit to for the first many thousands of gradient steps. Until the policy starts succeeding regularly, the critic's "predictions" are essentially the empirical mean reward, which is what GRPO uses as the group baseline anyway, without the cost of training a separate network.

**3. The state is unstructured text.** Tabular value functions exploit state structure (this state is similar to that state). In LLM-agent settings, two states that look superficially similar (same files open, same error messages) can have wildly different value because of subtle differences in conversation history. Generalization across states is hard, and a value function that doesn't generalize well doesn't reduce variance; it just adds bias.

**4. Per-step rewards usually don't exist.** Even if you had a value function, GAE-style bootstrapping needs per-step rewards (the δₜ terms). For most agentic tasks, the only nonzero reward is at the end. You can fake intermediate rewards (a process reward model, see below), but now you're back to the question the lecture is about: how do you generate trustworthy per-step credit when you don't have a way to verify per-step correctness?

This is why standard RLHF training pipelines for agentic tasks usually skip GAE and use trajectory-level outcome rewards with GRPO-style group baselines (as in [Lecture 16](./16-agentic-rl.md) and the DeepSeek-R1 training recipe; DeepSeek-AI 2025, [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)). They eat the high variance, throw more rollouts at the problem, and hope the policy still moves in a useful direction. For tasks with T ≈ 30 or so, this works. For T in the hundreds, the wall starts showing.

---

## Trajectory-level outcome reward with group baselines

The dominant approach in 2024-2025, used in essentially every public reasoning-model training recipe, is outcome reward + GRPO-style group baselines. The setup is the one from [Lecture 15](./15-rl-verifiable-rewards.md): for each prompt or task, sample K trajectories, score each by its terminal reward, and use the within-group mean and std to compute advantages:

```
A_i = (R(τ_i) - mean(R_1..R_K)) / (std(R_1..R_K) + ε)
```

Each token in trajectory τ_i is weighted by A_i during the policy gradient update. No critic. No per-step reward. The "credit assignment" is: this whole trajectory got credit A_i, and we apply it uniformly across all the tokens the policy produced in this trajectory.

### Why this works when it works

When K is large enough and the trajectories are independent samples (e.g., K independent attempts at solving the same problem), the group-relative advantage is well-defined: trajectories that did better than the group get pushed up, ones that did worse get pushed down. The policy moves toward what works.

The horizon-length issue is real but bounded. For math reasoning tasks, T is roughly the length of the chain-of-thought: say, 1000 tokens but only one "step" in the agentic sense. For agentic coding tasks at T ≈ 20-30, K = 8 to 16 trajectories per task is enough to get a usable signal as long as the success rate is in a reasonable range (5% to 95%). DeepSeek-R1's training recipe uses this structure for math and code; the SWE-RL paper (Wei et al. 2025, [arXiv:2502.18449](https://arxiv.org/abs/2502.18449)) uses a similar structure for software-engineering-style tasks.

### Why this breaks at scale

A few failure modes show up as the horizon grows.

**Effective batch size collapses.** When the success rate is very low (e.g., 1% on a hard task), most batches contain only failures. The group mean is zero, the std is zero, and the advantage is undefined or noise. The policy doesn't move. You need either (a) a much larger K (which is expensive: K × T tokens of rollout per task per update) or (b) a curriculum that keeps the task in a difficulty range where the success rate is nonzero (more on this below).

**All-or-nothing credit.** Uniform weighting across all tokens in a successful trajectory means the model gets equally credited for the action at step 1 (which mattered) and the action at step 312 (which was a cleanup commit). For long horizons, this means most of the gradient mass goes toward tokens that weren't causally responsible for the outcome. The policy still moves in the right direction on average, but slowly, and it can pick up superficial patterns that correlate with success without being the cause of it.

**Long-trajectory rollouts are slow.** Each step in an agentic trajectory is a model call plus an environment call. At T = 500, K = 8, batch size 32, that's 128,000 environment calls per gradient update. If each call takes 1 second, a single update takes ~36 hours. Real systems batch and parallelize aggressively, but the cost remains the dominant practical bottleneck.

GRPO scales fine to the horizons used in reasoning models (where T is the chain-of-thought length and the agent is effectively single-step) and to bounded agentic tasks (T ≈ 30). For tasks with T in the hundreds or thousands, the trajectory-level approach is what people use in the absence of anything better, and the resulting training is fragile.

---

## LLM as critic: process reward models

A process reward model (PRM) replaces the missing value function with a learned per-step judge. Given a (history, action) pair, the PRM outputs a scalar estimating whether that step looks like a good step. During training, the agent gets the PRM's per-step scores as a dense reward signal, which the policy gradient can then assign as credit per step.

This was first developed seriously for math reasoning. Lightman et al. (2023, [arXiv:2305.20050](https://arxiv.org/abs/2305.20050)) ("Let's Verify Step by Step") trained a PRM on PRM800K, a dataset of 800K human-labeled step-level judgments on math problems, and showed that process supervision outperformed outcome supervision for both filtering and reranking solutions. Math-Shepherd (Wang et al. 2023, [arXiv:2312.08935](https://arxiv.org/abs/2312.08935)) extended this without requiring human step labels by using Monte Carlo rollouts to score steps: a step is "good" to the extent that rollouts starting from it tend to succeed.

The lecture-17 derivation aside, the PRM gives you something like a value function: a per-step scalar you can use for credit assignment. The shape of the modified policy gradient is then:

```
∇_θ J(θ) ≈ Σₜ ∇_θ log π_θ(aₜ | sₜ) · ( PRM(sₜ, aₜ) - baseline )
```

### What PRMs buy

For math reasoning specifically, the wins are real. Process supervision detects intermediate errors (a sign flip, a misapplied identity) that outcome supervision can only detect via end-to-end failure. PRMs work as rerankers (run K rollouts, pick the one the PRM rates highest) and they work as dense reward signals during RL.

### What PRMs cost, and where they break

**Cost: PRM training data is expensive.** PRM800K had 800K human-judged steps. Most domains don't have a dataset of that size, and creating one is expensive enough that it bottlenecks the approach. Math-Shepherd's MC-rollout-based labeling reduces the human-labeling cost but multiplies the compute cost: to label one step, you run K rollouts from that step to estimate the success rate, which is exactly the variance problem the PRM was supposed to solve, just pushed into the label-generation stage.

**Break case 1: PRMs are themselves gameable.** A PRM is a learned model. The policy will eventually find inputs that the PRM scores highly without actually being good steps, the standard reward hacking failure mode from [Lecture 09](./09-reward-modeling.md) and [Lecture 15](./15-rl-verifiable-rewards.md). The policy might learn to produce text that looks like a productive reasoning step (using PRM-favored phrases, structure, vocabulary) while not actually solving the problem.

**Break case 2: PRM generalization is brittle.** A PRM trained on math reasoning doesn't transfer to code editing. A PRM trained on SWE-bench-style tasks doesn't transfer to web browsing. The "what is a good step" judgment is task-dependent, and each new domain requires either fresh training data or a much more general-purpose judge. As of 2026, general-purpose process reward models that work across agentic domains do not exist publicly.

**Break case 3: Long-horizon PRMs are unstable.** Even where PRMs work for short trajectories (math, T ≈ 10 reasoning steps), they degrade as the trajectory length grows. A PRM that's 90% accurate per step compounds to 0.9⁵⁰⁰ ≈ 0% over 500 steps; a single misjudged step early can propagate. This is the same compounding-error problem as in any long-horizon learned model.

### The fundamental tension

Dense process reward gives low-variance credit assignment but is gameable and expensive to label. Sparse outcome reward is honest but high-variance. There is no version of this tradeoff that gives you both at once. The current best practice, for the small set of domains where it works, is to use outcome reward as the ground truth and PRM-derived dense rewards as a variance-reduction tool that you weight carefully, and to be ready to retrain the PRM as the policy distribution shifts.

---

## Hindsight relabeling

Hindsight Experience Replay (HER; Andrychowicz, Wolski, Ray, Schneider, Fong, Welinder, McGrew, Tobin, Abbeel, Zaremba 2017, [arXiv:1707.01495](https://arxiv.org/abs/1707.01495)) is the foundational idea: if the agent failed to achieve goal g but did achieve some other state s', you can treat the trajectory as a successful attempt to achieve s'. The trajectory is "relabeled": the goal in the data is rewritten to match what was actually accomplished, and the agent gets a positive reward signal where there would otherwise have been zero.

This is a particularly clean fit for goal-conditioned policies in physical control (where "achieve some state" is a natural goal description) but it generalizes: the underlying idea is that a failed trajectory under one specification is a successful trajectory under a different specification, and you can extract learning signal from the second specification even though the first one returned zero.

### HER in the LLM-agent setting

The LLM-agent variant is less mechanical. A failed SWE-bench attempt (the agent's patch didn't pass the hidden tests) is not obviously a successful attempt at anything. But several relabeling strategies have been explored in the 2023-2025 period:

**1. Substring goals.** If the task was "fix the bug so the tests pass" and the agent didn't fix the bug but did refactor function X, relabel the trajectory as "refactor function X" and use that as a positive example for a different distribution of tasks. Useful for building a curriculum of subtasks; less useful for the main reward.

**2. Self-generated goals.** Ask the model to look at its own trajectory and describe what it actually did: "I tried to fix the off-by-one error in `parse_args`. I ended up adding a `--verbose` flag." That description becomes the relabeled task. Trains the model to be a faithful executor of self-described instructions, at the cost of training it on noisy self-descriptions.

**3. Capability extraction.** Across a population of failed trajectories, look for recurring useful sub-behaviors (the agent reliably reads the right file before editing; the agent reliably runs tests before declaring success). Relabel those sub-behaviors as the targets of separate, smaller tasks and train on them. This is closer to skill discovery than HER proper.

The risk in all these is the same: relabeling generates apparent positive signal where there was none, but it's signal toward a goal the user did not specify. The policy can learn to be very good at "achieve some state I can describe after the fact" while remaining bad at "achieve the goal the user actually wanted." This is a flavor of the goal-misgeneralization failure mode discussed in alignment work, and there is no clean technical fix; it has to be managed via the choice of which relabelings to allow and how to weight them against the original-goal outcome reward.

### Where it helps

Hindsight relabeling is most useful as a way to bootstrap learning when the success rate on the real task is near zero. A policy that has never succeeded at the real goal but has accidentally done many useful sub-things can use relabeling to build up component skills, which then improve the success rate on the real goal once they're integrated. After the real success rate climbs above some threshold (1-5%, very roughly), the value of relabeling drops and the main reward signal takes over. Treat it as a warm-start trick, not a primary training signal.

---

## Self-critique and self-reflection loops

Reflexion (Shinn, Cassano, Berman, Gopinath, Narasimhan, Yao 2023, [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)) introduced a different style of credit assignment: the agent generates a verbal critique of its own failed trajectory, stores that critique in a separate memory, and uses it as additional context for the next attempt. The original work was inference-time only, no gradient updates, but the same idea has been extended to training-time procedures where the critique itself becomes a learning signal.

The schematic:

```
1. Agent attempts task; gets terminal reward (success/failure).
2. If failure: agent (or a separate critic model) reads the trajectory and writes a critique:
   "I should have checked the error log before editing. I assumed the bug was in
    parse_args but it was actually in the import path."
3. The critique is stored.
4. On the next attempt at this task (or a related task), the agent sees the critique
   prepended to its context.
5. Optionally: the critique-generation model is itself trained, using a signal like
   "did the critique help the next attempt succeed?"
```

For credit assignment, the key idea is that the critique provides a localized signal: it points at a specific step or pattern in the trajectory that was responsible for failure. This is a denser signal than the terminal reward and a more interpretable one than a PRM score. The cost is that the critique is generated by a model, so it has all the usual reliability problems of LLM-generated content.

### What this gets you

In settings where the same task gets retried multiple times, self-critique can dramatically improve sample efficiency. The original Reflexion paper reported large gains on HumanEval and on a decision-making benchmark when allowing 3-5 self-reflection iterations.

For training (not just inference), critiques can be used in at least two ways:

**1. As input features.** Train the policy to perform well when conditioned on prior critiques. This is essentially in-context learning at training time: the policy learns to use the critique signal effectively.

**2. As targets.** Train a separate critique-generator model to produce critiques that, when used as context for a new attempt, improve success rate. This requires a measurable feedback loop (run the next attempt, see if it works) and is expensive.

### Where this breaks

The critic is part of the agent. If the agent's critique-generation is wrong (the critic blames the wrong step, identifies a pattern that wasn't actually present), the next attempt's context is contaminated with confidently-wrong analysis. This can be worse than no critique at all: the agent may "fix" things that weren't broken and break things that worked.

Self-critique works best when there is some external validation between attempts: a partial-credit check, a test runner, a human in the loop. Without that, the agent is just talking to itself with confident analysis, and the failure mode is exactly what you would expect.

---

## Tree search at training time

The general recipe: instead of sampling K independent trajectories per task and using GRPO over them, search the trajectory space more systematically. Find a high-reward trajectory using search (MCTS, beam search, ToT-style branching), then train the policy on that trajectory as if it had been sampled from a normal rollout. This is the AlphaGo-style approach generalized to language agents.

Tree of Thoughts (Yao, Yu, Zhao, Shafran, Griffiths, Cao, Narasimhan 2023, [arXiv:2305.10601](https://arxiv.org/abs/2305.10601)) is the canonical reference for the inference-time version. ToT branches the model's reasoning into multiple candidate thoughts at each step, evaluates them with the model itself (using LM-as-judge), and searches across the tree. The original ToT was inference-time only, no training, but the recipe extends naturally to training-time use: run ToT to find a successful trajectory on a task where greedy decoding fails, then use that trajectory as a training example.

### The recipe

```
1. For each task, run search (MCTS or similar) using the current policy as the prior.
2. The search expands many partial trajectories, scoring them either with terminal
   reward (if reachable in the search budget) or with a learned value/heuristic.
3. Identify the highest-scoring complete trajectory found during search.
4. Use that trajectory as a positive example for the policy: maximize the policy's
   log-probability of producing the actions in that trajectory, given the histories.
5. Optionally: also train the value/heuristic on the search results.
```

This works because search effectively concentrates the gradient signal. Random sampling at K = 8 on a 1% success task gives you almost no usable signal; targeted search at the same compute budget might find 3-4 successful trajectories per task by exploiting model-prior structure to prune the search space.

### What this gets you

Search lets you train on problems that are out of reach for plain sampling. AlphaGo-style results in formal math (theorem proving) come from this recipe: the model proposes proof steps, a tree search explores combinations of them, and the policy is trained on the proofs that the search finds. As of early 2026, the most visible academic example is in formal theorem proving (Lean / Coq), where the binary reward (proof checker accepts) makes search-based training particularly clean.

Monte-Carlo Tree Self-Refine variants for math (e.g., Zhang, Huang, Zhou, Li, Ouyang 2024, [arXiv:2406.07394](https://arxiv.org/abs/2406.07394)) apply similar ideas in less verified domains; treat the specific claims in such work with the usual unreviewed-citation caution, but the structural idea (search to find good trajectories, train on them) is well-established.

### What this costs

Search is expensive. A tree search that expands B branches at each of T steps explores up to B^T trajectories; even with aggressive pruning, the per-task compute can be 10-100× the cost of plain K-sample rollouts. The compute is well-spent on hard tasks where plain sampling fails, but you don't want to pay it on easy tasks where greedy decoding already works. The practical engineering is in deciding when to deploy search and when to rely on cheaper sampling.

The other cost is that training on search-discovered trajectories has a distribution-shift hazard: the search distribution is different from the policy's sampling distribution. You are training the policy to imitate trajectories that were generated by a search procedure on top of the policy, not by the policy alone. This can cause the policy to learn behaviors that are only useful in conjunction with search, which then degrades pure-greedy performance. The standard mitigation is to also train on regular sampled trajectories (mixed updates) rather than only on search outputs.

---

## Curriculum and task decomposition

The brute-force way to make long-horizon problems tractable is to not give the model a long-horizon problem. Decompose the task into shorter pieces, train on the pieces first, and scale up the horizon gradually.

### Curriculum on horizon length

If the target task has T = 500, start training on T = 20 instances. Get the success rate up. Move to T = 50. Get the success rate up. Move to T = 100. Eventually reach T = 500 with a policy that has working primitives for each sub-skill.

This is the standard curriculum-learning idea, applied to horizon as the difficulty axis. It works when there is a natural decomposition (a 500-step coding task is approximately a sequence of independent 20-step coding tasks). It doesn't work when the long horizon comes from genuinely long dependencies: a 500-step task where the action at step 7 is only useful in combination with the action at step 314 cannot be solved by mastering 20-step segments in isolation.

### Hierarchical RL with subgoals

A manager policy picks a subgoal. A worker policy executes a short trajectory to achieve that subgoal. The manager's "actions" are the subgoals; the worker's "actions" are the primitive steps. The credit assignment becomes a two-level problem:

```
Manager:  every K steps, picks a new subgoal g.
Worker:   for the next K steps, acts to achieve g; gets reward when g is achieved.
```

The manager operates on horizon T/K (much shorter than T), and the worker operates on horizon K. Each level is easier to train than the flat T-horizon problem.

This works in narrow settings where the subgoal language is rich enough to express what's needed but compact enough that the manager has a tractable action space. In LLM-agent settings, the subgoal language could be natural-language descriptions ("read the README," "find the failing test"), but designing the right granularity is a nontrivial open problem. Most published work on hierarchical RL in LLM agents either uses hand-designed subtask schemata (one per domain) or learns very coarse "skills" (write code, run tests, search documentation) that look more like task tags than real subgoals.

### Skill-based RL

Closely related: learn a library of reusable skills (each a short policy), then train the agent to invoke them as composite actions. The agent's action space becomes the set of skills plus the primitive vocabulary. Each skill is trained on its own short-horizon reward signal; the agent on top is trained on the original task reward but with much shorter effective trajectories (because each skill call covers many primitive steps).

The hard part is skill discovery. Hand-designed skill sets work but limit the agent to what the designer thought of. Auto-discovered skills (via clustering, options frameworks, etc.) tend to be either too narrow (specific to the training tasks and useless on new ones) or too coarse (just "do task X" with X being the original task). As of 2026, skill discovery for general-purpose LLM agents is not a solved problem.

---

## Code: a hindsight relabel loop

A minimal sketch of how a hindsight-relabel update would look in code. The shapes match the abstractions in the [Lecture 16 rollout code](./16-agentic-rl.md) (`Trajectory`, `Step`).

```python
from dataclasses import dataclass, replace
from typing import Callable


@dataclass
class Task:
    task_id: str
    goal: str           # original goal, e.g., "fix the bug in parse_args"
    metadata: dict      # any additional task description fields


# --- Relabeling primitives ---

def describe_what_happened(trajectory) -> str:
    """
    Generate a description of what the agent actually accomplished in the trajectory.

    In practice this is a separate LM call (or a deterministic summary): read the
    final state of the environment plus the actions taken, and produce a one-line
    goal description that this trajectory could be considered a successful attempt at.

    Returns a string like "renamed parse_args to parse_arguments and added a docstring",
    or "" if the trajectory accomplished nothing identifiable.
    """
    raise NotImplementedError  # implementation depends on the environment


def goal_match_reward(trajectory, goal: str) -> float:
    """
    Verify whether `trajectory` actually achieves `goal`.

    For relabeling to be safe, this verifier should be cheap and reliable for the
    relabeled goal. Typical implementations: regex on the final file diff, schema
    check on the final output, exact-match on a final string. The verifier here is
    NOT the original task's verifier; it's a goal-specific check.

    Returns 1.0 if the trajectory achieves goal, 0.0 otherwise.
    """
    raise NotImplementedError


# --- Relabeling step ---

def hindsight_relabel(trajectory, original_task: Task) -> list[tuple]:
    """
    Given a (possibly failed) trajectory on `original_task`, produce a list of
    (relabeled_task, relabeled_trajectory, relabeled_reward) tuples that can be
    used as additional training data.

    Always includes the original (task, trajectory, original_reward) pair so that
    the real goal is still trained on. The relabeled examples are extras.
    """
    examples = []

    # 1. Always include the original outcome.
    original_reward = trajectory.total_reward
    examples.append((original_task, trajectory, original_reward))

    # 2. If the original task failed, attempt to relabel.
    if original_reward < 0.5:  # treat as failure for binary tasks
        accomplished_goal = describe_what_happened(trajectory)
        if accomplished_goal:
            # Build a new task with the relabeled goal.
            relabeled_task = replace(
                original_task,
                task_id=original_task.task_id + "_hindsight",
                goal=accomplished_goal,
                metadata={
                    **original_task.metadata,
                    "relabeled_from": original_task.task_id,
                    "relabel_kind": "describe-what-happened",
                },
            )
            # Verify the relabeled goal really was achieved.
            relabel_r = goal_match_reward(trajectory, accomplished_goal)
            if relabel_r > 0.5:
                examples.append((relabeled_task, trajectory, relabel_r))

    return examples


# --- Training loop with relabeling ---

def train_with_hindsight(
    policy,
    env,
    tasks: list[Task],
    rollout_fn: Callable,   # signature: (policy, env, task) -> Trajectory
    update_fn: Callable,    # signature: (policy, examples) -> loss
    n_steps: int = 1000,
    relabel_weight: float = 0.3,  # how much to upweight relabeled data
):
    """
    Standard outer training loop, but each rollout is potentially augmented with
    relabeled (task, trajectory, reward) examples.

    `relabel_weight` controls how much credit relabeled examples get; tuning this
    is empirical. A small value (0.1-0.3) keeps the original-goal signal dominant.
    """
    for step_idx in range(n_steps):
        all_examples = []
        for task in tasks:
            trajectory = rollout_fn(policy, env, task)
            examples = hindsight_relabel(trajectory, task)

            # Apply relabel_weight to the relabeled examples (not the original).
            weighted = []
            for (t, tau, r) in examples:
                if t.task_id != task.task_id:
                    # this is a relabeled example
                    weighted.append((t, tau, r * relabel_weight))
                else:
                    weighted.append((t, tau, r))
            all_examples.extend(weighted)

        loss = update_fn(policy, all_examples)
        if step_idx % 50 == 0:
            print(f"step {step_idx}: loss={loss:.4f} n_examples={len(all_examples)}")
```

Notes on this sketch.

`describe_what_happened` is doing a lot of work in not many lines. In practice, this is the hardest part of an LLM-agent hindsight pipeline: generating accurate descriptions of what was actually accomplished, in a vocabulary that the policy can then re-target. The dumb version is "look at the final file diff and describe it"; the smart version involves running a separate LM to read the trajectory and write the goal description. Either way, the description's quality bounds the quality of the relabeling.

`goal_match_reward` is a per-relabeled-goal verifier. This is a different verifier from the one used for the original task. It needs to be cheap (called on every relabel) and reliable (otherwise the relabel signal is just noise). Typical patterns: regex match on the final state, schema validation, exact-match on a target string.

`relabel_weight` is the knob for how much you trust the relabeled signal. If relabeled data dominates the gradient, the policy will learn to be good at achieving descriptions-of-what-it-already-did, which is not the same as achieving user-specified goals. Keep it small.

The whole loop is most useful early in training, when the original success rate is near zero. Once the real reward signal is reliable (success rate > 5-10%), the relabeled signal becomes less valuable and can be turned off or reduced further.

---

## Code: tree search then train

A minimal sketch of the tree-search-then-train recipe. The idea: use the current policy as a prior for a tree search, find a successful trajectory, then add it to the training batch as a positive example.

```python
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Node:
    history: list[dict]               # state at this node
    action: Optional[str]             # action taken from parent to reach here
    parent: Optional["Node"]
    children: list["Node"] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    is_terminal: bool = False
    terminal_reward: float = 0.0

    @property
    def mean_value(self) -> float:
        return self.value_sum / max(1, self.visits)


def ucb_score(child: Node, parent_visits: int, c: float = 1.4) -> float:
    """Standard UCB1 selection criterion for tree search."""
    if child.visits == 0:
        return float("inf")
    import math
    exploit = child.mean_value
    explore = c * math.sqrt(math.log(parent_visits) / child.visits)
    return exploit + explore


def tree_search(
    policy,
    env,
    task,
    n_simulations: int = 64,
    branching: int = 4,
    max_depth: int = 30,
) -> Optional["Trajectory"]:
    """
    Search the trajectory tree using the policy as a prior. Returns the
    highest-reward complete trajectory found during search, or None if no
    successful trajectory was found within the search budget.

    Each simulation:
      1. Select: walk down from the root using UCB until reaching a leaf.
      2. Expand: sample `branching` actions from the policy at that leaf,
         step the environment, attach the new children.
      3. Rollout: from one of the new children, run a fast greedy rollout
         to terminal state to get a reward estimate.
      4. Backprop: propagate the reward up the tree.

    At the end, the best complete trajectory found is reconstructed and returned.
    """
    initial_obs = env.reset(task)
    root = Node(
        history=[
            {"role": "system", "content": "You are an agent."},
            {"role": "user",   "content": task["description"]},
            {"role": "env",    "content": initial_obs},
        ],
        action=None,
        parent=None,
    )

    best_terminal: Optional[Node] = None
    best_reward = -float("inf")

    for sim in range(n_simulations):
        # --- 1. SELECT ---
        node = root
        env.restore(task)  # reset env to root state; in practice this is the hard part:
                           # the env must support deterministic resets, or you must
                           # replay the trajectory of actions to reach the node
        while node.children and not node.is_terminal:
            scores = [ucb_score(c, node.visits) for c in node.children]
            node = node.children[scores.index(max(scores))]
            if node.action is not None:
                obs, _, done = env.step(node.action)
                if done:
                    node.is_terminal = True

        if node.is_terminal:
            # nothing to expand
            reward = node.terminal_reward
        elif len(node.history) > max_depth * 2:  # rough depth check
            reward = 0.0
        else:
            # --- 2. EXPAND ---
            for _ in range(branching):
                action = policy.act(node.history)  # sample with temperature > 0
                # snapshot env state, step, build child
                obs_after, reward_after, done_after = env.step(action)
                child_history = node.history + [
                    {"role": "assistant", "content": action},
                    {"role": "env",       "content": obs_after},
                ]
                child = Node(
                    history=child_history,
                    action=action,
                    parent=node,
                    is_terminal=done_after,
                    terminal_reward=reward_after if done_after else 0.0,
                )
                node.children.append(child)
                # undo step for the next sibling (env must support undo,
                # or you re-walk from root for each child)
                env.restore(task)
                for ancestor_action in _path_actions(node):
                    env.step(ancestor_action)

            # --- 3. ROLLOUT ---
            # pick one child arbitrarily, run greedy to terminal
            sim_node = node.children[0]
            sim_history = list(sim_node.history)
            depth = 0
            done = sim_node.is_terminal
            reward = sim_node.terminal_reward
            while not done and depth < max_depth:
                action = policy.act_greedy(sim_history)
                obs, reward, done = env.step(action)
                sim_history.append({"role": "assistant", "content": action})
                sim_history.append({"role": "env",       "content": obs})
                depth += 1

        # --- 4. BACKPROP ---
        n = node
        while n is not None:
            n.visits += 1
            n.value_sum += reward
            n = n.parent

        # Track the best complete trajectory seen so far
        if node.is_terminal and node.terminal_reward > best_reward:
            best_reward = node.terminal_reward
            best_terminal = node

    if best_terminal is None:
        return None

    return _reconstruct_trajectory(best_terminal)


def _path_actions(node: Node) -> list[str]:
    """Walk from root to node, collecting the actions taken."""
    actions = []
    n = node
    while n.parent is not None:
        actions.append(n.action)
        n = n.parent
    return list(reversed(actions))


def _reconstruct_trajectory(terminal_node: Node):
    """Build a Trajectory object from a path through the tree."""
    from collections import namedtuple
    Step = namedtuple("Step", ["history", "action", "reward"])
    Trajectory = namedtuple("Trajectory", ["steps", "total_reward"])
    steps = []
    n = terminal_node
    chain = []
    while n.parent is not None:
        chain.append(n)
        n = n.parent
    chain.reverse()
    for node in chain:
        steps.append(Step(
            history=node.parent.history,
            action=node.action,
            reward=node.terminal_reward if node.is_terminal else 0.0,
        ))
    return Trajectory(steps=steps, total_reward=terminal_node.terminal_reward)


# --- Training loop combining search and gradient updates ---

def train_with_search(
    policy,
    env,
    tasks,
    sample_fn,           # plain GRPO-style sampling rollout
    search_fn,           # tree_search above
    update_fn,           # gradient update given trajectories
    n_steps: int = 1000,
    search_fraction: float = 0.3,
    sample_K: int = 8,
):
    """
    Mix plain GRPO sampling with occasional tree search on hard tasks.

    A task is "hard" if plain sampling produces no successful trajectories;
    on hard tasks, deploy search to try to find one positive example.
    """
    for step_idx in range(n_steps):
        all_trajectories = []
        for task in tasks:
            # Plain sampling first
            sampled = [sample_fn(policy, env, task) for _ in range(sample_K)]
            successes = [t for t in sampled if t.total_reward > 0.5]

            if not successes and step_idx % int(1 / search_fraction) == 0:
                # No success from plain sampling; try search
                searched = search_fn(policy, env, task)
                if searched is not None and searched.total_reward > 0.5:
                    sampled.append(searched)

            all_trajectories.extend(sampled)

        loss = update_fn(policy, all_trajectories)
        if step_idx % 50 == 0:
            n_success = sum(1 for t in all_trajectories if t.total_reward > 0.5)
            print(f"step {step_idx}: loss={loss:.4f} success={n_success}/{len(all_trajectories)}")
```

Notes on this sketch.

The `env.restore(task)` API is the hardest engineering problem in practice. Real environments (a shell, a browser, a build system) are not trivially resettable to an arbitrary intermediate state. The two practical approaches: (a) the environment supports snapshotting (containers, virtual machines), which is expensive; (b) re-walk from the initial state by replaying actions (deterministic only, slow at depth). Most research implementations use (a) with aggressive caching.

The search budget (`n_simulations`, `branching`, `max_depth`) is set per-task and depends on how much compute you can afford. A typical setting for SWE-bench-style tasks is `n_simulations = 64-256`, `branching = 4`, `max_depth = 30`, but the right values are empirical.

The mixing strategy (`search_fraction`, the "hard task" detection) is a design choice. Always-search is too expensive; never-search loses the variance-reduction benefit. The conditional-on-failure approach in the sketch is a reasonable starting point but it has a subtle bias: tasks that are at the very edge of the policy's ability will get search applied repeatedly while easy tasks won't, which can over-weight the hard tasks in the training distribution. This may or may not be what you want.

Training on search-discovered trajectories will partially imitate the search's distribution rather than the policy's. The mitigation is mixing in plain sampled trajectories (the loop does this) and limiting the search-discovered examples to a fraction of the gradient mass.

---

## Open questions in 2026

Long-horizon credit assignment doesn't have a clean solution. A few directions are visibly being explored.

### Long-context value functions

A value function that can process entire 50K-token agentic trajectories would, in principle, give per-step credit assignment via standard GAE. The blockers are (a) compute: a critic that scales with context length is expensive to train and run; (b) data: the value function needs to be fit to something, which means either ground-truth Monte Carlo returns (expensive to collect for long-horizon tasks) or bootstrapped TD estimates (which compound bias).

A few research directions: shared trunks between policy and value (the value head is a small projection on top of the policy's representation, reducing compute), value functions that operate on summaries of history rather than full context (lossy but cheap), value functions trained on a curriculum of horizons (start short, scale up). None of these are obviously dominant yet.

### Learned credit assignment networks

A separate model that takes a full trajectory plus a terminal reward and outputs per-step credit weights. The idea: if you can't fit a value function, maybe you can fit a function that says "for this kind of trajectory, this step was 30% responsible for the outcome." Train the credit network end-to-end using the policy gradient on the policy as the downstream signal.

This is appealing in principle and has been explored in classical RL (under names like "return decomposition" and "RUDDER-style attribution"). For LLM-agent settings, the open problem is the same as for value functions: how to fit the credit network when the data is sparse and the contexts are huge. The advantage over a value function is that the credit network doesn't need to predict the reward; it only needs to attribute it, which is potentially a more tractable problem.

### Persistent memory across trajectories

A trajectory ends. The agent forgets everything. The next trajectory starts from scratch. This is grossly wasteful: across many attempts at similar tasks, the agent re-learns the same primitive facts (this file is at this path; this command takes these flags; this test failure usually means this thing) every time.

A persistent memory, a structured store that survives between trajectories and can be queried by the agent at relevant moments, would compress the effective horizon. If the agent can pull "I learned last time that the test runner needs the --no-cache flag" from memory at step 3, the trajectory needed to solve the task is much shorter. Long-horizon credit assignment becomes easier when the horizon is shorter.

The open questions are: what to put in memory (everything is too much; nothing is too little), how to retrieve from it (the retrieval model is itself part of the policy and needs to be trained), and how to update it (memory should grow but also should not accumulate noise that degrades retrieval). None of these are solved as of 2026, though several research efforts are visibly working on them.

### Better trajectory-level baselines

The GRPO group baseline works for tasks where you can afford K independent rollouts. For some agentic settings (slow environments, expensive rollouts), even K = 8 is too expensive. Better baselines that exploit cross-task structure (the success rate on similar previous tasks, an LM's estimate of task difficulty, a learned baseline conditioned on the prompt) could reduce variance without requiring more rollouts.

A few public efforts: prompt-difficulty conditioning (the baseline is a learned function of the task description, not just the within-batch mean), task-level value functions (one value per task rather than one per state), historical-mean baselines (running average of success rate on this task). These are incremental improvements over plain group baselines, not breakthroughs.

### What might not be solvable

It is worth naming the possibility that long-horizon credit assignment in the strict sense, figuring out which action at step 7 was responsible for success at step 500, may not be solvable without a lot of extra structure. The information just isn't in the data. A 500-step trajectory with one bit of terminal reward has, by definition, one bit of signal about 500 actions. Even an oracle credit-assignment scheme cannot recover more than that from a single trajectory.

The way out is to add structure: more rollouts (group baselines), denser reward (PRMs), search-based exploration (tree search), task decomposition (curriculum), or memory (persistent context). Each of these is a way of injecting prior information or computational effort to overcome the information-theoretic limit. The interesting research is not in "solving" credit assignment but in figuring out which combinations of these tricks work for which classes of tasks, and at what compute cost.

---

## When to use what

A rough decision tree, based on what's tractable as of 2026.

**Short horizon (T < 30), binary reward, plenty of compute.** GRPO with K = 8 group rollouts. This is the baseline. Most "agentic RL" public training runs are this.

**Long horizon (T = 30-500), binary reward, low success rate.** Combine GRPO with one or more of: (a) hindsight relabeling to bootstrap from near-zero success rate, (b) curriculum starting from shorter horizons, (c) tree search on the hardest tasks. Expect the training to be brittle and the loss curves to look noisy. Verify on held-out tasks frequently.

**Long horizon, dense reward available.** Use PRM-style per-step rewards combined with terminal outcome reward. Weight terminal heavily, PRM lightly. Watch for PRM gaming.

**Very long horizon (T > 500).** No publicly-validated recipe. The state of the art is some combination of hierarchical decomposition, persistent memory, and aggressive compute spending. Expect to do significant engineering on the environment and the reward structure before any RL training is worth running.

**Research / formal theorem proving.** Tree search dominated. The binary reward (proof checker accepts) and the search-friendly action space (single proof step at a time) make this a particularly clean fit for AlphaGo-style train-on-search-results recipes.

---

## Exercises

These are projects, not 30-minute problems. Treat them as integration exercises that build on [Lecture 16's](./16-agentic-rl.md) agentic-RL scaffold rather than as new self-contained problems.

**1. Variance scaling.** Take the agentic RL loop from [Lecture 16](./16-agentic-rl.md). Generate 100 trajectories of horizon T from a random policy on a toy environment (any environment where you can vary T). Measure the variance of the policy-gradient estimator (the sample std of ∇_θ log π · R(τ) across the 100 trajectories) as you vary T = 10, 30, 100, 300. Verify that variance grows roughly linearly with T. Try the same with returns-to-go (per-step discounted returns) and verify the variance is lower. Plot both.

**2. GRPO advantage degradation.** With the same toy environment, vary the success rate by changing task difficulty. At each difficulty level, run GRPO with K = 8 and measure the fraction of batches where all K trajectories have the same reward (so the group-relative advantage is zero or NaN). Plot this against difficulty. Identify the difficulty range where GRPO is informative.

**3. Hindsight relabel toy.** Pick a simple environment with multiple possible "goals" (e.g., a gridworld with multiple target cells, or a tiny code editor with several possible final states). Implement the `describe_what_happened` and `goal_match_reward` functions from the hindsight code sketch. Verify that failed trajectories on goal A can be relabeled as successful trajectories on goal B when B is something the agent actually did. Train a goal-conditioned policy with and without hindsight relabeling; compare sample efficiency.

**4. PRM toy.** Train a tiny PRM on a simple step-judgment task (e.g., the PRM rates whether each step in a chain-of-thought math solution is correct, using synthetic data). Use it as a dense reward signal for a small policy and observe what happens: does the policy learn to solve problems, or does it learn to produce text that the PRM rates highly?

**5. Mini tree search.** Implement a stripped-down version of the tree search sketch above on a tiny environment where you can afford full enumeration. Verify that for hard tasks (low success rate from sampling), search finds successful trajectories that sampling does not. Measure the compute multiplier (search cost / sampling cost) and the success-rate multiplier (search successes / sampling successes).

---

## References

**Schulman, Moritz, Levine, Jordan, Abbeel (2015)**: "High-Dimensional Continuous Control Using Generalized Advantage Estimation." [arXiv:1506.02438](https://arxiv.org/abs/1506.02438). Verified. The GAE estimator; introduces λ-interpolation between Monte Carlo and TD for variance reduction. Foundational for any modern policy-gradient method.

**Andrychowicz, Wolski, Ray, Schneider, Fong, Welinder, McGrew, Tobin, Abbeel, Zaremba (2017)**: "Hindsight Experience Replay." [arXiv:1707.01495](https://arxiv.org/abs/1707.01495). Verified. The original HER paper; relabels failed trajectories with the goals they actually accomplished. Critical primitive for the hindsight-relabel approach in LLM-agent settings.

**Schulman, Wolski, Dhariwal, Radford, Klimov (2017)**: "Proximal Policy Optimization Algorithms." [arXiv:1707.06347](https://arxiv.org/abs/1707.06347). Verified. Background for the clipped surrogate used in most modern policy-gradient implementations.

**Yao, Zhao, Yu, Du, Shafran, Narasimhan, Cao (2022)**: "ReAct: Synergizing Reasoning and Acting in Language Models." [arXiv:2210.03629](https://arxiv.org/abs/2210.03629). Verified. The scaffold ([Lecture 16](./16-agentic-rl.md)) on which most agentic RL training operates.

**Shinn, Cassano, Berman, Gopinath, Narasimhan, Yao (2023)**: "Reflexion: Language Agents with Verbal Reinforcement Learning." [arXiv:2303.11366](https://arxiv.org/abs/2303.11366). Verified. Self-critique loop where the agent generates verbal reflections between attempts; influential for the self-critique approach to credit assignment.

**Yao, Yu, Zhao, Shafran, Griffiths, Cao, Narasimhan (2023)**: "Tree of Thoughts: Deliberate Problem Solving with Large Language Models." [arXiv:2305.10601](https://arxiv.org/abs/2305.10601). Verified, NeurIPS 2023. ToT-style branching at inference time; the structural basis for tree-search-then-train recipes.

**Lightman, Kosaraju, Burda, Edwards, Baker, Lee, Leike, Schulman, Sutskever, Cobbe (2023)**: "Let's Verify Step by Step." [arXiv:2305.20050](https://arxiv.org/abs/2305.20050). Verified. The PRM800K dataset and process-reward-model approach for math reasoning; shows process supervision outperforms outcome supervision in that setting.

**Jimenez, Yang, Wettig, Yao, Pei, Press, Narasimhan (2023)**: "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" [arXiv:2310.06770](https://arxiv.org/abs/2310.06770), ICLR 2024. Verified. The benchmark that defines the long-horizon agentic coding setting referenced throughout this lecture.

**Wang, Li, Shao, Xu, Dai, Li, Chen, Wu, Sui (2023)**: "Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations." [arXiv:2312.08935](https://arxiv.org/abs/2312.08935). Verified. PRM training without human step labels by using MC-rollout-based step scoring; extends "Let's Verify Step by Step" to a cheaper labeling scheme.

**Shao, Wang, Zhu, Xu, Song, Bi, Zhang, Zhang, Li, Wu, Guo (2024)**: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." [arXiv:2402.03300](https://arxiv.org/abs/2402.03300). Verified. Introduces GRPO; the trajectory-level outcome reward + group baseline approach that dominates current practice.

**Zhang, Huang, Zhou, Li, Ouyang (2024)**: "Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B." [arXiv:2406.07394](https://arxiv.org/abs/2406.07394). Verified (existence and metadata). Representative MCTS-style application of search to math reasoning; treat specific performance claims as unverified.

**DeepSeek-AI (2025)**: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." [arXiv:2501.12948](https://arxiv.org/abs/2501.12948). Verified. The public R1 training recipe; uses GRPO-style outcome rewards on verifiable tasks (math, code, STEM).

**Wei, Duchenne, Copet, Carbonneaux, Zhang, Fried, Synnaeve, Singh, Wang (2025)**: "SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution." [arXiv:2502.18449](https://arxiv.org/abs/2502.18449), NeurIPS 2025. Verified. RL on agentic software evolution data; representative of the current published state of long-horizon agentic training, with the limitations discussed above.

**Systems with proprietary training details.** Several of the highest-performing agentic coding systems and research agents in 2024-2026 are described in blog posts or technical reports without specifying their credit-assignment approach. References in this lecture to "what works at scale" are necessarily based on the public literature; the proprietary systems may use techniques that are not documented anywhere referenceable.

---

## Next lecture

Forthcoming. Possibilities include: persistent memory and retrieval in agentic RL; safety and reward-hacking in long-horizon settings; or a deeper treatment of search-based RL for formal mathematics.
