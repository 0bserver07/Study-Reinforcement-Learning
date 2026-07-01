<!-- status: unreviewed | last-reviewed: never -->

# Lecture 21: Multi-agent RL and self-play

_Unreviewed: no one has checked this end to end. Treat the math, code, and citations as unverified._

**Time**: ~3 h · **Prerequisites**: Lectures 02, 04, 06

---

## What changes when there's more than one learner

Lectures 01–08 treat the world as a single MDP. There's an environment with stationary dynamics, a reward function defined on (state, action), and a single agent whose only job is to act well in it. The textbook (Sutton & Barto, 2018) is built around this assumption: tabular methods, function approximation, policy gradients, all of it lives inside that single-agent frame.

Multi-agent RL relaxes one assumption: there is more than one decision maker. The environment now contains several agents, each picking actions, each receiving rewards. The transition function takes the *joint* action of all agents and returns the next state. From any one agent's perspective, the rest of the agents are part of the environment, but a part that is itself learning, which means that "environment" is non-stationary in a way the single-agent algorithms aren't built to handle.

This shows up in three settings that are worth distinguishing from the start:

- **Cooperative**: all agents share one reward function. Examples: a team of robots collaborating on a warehouse task, multiple LLM agents coordinating on a coding job. The hard part is credit assignment: which agent caused the reward to go up?
- **Competitive (zero-sum)**: one agent's reward is the negative of another's. Two-player chess, Go, poker, StarCraft 1v1. The hard part is opponent modeling: your best response depends on what they do, and theirs depends on what you do.
- **General-sum**: rewards are neither identical nor opposite. Most realistic settings: markets, traffic, social dilemmas, multi-team games. The hard part is everything: cooperation can emerge or fail, equilibria are not unique, and "optimal" stops being a well-defined target.

The single-agent toolkit doesn't transfer cleanly to any of these. The next sections walk through why, then through the modern algorithmic responses.

---

## Stochastic games: the formalism

The multi-agent generalization of the MDP is the **stochastic game** (also called a Markov game). It was introduced by Shapley in 1953 for the two-player zero-sum case and generalized by Littman 1994 to the deep-RL-relevant formulation. A stochastic game is a tuple:

```
G = (N, S, {A_i}, P, {R_i}, gamma)
```

- `N`: the set of agents, indexed `i = 1, ..., n`.
- `S`: the state space (shared across agents).
- `A_i`: agent `i`'s action set. The joint action space is `A = A_1 × A_2 × ... × A_n`.
- `P : S × A → Δ(S)`: the transition function. It depends on the joint action.
- `R_i : S × A × S → R`: agent `i`'s reward function. Different agents can have different rewards from the same transition.
- `gamma` in [0, 1): the discount factor.

The two limiting cases:

- All `R_i` identical → fully cooperative. The game collapses to a single MDP if you treat the joint action as a single agent's action.
- `n = 2` and `R_1 = -R_2` → two-player zero-sum.

The cooperative case looks like it should reduce to single-agent RL, and in principle it does, if you can centralize. The problem is that the joint action space `A` grows multiplicatively in `n`. With 10 agents and 5 actions each, you have `5^10 ≈ 10^7` joint actions per state. That kills tabular methods immediately and stresses any function approximator that takes the joint action as input.

The competitive and general-sum cases have a different problem: there is no single objective to optimize. Each agent has its own `R_i`, and the question "what's the best policy for agent `i`?" is ill-posed without specifying what the other agents are doing. This is where game theory enters.

---

## Equilibrium concepts

In single-agent RL, "optimal" means "maximizes expected discounted return." In multi-agent RL, "optimal for one agent" depends on the other agents, so the natural object isn't a single optimal policy but a *joint* policy where no agent has an incentive to unilaterally deviate. This is a **Nash equilibrium**.

Formally, a joint policy `(pi_1*, ..., pi_n*)` is a Nash equilibrium if for every agent `i`:

```
V_i(pi_i*, pi_{-i}*) >= V_i(pi_i, pi_{-i}*)   for all pi_i
```

where `pi_{-i}*` denotes the policies of all agents except `i`. In words: holding the others' policies fixed, no single agent can improve by switching strategies.

Three things to know about Nash equilibria in stochastic games:

- **Existence**: under mild conditions (finite states, finite actions, discounted), at least one Nash equilibrium exists. This is from Fink 1964; the proof generalizes Nash's 1950 result for normal-form games.
- **Non-uniqueness**: there can be many. Coordination games (drive-on-the-left vs drive-on-the-right) have at least two equilibria, and the agents have to converge on which one to play.
- **Computational hardness**: finding a Nash equilibrium is PPAD-complete in general (Daskalakis, Goldberg, Papadimitriou 2009; for specific classes it can be polynomial). Approximating one is also hard.

For two-player zero-sum games the picture is much cleaner. The Nash equilibrium is unique in value (different equilibrium policies all give the same game value), and you can compute it via linear programming for the matrix-game case or via fictitious play / regret matching for the sequential case. This is why two-player zero-sum work (chess, Go, poker) has a cleaner algorithmic story than general-sum work.

For cooperative games, the natural target is not Nash but a joint optimum: the joint policy that maximizes the shared return. Every joint optimum is a Nash equilibrium (no one wants to deviate from the best joint policy), but not every Nash equilibrium is a joint optimum. Cooperative MARL methods explicitly target joint optima.

---

## Why naive independent learning fails

The simplest thing you can try: give each agent its own Q-learning agent. Treat the other agents as part of the environment. This is **independent Q-learning** (IQL), and it almost works, except for two reasons it usually doesn't.

**Non-stationarity**. Agent `i` learns a Q-function `Q_i(s, a_i)` that estimates expected return assuming the other agents' policies stay fixed. But the other agents are learning too. Their policies change with each update. So the Q-target that agent `i` was learning toward is actively moving. The convergence guarantees for tabular Q-learning (Watkins, Dayan 1992) assume a stationary MDP. They do not apply here.

In practice: independent Q-learning often oscillates instead of converging. Two agents update against each other's old policies, then the new policies make those updates wrong, and you get cycles. In zero-sum matrix games like rock-paper-scissors, two independent Q-learners will not converge to the (1/3, 1/3, 1/3) Nash policy: they'll cycle.

**Credit assignment in cooperative settings**. If all agents share a reward, agent `i` sees the team reward and has to figure out how much of it was caused by its own action versus the others'. With one shared reward signal and `n` agents, the per-agent gradient is essentially the team gradient, plus noise from everyone else's randomness. Variance scales badly with `n`.

These two failure modes, non-stationarity and joint credit assignment, are what most modern MARL algorithms try to fix.

A short note on the convergence claims you'll see in older literature. **Joint-action learners** (Claus and Boutilier 1998) extend Q-learning to take the joint action as the input: `Q(s, a_1, ..., a_n)`. This restores convergence in the cooperative case (it's just a single Q-table over the joint space) but doesn't scale beyond a handful of agents. **Hyper-Q** (Tesauro 2003) maintains a Q-function over `(state, opponent strategy)` pairs, which is well-defined but requires estimating opponent strategies and again doesn't scale. These are mostly historical now; the modern methods are CTDE (next section) and self-play (the section after).

---

## Centralized training, decentralized execution (CTDE)

CTDE is the dominant architectural pattern for cooperative and mixed MARL. The idea:

- **At training time**, you can use information that wouldn't be available at deployment: the other agents' observations, their hidden states, the global state, the joint action. Use this extra information to train the policies.
- **At execution time**, each agent acts based only on its local observation. No centralized coordinator, no inter-agent communication beyond what the policies have learned to send.

This buys you the gradient quality of centralized training (because you can condition the critic on everything) without the deployment cost of centralized execution (each agent runs its own policy, locally). Three of the canonical CTDE algorithms are MADDPG, QMIX, and MAPPO.

### MADDPG (Lowe et al. 2017, arXiv:1706.02275)

MADDPG (Multi-Agent DDPG) extends the deterministic policy gradient method from [Lecture 07](./07-off-policy-rl.md) to the multi-agent setting. Each agent `i` has:

- A deterministic policy `mu_i(o_i; theta_i)` that takes only the local observation `o_i` and outputs an action.
- A centralized critic `Q_i(s, a_1, ..., a_n; phi_i)` that takes the global state and the joint action and predicts agent `i`'s return.

The critic update is the standard TD loss, but conditioned on the joint action:

```
L(phi_i) = E[(Q_i(s, a_1, ..., a_n) - y)^2]
y = r_i + gamma * Q_i'(s', mu_1'(o_1'), ..., mu_n'(o_n'))
```

The policy gradient for agent `i` flows through its own policy only:

```
nabla_theta_i J = E[ nabla_theta_i mu_i(o_i) * nabla_a_i Q_i(s, a_1, ..., a_n) |_{a_i = mu_i(o_i)} ]
```

The critic is centralized (it sees everyone's actions), so it gives agent `i` a stationary learning target even though the other agents' policies are changing. From the critic's perspective, the other agents' actions are inputs; the critic's job is just to predict return given those inputs. The non-stationarity is absorbed into the critic's input space.

What MADDPG won't fix:
- It doesn't handle large numbers of agents well. The critic input grows linearly with `n` (one action per agent), and the variance of the policy gradient grows with `n` too.
- It assumes you know the other agents during training. In an adversarial deployment this is fine for self-play; in a heterogeneous setting it requires either co-training or opponent modeling.
- The original paper uses a deterministic policy with Gaussian exploration noise. For discrete action spaces you switch to Gumbel-softmax tricks or use a stochastic variant.

The MADDPG paper reports results on cooperative communication, predator-prey, and physical-deception toy tasks. Performance gains are clear in the small-agent regime (2–4 agents); the paper doesn't push past that.

### Value decomposition: VDN and QMIX

For purely cooperative settings with a shared team reward, a different family of methods asks: can you express the joint Q-function as a function of per-agent Q-functions, in a way that makes decentralized execution trivial?

The decomposition you want is **individual-global max consistency (IGM)**: the joint action that maximizes `Q_tot(s, a_1, ..., a_n)` should be the joint action where each `a_i` maximizes its individual `Q_i`. If IGM holds, each agent can act greedily on its own `Q_i` and the resulting joint action is the joint-greedy action. No centralized coordinator at execution time.

**VDN** (Sunehag et al. 2017, arXiv:1706.05296) takes the simplest approach: assume `Q_tot` is a sum of per-agent Q-functions:

```
Q_tot(s, a_1, ..., a_n) = sum_i Q_i(o_i, a_i)
```

This trivially satisfies IGM (the argmax of a sum of independent terms is the joint argmax of each term). VDN is trained end-to-end: you compute the team TD loss using `Q_tot`, and gradients flow back through the per-agent Q-functions. It's surprisingly effective on simple coordination tasks but fails on tasks where credit assignment is genuinely non-additive, e.g., where one agent's action only matters in combination with another's.

**QMIX** (Rashid et al. 2018, arXiv:1803.11485) generalizes VDN by allowing a more flexible mixing function:

```
Q_tot(s, a_1, ..., a_n) = f(Q_1(o_1, a_1), ..., Q_n(o_n, a_n); s)
```

where `f` is a neural network whose weights are produced by a hyper-network that takes the global state `s`. The catch is that `f` is constrained to be **monotonic** in each `Q_i`: increasing one agent's `Q_i` cannot decrease `Q_tot`. The constraint is enforced by making the hyper-network output non-negative weights.

Monotonicity is sufficient (though not necessary) for IGM. It's a real restriction (there exist cooperative tasks where the optimal `Q_tot` is non-monotonic in the per-agent values), but it's general enough to handle most of the StarCraft multi-agent challenge benchmark, where QMIX was originally evaluated.

Concretely, the StarCraft Multi-Agent Challenge (SMAC, Samvelyan et al. 2019, arXiv:1902.04043) is the de facto benchmark for cooperative MARL. QMIX outperforms VDN, IQL, and the various baseline methods on the harder maps in SMAC. Subsequent work (QTRAN, MAVEN, Weighted QMIX) tried to relax the monotonicity constraint while preserving IGM, with mixed practical success: the simple QMIX recipe remains a strong baseline.

### MAPPO (Yu et al. 2021, arXiv:2103.01955)

The full title of this paper is "The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games", and it's worth knowing the title because it tells you what the paper is for. The community had spent years developing custom MARL algorithms (MADDPG, QMIX, COMA, etc.); the question Yu et al. asked was: if you just take PPO ([Lecture 06](./06-ppo.md)) and apply it in a CTDE setup, how badly does it do?

The answer: it's competitive with the specialized methods on most cooperative benchmarks, including SMAC and Hanabi. The CTDE recipe is straightforward:

- Each agent has its own policy `pi_i(a_i | o_i)`, shares parameters across homogeneous agents (parameter sharing helps in symmetric tasks).
- A centralized critic `V(s)` takes the global state (or a concatenation of all observations) and predicts the team return.
- Advantage is computed from the centralized critic; policy is updated with PPO's clipped surrogate, just like in single-agent.

The paper documents specific tricks that matter: value normalization, generalized advantage estimation with shared `lambda`, careful initialization. None individually surprising, but the bundle gets PPO to match or exceed QMIX and other specialized methods.

For practical work in 2024–2025, MAPPO is often the right starting point. It's simple, it inherits the well-tuned PPO machinery, and the parameter-sharing trick controls the variance growth in `n`. The specialized methods are still relevant when you need value decomposition for downstream interpretability, or when continuous actions push you toward MADDPG.

### COMA: counterfactual multi-agent policy gradients

One more CTDE method worth knowing: **COMA** (Foerster et al. 2017, arXiv:1705.08926). COMA tackles the credit assignment problem head-on by computing a *counterfactual baseline*: for each agent, what would the team reward have been if that agent had taken a different action, holding everyone else fixed?

The COMA advantage for agent `i`:

```
A_i(s, a) = Q(s, a) - sum_{a_i'} pi_i(a_i' | o_i) * Q(s, (a_i', a_{-i}))
```

The first term is the actual joint Q-value; the second is the expected Q-value if agent `i` had marginalized over its own action distribution while everyone else's actions stayed fixed. The difference isolates agent `i`'s contribution.

This requires a centralized critic that can be queried for arbitrary actions, and it's expensive (you sum over `|A_i|` actions per gradient step). COMA is rarely the first choice today, but the counterfactual baseline idea shows up in various places (process reward modeling, agentic credit assignment) and is worth understanding.

---

## Self-play

Self-play is the multi-agent special case where all agents are copies (or near-copies) of the same policy, playing each other. It's the workhorse for two-player zero-sum games (chess, Go, poker, StarCraft, Dota) and it sidesteps the opponent-distribution problem by definition: the opponent is just yourself, or a slightly older version of yourself.

The historical milestones, in rough order:

- **TD-Gammon** (Tesauro 1995) was a backgammon program that trained a value function via TD-learning while playing itself. It reached world-champion level for backgammon in the early 1990s. The stochasticity of backgammon (dice rolls) helped: it provided exploration "for free" and prevented some of the convergence pathologies that plague deterministic-game self-play.
- **AlphaGo** (Silver et al. 2016, Nature) combined supervised pre-training on human games with self-play RL and Monte Carlo tree search. It beat Lee Sedol in 2016.
- **AlphaGo Zero / AlphaZero** (Silver et al. 2017, Nature; arXiv:1712.01815) dropped the human-game pre-training entirely. AlphaZero learned chess, shogi, and Go from random initialization, using only self-play and MCTS-guided policy improvement. The learning loop became canonical: at each iteration, run MCTS with the current network to generate "improved" policies, train the network to imitate those improved policies, repeat.
- **AlphaStar** (Vinyals et al. 2019, Nature) extended the self-play recipe to StarCraft II: partial information, real-time, large action space, long horizons. The training relied on a *league* of policies (more on populations below) rather than naive self-play.
- **OpenAI Five** (Berner et al. 2019, arXiv:1912.06680) trained a Dota 2 team via large-scale PPO with self-play and a "team spirit" parameter that interpolated between individual and team rewards. The compute scale was significant: roughly 180 years of game time per day across the training cluster.
- **Pluribus** (Brown and Sandholm 2019, Science) reached superhuman level in 6-player no-limit Texas hold'em poker using Monte Carlo CFR with abstraction and a depth-limited search at play time. Not self-play in the AlphaZero sense (CFR has a different algorithmic flavor), but conceptually adjacent.
- **DeepNash** (Perolat et al. 2022, arXiv:2206.15378) reached expert-human level at Stratego using a regularized Nash dynamics method (R-NaD) with self-play. Stratego is interesting because of its enormous game tree (about 10^535 nodes by some estimates), making MCTS-style methods infeasible.

The pattern across these is consistent: self-play turns a competitive game into a curriculum that improves with the policy. Easy openings get solved first, the policy gets stronger, the openings it now faces are harder, and so on. The opponent's strength scales with yours.

### The basic self-play loop

The skeleton is short:

```python
def self_play_train(
    policy,            # neural network, shared across both players
    env,               # two-player zero-sum game with .reset(), .step()
    n_iterations,
    games_per_iter,
    eval_every,
):
    """
    Vanilla self-play: latest policy plays itself, learns from the games.
    """
    history = []  # list of past policy snapshots

    for it in range(n_iterations):
        # 1. Generate self-play data: latest policy vs latest policy.
        trajectories = []
        for _ in range(games_per_iter):
            traj = play_game(env, player_1=policy, player_2=policy)
            trajectories.append(traj)

        # 2. Build training targets from the game outcomes.
        #    For each (state, action) by the winning side: positive advantage.
        #    For each (state, action) by the losing side: negative advantage.
        batch = build_batch(trajectories)

        # 3. Update the policy (PPO, REINFORCE, AlphaZero-style imitation
        #    of MCTS-improved policy: depends on the recipe).
        policy.update(batch)

        # 4. Periodically snapshot for evaluation / league play.
        if it % eval_every == 0:
            history.append(deepcopy(policy))
            evaluate_against(policy, history)

    return policy, history
```

A few things this skeleton hides:

- `play_game` has to assign the game outcome (+1, 0, or -1) back to all states visited by each player. The reward is sparse, one scalar per game, so credit assignment relies on the value function or Monte Carlo returns, exactly as in single-agent sparse-reward RL.
- `policy.update` could be PPO with the latest games as a batch (OpenAI Five), AlphaZero-style supervised learning on MCTS-improved targets (AlphaZero), or a CFR-like update (Pluribus). The training algorithm is decoupled from the self-play data-collection loop.
- Snapshotting is necessary because you usually evaluate against historical opponents, not just the current one. A policy that beats its current self every time may have specialized to a narrow style; periodic evaluation against older versions catches this.

### Why naive self-play breaks for non-transitive games

The simplest self-play setup (current policy plays current policy, update on the games) works well for **transitive** games. A game is roughly transitive if "A beats B and B beats C" tends to imply "A beats C." Chess and Go are mostly transitive at the high level; the strongest engines beat each other in fairly stable orderings.

For **non-transitive** games (rock-paper-scissors being the canonical example), naive self-play cycles forever. If you're currently playing all-rock, the gradient pushes you toward all-paper (because paper beats rock). Your opponent (your old self) is also pushed toward all-paper. Then both of you are playing paper, and the gradient pushes toward scissors. Then to rock. The policies cycle through pure strategies and never converge to the mixed Nash equilibrium (1/3, 1/3, 1/3).

Real games aren't pure rock-paper-scissors but they often have non-transitive *components*. StarCraft has rock-paper-scissors-like matchups between unit types and strategies. Stratego has counters that counter counters. A self-play training that never plays against historical versions of itself can get stuck in cycles, repeatedly "unlearning" defenses against strategies the current opponent has temporarily abandoned.

The fix is **fictitious self-play** (FSP).

### Fictitious self-play

The classical fictitious play algorithm (Brown 1951, in normal-form games) has each player at each round play the best response to the *empirical average* of the opponent's history. This converges to Nash in two-player zero-sum matrix games.

**Neural Fictitious Self-Play** (Heinrich and Silver 2016, arXiv:1603.01121) ports this idea to deep RL. Each agent maintains two networks:

- A **best-response network** trained by RL to beat the average opponent.
- An **average policy network** trained by supervised learning to imitate the best-response actions taken so far.

At each step, the agent samples either from the best-response network (exploration into new strategies) or from the average policy network (playing the historical mixture). Opponents see actions drawn from the average network, which approximates the empirical distribution over historical best responses. This mirrors the fictitious play guarantee while using neural function approximation.

NFSP was demonstrated on limit Texas hold'em poker. It converged to near-Nash play in a setting where naive self-play cycles. It's not the only solution (population-based training (next section) is the more popular modern approach), but it's the conceptually clearest fix for the non-transitivity problem.

### Exploitability

The metric you actually care about in zero-sum self-play work is **exploitability**, not win rate.

Win rate against a fixed opponent measures how well you beat that opponent. It says nothing about whether your policy is exploitable by some other opponent. A policy that beats its current self 50% of the time (the symmetric self-play null) might still be terrible: both copies might be playing badly in a way that a third strategy could exploit.

Exploitability is the upper bound on how much a best-response opponent could win against you. Formally, for a policy `pi` in a two-player zero-sum game:

```
exploitability(pi) = max_{pi'} V(pi', pi)
```

where `V(pi', pi)` is the expected return for player 1 playing `pi'` against player 2 playing `pi`. A Nash equilibrium policy has exploitability equal to the game value (often zero by symmetry). A strictly worse-than-Nash policy has higher exploitability, because some opponent can take advantage of its weaknesses.

In practice you can't compute exploitability exactly for large games: that requires solving for a best response, which is itself hard. You approximate it by training a best-response opponent against the policy and reporting its win rate. This is sometimes called the "best-response gap" or "approximate exploitability."

The reason this matters: a self-play training loop can drive win-rate-against-self to 50% (which it must always be by symmetry) while exploitability climbs. This is what cycling looks like: both copies update against each other, converge to some pattern, and a separate opponent that doesn't follow the pattern can crush both of them.

OpenSpiel (Lanctot et al. 2019, arXiv:1908.09453) is a software library that implements exploitability calculations for many small games, plus reference implementations of CFR, fictitious play, NFSP, and PSRO. It's the standard tool for verifying your self-play implementation against ground-truth equilibria on small games before scaling up.

---

## Population-based training and PSRO

Self-play with a single current policy and a single average opponent is one point on a spectrum. The other end is **population-based**: maintain a population of policies, each playing against various others, and use the population dynamics to drive improvement.

The key reference is **PSRO** (Policy Space Response Oracles, Lanctot et al. 2017, arXiv:1711.00832). PSRO unifies a wide class of multi-agent algorithms (including independent learning, fictitious play, and double oracle) into one schema:

1. Maintain a population `P = {pi_1, pi_2, ..., pi_K}` of policies.
2. Compute the *meta-game* payoff matrix: how does each `pi_i` do against each `pi_j`? This gives a `K × K` matrix `M` of expected returns.
3. Compute a **meta-strategy** `sigma` over the population: a probability distribution over which policies to play. The meta-strategy is found by solving the meta-game (e.g., via a game-theoretic solver like fictitious play on the matrix `M`).
4. Train a new policy `pi_{K+1}` as the **best response** to opponents sampled from `sigma`. This is a standard RL training run with the opponent distribution fixed during the iteration.
5. Add `pi_{K+1}` to the population. Recompute `M` (only the new row/column). Repeat.

The "policy space response oracle" name refers to step 4: you have an oracle that, given a distribution over opponent policies, returns a best-response policy. In PSRO that oracle is approximate (it's an RL algorithm), but the structure is what unifies the family.

Different choices of meta-strategy solver give different PSRO instantiations:

- **Uniform meta-strategy** → fictitious play (you train against an average over history, like NFSP).
- **Nash meta-strategy** → double oracle (you train against the Nash mixture over the current population).
- **No-regret meta-strategy** → variants like Mirror Descent PSRO.

PSRO scales to games with deep non-transitivity. AlphaStar's "league" training (Vinyals et al. 2019) is essentially PSRO with three classes of agents (main agents, main exploiters, league exploiters) and a more elaborate matching scheme that's tuned for StarCraft's specific non-transitive structure.

The cost of PSRO: you need a lot of compute. Each iteration trains a new policy from scratch (or near-scratch) as a best response, and you need many iterations to cover the policy space. The DeepNash paper notes that the equivalent of ~10^4 to 10^5 best-response generations are needed for non-trivial games; for Stratego specifically, DeepNash uses a different approach (R-NaD with regularized self-play) precisely because PSRO's per-iteration cost is too high for that game's complexity. AlphaStar was reported to use thousands of TPU-years over its training horizon.

For practical work outside the AlphaStar / DeepNash regime, the cheaper alternatives are usually NFSP-like methods or PSRO variants with smaller populations and shorter inner-loop training.

---

## Mixed cooperative-competitive: the general-sum case

Cooperative MARL has a single shared reward and a clean target (joint optimum). Two-player zero-sum has a clean equilibrium concept (Nash, computable) and self-play. General-sum is messier than either.

The reward structure is now `R_1, ..., R_n` with no necessary relationship between them. Coordination, defection, partial cooperation, betrayal, all are possible equilibrium behaviors depending on the game structure. The classical examples are:

- **Iterated Prisoner's Dilemma**: in single-shot, defection is the dominant strategy and the unique Nash equilibrium. Iterated, with discounting, cooperation can be sustained by tit-for-tat strategies (Axelrod 1984 for the empirical work).
- **Stag Hunt**: two equilibria: both cooperate for high payoff, both defect for low payoff. Coordination on the better equilibrium isn't automatic.
- **Battle of the Sexes**: two equilibria with asymmetric payoffs; the players have to coordinate on which one.

In deep MARL, general-sum has produced two strands of work worth knowing.

**LOLA** (Learning with Opponent-Learning Awareness, Foerster et al. 2017, arXiv:1709.04326) modifies the policy gradient to account for the fact that the opponent will *also* update in response to the current step. The standard policy gradient assumes a fixed opponent. LOLA differentiates through the opponent's expected update, adjusting the gradient to favor outcomes that account for how the opponent's future policy will shift. On iterated prisoner's dilemma, LOLA agents learn to cooperate more reliably than naive policy gradient agents.

LOLA was notable for showing that opponent-aware learning can change equilibrium selection in social dilemmas. The downsides: it requires access to opponent gradients (or estimates thereof), and the meta-gradient is expensive to compute and unstable to optimize. Subsequent work (Stable Opponent Shaping, M-FOS) has tried to make this more practical.

**Emergent autocurricula in mixed-motive environments**. The "Hide and Seek" paper (Baker et al. 2019, arXiv:1909.07528) trained agents in a multi-agent physics environment with hide-and-seek dynamics and a few simple shared and individual reward terms. Without explicit curriculum, the population went through six distinct strategy regimes (hiders learn to use walls, seekers learn ramps to climb walls, hiders learn to lock ramps, seekers learn to surf on top of boxes, etc.), each emerging in response to the previous opponent strategy. This is a general-sum (not zero-sum) game, but the autocurriculum dynamic is similar to the population dynamics in zero-sum self-play: opponents drive each other's learning.

For general-sum work, the practical advice is short: there isn't a single right algorithm, and equilibrium selection is your main difficulty. Independent learners often coordinate poorly (or settle on bad equilibria); centralized critics with shaped rewards work in specific domains; opponent-aware methods work on small problems but don't yet scale. This is one of the more open areas of MARL.

---

## A short note on multi-agent LLMs

In 2024 and 2025 there's been significant activity around multi-agent LLM systems: multiple language model agents that talk to each other, debate, critique, or coordinate on tasks. Most of this work uses *frozen* LLMs in elaborate scaffolds (multi-agent debate, role-played panels, judge-based ensembles). It's not RL training in the sense the rest of this lecture covers; it's prompting and inference-time orchestration.

Two pieces worth distinguishing:

- **Multi-agent debate at inference time** (Du et al. 2023, arXiv:2305.14325) shows that having multiple LLM instances debate and converge on an answer improves factuality on certain tasks. This is inference-time only: no model parameters are updated.
- **Multi-agent training with RL** is the harder version: actually training LLM policies via something like MAPPO or self-play, where multiple LLM agents update against each other. This is much less developed as of late 2025. Work exists on adversarial LLM training (red-team vs blue-team setups, debate as a training signal), but a clean canonical reference for "multi-agent RL fine-tuning of LLMs" does not exist as of this writing; at least, not one this lecture's author has verified. If you're searching for this, look for terms like "LLM debate training," "multi-agent RLHF," "constitutional debate," and verify the citations carefully before relying on them.

The connection to the rest of this lecture: if you do attempt RL training of multi-agent LLM systems, the failure modes covered above all apply. Non-stationarity from co-trained partners, credit assignment in cooperative tasks, equilibrium selection in general-sum settings, exploitability concerns in adversarial settings, none of these go away because the agents happen to be language models. The CTDE pattern (centralized critic at training, decentralized policies at inference) maps cleanly onto multi-LLM training pipelines.

---

## What to use when

A rough decision sketch:

```
What's the reward structure?
├── Single shared reward (cooperative)
│     ├── Few agents (2–4) → MAPPO with shared parameters
│     ├── Many agents but value-decomposable → QMIX
│     └── Value-decomposable doesn't apply → MAPPO with centralized critic
│
├── Strict zero-sum (two-player competitive)
│     ├── Mostly transitive (chess, Go) → AlphaZero-style self-play + MCTS
│     ├── Non-transitive (poker, RPS-like) → NFSP or PSRO variants
│     └── Very large state space (StarCraft, Stratego) → league training, R-NaD
│
└── General-sum
      ├── Cooperation dilemmas (iterated PD, stag hunt) → LOLA, opponent-aware
      ├── Mixed-motive environments → independent learners + reward shaping
      └── No single right answer; report exploitability bounds where possible
```

A few rules of thumb that hold across the categories:

- **Always evaluate against a held-out opponent population.** Self-play win rate against the current policy is symmetric and uninformative. Pull out historical snapshots, or train a separate exploiter, or use OpenSpiel's exact exploitability for small games.
- **Watch for non-stationarity in your training curves.** Reward against fixed opponents going up while reward against the current self stays at 50% is healthy. Reward against fixed opponents oscillating or declining is the cycling failure mode.
- **Parameter sharing across symmetric agents helps.** In MAPPO and MADDPG, sharing the policy network across homogeneous agents reduces variance and accelerates learning. Don't share if the agents have qualitatively different roles.
- **Exploration is harder.** In single-agent RL you explore by perturbing your policy. In multi-agent, exploration includes "what if the other agents try a different strategy?" which you don't directly control. Population methods address this; single-policy methods often don't.

---

## A worked self-play skeleton

Below is a more complete skeleton, slightly above pseudocode, of a self-play training loop with periodic evaluation against historical snapshots. It assumes a two-player zero-sum game and a PPO-style policy update. It's not a runnable file (the env, model, and PPO update are stubbed) but the structure is concrete enough to translate.

```python
import copy
import random
from collections import deque
from dataclasses import dataclass

import torch


@dataclass
class GameTrajectory:
    states: list
    actions: list
    log_probs: list
    rewards: list  # per-step rewards, often zero except at game end
    winner: int    # +1 if player 1 won, -1 if player 2, 0 if draw


def play_game(env, policy_a, policy_b, max_turns=500):
    """
    Play one game of a two-player zero-sum game.
    Returns trajectories from each player's perspective.
    """
    state = env.reset()
    traj_a = GameTrajectory([], [], [], [], winner=0)
    traj_b = GameTrajectory([], [], [], [], winner=0)

    current = "a"  # alternating turns; could also be determined by env

    for _ in range(max_turns):
        policy = policy_a if current == "a" else policy_b
        traj   = traj_a   if current == "a" else traj_b

        action, log_prob = policy.act(state)
        next_state, reward, done, info = env.step(action)

        traj.states.append(state)
        traj.actions.append(action)
        traj.log_probs.append(log_prob)
        traj.rewards.append(reward)

        if done:
            # Final reward is typically +1 / -1 / 0 from player 1's perspective.
            winner = info.get("winner", 0)
            traj_a.winner = winner
            traj_b.winner = -winner
            # Backfill the terminal reward to each player's last step.
            if traj_a.rewards: traj_a.rewards[-1] = float(winner)
            if traj_b.rewards: traj_b.rewards[-1] = float(-winner)
            break

        state = next_state
        current = "b" if current == "a" else "a"

    return traj_a, traj_b


def compute_returns(traj, gamma=1.0):
    """Discounted returns; for board games gamma is often 1.0."""
    returns = []
    G = 0.0
    for r in reversed(traj.rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def ppo_update(policy, batch, clip_eps=0.2, lr=3e-4):
    """
    Stub for PPO update. See Lecture 06 for the full algorithm.
    `batch` is a list of (state, action, old_log_prob, advantage) tuples.
    """
    # ... implementation per Lecture 06 ...
    pass


def evaluate(policy, opponents, env, n_games=100):
    """
    Evaluate `policy` against each opponent in `opponents`.
    Returns a dict of opponent_id -> win rate.
    """
    results = {}
    for opp_id, opp in enumerate(opponents):
        wins = 0
        for _ in range(n_games):
            # Alternate which side `policy` plays to avoid first-mover bias.
            if random.random() < 0.5:
                ta, _ = play_game(env, policy, opp)
                wins += int(ta.winner == 1)
            else:
                _, tb = play_game(env, opp, policy)
                wins += int(tb.winner == 1)
        results[opp_id] = wins / n_games
    return results


def self_play_train(
    env,
    policy,
    n_iters=10_000,
    games_per_iter=64,
    snapshot_every=100,
    eval_every=500,
    history_size=20,
    sample_historical_prob=0.2,
):
    """
    Self-play training with a small historical snapshot pool.
    With probability `sample_historical_prob`, the opponent is drawn
    from the snapshot pool instead of being the current policy.
    This is the simplest defense against cycling.
    """
    history = deque(maxlen=history_size)
    history.append(copy.deepcopy(policy))

    for it in range(n_iters):
        # 1. Generate self-play data.
        trajectories = []
        for _ in range(games_per_iter):
            if random.random() < sample_historical_prob and len(history) > 1:
                opponent = random.choice(list(history))
            else:
                opponent = policy
            ta, tb = play_game(env, policy, opponent)
            # Only learn from `policy`'s side of the game.
            trajectories.append(ta)

        # 2. Compute returns (and ideally advantages via a value network).
        batch = []
        for traj in trajectories:
            returns = compute_returns(traj)
            for s, a, lp, R in zip(traj.states, traj.actions, traj.log_probs, returns):
                # In real code: advantage = R - V(s); here we use return as advantage.
                batch.append((s, a, lp, R))

        # 3. Update the policy.
        ppo_update(policy, batch)

        # 4. Snapshot.
        if it % snapshot_every == 0:
            history.append(copy.deepcopy(policy))

        # 5. Evaluate against the snapshot pool.
        if it % eval_every == 0:
            win_rates = evaluate(policy, list(history), env)
            print(f"iter={it} | win_rates_vs_history={win_rates}")
```

A few notes on what this does and doesn't capture:

- **Playing against historical snapshots** is the simplest cycling defense. With probability 20%, the opponent is a randomly-chosen past version. This is a poor man's fictitious play. NFSP and PSRO are more principled but more involved.
- **The reward is sparse**: usually zero for every step except the last. PPO with a value network handles this; pure REINFORCE will be very high-variance unless games are short.
- **No MCTS**. AlphaZero-style training would replace `policy.act` with `mcts(policy, state)` to generate higher-quality actions, then train the policy to imitate the MCTS output. The RL update becomes supervised distillation onto an improved target. That's a different recipe; it's worth building this simpler version first.
- **No symmetry exploitation**. In games where the position can be reflected/rotated without changing the value (Go, chess to a degree), you can augment the training data with symmetric copies. This effectively gives 8x data on a Go board. AlphaZero does this; the skeleton above doesn't.
- **No league**. AlphaStar's league training has multiple agent types with explicit roles (main, exploiter, league-exploiter). Implementing that requires more bookkeeping; the historical-snapshot pool above is the simplest stand-in.

To actually run this on a real game, plug in OpenSpiel for `env` (it gives you Tic-Tac-Toe, Connect Four, Kuhn poker, Leduc poker, etc.), implement a small policy and value network, and run for a few hundred thousand iterations. For Tic-Tac-Toe you'll converge to perfect play within minutes on a CPU; for Connect Four it takes longer but is still feasible on a single GPU.

---

## Common failure modes

A few patterns that show up across MARL settings, worth knowing in advance.

**Cycling in self-play**. Policies oscillate between strategies without convergence. The training reward against the current self stays around 50% (it must, by symmetry), but evaluation against historical opponents shows declining or oscillating win rates. Fix: introduce historical opponents, switch to NFSP/PSRO, or add regularization toward the average policy.

**Reward hacking in shared rewards**. In a cooperative setup with a shared reward, one agent can learn a strategy that makes the team reward go up but does so by exploiting another agent's behavior in a brittle way. When the other agent's policy changes, the strategy collapses. Symptom: training reward is high but evaluation in slightly perturbed conditions degrades sharply.

**Lazy agent problem**. With shared rewards and value decomposition like VDN, one agent can become "lazy": it doesn't contribute much, but as long as other agents do enough work, the team reward stays positive and the lazy agent's `Q_i` doesn't strongly point in any direction. QMIX's monotonic mixer doesn't fully solve this; explicit exploration bonuses (entropy on per-agent policies) help.

**Equilibrium selection in cooperative tasks**. Two agents in a coordination task can converge to a mutually-suboptimal equilibrium and stay there. The optimization is locally fine (neither agent can improve unilaterally), but the joint policy is bad. This is harder to detect than a pure failure mode because nothing visibly breaks; you need to compare against an alternative equilibrium to know.

**Compute non-monotonicity**. Adding more compute to a self-play training run does not always produce a monotonically stronger policy. AlphaStar reports cases where extending training caused regression on certain matchups before recovery. This is the non-transitive cycling at training-trajectory scale: more compute lets the policy explore more strategies, some of which are non-transitive responses to current strategies. Periodic evaluation against a frozen reference is the diagnostic.

**Exploitability climbing while win rate is stable**. The most insidious failure. Win rate against the symmetric self stays at 50%, training metrics look healthy, but if you train an exploiter against the current policy it can win 80%+. The policy has converged to a narrow style that an off-distribution opponent can crush. This is why exploitability (not win rate against self) is the right metric for two-player zero-sum work.

---

## Connections back to the rest of the curriculum

- **From [Lecture 02](./02-policy-gradients.md)**: policy gradient is the underlying optimization for MAPPO, MADDPG, COMA, and most modern MARL. The objective is the same (maximize expected return); the change is what "expected return" means when the environment includes other learners.
- **From [Lecture 04](./04-actor-critic.md)**: the centralized critic in CTDE is a direct application of actor-critic: the critic is the value function, the actor is the per-agent policy. The change is that the critic conditions on the global state and joint action.
- **From [Lecture 06](./06-ppo.md)**: MAPPO is literally PPO applied in a CTDE setting with parameter sharing. The clipping, the GAE, the value normalization, all unchanged. This is part of why MAPPO works as well as it does: it inherits the well-tuned PPO machinery.
- **From [Lecture 07](./07-off-policy-rl.md)**: MADDPG extends DDPG to multi-agent. The replay buffer, the target networks, the exploration noise, all standard DDPG. The change is the centralized critic.
- **From [Lecture 15](./15-rl-verifiable-rewards.md)**: GRPO's group-relative advantages have a structural similarity to self-play populations. In both cases, you score multiple samples (completions / opponents) and use the relative scoring as the gradient signal. The mechanisms differ but the variance-reduction logic is similar.
- **Forward to [Lecture 16](./16-agentic-rl.md)**: agentic LLM systems with multiple coordinating agents are where MARL methods plausibly meet the modern LLM stack. Most current multi-agent LLM work is inference-time only, but if you do attempt RL training of multi-agent LLMs, the methods in this lecture are the relevant prior art.

---

## Exercises

There is no exercise directory for this lecture yet. Reasonable things to build, in increasing order of effort:

- **Independent Q-learning on rock-paper-scissors**. Two tabular Q-learners against each other. Plot the action distribution over training. You should see cycling, not convergence to (1/3, 1/3, 1/3). Then add an opponent-strategy estimator to one player and see whether it stabilizes.
- **MAPPO on a small SMAC map**. The SMAC code is open-source (Samvelyan et al. arXiv:1902.04043). The 3m map is the easiest. Use the reference MAPPO implementation if available; the goal is to see a CTDE setup work end to end.
- **Self-play on Tic-Tac-Toe with the skeleton above**. Plug in OpenSpiel's Tic-Tac-Toe environment. You should converge to optimal play (always-draw against optimal opponent) within a short training run. Then add an MCTS step on top of the policy and check if learning is faster.
- **NFSP on Kuhn poker**. Kuhn poker has a known Nash equilibrium and a small enough state space to compute exact exploitability. Implement NFSP and verify that exploitability decreases over training. Compare to naive self-play (which won't converge).
- **PSRO on Leduc poker**. Leduc is one step up from Kuhn. Use OpenSpiel's PSRO implementation as a reference; tune the meta-strategy solver and the number of best-response training iterations.

For all of these: report exploitability where the game is small enough to compute it; report win rate against a held-out opponent population otherwise. Don't trust win-rate-against-self.

---

## References

All arXiv IDs verified by resolving against arxiv.org during writing.

**Cooperative MARL with CTDE**

- Lowe, Wu, Tamar et al. 2017. "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments." NeurIPS 2017. arXiv:1706.02275: MADDPG. The canonical CTDE algorithm; centralized critic, decentralized deterministic policies.
- Sunehag, Lever, Gruslys et al. 2017. "Value-Decomposition Networks For Cooperative Multi-Agent Learning." arXiv:1706.05296: VDN. Additive decomposition of the joint Q-function.
- Rashid, Samvelyan, de Witt et al. 2018. "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning." ICML 2018. arXiv:1803.11485: QMIX. Generalizes VDN with a state-conditioned monotonic mixer; the dominant value-decomposition method for cooperative MARL.
- Foerster, Farquhar, Afouras et al. 2017. "Counterfactual Multi-Agent Policy Gradients." AAAI 2018. arXiv:1705.08926: COMA. Counterfactual baseline for credit assignment in cooperative settings.
- Yu, Velu, Vinitsky et al. 2021. "The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games." NeurIPS 2022. arXiv:2103.01955: MAPPO. Vanilla PPO in CTDE matches or beats specialized MARL algorithms.
- Samvelyan, Rashid, de Witt et al. 2019. "The StarCraft Multi-Agent Challenge." arXiv:1902.04043: SMAC. Standard cooperative MARL benchmark.

**Self-play and competitive game-playing**

- Silver, Hubert, Schrittwieser et al. 2017. "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm." arXiv:1712.01815: AlphaZero. Self-play + MCTS, generalized to chess and shogi from the AlphaGo Zero recipe.
- Heinrich and Silver. 2016. "Deep Reinforcement Learning from Self-Play in Imperfect-Information Games." arXiv:1603.01121: Neural Fictitious Self-Play (NFSP). The deep-RL extension of fictitious play for non-transitive games like poker.
- Berner, Brockman, Chan et al. 2019. "Dota 2 with Large Scale Deep Reinforcement Learning." OpenAI. arXiv:1912.06680: OpenAI Five. Large-scale PPO with self-play on a 5v5 team game; the engineering paper rather than an algorithmic novelty paper.
- Vinyals, Babuschkin, Czarnecki et al. 2019. "Grandmaster level in StarCraft II using multi-agent reinforcement learning." Nature 575:350–354: AlphaStar. League training (a form of PSRO) for StarCraft II.
- Perolat, de Vylder, Hennes et al. 2022. "Mastering the Game of Stratego with Model-Free Multiagent Reinforcement Learning." Science 378:990–996; arXiv:2206.15378: DeepNash. Regularized Nash dynamics (R-NaD) on Stratego, a game with too many states for MCTS-style methods.

**Population-based and game-theoretic methods**

- Lanctot, Zambaldi, Gruslys et al. 2017. "A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning." NeurIPS 2017. arXiv:1711.00832: PSRO. Unifies independent learning, fictitious play, and double oracle.
- Lanctot, Lockhart, Lespiau et al. 2019. "OpenSpiel: A Framework for Reinforcement Learning in Games." arXiv:1908.09453: Software library with reference implementations of CFR, NFSP, PSRO, and exact exploitability solvers for small games.

**General-sum and opponent-aware learning**

- Foerster, Chen, Al-Shedivat et al. 2017. "Learning with Opponent-Learning Awareness." arXiv:1709.04326: LOLA. Modify the policy gradient to account for opponent learning dynamics; changes equilibrium selection in social dilemmas.
- Baker, Kanitscheider, Markov et al. 2019. "Emergent Tool Use From Multi-Agent Autocurricula." arXiv:1909.07528: Hide and Seek. Multi-agent autocurriculum in a physics environment; six emergent strategy regimes.

**Foundations**

- Sutton and Barto. 2018. "Reinforcement Learning: An Introduction" (2nd ed). MIT Press: Chapter 17 mentions multi-agent and game-theoretic settings briefly. The book's bias is toward single-agent, but the foundations (TD, policy gradient, function approximation) all carry over.
- Shapley. 1953. "Stochastic Games." PNAS 39(10):1095–1100: Original definition of the stochastic game model.
- Littman. 1994. "Markov games as a framework for multi-agent reinforcement learning." ICML 1994: Connects stochastic games to RL; introduces minimax-Q for two-player zero-sum.

**Multi-agent LLM systems (inference-time)**

- Du, Li, Torralba et al. 2023. "Improving Factuality and Reasoning in Language Models through Multiagent Debate." arXiv:2305.14325: Inference-time multi-LLM debate. No model training; included as a reference for the connection to current LLM work.

**Not cited but worth knowing about**

- TD-Gammon (Tesauro 1995, "Temporal difference learning and TD-Gammon," CACM 38(3):58–68). The foundational self-play work, predates the deep RL era.
- AlphaGo (Silver et al. 2016, Nature 529:484–489). The chess/shogi/Stratego papers above are the algorithmically cleaner descendants.
- Pluribus (Brown and Sandholm 2019, Science 365:885–890). Six-player no-limit Texas hold'em via Monte Carlo CFR with abstraction. CFR is a different family of algorithms (regret-matching rather than gradient-based) and gets its own treatment in the imperfect-information game literature.

---

## Next lecture

There's no lecture 22 yet. Plausible directions: imperfect-information game-solving (CFR, regret minimization), exploration in MARL (intrinsic motivation in multi-agent settings), or a deeper treatment of opponent modeling and meta-learning across opponents.
