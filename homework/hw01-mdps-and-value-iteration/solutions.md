<!-- status: unreviewed | last-reviewed: never -->

# HW1 solutions

> Worked solutions. Try the problems in [`problems.md`](./problems.md) first.

---

## Problem 1: solutions

(a) **Deterministic gridworld.**
- `S` = {(i, j) : 0 ≤ i, j ≤ 4} ∪ {terminal}. Or more compactly: the 25 cells with the goal cell as an absorbing state.
- `A` = {up, right, down, left}.
- `P(s' | s, a)` = 1 for the unique cell you land in (clamped at walls); 0 elsewhere. From the goal: `P(goal | goal, a) = 1` for all `a` (absorbing).
- `R(s, a, s')` = +10 if `s' = goal` and `s ≠ goal`; 0 if `s = s' = goal`; −1 otherwise.
- `γ` ∈ (0, 1): pick anything; the lecture and exercise use 0.99.
- The Markov property holds: the next cell only depends on `(s, a)`, not on history.

(b) **Slippery gridworld.** Same `S`, `A`, `R`, `γ`. The transitions become stochastic:
- `P(s_intended | s, a) = 0.8`
- `P(s_left_of_intended | s, a) = 0.1` (clamped if it'd be off-grid)
- `P(s_right_of_intended | s, a) = 0.1`

The Markov property still holds because slipping is independent of history. (If slipping had memory: "if you slipped last step you slip again": you'd need to add the last slip outcome to the state to keep it Markov.)

(c) **Chain-of-thought MDP.**
- `S` = the set of partial token sequences (the prompt plus whatever's been generated so far).
- `A` = the vocabulary (each action is a next token).
- `P(s' | s, a) = 1` if `s' = s ⧺ a` (concatenation), 0 otherwise. The transition is deterministic: generation just appends a token.
- `R(s, a, s')` = +1 if `s'` ends the generation (e.g. `<eos>`) and the verifier judges the parsed-out answer correct; 0 otherwise.
- `γ` close to 1 (you don't want to discount within a single response much).

The Markov property is automatic because the "state" *is* the full conversation history; nothing past is hidden.

> What this teaches: an "MDP" is just five symbols. Picking what to put in each one is the modeling work. The Markov property is about whether your state captures everything that matters, if it doesn't, your policy can't be deterministic on the state alone.

---

## Problem 2: derivation

Start from `G_t = Σ_{k=0}^∞ γ^k R_{t+k+1}`. Split off the first term:

```
G_t = R_{t+1} + γ Σ_{k=0}^∞ γ^k R_{t+k+2}
    = R_{t+1} + γ G_{t+1}
```

Take the conditional expectation under π given `S_t = s`:

```
V^π(s) = E_π[G_t | S_t = s]
       = E_π[R_{t+1} + γ G_{t+1} | S_t = s]
       = E_π[R_{t+1} | S_t = s]  +  γ E_π[G_{t+1} | S_t = s]
```

Now condition the inner expectations on `A_t` and `S_{t+1}` (tower property):

```
E_π[R_{t+1} | S_t = s]
  = Σ_a π(a | s) Σ_{s'} P(s' | s, a) R(s, a, s')

E_π[G_{t+1} | S_t = s]
  = Σ_a π(a | s) Σ_{s'} P(s' | s, a) E_π[G_{t+1} | S_{t+1} = s']
```

The last conditional expectation: `E_π[G_{t+1} | S_{t+1} = s'] = V^π(s')`, by the Markov property, the future depends only on `S_{t+1}`.

Putting it together:

```
V^π(s) = Σ_a π(a | s) Σ_{s'} P(s' | s, a) [R(s, a, s') + γ V^π(s')]   ∎
```

> What this teaches: the Bellman equation isn't a definition, it's a *consequence* of the definition of value plus the Markov property. The recursive structure is forced.

---

## Problem 3: contraction proof

(a) For any state `s`,

```
|(TV)(s) - (TW)(s)|
  = |max_a Σ_{s'} P(s' | s, a) [R(s, a, s') + γ V(s')]
   − max_a Σ_{s'} P(s' | s, a) [R(s, a, s') + γ W(s')]|
  ≤ max_a |Σ_{s'} P(s' | s, a) [γ V(s') − γ W(s')]|              (max inequality)
  ≤ max_a Σ_{s'} P(s' | s, a) γ |V(s') − W(s')|                    (triangle)
  ≤ γ Σ_{s'} P(s' | s, a) ||V − W||_∞                              (bound each term)
  = γ ||V − W||_∞                                                 (probabilities sum to 1)
```

The bound is uniform in `s`, so `||TV − TW||_∞ ≤ γ ||V − W||_∞`. ∎

The lemma `|max_a f(a) − max_a g(a)| ≤ max_a |f(a) − g(a)|`: WLOG `max f ≥ max g`. Pick `a* = argmax f`. Then `max f − max g ≤ f(a*) − g(a*) ≤ |f(a*) − g(a*)| ≤ max_a |f − g|`.

(b)
- (i) `T` is a contraction on the complete metric space `(R^|S|, ||·||_∞)`. By the Banach fixed-point theorem, it has a unique fixed point. Call it `V*`. (And `V*` satisfies the Bellman optimality equation by construction.)
- (ii) The same theorem says `V_k = T^k V_0` converges to `V*` for any `V_0`.
- (iii) `||V_k − V*||_∞ = ||T V_{k−1} − T V*||_∞ ≤ γ ||V_{k−1} − V*||_∞`, so by induction `||V_k − V*||_∞ ≤ γ^k ||V_0 − V*||_∞`. Geometric in γ.

> What this teaches: "value iteration converges" isn't magic, it's the Banach fixed-point theorem applied to a γ-contraction. The discount factor isn't just for finite returns; it's what makes the operator a contraction at all. Set γ = 1 and you lose the contraction (and may lose convergence).

---

## Problem 4: V* on the gridworld

Under the optimal policy from a non-terminal cell: take a shortest path to the goal. Let `d` be the Manhattan distance (positive integer; `d = 0` means we're at the goal, terminal).

The optimal trajectory of length `d`: take `d − 1` steps of reward `−1` (since none of the first `d − 1` transitions lands on the goal), then one step of reward `+10` (the step *into* the goal). After that, the goal is terminal, so the return ends.

Discounted return:

```
V*(s) = (-1) + γ(-1) + γ²(-1) + … + γ^(d-2)(-1) + γ^(d-1)(+10)
      = -(1 - γ^(d-1)) / (1 - γ) + 10 γ^(d-1)
```

For γ = 0.99:

(a) `d = 1` for `(0, 3)`: `V* = 10 · 0.99^0 = 10.0` exactly.

(b) `d = 2` for `(0, 2)`: `V* = -1 + 0.99 · 10 = -1 + 9.9 = 8.9` exactly.

(c) Closed form: `V*(s) = -(1 - 0.99^(d-1)) / (1 - 0.99) + 10 · 0.99^(d-1)` where `d = |row| + |4 − col|`. For `d = 0` (the goal itself), `V* = 0` (terminal).

Spot-check the recursion: `V*_d = -1 + 0.99 · V*_{d-1}`, with `V*_1 = 10`. So `V*_2 = -1 + 9.9 = 8.9`, `V*_3 = -1 + 0.99·8.9 ≈ 7.811`, `V*_4 ≈ 6.733`, `V*_5 ≈ 5.666`, `V*_6 ≈ 4.609`, `V*_7 ≈ 3.563`, `V*_8 ≈ 2.527`. The corner `(4, 0)` has `d = 8` so `V* ≈ 2.527`.

> What this teaches: when the rewards are simple, you can derive closed-form values without running anything, and the closed form gives you a free correctness check on your value-iteration code.

---

## Problem 5: coding (sanity check)

When you implement value iteration correctly, the printed `V*` matrix on `GridWorldMDP(size=5)` should be:

```
[[ 6.73  7.81  8.90 10.00  0.00]
 [ 5.67  6.73  7.81  8.90 10.00]
 [ 4.61  5.67  6.73  7.81  8.90]
 [ 3.56  4.61  5.67  6.73  7.81]
 [ 2.53  3.56  4.61  5.67  6.73]]
```

Every entry matches `V*_d` from problem 4(c) for `d = i + (4 − j)`, except the goal cell `[0, 4]` which prints `0.00` because it's terminal and value iteration skips it.

The greedy policy points every non-goal cell toward the goal: `→ → → → G` along the top row, `↑` everywhere else (since ties between "up" and "right" both leading to a `d − 1` cell are broken by `argmax`'s lowest-index rule, which picks `up` = action 0).

> What this teaches: the test you actually want is "do the values match the closed form I derived", that catches bugs the unit tests won't.

---

## Problem 6: γ matters

(a) **γ = 1.** In the *deterministic* gridworld with a terminal goal, value iteration still converges, the goal terminates every trajectory in finite steps, returns are bounded, and the operator with γ = 1 is *not* a strict contraction but still converges on this specific MDP because it's an *episodic* problem with bounded length. (In general, γ = 1 plus a non-terminating MDP can give infinite returns and the operator isn't a contraction at all, value iteration may not converge.) `V*` exists here because every state reaches the goal in ≤ 8 steps, so values are bounded by `−7 + 10 = 3` from below and `+10` from above. With γ = 1: `V*_d = −(d − 1) + 10 = 11 − d`. So `V*_1 = 10`, `V*_8 = 3`. Linear, not geometric.

(b) **γ = 0.5.** `V*_d = -(1 - 0.5^{d-1}) / 0.5 + 10 · 0.5^{d-1} = -2(1 - 0.5^{d-1}) + 10 · 0.5^{d-1} = -2 + 12 · 0.5^{d-1}`. So `V*_1 = -2 + 12 = 10`, `V*_2 = -2 + 6 = 4`, `V*_3 = -2 + 3 = 1`, `V*_4 = -2 + 1.5 = -0.5`, `V*_5 = -2 + 0.75 = -1.25`. Far cells have *negative* value: the discounted +10 is no longer worth the cost of getting there. The greedy policy still tries to reach the goal because `V` decreases sharply with distance, but values past `d = 3` are essentially "step costs accumulated."

(c) **No terminal goal, +1 per visit, γ = 0.99.** Standing on the goal forever yields return `Σ_{k=0}^∞ 0.99^k · 1 = 1 / (1 − 0.99) = 100`. So `V*(goal) = 100`, and other cells have values approaching that minus the cost of getting there. With γ = 1 the geometric series diverges: `V*(goal) = ∞`. This is why average-reward formulations exist for non-terminating MDPs without discounting.

> What this teaches: γ isn't a tuning knob for "how much do I care about the future", it's mathematically what keeps value iteration well-defined. In episodic problems with bounded length you can get away with γ = 1; otherwise you need γ < 1 (or to switch to average-reward).

---

## Problem 7: PI vs. VI (reading)

(a) Policy iteration improves the policy in big jumps: each round, `π_{k+1}` is greedy w.r.t. `V^{π_k}` (which itself was solved exactly). For small finite MDPs, this often converges in just a few rounds (~`|S|` in the worst case, but commonly far fewer). Value iteration improves the *value function* one Bellman backup at a time, you only get the optimal policy at the end (after VI converges), and convergence is geometric in γ rather than polynomial in `|S|`.

(b) Value iteration is preferred when: (i) `|S|` is large and exact policy evaluation would be expensive (it's an `|S|`-by-`|S|` linear solve, or many sweeps); (ii) you only need an approximately-optimal policy and can stop VI early; (iii) you're going to throw a function approximator at the value function anyway. In modern deep RL, "VI" essentially means "do Bellman backups indefinitely", Q-learning is asynchronous-stochastic VI on `Q*`.

(c) Generalized policy iteration is the observation that PI and VI are two extremes of a spectrum: PI does *full* policy evaluation between improvements; VI does *one* Bellman backup before improving. *Generalized* PI is anything in between, partial evaluation (a few sweeps), then improvement, repeat. It's the right framing because most modern RL algorithms (actor-critic, PPO, ...) are GPI: the critic does some evaluation, the actor improves, they alternate. The dichotomy "PI vs VI" is a special-case mindset; "GPI" is the general one.

> What this teaches: PI and VI aren't separate algorithms, they're endpoints of a continuum. The continuum (do *some* evaluation, then improve, repeat) is what scales to deep RL.
