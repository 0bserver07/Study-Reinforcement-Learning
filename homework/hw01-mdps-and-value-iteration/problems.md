<!-- status: unreviewed | last-reviewed: never -->

# HW1 problems

Solutions are in [`solutions.md`](./solutions.md) — try every problem before looking.

---

## Problem 1 — write down an MDP (warm-up, ~10 min)

Frame each of the following as an MDP. For each, state explicitly what `S`, `A`, `P(s' | s, a)`, `R(s, a, s')`, and `γ` are. Note where a Markov state isn't obvious — what would you have to add to the state for the Markov property to hold?

(a) A robot navigating a 5×5 gridworld to reach a goal cell. Each step costs 1; reaching the goal pays 10; the goal is terminal. The actions are {up, right, down, left}; movement off the grid is clamped.

(b) The same gridworld, but slippery: 80% of the time the action does what you asked, 20% it slips one cell perpendicular (10% each side).

(c) Optimizing a chain-of-thought response from an LLM: states are partial generations, actions are next tokens, the reward is +1 if the final answer (parsed out of the generation) is correct as judged by a verifier, and 0 otherwise.

---

## Problem 2 — derive the Bellman expectation equation (theory, ~20 min)

Starting from the definition

```
V^π(s) = E_π[G_t | S_t = s]
       = E_π[Σ_{k=0}^∞ γ^k R_{t+k+1} | S_t = s]
```

show that

```
V^π(s) = Σ_a π(a | s) Σ_{s'} P(s' | s, a) [R(s, a, s') + γ V^π(s')]
```

**Hint.** Split `G_t = R_{t+1} + γ G_{t+1}`. Take the conditional expectation, condition on `(A_t, S_{t+1})`, and use the Markov property. The conditional expectation of `G_{t+1}` given `S_{t+1} = s'` (and `S_t, A_t`, by Markov) is just `V^π(s')`.

---

## Problem 3 — value iteration converges (theory, ~30 min)

Define the Bellman optimality operator `T : R^|S| → R^|S|` by

```
(TV)(s) = max_a Σ_{s'} P(s' | s, a) [R(s, a, s') + γ V(s')]
```

Use the max-norm `||V||_∞ = max_s |V(s)|`.

(a) Prove that `T` is a γ-contraction in `||·||_∞`:

```
||TV - TW||_∞ ≤ γ ||V - W||_∞
```

You will need the inequality `|max_a f(a) - max_a g(a)| ≤ max_a |f(a) - g(a)|` (prove it in one line if you haven't seen it).

(b) Conclude (one sentence each):
- (i) `T` has a unique fixed point `V*`.
- (ii) Value iteration `V_{k+1} = T V_k` converges to `V*` from any starting `V_0`.
- (iii) The convergence rate is geometric: `||V_k - V*||_∞ ≤ γ^k ||V_0 - V*||_∞`.

---

## Problem 4 — V* on the deterministic gridworld (numeric, ~15 min)

The 5×5 deterministic gridworld from problem 1(a): goal at row 0, column 4; reward −1 per step; +10 for the transition that lands on the goal; γ = 0.99; the goal is terminal so `V*(goal) = 0`.

(a) Compute `V*((0, 3))` exactly (the cell directly left of the goal, one step away).

(b) Compute `V*((0, 2))` exactly.

(c) Without computing every cell, write a closed-form expression for `V*(s)` as a function of the Manhattan distance `d` from `s` to the goal cell. (`d = |row| + |4 − col|`.)

---

## Problem 5 — implement value iteration (coding, ~45 min)

Do [`exercises/01-mdps/`](../../exercises/01-mdps/):

```bash
pip install -r exercises/requirements.txt
pytest exercises/01-mdps/
```

The tests cover `value_iteration` on tiny MDPs (the right fixed points), `greedy_policy` extraction, and an integration test that the greedy policy from `V*` reaches the goal from every cell of the 5×5 gridworld.

Submit: your filled-in `starter.py`. Look at `solution/value_iteration.py` only after a real attempt.

**Sanity check against problem 4.** When your value iteration runs on `GridWorldMDP(size=5)`, the printed `V*` matrix should match what you derived in problem 4(c) for every non-terminal cell. If it doesn't, one of the two is wrong.

---

## Problem 6 — γ matters (conceptual, ~15 min)

In the deterministic gridworld of problem 4:

(a) What changes if you set γ = 1 (no discounting)? Does value iteration still converge? Why or why not? Does `V*` still exist?

(b) Sketch (in words or numbers) how `V*` would look with γ = 0.5. Which cells still have meaningfully positive value? Which are essentially zero?

(c) Now imagine the grid has no terminal goal — instead, the agent receives reward +1 every time it stands on the goal cell, and continues forever. With γ = 0.99, what is `V*(goal)`? With γ = 1?

---

## Problem 7 — policy iteration vs. value iteration (reading, ~15 min)

Read Sutton & Barto sections 4.3 and 4.4. In your own words, ~3–5 sentences each:

(a) Why does policy iteration sometimes converge faster (in number of policy improvements) than value iteration?

(b) When would you prefer value iteration anyway?

(c) What is "generalized policy iteration"? Why is it the right way to think about most modern RL algorithms?
