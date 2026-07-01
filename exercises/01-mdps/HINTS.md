# Hints: exercise 01

Read one at a time. Try after each before reading the next.

---

**Hint 1: the Q-value helper.** Both functions need "the value of taking action `a` in state `s`, then acting optimally": `q(s, a) = sum(p * (r + gamma * V[sp]) for (p, sp, r) in mdp.transitions(s, a))`. Write that once (a small helper, or inline) and reuse it in both `value_iteration` and `greedy_policy`.

---

**Hint 2: `value_iteration`.** `V = np.zeros(mdp.n_states)`. Then loop forever: track `delta = 0.0`; for each state `s`, if `mdp.is_terminal(s)` skip it; else `v_old = V[s]`, `V[s] = max(q(s, a) for a in range(mdp.n_actions))`, `delta = max(delta, abs(v_old - V[s]))`. After the sweep, `if delta < theta: return V`. Updating `V` in place during the sweep (Gauss-Seidel) is fine; it just converges a bit faster.

---

**Hint 3: why skip terminal states.** A terminal state has no future, so `V[terminal] = 0` by definition. If you run the backup on it, `transitions` returns a self-loop `[(1.0, s, 0.0)]`, so `V[s] = max_a (0 + gamma * V[s])`, which forces `V[s] = 0` here, but in a different MDP a self-loop could blow up or stall. Cleaner to leave terminal states at 0 and never touch them.

---

**Hint 4: `greedy_policy`.** `policy = np.zeros(mdp.n_states, dtype=int)`; for each non-terminal `s`, `policy[s] = int(np.argmax([q(s, a) for a in range(mdp.n_actions)]))`. Leave terminal states at 0; the action there is never used. (Ties: `np.argmax` picks the lowest index; that's fine here, both tied actions make progress toward the goal.)

---

**Hint 5: if the gridworld tests fail.** Print `np.round(V.reshape(mdp.size, mdp.size), 2)`. The goal cell should be `0` (terminal), its left neighbor should be `10` (one "right" step lands on the goal for +10), and the numbers should decrease as you move down and left away from the goal. If the goal-adjacent cell isn't ≈ 10, you're probably bootstrapping the goal's value into it incorrectly. Remember the reward is on the *transition into* the goal, and `transitions` already includes that `+10` reward.

---

If you're past hint 5 and still stuck, read [`solution/value_iteration.py`](./solution/value_iteration.py). Then re-derive value iteration from the Bellman-optimality equation without looking.
