# Hints: exercise 03

Read one at a time. Try after each before reading the next.

---

**Hint 1: `epsilon_greedy`.** `if rng.random() < epsilon: return int(rng.integers(n_actions))` else `return int(np.argmax(q_row))`. The `int(...)` matters: `np.argmax` gives a numpy int, and the test checks for a plain Python `int`.

---

**Hint 2: `q_update`.** Compute the target first: `target = reward if done else reward + gamma * np.max(q_table[next_state])`. Then `td_error = target - q_table[state, action]`, then `q_table[state, action] += alpha * td_error`, then `return float(td_error)`. The `done` branch matters: a terminal transition has no future, so the target is just the reward; if you bootstrap off `q_table[next_state]` for a terminal state you'll learn garbage.

---

**Hint 3: `train`, the update step.** Right after `env.step(...)`, before `state = next_state`:

```python
q_update(q_table, state, action, reward, next_state, done, alpha, gamma)
```

That's it. One update per transition, on the transition you just observed. (`q_update` mutates `q_table` in place, so you don't need to reassign anything.)

---

**Hint 4: if it runs but the greedy policy doesn't reach the goal.** Two things to check:

- Are you bootstrapping off the **max** over next-state actions? `np.max(q_table[next_state])`, not `q_table[next_state, action]` (that would be SARSA-ish and isn't what's asked).
- Is `done` being handled? On FrozenLake, falling in a hole *also* sets `terminated=True` with reward 0, so the target for that transition is 0, which is correct (that's the agent learning "this action here is bad"). Make sure you're not adding `gamma * max(q_table[next_state])` when `done`.

If both are fine and it still doesn't work, print `q_table` after training and look at the row for the start state (state 0): the argmax there should point along a path that avoids the holes at states 5, 7, 11, 12. (If `q_table` is *all zeros* after training, your `q_update` isn't writing into it: check it mutates `q_table` in place rather than a copy. Note `train` deliberately starts the table at all-ones, not zeros: optimistic init, so the agent explores instead of walking into a wall.)

---

If you're past hint 4 and still stuck, read [`solution/q_learning.py`](./solution/q_learning.py). Then re-derive the update from the Bellman equation without looking.
