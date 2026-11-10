# Exercise 20: RND exploration on a combination lock

Goes with [Lecture 20: Exploration](../../notes/lectures/20-exploration.md).

You'll build a 16-square combination lock where the only reward is at the far end, watch ε-greedy Q-learning fail to open it, and then add a Random Network Distillation (RND) novelty bonus that opens it within a few dozen episodes. It's the lock from the figure on the site's [lecture 20 page](https://rl.yad.codes/posts/20-exploration/), which uses a count bonus where this exercise uses RND.

## The lock

Squares 0 to 15 in a row. The agent starts on square 0 and has two moves. Right goes to the next square. Left is the wrong move on every square, square 0 included: the episode ends with reward 0. Reaching square 15 pays +1 and ends the episode. An episode lasts at most 15 moves, so it needs no time limit, and `done=True` always means the episode really ended.

The agent is tabular Q-learning with α = 0.5, γ = 0.99, ε = 0.1 and Q starting at 0. When both moves have the same Q-value, it picks one at random.

## Why ε-greedy fails here

Until the agent first reaches the goal, none of its Q-values can move off 0. The wrong move's target is its reward, 0. A right move's target is 0 + γ · max Q(s'), and every Q(s') is still 0. So every greedy choice is a tie that goes either way with probability 1/2, and an ε-random move is right half the time too. Each move is a coin flip, whatever happened in earlier episodes, and opening the lock takes 15 rights in a row: probability 2^-15, once in 32,768 episodes. With N squares it's 2^-(N-1), so each extra square halves the odds.

After n episodes the chance of at least one success is 1 − (1 − 2^-15)^n: 0.15% after 50 episodes, 0.30% after 100, 0.61% after 200. The median wait is about 22,700 episodes. Over 2,000 seeds of `train_q_learning_alone`, 3, 4 and 8 runs had reached the goal by episodes 50, 100 and 200 (0.15%, 0.20%, 0.40%), and none reached it twice.

Failed episodes teach the agent nothing, since they leave every Q-value at 0. Only a +1 changes the table, and the +1 is what it can't find.

The tie-breaking matters here. `np.argmax` returns the first of several tied actions, which is action 0, the wrong move. An agent built on it goes left 95% of the time and fails for that reason alone, so `act()` has to pick among tied actions at random.

## What a bonus changes

Give the agent a small extra reward for each right move that lands on a square without ending the episode. The first time it tries right on a square, Q(s, right) turns positive, while Q(s, left) stays at exactly 0, so from then on the greedy move on that square is right. Each episode the agent walks right through the squares where it has tried right before and flips a coin at the first square where it hasn't. Every episode that gets that far moves this frontier on by one square on average. ε still sends the agent left on 5% of its moves, which ends the episode.

Measured with the reference solution over 1,000 seeds: the first +1 came by episode 10, 25, 50 and 100 in 6.5%, 65.4%, 99.6% and 100% of runs, at a median of episode 22 and never later than episode 53. After 200 episodes the greedy move was right on every square in every run. With ε = 0.1 still on, each of the 15 moves is a random left 5% of the time, so only 0.95^15 ≈ 46% of episodes reach the goal, and that caps the return. The mean return over the last 50 of the 200 episodes averaged 0.46 across the seeds, and the lowest was 0.28.

The argument above needs only one thing from the bonus: that it's positive. Its size doesn't matter, and neither does whether it shrinks or measures novelty at all. On this lock a constant 0.1 for every such move, and the count bonus 0.1/√N(s') from the site's figure, gave exactly the same runs as the RND solution, seed for seed over 1,000 seeds. So did RND without the normalization, and RND with a predictor that never trains (200 seeds each). What RND adds is a bonus that shrinks as a state gets familiar, computed by a network instead of a table of counts, so the same code works on states you can't keep a table for, such as game frames. A shrinking bonus can tell a move back to somewhere the agent has already been from a move onto a new square, and a constant bonus can't, but the lock has no move of the first kind. What the lock does show is the part every positive bonus shares: a move that was tried and didn't end the episode ends up worth more than the wrong move.

The bonus does need one rule: no bonus on a move that ends the episode. After the wrong move the agent is still standing on its square. If you score that square's novelty as the wrong move's bonus, Q(s, left) turns positive, and on the frontier square, where right is untried and still worth 0, the greedy move becomes the wrong one. Over 200 seeds, paying the bonus on moves that end the episode got the agent to the goal within 200 episodes in 16.5% of runs, against 100% for the reference solution.

## What RND does

RND replaces "I've never seen this state" (which would need an explicit visit count) with "my predictor doesn't yet output what the target outputs for this state":

1. Build two MLPs with identical architecture, mapping a one-hot state to a 16-dim feature vector.
2. Freeze the first (the *target*) at its random initialization. It never updates.
3. Train the second (the *predictor*) to match the target's output on visited states, by MSE.
4. The intrinsic reward at state `s` is `||predictor(s) - target(s)||²`, normalized by a running standard deviation of those errors.

A state the predictor has never trained on produces a large error: two different randomly-initialized networks output very different things on the same input. After many gradient steps on a state, the predictor matches the target there and the error drops toward 0. So the bonus is large on new states and decays on well-visited ones, which is what a count bonus does, without needing a table of states.

Q-learning then trains on `r_extrinsic + intrinsic_coef * r_intrinsic`, with `intrinsic_coef = 0.1`.

RND comes from Burda et al. 2018 (arXiv:1810.12894), who used it on Atari games, Montezuma's Revenge among them, with game frames as the input.

## The task

Fill in the TODOs in [`starter.py`](./starter.py). Four pieces:

1. `CombinationLock.step`: the transition. Right moves one square, and arriving on square 15 pays +1 and ends the episode. Left ends the episode with reward 0 and leaves the agent where it was.
2. `QLearningAgent.act` and `.update`: ε-greedy with ties broken at random, and the standard Q-learning update. Watch the terminal case: on `done=True` the target is `r`, not `r + gamma * max(Q[s'])`. On this lock every `done` is a real ending. A time limit would be different: a timeout cuts the episode short, but the state it stops in still has a future, so you'd bootstrap there.
3. `RNDIntrinsicReward.intrinsic_reward` and `.update`: compute the normalized prediction error, and run one Adam step on the predictor toward the (frozen) target output. The `_raw_error` helper and the running-stats update are given.
4. `train_with_intrinsic`: the inner loop. Act and step. If the move didn't end the episode, get the bonus for the *next* state and take one RND predictor step on it. Then update Q on `r_ext + intrinsic_coef * r_int`.

`train_q_learning_alone` is given. Run it as a sanity check before you wire up RND: it should almost never reach the goal.

## Acceptance criteria

`pytest exercises/20-exploration/` passes. That's:

- `CombinationLock.step`: right moves one square with reward 0 up to square 14, and the move onto square 15 pays 1.0 and ends the episode. Left ends the episode with reward 0 on any square, square 0 included, and leaves the agent where it was.
- `QLearningAgent.act` returns an int, takes the greedy move when there's no tie, and picks each of two tied moves about half the time.
- `QLearningAgent.update` uses `r` alone on `done=True` (for the goal and for the wrong move), bootstraps off `max(Q[s_next])` otherwise, moves `Q` by `alpha` times the TD error, and returns the TD error.
- `RNDIntrinsicReward.intrinsic_reward` is positive for a never-trained state and falls below 10% of its first reading after 200 updates on that state. The target network's parameters never move; the predictor's do.
- `train_q_learning_alone(seed=s, n_episodes=200)` reaches the goal at most once, for seeds 0, 1 and 2.
- `train_with_intrinsic(seed=s, n_episodes=200, intrinsic_coef=0.1)`, for seeds 0, 1 and 2: Q(s, left) stays exactly 0 on every square, the RND predictor has been trained, the first +1 comes by episode 100, the mean extrinsic return over the last 50 episodes is at least 0.2, and the greedy move is right on every square.

The suite takes under 4 seconds on a laptop CPU (2.8 to 3.7 seconds on an Apple M2 Max).

You're done when the tests pass and you can explain why plain ε-greedy's Q-values stay at 0 until its first success, why any positive bonus opens this lock, why the bonus mustn't be paid on a move that ends the episode, and what RND would give you over a constant bonus on a problem where a move can lead back to a state you've already visited.

## If you get stuck

Read [`HINTS.md`](./HINTS.md), one hint at a time. The reference implementation is in [`solution/exploration.py`](./solution/exploration.py); look at it after you've made a real attempt.

## Going further (optional)

- Check that any positive bonus does the same here. In `train_with_intrinsic`, replace `r_int` with `1.0` and compare the episode of the first +1 with the RND run, seed by seed. Then drop the normalization, or the `rnd.update` call, and compare again. Before you run them, work out which numbers can change: the Q-values will, but what about the moves the agent makes?
- Make the lock longer. At 24 or 32 squares plain ε-greedy's odds fall to 2^-23 or 2^-31 per episode. The bonus agent still opens the lock, but more slowly, because walking to square k without an ε-random left has probability 0.95^k. With RND the median first +1 moved from episode 22 at 16 squares to 41 at 24 squares and about 72 at 32, and the mean return over the last 50 of 400 episodes fell to 0.30 and 0.20, close to 0.95^23 and 0.95^31 (200 seeds each). Try decaying ε once the greedy move is right on every square.
- Find a problem where the novelty matters. Change the lock so the wrong move sends the agent back to square 0 instead of ending the episode. Episodes then need a step limit, and a timeout isn't a real ending, so bootstrap there: pass `done=False` to the update and stop the loop yourself. Now a constant bonus pays as much for a move back to square 0 as for a move onto a new square, and RND's bonus doesn't. Before you run it, work out what a constant 0.1 per move is worth at γ = 0.99 to an agent that never stops, 0.1 / (1 − 0.99) = 10, and compare that with the +1 for reaching the goal.
