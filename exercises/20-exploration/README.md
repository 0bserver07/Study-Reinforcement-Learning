# Exercise 20: RND exploration on a sparse-reward chain

Goes with [Lecture 20: Exploration](../../notes/lectures/20-exploration.md).

You'll build a 1D chain MDP where the reward is so sparse that vanilla epsilon-greedy Q-learning never finds it, then fix it by adding a Random Network Distillation (RND) intrinsic-motivation bonus. The point is to see the failure mode of undirected exploration with your own eyes (the mean return stays at 0.0 forever) and then to see a tiny novelty signal flip it to ~1.0 within a hundred episodes.

## Why this env is hard

The chain has 20 states. Agent starts at 0. Actions are {left, right}. Reward is 1.0 only on entering state 19. Episodes terminate at 100 steps.

With epsilon-greedy at `epsilon = 0.1`, the agent's behavior is essentially a random walk over the chain: the Q-values stay at zero, so `argmax` always picks action 0 (left), and the only progress to the right comes from the epsilon-random branches. A symmetric random walk on a chain of length N takes O(N²) steps to reach the far end. For N=20 and a 100-step cap, the probability of reaching the goal in any given episode is roughly 1%, and even that vastly overstates it for a near-greedy walker biased toward state 0.

So the agent never sees a reward. Without seeing a reward, the Q-table never updates away from zeros. Without a non-zero Q-table, the agent has no reason to do anything different. The training loop runs without ever learning anything. This is the structural pathology that ε-greedy can't fix on sparse-reward environments.

## What RND does

RND replaces "I've never seen this state" (which would need an explicit visit count) with "my predictor doesn't yet output what the target outputs for this state":

1. Build two MLPs with identical architecture, mapping one-hot state to a 16-dim feature vector.
2. Freeze the first (the *target*) at its random initialization. It never updates.
3. Train the second (the *predictor*) to match the target's output on visited states, by MSE.
4. The intrinsic reward at state `s` is `||predictor(s) - target(s)||²`, normalized by a running standard deviation of those errors.

A state the predictor has never trained on produces a large error: two different randomly-initialized networks output very different things on the same input. After many gradient steps on a state, the predictor matches the target there and the error drops to ~0. So the bonus is large on novel states and decays toward zero on well-visited ones, which is what a count-bonus would do, but without needing to enumerate states.

You then run Q-learning where the reward is `r_extrinsic + intrinsic_coef * r_intrinsic`. The agent starts getting positive reward signal for moving to states it hasn't visited yet. That pushes it past the initial cluster around state 0. By the time it reaches state 19, it gets the real extrinsic reward and Q-learning takes over the rest of the work.

This works on Montezuma's Revenge (Burda et al. 2018, arXiv:1810.12894) for the same structural reason it works on the chain: a novelty bonus turns "I have no signal anywhere" into "I have a directional signal toward parts of the state space I haven't covered yet."

## The task

Fill in the TODOs in [`starter.py`](./starter.py). Four pieces:

1. `ChainEnv.step`: the env transition. Left/right, clamp at 0, terminate at the goal or at `max_steps`.
2. `QLearningAgent.act` and `.update`: epsilon-greedy action selection and the standard Q-learning update. Watch the terminal case: target is `r`, not `r + gamma * max(Q[s'])`, on `done=True`.
3. `RNDIntrinsicReward.intrinsic_reward` and `.update`: compute the normalized prediction error, and run one Adam step on the predictor toward the (frozen) target output. The `_raw_error` helper and the running-stats update are given.
4. `train_with_intrinsic`: the inner loop: act, step, get the intrinsic bonus on the *next* state, update Q on `r_ext + intrinsic_coef * r_int`, and take one RND predictor step on the next state.

`train_q_learning_alone` is given. Run it as a sanity check before you wire up RND: it should flatline at 0 returns.

## Acceptance criteria

`pytest exercises/20-exploration/` passes. That's:

- `ChainEnv.step` handles the three cases (right, goal-reaching with reward 1 and done, timeout with done but no reward) and the left-at-zero boundary.
- `QLearningAgent.update` uses `r` only on `done=True` and bootstraps off `max(Q[s_next])` otherwise.
- `RNDIntrinsicReward.intrinsic_reward` returns positive numbers for never-trained states, and the value drops to <10% of the initial reading after 200 updates on the same state.
- The target network's parameters never move; the predictor's do.
- `train_q_learning_alone(seed=0, n_episodes=200)` has mean return below 0.1 over the last 50 episodes: vanilla Q-learning fails as expected.
- `train_with_intrinsic(seed=0, n_episodes=200, intrinsic_coef=0.1)` has mean *extrinsic* return above 0.5 over the last 50 episodes. (Reference solution hits ~1.0 within 100 episodes.)

Full test suite runs in under 15 seconds on CPU.

You're done when the tests pass and you can explain: why the intrinsic bonus has to *decay* on well-visited states for the agent to eventually exploit, and why we normalize the intrinsic reward by a running std instead of using the raw squared error.

## If you get stuck

Read [`HINTS.md`](./HINTS.md), one hint at a time. The reference implementation is in [`solution/exploration.py`](./solution/exploration.py); look at it after you've made a real attempt.

## Going further (optional)

- Drop the running-std normalization and watch what happens. With raw errors, the bonus is huge at the start of training (the predictor knows nothing, so the error is large everywhere) and tiny later (after a few states are learned). The effective `intrinsic_coef` is moving by orders of magnitude without you controlling it. On a longer chain (say N=40) the un-normalized version can either get stuck or chase noise.
- Replace RND with a literal count bonus: `r_intrinsic(s) = beta / sqrt(N(s) + 1)` where `N(s)` is the number of times you've visited `s`. This is exactly what RND approximates: when the state space is discrete and small, the direct version is simpler and often just as good. The interesting comparison is: at what chain length does the count version start to lose to RND? (For the discrete one-hot case here, it actually doesn't: RND is the right baseline to study because it generalizes to continuous state where you can't tabulate counts.)
- Try the deceptive-reward variant: add a small reward of 0.1 at state 2, the much bigger reward of 1.0 still at state 19. RND will help the agent past state 2, but a greedy-on-Q policy with a less aggressive epsilon can get stuck on the 0.1. This is the "novelty bonus helps with sparse reward, not with deceptive reward" point from the lecture, in 5 lines of code.
- Implement ICM (Pathak et al. 2017, arXiv:1705.05363) on the same env: train an inverse model `a_hat = g(phi(s), phi(s'))`, use the forward-model prediction error as the bonus. On this simple env you won't see ICM beat RND, but you'll see that getting the encoder right is harder than just freezing a random one.
