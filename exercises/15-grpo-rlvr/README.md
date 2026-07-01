# Exercise 15: GRPO on a verifiable toy task

Goes with the lecture on GRPO (Group-Relative Policy Optimization) and RLVR (Reinforcement Learning from Verifiable Rewards).

You'll implement the core GRPO update from scratch on a tiny arithmetic task: the policy learns to output the correct answer to addition problems like "what is 2 + 3?". No language model, no transformer: just a per-prompt categorical distribution, so everything runs in seconds and the machinery is visible without the noise.

The point: this is the same algorithm that powers DeepSeekMath and DeepSeek-R1, just scaled down. Once you understand why each piece is here, the full-scale version is conceptually the same.

## Why GRPO works

Standard policy gradient needs a baseline to reduce variance: usually a learned value function `V(s)`. That means training two networks and making sure the critic doesn't lag the actor.

GRPO replaces the value-function baseline with a *group baseline*: sample K completions for the same prompt, compute a reward for each, and use the mean reward of the group as the baseline. The advantage for completion `i` is:

```
A_i = (r_i - mean(r_1..r_K)) / (std(r_1..r_K) + eps)
```

This is cheap (no extra network), prompt-specific (the baseline adapts per prompt), and unbiased in expectation. When all K completions are equally good or bad, the advantages are all zero: no noisy update that step. When some are better than others, the policy gets a clear signal.

The reward is a *verifier* (a pure checker function), not a learned reward model. You don't need RLHF-style preference data; you just need a function that can check correctness. That's why this style of training works so well for math and code: the output is either right or wrong.

## The task

Fill in the TODOs in [`starter.py`](./starter.py). There are five pieces:

1. `GRPOPolicy`: an `nn.Embedding(n_prompts, n_answers)` that outputs logits. One row per prompt, one column per possible answer. Dead simple.
2. `sample_group`: sample K answers from the policy for a given prompt; return the answers and their log-probs (with gradient).
3. `group_advantages`: compute `(r - r.mean()) / (r.std() + eps)` for a length-K reward tensor. Handle the edge cases: K==1 or all rewards equal → return zeros.
4. `grpo_loss`: the PPO-clipped surrogate loss, averaged over the group. The clip keeps the policy from taking a step that's too large in one update.
5. `train`: the loop: sample a prompt, sample a group of K answers, get rewards from the verifier, compute advantages, compute the loss, update. Return per-step mean reward so you can see learning.

## Setup

```bash
pip install -r exercises/requirements.txt
pytest exercises/15-grpo-rlvr/
```

## Acceptance criteria

`pytest exercises/15-grpo-rlvr/` passes. That's:

- `verifier` returns 1.0 for correct answers, 0.0 for wrong ones.
- `group_advantages` produces a zero-mean, approximately unit-std tensor, and returns zeros for degenerate inputs (K==1 or all rewards equal).
- `grpo_loss` returns a scalar tensor that backpropagates into the policy parameters.
- Integration test: `train(seed=0)` converges so that the last 200 steps average above 0.6 mean reward per group. Random guessing gets about 0.1 (1/10 answers). The test runs in well under 10 seconds.

You're done when the tests pass and you can explain: why the group baseline doesn't require a value network, and why clipping the probability ratio stabilizes training.

## If you get stuck

Read [`HINTS.md`](./HINTS.md), one hint at a time. The reference implementation is in [`solution/grpo_rlvr.py`](./solution/grpo_rlvr.py); look at it after you've made a real attempt.

## Going further (optional)

- Add a KL-penalty term: `+ beta * KL(pi || pi_ref)` where `pi_ref` is a frozen copy of the initial policy. This is the full GRPO objective. Watch what happens to convergence speed as you raise `beta`.
- Try increasing K and watching the advantage estimates get less noisy: this is the variance-reduction effect of more samples.
- Try a harder task: the policy maps `(a, b, c)` triples to `a + b + c`, larger prompt space, more possible answers. At what scale does the simple embedding approach start to need more structure?
- Read the DeepSeekMath paper and find the exact equation that matches what you implemented here.
