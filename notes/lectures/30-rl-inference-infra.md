<!-- status: unreviewed | last-reviewed: never -->

# Lecture 30: RL inference infrastructure for LLMs

_Unreviewed: no one has checked this end to end. Treat the math, code, and citations as unverified._

**Time**: ~2 h · **Prerequisites**: Lecture 10

---

## Why this lecture exists

Profile a GRPO or PPO run on a modern LLM (say, a 7B-to-70B policy with a group size of 32 and a few thousand prompts per step). You will find that something like 70–90% of wall-clock time goes to generation, running the policy in inference mode to produce the rollouts you'll train on, and the actual gradient step, with all its backwards passes and optimizer updates, takes the remainder.

That ratio is uncomfortable until you internalize it. It means optimizing rollouts is the single highest-leverage thing you can do for training throughput. A 2x speedup on the sampler roughly halves your iteration time. A 2x speedup on the gradient step buys you maybe 10–15%. The lecture is about why the imbalance exists and what people do about it.

The shape of the work matters. Production LLM serving (chatbots, completions APIs) and RL rollout generation look superficially similar (both run an autoregressive model on prompts to produce completions), but the constraints are different enough that the engineering trade-offs end up in different places. You can take vLLM off the shelf and use it for rollouts, and many people do, but the workloads it was tuned for are not exactly the workload of an RL training loop.

This is a 2026-era snapshot. The tooling moves fast; treat the names of specific systems as anchor points rather than the canonical answers.

---

## What makes RL rollout workloads different

Compared to a chatbot serving workload, RL rollouts have a particular shape:

**Many short prompts, K samples each.** A typical training batch is something like 256–1024 unique prompts, each sampled K=16–64 times. That gives you 4k–64k completions to generate per gradient step. The prompts themselves are usually a few hundred to a few thousand tokens; the completions can be much longer (reasoning traces, code with tests, multi-turn dialogue).

**All K completions must finish before the gradient step.** A chatbot can stream tokens to a user as they're generated. The RL learner can't start its update until every completion in the batch is ready, or until you decide to accept partial batches and pay an asynchrony tax. The slowest completion in the batch sets the wall-clock floor.

**The prompts repeat.** Each of the K samples for a given prompt shares the same prefill. That prefix's KV cache is the same across all K rollouts, up to the point where they diverge. A serving system that handles every request as an independent context is leaving that sharing on the floor.

**Variable output length, with a hard ceiling.** Some completions hit `max_new_tokens` (often 1024–8192 for reasoning); others stop early on EOS. Mean length might be 600 tokens, max 4096. Scheduling that mix on shared GPU memory is the central challenge; naive batching pads the short completions out to the longest one and wastes the rest.

**The policy you sample from will be discarded soon.** Within a few hundred or thousand gradient steps, the policy weights will be replaced. There's no point in elaborate compilation or warm-up if the weights you're optimizing for are obsolete by the time you finish. This biases the engineering toward fast iteration over peak per-token throughput.

**Decoder-only, autoregressive.** Same as inference for any GPT-style model. Prefill is parallel across the prompt tokens; decode is sequential, one token at a time per sequence, but parallel across the batch.

The first three of these are the ones that distinguish RL serving from chatbot serving and motivate most of what follows.

---

## A mental model: the pipeline

Here's the pipeline a typical 2025-era GRPO trainer runs, in ASCII:

```
                    +--------------------+
                    |   prompt source    |
                    | (dataset, replay)  |
                    +---------+----------+
                              |
                              | prompts
                              v
              +---------------+----------------+
              |        rollout orchestrator    |
              |  - replicates prompt K times   |
              |  - load-balances across pool   |
              |  - tracks in-flight requests   |
              +-+--------+--------+--------+---+
                |        |        |        |
                v        v        v        v
            +-----+  +-----+  +-----+  +-----+
            |vLLM |  |vLLM |  |vLLM |  |vLLM |   each holds
            | #0  |  | #1  |  | #2  |  | #N  |   policy weights
            +-----+  +-----+  +-----+  +-----+   (fp8 or fp16)
                |        |        |        |
                +--------+---+----+--------+
                             |
                             | completions
                             v
                  +----------+-----------+
                  |   reward computation |
                  | - code sandbox       |
                  | - judge model        |
                  | - regex / verifier   |
                  +----------+-----------+
                             |
                             | (prompt, completion, reward, sampling logprobs)
                             v
                  +----------+-----------+
                  |    rollout buffer    |
                  +----------+-----------+
                             |
                             v
                  +----------+-----------+
                  |       learner        |
                  | (training fwd/bwd,   |
                  |  optimizer step,     |
                  |  KL to ref policy)   |
                  +----------+-----------+
                             |
                             | new weights
                             v
                  +----------+-----------+
                  | weight sync broker   |
                  | (NCCL / IPC / disk)  |
                  +----------+-----------+
                             |
                             | push weights back to pool
                             +---> back to vLLM pool
```

The arrows hide an enormous amount of work. The dotted boundary between "rollout pool" and "learner" is the place where most of the engineering pain lives. Two big questions:

1. How fresh are the rollouts? If the learner is on update 100 and the rollout pool is still serving completions sampled from update 95, you have stale rollouts. Stale rollouts need importance correction, otherwise the policy gradient is biased.
2. How much do you spend on weight sync? Full-state broadcast for a 70B model over NCCL takes seconds; if you sync every step, that's seconds-per-step of pure overhead.

These two trade off against each other. Section on weight sync below.

---

## vLLM and PagedAttention

The state of the art for open-source LLM serving in 2025 is vLLM, introduced in Kwon et al. 2023 (arXiv:2309.06180). The headline contribution is PagedAttention, a KV-cache memory manager that borrows the virtual-memory-and-paging idea from operating systems.

The problem it solves: in a transformer, every active sequence in a batch needs a KV cache (the cached keys and values from every attention layer for every token in the sequence so far). Naively, you pre-allocate the worst case, `max_seq_len` slots per request, and live with the waste. For a model with 32 layers, hidden dim 4096, half-precision keys+values, and `max_seq_len=8192`, that's:

```
2 * 32 * 4096 * 8192 * 2 bytes = 4 GB per request
```

Hold a batch of 32 requests at that ceiling and you've burned 128 GB on KV cache alone, most of it never used because most sequences finish well short of `max_seq_len`. This is internal fragmentation in the OS sense: allocated but unused space.

PagedAttention chops the KV cache into fixed-size pages (typically 16 tokens each) and allocates pages on demand as sequences grow. A page table per request maps logical positions to physical pages. When the request finishes, its pages return to the pool. The waste drops to at most a partial last page per request, plus a small bookkeeping overhead.

Why this matters for RL: with PagedAttention, you can hold many more in-flight rollouts on the same GPU. A 4x reduction in KV-cache waste means roughly a 4x larger batch, which means roughly a 4x throughput multiplier on the decode stage (since decode is memory-bandwidth-bound and bigger batches amortize the cost of streaming weights from HBM). Kwon et al. report 2–4x throughput over earlier systems on production workloads; the gain on RL rollout workloads, with their many parallel completions, is in the same range.

A related vLLM feature: **prefix caching**. If two requests share the first N tokens (same system prompt, same few-shot examples, same problem statement), the KV-cache pages for those tokens can be shared. The K samples per prompt in a GRPO step are exactly this case: the prefill is shared across all K, the decode diverges per-sample. The shared pages are reference-counted; they're freed when the last sample finishes. For a problem statement of 500 tokens and K=32, that's a 32x reduction in prefill compute and 31x reduction in KV memory for those prefix pages. The savings are massive on long prompts.

---

## Continuous batching (iteration-level scheduling)

Continuous batching, also called iteration-level scheduling, was introduced as Orca (Yu et al., OSDI 2022; "Orca: A Distributed Serving System for Transformer-Based Generative Models"). The vLLM serving loop implements it.

The naive batching strategy is **request-level**: pick B requests, batch them together, run forward passes until all B are done, return results, pick the next B. The problem: requests finish at different times. The first request might emit EOS after 50 tokens; the last might run to `max_new_tokens=2048`. The batch occupies the GPU for the duration of the longest sequence, and the slots where shorter sequences finished sit idle.

Continuous batching does the scheduling at the granularity of a single decode step:

1. After every decode step (one token generated per active sequence), the scheduler checks which sequences have finished.
2. Finished sequences are returned to the client and their slots freed.
3. Pending requests from the queue can take those freed slots immediately.

The result is that the GPU is always running a full batch of active sequences. Throughput goes up by roughly the ratio of longest-to-mean sequence length, which on real distributions is often 3–5x.

For RL rollouts, continuous batching matters because the length distribution is wide. A reasoning model might emit 200 tokens for an easy problem and 4000 for a hard one. Without continuous batching, the GPU is sitting on the easy ones for as long as the hard ones run.

There's a subtlety with PPO-style importance sampling. The token at position t in a sequence might be generated under slightly different conditions (different KV cache eviction patterns, different paging) depending on what else is in the batch at that moment. In practice this doesn't perturb the sampled tokens (sampling is deterministic given logits and seed), but it can perturb timing measurements. If you're profiling rollout throughput, profile under load, not in isolation.

---

## SGLang and RadixAttention

vLLM's prefix caching is most useful when prompts share a literal prefix. SGLang (Zheng et al. 2023, arXiv:2312.07104) generalizes this with RadixAttention: a radix tree of KV-cache pages indexed by token sequences. When a new request comes in, its prefix is matched against the tree, and any matching pages are reused.

This buys you sharing across requests that share intermediate prefixes, not just the head. For RL workloads where many rollouts share a system prompt + few-shot exemplars + problem statement, the savings compound. If you have:

- A 500-token system prompt shared by all 4000 rollouts in a batch
- A 200-token problem-class instruction shared by the 100 rollouts on each problem class
- A 500-token problem statement shared by the 32 K samples for each problem

then RadixAttention can serve all three layers of sharing from one KV cache, instead of just the outermost layer.

The bookkeeping cost is nontrivial (maintaining the radix tree, doing the longest-prefix lookups, evicting cold pages), but the win is large enough for RL rollout workloads that several training frameworks now use SGLang as their rollout backend by default. As of late 2025, SGLang and vLLM are roughly co-equal options for this; pick on the basis of which integrates more cleanly with your trainer.

---

## Speculative decoding: useful but careful

Speculative decoding (Leviathan et al. 2022, arXiv:2211.17192; Chen et al. 2023, arXiv:2302.01318) uses a small "draft" model to propose K candidate tokens, then runs the large target model on all K in parallel to verify. If the draft model agrees with the target on a prefix of length k ≤ K, you accept that prefix and use the target's k+1-th token; otherwise you reject at the first disagreement. The math is set up so the output distribution is exactly the target's: no accuracy loss for inference.

For production serving, this is great: a 2–3x speedup on decode is reported in the Chen et al. paper for Chinchilla 70B.

For RL training, it's complicated. The policy gradient needs the exact log-probability of each sampled token under the policy at sampling time: that's the `log_probs_old` from Lecture 10's PPO surrogate, or the `pi_old` in the importance ratio. Speculative decoding gives you the sampled tokens, but the per-token logprobs available naturally from the spec-decode loop are the draft model's logprobs, not the target's. You have two options:

**Option A: pay for a verification pass.** Run a separate forward pass of the target model over the full sampled sequence to extract exact target logprobs. This costs about as much as the prefill it would have done anyway, but you've lost the savings from speculation. You're back to vanilla decoding cost, plus the draft model's overhead.

**Option B: accept the approximation.** Treat the spec-decode-generated tokens as samples from the target distribution (which they are, by construction), but use the draft model's logprobs as a stand-in for the target's in the importance ratio. This is technically wrong: the ratio is no longer `pi_current/pi_old` but something biased. In practice, if the draft model is close to the target (e.g., a smaller distilled version), the bias may be small enough not to destabilize training. The literature on this for RL training is thin and the answer depends on your tolerance for off-policy correction error.

A safer middle ground: use spec decode for the parts of training where exact logprobs aren't needed. If you're using RL with verifiable rewards (Lecture 15) where the reward depends only on the final answer, and you've chosen to do without a per-token importance correction (treating the whole completion as one action), then the per-token logprobs don't enter the gradient at all and spec decode is free to use.

In practice, the bigger 2025 trick for this regime is **draft model self-speculation**: the target model itself proposes drafts using early-exit heads or smaller layer subsets, avoiding a separate model entirely. This eliminates one of the bookkeeping problems but has the same logprob-validity question.

---

## The reference policy: the other model in the room

PPO-LLM and GRPO both include a KL penalty to a reference policy (Lecture 10, Section 4; Lecture 15, Section on GRPO loss):

```
L_KL = beta * KL(pi_theta || pi_ref)
```

The reference policy is usually the initial SFT model or the pre-RL policy. To compute the KL term, you need `log pi_ref(y_i | x)` for every token of every sampled completion. If the reference model is LLM-sized, typically the same size as the policy, you have a second inference-cost-equivalent workload sitting next to the rollout pool.

A few common tricks:

**Cache reference logprobs at sampling time.** When the rollout pool generates a completion, run the reference model in parallel on the same prompt+completion and save the per-token logprobs alongside the response. Store them in the rollout buffer. The learner then doesn't need to recompute them; it reads them off the buffer. Cost: roughly doubles the rollout-pool compute, but avoids running the reference model inside the gradient step's forward pass. This is the usual default.

**Co-locate the reference model with the rollout instance.** Same GPU, different inference engine. If you have spare HBM, you save weight loading and can interleave the two forward passes.

**Use a smaller proxy.** A distilled or quantized version of the reference model approximates `pi_ref`. The KL penalty is now to the proxy, not the true reference, which changes the optimization slightly. In practice this is fine if the proxy is close, and the cost savings are substantial when the policy is large.

**Skip the reference entirely.** Some recent papers (e.g., the family of "reference-model-free" variants of DPO-like methods) sidestep the reference model with various reparameterizations. Worth being aware of for the algorithm choice; doesn't help if you've committed to PPO or GRPO.

The reference-policy KV cache is, of course, separate from the policy's KV cache. The total HBM footprint of an RL rollout setup is meaningfully higher than for serving one model.

---

## The actor pool and the orchestrator

For a serious-scale training run (anything beyond toy), you don't have one vLLM instance; you have a pool. Common shapes:

- 8–32 vLLM instances per training run, each on its own GPU (or a small TP group).
- Each holds a full copy of the policy weights, usually quantized to fp8 or int8 for inference.
- A central orchestrator (often a Ray actor, or a custom Python process) receives the training step's prompt list, replicates each prompt K times for the group size, and dispatches the K*B individual requests to the pool.
- Load balancing is approximate: usually round-robin, sometimes load-aware based on each instance's queue depth.

The pool model is necessary because a single GPU can't usually hold enough rollouts in flight to saturate the learner. At a 4000-completion batch with mean length 1000 tokens, you're generating 4M tokens per step. A single H100 at, say, 5k tokens/sec sustained throughput on a 70B model would take 800 seconds per step: call it 13 minutes. With 16 instances you're at under a minute. The gradient step itself is on the order of 10–30 seconds on a similarly-sized cluster, so the math starts working.

The orchestrator's job is unglamorous:

- Hold a queue of pending prompts.
- Dispatch to whichever vLLM instance has the shortest queue.
- Collect completions as they come back.
- When all K samples for a prompt are done, push the prompt's group to the rollout buffer.
- Manage retries when a vLLM instance crashes mid-request (it happens).

A common bug at this layer: head-of-line blocking. If the orchestrator dispatches all K samples for prompt P to the same instance, and prompt P happens to be a hard one that runs long, that instance is bottlenecked while others finish their work. Spreading the K samples across multiple instances is the standard fix; the cost is that you lose the prefix-cache locality for that prompt (each instance has to fill its own prefix cache once). The trade-off is usually worth it for non-trivial K and non-trivial prompt lengths.

---

## Weight synchronization

The single most painful piece of infra in an RL-LLM training loop is getting the new policy weights from the learner to the rollout pool. Three approaches in common use:

**Full state-dict broadcast.** After each gradient step (or every N steps), the learner serializes the full parameter set and broadcasts it to every rollout instance over NCCL or a similar high-bandwidth collective. For a 70B model at fp16, that's 140 GB per broadcast. On a 200 Gbps InfiniBand interconnect, that's roughly 6 seconds of network time at peak utilization, more in practice. For a 7B model it's well under a second. This is the simplest approach and the one most frameworks default to.

**Delta updates.** Only send the parameters that changed meaningfully, or send low-rank decompositions of the delta. The bookkeeping is heavier and the savings depend on how concentrated the gradient updates are. In practice, for full-parameter fine-tuning, almost everything changes a little, so delta updates don't help much. For LoRA-based RL, where only the adapter weights change, this dominates: you ship megabytes, not gigabytes, and weight sync becomes negligible.

**In-place same-machine memcpy.** When the learner and a rollout instance are on the same machine (in different processes, with the policy in shared memory or accessible via CUDA IPC handles), the sync is just a `cudaMemcpyPeer`-style copy at PCIe or NVLink speeds. This is the fastest option when you can arrange the topology for it; it doesn't help across machines.

The frequency trade-off is the core knob:

- **Sync every gradient step.** Rollouts are always fresh: sampled from the latest policy. The importance ratio in PPO/GRPO is exactly 1.0 (modulo numerical noise), so no off-policy correction is needed. Cost: high sync overhead. If sync takes 10s and a gradient step takes 20s, you've lost a third of throughput.

- **Sync every N steps.** Rollouts up to N steps stale. The importance ratio diverges from 1.0; PPO's clipping handles small deviations, but at some point the policy has moved too far and the clipped objective stops giving useful gradients. The cap on N depends on the size of the gradient step. With a small LR and KL-constrained updates, N=8 or N=16 is usually fine. With aggressive updates, N=1 may be needed.

- **Asynchronous sync.** Decouple weight push from gradient step. The learner keeps stepping; the rollout pool picks up new weights when they arrive. The staleness is variable but bounded. This requires careful importance-ratio tracking but maximizes throughput.

The frontier in 2025 is mostly the asynchronous variants, with importance-corrected GRPO variants that explicitly handle the staleness in their loss. Be careful, though: a paper might report excellent results with async training, but it usually also reports careful tuning of the maximum allowed staleness, the importance-ratio clipping, and the buffer drainage policy. Replicating async training is hard.

---

## Reward computation at scale

The rollouts come back. Now you have to compute rewards for them. For some reward types this is fast; for others, it's the next bottleneck after rollouts.

**Cheap rewards.**

- Regex/string-match on a final answer (math problems with extracted numeric answers): microseconds per completion. Negligible.
- Format checks (does the output have `<think>` tags, an answer box): microseconds per completion. Negligible.

**Expensive rewards.**

- **Code execution.** Each code completion is run against a test suite in a sandbox. Sandbox spin-up (Firejail, gVisor, Docker, e2b-style isolated VMs) takes 50–500 ms; test execution time varies wildly with the problem (10 ms to several seconds). For a batch of 4000 code completions, you can be looking at minutes of wall-clock just for reward computation, even with parallelism across many sandbox workers.
- **Judge-model calls.** If your reward is "a frontier LLM rates this completion 1–10," you're paying API latency (hundreds of ms per call) plus token cost for every completion. Batching helps; using a smaller local judge model helps more.
- **Reward-model inference.** If you have a trained reward model (Lecture 09), it's another LLM-sized forward pass per completion. Smaller than the policy usually, but not free. Batched, this is bearable.

For code rewards, the standard mitigations are: pre-warm sandbox pools so you don't pay spin-up per execution; batch the test executions per sandbox; cap execution time per test aggressively (kill anything over a few seconds); colocate the reward workers near the rollout pool so completions don't have to traverse the network.

For judge-model rewards, the question is whether to spend on a frontier judge or a cheap judge. The cheap judge introduces noise; the noise translates directly into noisier gradient estimates. Calibration studies on judge agreement with humans (Lecture 14 covers RLAIF) are the input to this decision.

Reward latency can be the next bottleneck after rollout, especially in code or tool-use domains. The orchestrator usually handles this by streaming completions to a reward queue, with reward workers consuming asynchronously and pushing scored rollouts to the buffer. The learner drains the buffer when it has enough.

---

## FP8/INT8 rollouts vs FP16/BF16 training

A subtle issue. For inference speed and memory, you want the rollout pool running the policy in fp8 or int8. For training stability, the learner runs the same policy in bf16 (or fp32 for the optimizer state). After each weight sync, the bf16 master weights are quantized to fp8 for the rollout pool.

The implication: the policy you sample from (call it `pi_q`, the quantized policy) is not exactly the policy you're training (`pi_theta`). Quantization introduces a small distortion to the per-token distribution.

For the policy gradient, what you care about is the importance ratio:

```
ratio = pi_theta(y_t | x, y_{<t}) / pi_old(y_t | x, y_{<t})
```

Here `pi_old` is the policy that was used to sample, which is `pi_q`. The ratio is `pi_theta / pi_q`, and this is not 1.0 even immediately after a weight sync, because `pi_q` is the quantized version, while `pi_theta` is the un-quantized version that the gradient is computing through.

In practice this matters less than you might fear. The quantization error per token is small, and the PPO/GRPO clipping handles small ratios well. The published literature mostly papers over this: most training frameworks sample with fp8 and train with bf16 without explicit correction, and report fine results.

But: if you observe unexpected instability and your training framework is doing aggressive quantization, this is one place to look. Diagnostic: compute the empirical ratio `pi_theta(y_t) / pi_q(y_t)` over a fresh batch and look at its distribution. If most of the mass is between 0.9 and 1.1, you're fine. If it's spread wider, the quantization is biting and you may need to either reduce step size or recompute `log_probs_old` in bf16 right before the gradient step (which costs a full forward pass through the rollouts, but recovers exactness).

---

## The rollout buffer

The buffer is where rollouts wait between the rollout pool and the learner. Two design choices have outsized consequences:

**On-policy vs replay.** Strictly on-policy means the buffer is drained completely between gradient steps: every rollout you train on was sampled from the immediately-preceding policy. This is PPO's default assumption and what makes the importance ratio close to 1.0. Replay-style means rollouts hang around for multiple gradient steps; you sample minibatches from a buffer that holds rollouts from the last N policies. Replay improves sample efficiency but pushes you off-policy and requires the importance ratio to do real work.

The middle ground in modern GRPO trainers: drain the buffer each step, but allow some overlap; start collecting rollouts for step N+1 while step N's gradient is still running. The new rollouts come from the still-old weights (you haven't synced yet), so they're at most 1 step stale. This is the "1-step-behind async" pattern and it's usually safe.

**Grouped vs flat layout.** The buffer can store rollouts as flat (one row per completion) or grouped (one row per prompt, holding the K completions for that prompt as a list). GRPO needs the grouped form to compute group-relative advantages; PPO with GAE can use either. If you store flat and need grouped, you pay a regrouping pass: not expensive, but if you forget about it you'll end up with advantages computed over the wrong baseline.

A practical issue: in a partially-failed step (some prompts didn't get all K samples back), the orchestrator has to decide whether to drop the incomplete groups, retry the missing samples, or train with reduced K for those groups. Each option has a different statistical consequence (dropping is unbiased but reduces effective batch size; reduced K means higher-variance advantages for those prompts). Most production trainers drop, set a minimum-K threshold around K-2 or K-4, and call it done.

The buffer should also store, alongside the completion text and rewards, everything the learner needs to compute the loss without re-running the policy: the sampling logprobs (per token, or summed per completion depending on your loss form), the reference logprobs, the prompt token IDs, the completion token IDs, and any tool-call metadata (for multi-turn rollouts). Anything you forget to store, you'll pay to recompute, and recomputing usually means another inference pass.

---

## A small pseudo-code skeleton

Here's the rollout-batched loop in skeletal form. It elides most of the actual machinery (no error handling, no real load balancing, no real buffer), but the shape is right.

```python
import asyncio
from dataclasses import dataclass
from typing import List


@dataclass
class Rollout:
    prompt: str
    completion: str
    sampling_logprobs: List[float]   # per-token logprob under pi_old
    ref_logprobs: List[float]        # per-token logprob under pi_ref
    reward: float
    prompt_id: str
    group_idx: int                   # which K-sample is this within the group


@dataclass
class RolloutGroup:
    prompt: str
    prompt_id: str
    rollouts: List[Rollout]          # length K


class VLLMPool:
    """Pool of vLLM instances holding the current policy."""

    def __init__(self, n_instances, model_name):
        self.instances = [...]       # spin up vLLM workers
        self.queue_depths = [0] * n_instances

    async def generate(self, prompt: str, sampling_params) -> Rollout:
        # Pick least-loaded instance.
        i = min(range(len(self.instances)), key=lambda j: self.queue_depths[j])
        self.queue_depths[i] += 1
        try:
            result = await self.instances[i].agenerate(prompt, sampling_params)
            return Rollout(
                prompt=prompt,
                completion=result.text,
                sampling_logprobs=result.token_logprobs,
                ref_logprobs=[],     # filled in by reference pool
                reward=0.0,          # filled in by reward worker
                prompt_id="",
                group_idx=0,
            )
        finally:
            self.queue_depths[i] -= 1

    def sync_weights(self, new_weights):
        # Broadcast new weights to every instance.
        # This is the slow part; see weight-sync section above.
        for inst in self.instances:
            inst.load_weights(new_weights)


class RefPool:
    """Pool serving the frozen reference policy."""

    async def score(self, prompt: str, completion: str) -> List[float]:
        # Return per-token logprob of completion under pi_ref.
        ...


class RewardWorker:
    """Computes rewards. May be CPU-bound (regex), GPU-bound (RM), or
    sandbox-bound (code execution)."""

    async def score(self, prompt: str, completion: str) -> float:
        ...


async def collect_one_group(
    pool: VLLMPool,
    ref_pool: RefPool,
    reward_worker: RewardWorker,
    prompt: str,
    prompt_id: str,
    K: int,
    sampling_params,
) -> RolloutGroup:
    """Sample K completions for one prompt; score them; return the group."""
    # Fan out K parallel generations. The prefix is shared, so the vLLM
    # instances should be hitting cached pages for the prompt.
    completion_tasks = [
        pool.generate(prompt, sampling_params) for _ in range(K)
    ]
    rollouts = await asyncio.gather(*completion_tasks)

    # Fill in reference-model logprobs in parallel.
    ref_tasks = [
        ref_pool.score(prompt, r.completion) for r in rollouts
    ]
    ref_lps = await asyncio.gather(*ref_tasks)

    # Compute rewards in parallel.
    reward_tasks = [
        reward_worker.score(prompt, r.completion) for r in rollouts
    ]
    rewards = await asyncio.gather(*reward_tasks)

    # Stamp prompt_id, group_idx, reward, ref_logprobs onto each rollout.
    for i, r in enumerate(rollouts):
        r.prompt_id = prompt_id
        r.group_idx = i
        r.ref_logprobs = ref_lps[i]
        r.reward = rewards[i]

    return RolloutGroup(prompt=prompt, prompt_id=prompt_id, rollouts=rollouts)


async def collect_step_batch(
    pool: VLLMPool,
    ref_pool: RefPool,
    reward_worker: RewardWorker,
    prompts: List[str],
    K: int,
    sampling_params,
) -> List[RolloutGroup]:
    """Collect a full batch of B prompt-groups (B*K rollouts)."""
    tasks = [
        collect_one_group(pool, ref_pool, reward_worker, p, f"p{i}", K, sampling_params)
        for i, p in enumerate(prompts)
    ]
    return await asyncio.gather(*tasks)


# ── Outer training loop ──────────────────────────────────────────────────

def training_loop(pool, ref_pool, reward_worker, learner, prompt_iter):
    sampling_params = SamplingParams(temperature=1.0, max_tokens=2048)
    K = 16
    B = 256

    for step in range(num_steps):
        # 1. Sample a batch of prompts.
        prompts = [next(prompt_iter) for _ in range(B)]

        # 2. Collect rollouts. This is where 70-90% of wall-clock goes.
        groups = asyncio.run(collect_step_batch(
            pool, ref_pool, reward_worker, prompts, K, sampling_params
        ))

        # 3. Hand to the learner. Computes group-relative advantages,
        #    PPO-clipped loss, KL term, gradient step. See Lecture 15.
        new_weights = learner.update(groups)

        # 4. Sync weights to the rollout pool.
        pool.sync_weights(new_weights)
        # ref_pool stays frozen; no sync needed.
```

This is a synchronous loop: the learner waits for rollouts, then the rollout pool waits for sync. An asynchronous version overlaps steps 2 and 3 by always having a previous batch of rollouts in flight while the current one is being trained on. The async version is more performant but more complex; the staleness analysis from the weight-sync section starts to matter.

---

## What the rollout pool's GPU is actually doing

Worth knowing what the bottleneck is at the kernel level, because it shapes which optimizations help.

During **prefill** (processing the prompt all at once), the GPU is doing one big matmul per layer over the prompt. The arithmetic intensity is high: you stream the weights from HBM once and reuse them across all prompt tokens. This is compute-bound for prompts of moderate length, and a high-throughput regime.

During **decode** (generating one token at a time), the GPU is doing one matmul per layer for a single new token. The arithmetic intensity is low: you stream the weights from HBM, do a tiny amount of compute, repeat. This is memory-bandwidth-bound. The fix is to batch many sequences together so the same weight stream is reused across, say, 64 different sequences' decode steps. Continuous batching is what makes this possible.

The decoder's KV cache also gets read at every decode step, and that read scales with sequence length. For a 4096-token completion at decode step 4000, you're reading roughly 4000 tokens worth of KV cache plus the weights. At long sequence lengths the KV reads start to dominate over weight reads, and the GPU spends most of its time on attention rather than on the FFN layers. This is why throughput drops on long sequences: you're paging more KV data per token generated.

Quantization helps by reducing the bytes-per-parameter you have to stream. Going from bf16 (2 bytes) to int8 (1 byte) roughly doubles decode throughput on memory-bound workloads, modulo the quantization quality. fp8 is similar to int8 but with a different numerical profile. Quantizing the KV cache (in addition to the weights) further helps on long sequences.

The practical implication for RL: at the sequence lengths typical of reasoning rollouts (a few thousand tokens), the GPU is mostly memory-bandwidth-bound on decode, and the biggest single throughput lever after batch size is quantization. Frameworks that combine PagedAttention + continuous batching + fp8 weights + fp8 KV cache hit roughly an order of magnitude better tokens/sec than naive batch decode on the same hardware.

---

## Putting wall-clock numbers on it

Rough order-of-magnitude breakdown for a representative training step on a 7B policy:

| Stage | Wall-clock | Notes |
|---|---|---|
| Prompt batch assembly | <1 s | data loading, sampling from dataset |
| Rollouts (4096 completions, mean 800 tokens) | 30–60 s | dominated by decode |
| Reference logprob scoring | 5–10 s | if co-located and batched |
| Reward computation (math/regex) | 1–5 s | scales linearly with completions |
| Reward computation (code with sandbox) | 30–120 s | can dominate |
| Buffer assembly + advantage computation | <1 s | numpy/torch math on a CPU |
| Gradient step | 5–15 s | bf16, bf16 optimizer state |
| Weight sync (7B, fp8 quantize + broadcast) | 2–4 s | varies with interconnect |
| **Total per step** | **45–215 s** | rollouts dominate |

For a 70B policy, multiply rollouts and gradient steps by ~5x and weight sync by ~10x, and code-execution rewards stay roughly constant. The breakdown then has rollouts at maybe 4–8 minutes per step, gradient at 1–2 minutes, sync at 30–60 seconds, and code rewards still at 30–120 seconds. Rollouts still dominate.

Tokens/sec on a single H100 for a 7B model under continuous batching with PagedAttention is often quoted in the range of 5k–15k decode tokens/sec depending on batch composition and quantization. For a 70B model it's more like 500–2000 decode tokens/sec per instance. These are the levers throughput optimization has to pull on.

---

## What can go wrong, by category

A short list of failure modes specific to this layer (failures of the algorithm itself are covered in Lectures 10 and 15).

**Rollout pool crashes mid-step.** A vLLM instance OOMs, segfaults, or hits a CUDA assertion. The orchestrator has to detect this, requeue the in-flight requests to other instances, and restart the dead instance. Without this, you lose the step and your buffer becomes inconsistent (some prompts have K samples, some have fewer). Most production trainers handle this with a retry policy and a minimum-K threshold below which the group is dropped.

**Weight sync silently fails.** A few instances pick up the new weights, others don't. Subsequent rollouts come from a mix of policies: one set fresh, one set 1 step stale. Importance ratios for the stale samples will be off. Diagnostic: hash the parameters on each instance after sync and confirm they match. If they don't, retry the sync.

**Reward worker queue grows unboundedly.** Rollouts come in faster than rewards can be computed. RAM fills up, then the worker process is killed. Usual cause: code-execution reward, slow sandbox. Cap the queue length; apply backpressure by slowing the rollout pool when the queue is full.

**Prefix cache exhaustion.** With many distinct prompts (large dataset, sampled freshly each step), the KV cache eviction policy starts thrashing. Recently-cached prefixes are evicted before they're used. Detectable by monitoring the hit rate on prefix-cache lookups. Mitigations: bigger KV cache budget, hierarchical prompt selection that reuses prompts across steps.

**Tokenizer mismatch between pool and learner.** Two different versions of the same tokenizer (a HuggingFace bump, a vocab change) silently produce different token IDs for the same text. The learner trains on one tokenization; the rollout pool generates with the other. Tokens don't match, logprobs are computed against the wrong tokens, gradients are wrong. This one is brutal because it usually doesn't crash anything: training just doesn't work. Always verify tokenizer hashes match across components.

**Async staleness explodes.** In an async setup, if the learner steps faster than the rollout pool can swap weights (say, the gradient step is much faster than expected), staleness grows. The importance ratios start clipping heavily, the gradient signal weakens, and training plateaus. Throttle the learner or accept a maximum staleness window.

---

## How this connects back to the rest

The algorithm side of PPO-LLM (Lecture 10) and GRPO (Lecture 15) defines what the gradient step needs: per-token logprobs under the sampling policy, per-token logprobs under a reference policy, and a scalar reward per completion. The infrastructure side, this lecture, is about producing all of that fast enough and in a consistent enough form that the gradient step has something to chew on.

The two interact at a few specific points:

- The KL term ties the reference policy into every gradient step. Lecture 10 introduces it; this lecture explains why caching reference logprobs at sampling time is the standard optimization.
- The importance ratio in PPO/GRPO is tolerant of small drift between sampling and training. That tolerance is what makes any of the weight-sync trade-offs (every N steps, async) workable.
- The group-relative advantage in GRPO depends on having all K samples for a prompt available before you can compute advantages. The orchestrator's job of grouping completions back into K-sample sets is downstream of that algorithmic requirement.
- Verifiable rewards (Lecture 15) push the reward computation toward code execution and symbolic checks, which are where reward latency starts to matter and the sandbox infrastructure comes in.

For agentic RL (Lecture 16, multi-turn tool use), the rollout layer gets more complex: each rollout is a sequence of tool calls and model turns, not a single completion. The same building blocks apply, but the orchestrator has to handle multi-step episodes, partial rollouts, and tool-side latency. The base infrastructure here is a prerequisite for that.

---

## The trajectory of this stack

Two things to watch for, beyond 2025:

**Disaggregated prefill and decode.** The prefill stage is compute-bound (one big matmul per layer for the whole prompt at once); decode is memory-bandwidth-bound (one small matmul per token, with massive HBM traffic for the weights). They benefit from different hardware. Several systems are starting to split them onto different GPU pools, with the KV cache shipped between them. For RL workloads with shared prefixes across K samples, this is a natural fit: one prefill, K decodes.

**RL-specific serving systems.** vLLM and SGLang were built for general LLM serving. There's room for serving systems built specifically for RL: ones that hold the policy and the reference model in the same engine, expose logprobs natively, manage weight sync as a first-class operation, and make K-sampling efficient. As of late 2025 several projects in this space exist but none has clearly taken the lead. Worth checking what's current before committing.

**LoRA-based RL at scale.** If the policy update is a LoRA adapter rather than full-parameter, weight sync collapses from gigabytes to megabytes. Rollout instances can hold the base weights once and hot-swap adapters cheaply. The training dynamics are different (the model's update capacity is constrained) but the infra is much simpler. Many production setups are heading this way.

---

## Recap

The dominant cost in LLM RL training is rollout generation, not the gradient step. The infrastructure that makes rollouts fast (PagedAttention for KV cache management, continuous batching for handling variable-length completions, prefix-sharing for repeated prompts, pools of inference instances behind an orchestrator) is what determines training throughput in practice.

Speculative decoding speeds up sampling but interacts badly with the policy gradient's need for exact sampling logprobs. The reference model adds a second inference workload, usually handled by caching its logprobs at sampling time. Weight synchronization between the learner and the rollout pool is a knob that trades off staleness against overhead. Reward computation at scale is its own beast, especially for code execution.

Quantization at sampling time (fp8/int8) introduces a small distortion from the bf16 training policy that is usually absorbed by PPO clipping but worth being aware of when debugging instability.

Get the rollout infrastructure right and the rest of RL training becomes tractable. Get it wrong and you'll spend most of your compute waiting.

---

## Where to look next

- **[Lecture 10](./10-ppo-for-llms.md)** has the PPO-LLM algorithm this infrastructure feeds.
- **[Lecture 15](./15-rl-verifiable-rewards.md)** has GRPO, which is the dominant algorithm pairing with this infrastructure as of 2025.
- **[Lecture 16](./16-agentic-rl.md)** extends the rollout model to multi-turn tool-use; the orchestrator becomes much more complex.
- **vLLM**: https://github.com/vllm-project/vllm, the most common rollout backend.
- **SGLang**: https://github.com/sgl-project/sglang, alternative with stronger prefix-cache reuse.

---

## References

**Kwon et al. (2023)**: "Efficient Memory Management for Large Language Model Serving with PagedAttention." arXiv:2309.06180. The vLLM paper; PagedAttention and the open-source serving system.

**Yu et al. (2022)**: "Orca: A Distributed Serving System for Transformer-Based Generative Models." OSDI 2022. Iteration-level (continuous) batching for autoregressive serving.

**Zheng et al. (2023)**: "SGLang: Efficient Execution of Structured Language Model Programs." arXiv:2312.07104. RadixAttention for KV-cache prefix sharing across requests with shared intermediate prefixes.

**Leviathan et al. (2022)**: "Fast Inference from Transformers via Speculative Decoding." arXiv:2211.17192. The original speculative-decoding formulation with a small draft model.

**Chen et al. (2023)**: "Accelerating Large Language Model Decoding with Speculative Sampling." arXiv:2302.01318. DeepMind's concurrent independent formulation, applied to Chinchilla 70B.

---

## Debugging notes

A short list of things to print when rollout throughput is bad:

- **Active batch size on each vLLM instance**, averaged over a step. If it's well below the configured maximum, you're not feeding the engine fast enough; the orchestrator may be dispatching too sequentially.
- **Prefix-cache hit rate.** Should be high when K samples share a prompt; if it's low, either the cache is too small or eviction is too aggressive.
- **Distribution of completion lengths.** A long tail (mean 800, p99 4096) means continuous batching is critical; a flat distribution means it matters less.
- **Time to first token vs time per output token.** Disaggregating these tells you whether prefill or decode is the bottleneck.
- **Reward worker queue depth over time.** Should fluctuate around zero, not grow.
- **Weight sync wall-clock.** If it's growing as a fraction of step time, you may be syncing too often.
- **Tokenizer hash on each component.** Once, at startup, before anything goes wrong.

Most of the time when rollouts feel slow, the answer is one of: too small a batch reaching each vLLM instance, prefix cache thrashing, or a reward worker bottleneck.
