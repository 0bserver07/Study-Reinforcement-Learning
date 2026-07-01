<!-- status: unreviewed | last-reviewed: never -->

# Lecture 31: Hardware for RL

_Unreviewed — no one has checked this end to end. Treat the math, code, and citations as unverified._

**Time**: ~2 h · **Prerequisites**: Lecture 06 (any prior exposure to GPU training helps)

---

## Where this fits

Almost every other lecture in this series treats compute as a free variable. You write `loss.backward()` and assume the hardware sorts itself out. That works fine through Lecture 08 — CartPole and small MuJoCo tasks run on a laptop in minutes. It breaks the moment you try to run GRPO on a 7B model, never mind a 70B one.

This lecture is the missing piece: what actually constrains an RL run, why, and what to choose. There are two regimes, and they have almost nothing to do with each other.

The first is **classical RL** — Atari, MuJoCo, Procgen, the things in Lectures 02–08. The environment runs on CPU. The policy is small. Compute is dominated by stepping the environment, not by anything on the GPU. A workstation with a fast CPU and a modest GPU is enough; you don't need a cluster, you don't need an H100, you mostly need cores.

The second is **LLM-RL** — RLHF, DPO, GRPO with a 7B+ model. Rollouts and gradient updates both saturate the accelerator. The binding constraint shifts from CPU stepping to GPU memory bandwidth (during rollouts) and FLOPs plus interconnect (during gradient updates). Hardware choices here are nontrivial, the run takes days to weeks, and you can waste a lot of money getting it wrong.

The split is sharp enough that almost any "hardware for RL" advice that doesn't first ask which regime you're in is wrong. Most online discussion is about LLM-RL because that's where the money is, but if you're working through Block 1 of the curriculum, none of it applies to you.

---

## Regime 1: classical RL (env-step bottlenecked)

### What's actually slow

Take CartPole. The policy is a 2-layer MLP with maybe 10K parameters. A forward pass on CPU is microseconds; on a GPU, it's still microseconds plus the kernel-launch overhead. Stepping the environment — computing physics, applying the action, checking termination — is also microseconds, but it has to run sequentially per env: the state at step t+1 depends on the action at step t.

If you run one env at a time, you spend most of your wallclock waiting for `env.step()` to return. The GPU is idle 99% of the time. You can verify this by running `nvidia-smi dmon -s u` during a single-env REINFORCE loop; GPU utilization will sit near 0%.

The fix is **vectorized environments**: run N envs in parallel, each at a different state, then forward-pass the batched observations through the policy in one call. The policy compute amortizes over N envs; the per-step env work is what scales.

A few sample numbers (rough, your mileage will vary):

- Single-env CartPole on CPU: ~5,000–10,000 steps/sec.
- 64-env vectorized CartPole on CPU: ~100,000–300,000 steps/sec.
- 256-env vectorized CartPole on a fast CPU: can hit ~1M steps/sec with the right runner.

(Verify on your own box — these are not benchmark numbers from a paper, they're the order of magnitude you should expect.)

The right tool for this is one of the vectorized-env libraries:

- **Gymnasium** ([gymnasium.farama.org](https://gymnasium.farama.org/)) — the maintained fork of OpenAI Gym. Has `AsyncVectorEnv` and `SyncVectorEnv`. Good for everything in Block 1.
- **EnvPool** (Weng et al. 2022, arXiv:2206.10558) — C++ vectorized env runner. For Atari and MuJoCo, it's noticeably faster than `AsyncVectorEnv` because it avoids the Python GIL on the stepping loop. Reasonable to reach for once you're past simple tasks.

There's a subtle point about `AsyncVectorEnv`: it spawns subprocesses, each running its own copy of the env, and communicates via pipes. The overhead of pickling observations across processes can eat the parallelism gains for very fast envs (CartPole is on the edge). `SyncVectorEnv` runs the envs in a single process and is sometimes faster for trivial envs. For Atari (where each env step is heavier) async wins clearly.

### Where the GPU fits

In classical RL, the GPU is doing:

- Policy forward pass on the batched observations (cheap; the batch is small and the network is tiny).
- Gradient update on a small batch of trajectories (also cheap; PPO with a 1M-param policy and a 64K-step rollout fits in maybe 100 MB).

A single consumer GPU — a 3080, 4090, or even a 3060 — handles all of this comfortably. The thing you actually want to spend money on is **CPU cores**: 16–32 cores at high clock will run more envs in parallel than any GPU upgrade will. RAM matters less unless you're using huge replay buffers (off-policy methods, Lecture 07; even then 64 GB is usually plenty).

The mistake to avoid: provisioning an H100 for an Atari run. The H100 is bandwidth- and compute-monstrous for a network that's 99% idle, and you pay $30/hr (rough rule-of-thumb cloud price) for it to twiddle thumbs while your CPU strangles.

### Concrete advice for Block 1

A reasonable workstation:

- 16+ CPU cores (Ryzen 9 / Threadripper / a comparable Intel chip)
- 32–64 GB RAM
- One consumer GPU (4070 / 4080 / 4090, or any data-center card you already have)
- An NVMe SSD if you're going to log a lot

Cloud equivalents: any general-purpose instance with a T4 or A10G attached works; you don't need anything bigger. For learning, [Colab](https://colab.research.google.com/) free tier or a single g4dn.xlarge on EC2 is sufficient through Lecture 08.

If you find yourself looking at H100 price lists for an Atari run, stop. Wrong regime.

---

## Regime 2: LLM-RL (model-compute bottlenecked)

This is everything from Lecture 10 onward — PPO for language models, DPO, GRPO, the RLVR loops in Lecture 15. The model is large enough that the policy forward pass is no longer free; rollouts (autoregressive decoding) are the bulk of the wallclock, and gradient updates are the bulk of the FLOPs. Both saturate the accelerator. The classical-RL playbook of "more CPUs" does nothing for you.

The hardware question now becomes: HBM bandwidth, HBM capacity, FLOPs at the right precision, and interconnect. We'll work through each.

---

## The arithmetic-intensity argument

Before talking about specific chips, the single most useful concept in this regime is **arithmetic intensity** — the ratio of FLOPs done per byte of memory read. A kernel with high arithmetic intensity is **compute-bound**: speed is set by the chip's FLOPs/sec. Low intensity is **memory-bound**: speed is set by HBM bandwidth.

For a dense matmul with weights of size N and a single input vector of size d, the work is roughly:

- FLOPs: 2 · N (every weight gets one multiply-add per input element).
- Memory reads: N parameters × 2 bytes/parameter (BF16) = 2N bytes.
- Arithmetic intensity: 1 FLOP per byte.

That's terrible. With 1 FLOP/byte and HBM bandwidth around 3 TB/s, you're capped at 3 TFLOPs. The H100 can do 989 BF16 TFLOPs (NVIDIA H100 datasheet; the 1,979 number you see online is with 2:4 sparsity). The kernel will use ~0.3% of the chip's compute. Memory bandwidth is the binding constraint and the chip mostly sits idle.

This is what **autoregressive decode** looks like. Each generated token does one forward pass of the model with a batch of one (the new token). For an N-parameter model, that's 2N FLOPs done while reading N parameters from HBM. Memory-bound. You will not improve decode speed by buying a beefier-compute chip; you will improve it by buying a chip with more HBM bandwidth, or by batching more sequences together so the same N-parameter read amortizes over more FLOPs.

Now compare **prefill** — processing a B-token prompt to fill the KV cache. Work is 2 · N · B FLOPs, memory reads are still N (you read the weights once for the whole batch). Arithmetic intensity is now B FLOPs/byte. At B=512, you're at 512 FLOPs/byte, which the H100 can sustain comfortably; the chip becomes compute-bound. Prefill is fast per token because the matmul shape is right.

This asymmetry — **prefill is compute-bound, decode is memory-bound** — is the single most important fact about LLM serving. It applies directly to RL because RL rollouts are dominated by decode, not prefill: you have a short prompt and a long generation. The cost structure of an RL run is therefore set by HBM bandwidth, not by peak FLOPs.

The same FLOP-accounting framework is used in Hoffmann et al. 2022 (the Chinchilla paper, arXiv:2203.15556) for training-compute estimation — `C ≈ 6ND` for a model with N parameters trained on D tokens. The "6" is 2 for the forward pass plus ~4 for the backward pass. That formula is for training and assumes you're compute-bound (large batch). Decode is the opposite regime — small batch, memory-bound — and the FLOP count alone doesn't predict wallclock.

### Why this matters for RL specifically

A GRPO step (Lecture 15) does, per training iteration:

1. Sample K completions for each of B prompts (rollouts — decode-heavy).
2. Compute log-probs under the current policy on those K · B completions (prefill, since you have the full sequence in hand).
3. Compute log-probs under the reference policy (same).
4. Compute the loss and do a gradient step.

Steps 1, 2, 3 each touch every model parameter at least once. Step 1 is N · 2T FLOPs read at decode bandwidth (T tokens decoded). Steps 2 and 3 are N · T FLOPs each at prefill bandwidth (one shot per sequence). Step 4 is N · 6T FLOPs at training bandwidth — but spread across the whole batch.

The decode in step 1 is what dominates wallclock. If you can speed up decode by 2x — through a better kernel, a quantized rollout policy, paged attention to fit more sequences in HBM at once — you roughly halve the iteration time. Speeding up the gradient update by 2x doesn't help nearly as much, because the gradient update wasn't the bottleneck.

This is why the LLM-RL infrastructure stack looks the way it does: separate **rollout engines** (vLLM, SGLang, TensorRT-LLM) optimized for decode throughput, separate **training engines** (PyTorch FSDP, DeepSpeed, Megatron) optimized for gradient updates, and an orchestration layer (DeepSpeed-Chat, OpenRLHF, TRL, verl) gluing them together. See Lecture 30 for the rollout-engine side (this isn't a stable reference yet — Lecture 30 may not exist or may have changed; the relevant ideas are PagedAttention and continuous batching).

---

## Accelerators in 2024–2025

What's actually available. I'll give the spec numbers I can verify; treat anything not verified as "I haven't checked the datasheet, look it up before quoting it."

### NVIDIA H100

Released 2022. The 2024 default for serious LLM work. SXM5 form factor, 80 GB HBM3, 3.35 TB/s memory bandwidth, 989 BF16 TFLOPs dense (1,979 with 2:4 sparsity), 1,979 FP8 TFLOPs dense (3,958 with sparsity). NVLink at 900 GB/s within a node. Source: NVIDIA H100 datasheet on nvidia.com.

The H100 is what most published RLHF and GRPO runs are done on in 2024. If you want one number to anchor against, it's: 3 TB/s HBM bandwidth, 1 TB of HBM in an 8-GPU node (80 GB × 8 ÷ 1024 ≈ 0.64 TB, so actually closer to 640 GB), and 900 GB/s NVLink within the node.

### NVIDIA H200

A refresh of the H100 with more memory and more memory bandwidth, same compute. 141 GB HBM3e, 4.8 TB/s. Released 2024. Source: NVIDIA H200 page on nvidia.com.

The H200 is interesting for RL specifically because decode is memory-bound: 4.8/3.35 ≈ 1.43x the bandwidth at roughly the same FLOPs means decode throughput goes up by ~1.4x for free, and you can fit larger KV caches without offloading. For training (which is compute-bound) the speedup is small.

### NVIDIA Blackwell (B100/B200) and GB200

Announced March 2024, deliveries through 2024–2025. The B200 is a dual-die GPU with HBM3e, FP4 Tensor Cores, and a second-generation Transformer Engine. NVIDIA's announcement (nvidianews.nvidia.com, March 2024) describes "208 billion transistors" and "4-bit floating point AI inference" but the press release does not list specific HBM/bandwidth/FLOPs numbers I could verify directly — be skeptical of any specific spec I quote and look at the most recent product page.

FP4 is the headline. For decode (memory-bound), FP4 weights halve the memory traffic compared to FP8 and quarter it compared to BF16, which roughly doubles or quadruples decode throughput per dollar — if your model tolerates the quantization. Quantization quality for FP4 is an active research area as of 2025; expect some accuracy loss on edge tasks. For RL rollouts, where the rollout policy doesn't need to be the same precision as the training policy, this is potentially a big lever.

### NVIDIA GH200 (Grace Hopper)

A Hopper GPU paired with an ARM-based Grace CPU on the same board, connected by NVLink-C2C at 900 GB/s — meaning the GPU can read CPU memory at the same bandwidth as it reads its own HBM (just less of it). The Grace side has LPDDR5X, hundreds of GB. Source: NVIDIA Grace Hopper page; the page lists 900 GB/s C2C and a combined up-to-624 GB of fast memory.

For RL, the relevance is **offload**. If you can keep cold weights or large optimizer states on the CPU side and stream them across NVLink-C2C, you can fit much larger models in the same compute budget. Good for cases where memory rather than compute is the binding constraint.

### AMD MI300X

AMD's competitor to the H100. 192 GB HBM3 (verify on AMD's product page — the AMD site timed out for me, so treat this number as worth re-confirming), high memory bandwidth, FP8 and BF16 Tensor Core support. ROCm software stack — workable, but the PyTorch ecosystem still has more sharp edges on ROCm than on CUDA, especially for custom kernels.

If you're not infrastructure-constrained, the MI300X's main appeal is **more HBM per chip**. For a 70B model that needs to fit in HBM with a large KV cache, 192 GB on one chip means you can avoid sharding (no tensor parallel) and skip the NCCL all-reduce, which can win on simplicity even if peak FLOPs are lower. The catch: the kernel libraries you'd reach for on NVIDIA (FlashAttention, custom Triton kernels) may need adaptation or may not be as well-optimized on ROCm.

### Google TPU v5p, v5e, Trillium

TPUs are a different world — JAX/PyTorch-XLA, different memory model, different topology (3D torus rather than NVLink mesh). I haven't run RL on TPUs personally; the public picture in 2024:

- **TPU v5p**: training-focused. 95 GB HBM per chip, 2.575 TB/s HBM bandwidth, 459 BF16 TFLOPs per chip. Source: Google Cloud v5p docs page. Scales to "pods" of thousands of chips.
- **TPU v5e**: inference- and small-training-focused. Less HBM, lower bandwidth, cheaper.
- **Trillium (TPU v6)**: announced 2024, broader availability through 2025. Higher BF16 throughput and HBM bandwidth than v5p; I don't have spec numbers I can confidently quote from a primary source for this draft.

TPUs are typically used by Google internal projects and by external customers who run on GCP. Most open-source RL infrastructure (TRL, OpenRLHF, verl) is NVIDIA-first; running on TPU usually means either using JAX directly or accepting some friction.

### Quick comparison table

A rough cheat sheet for 2024–2025. **Verify any number you intend to cite externally — these are starting points, not authoritative.**

| Chip | HBM cap | HBM BW | BF16 TFLOPs (dense) | FP8/FP4 | Interconnect | Source |
|---|---|---|---|---|---|---|
| H100 SXM5 | 80 GB | 3.35 TB/s | 989 | 1,979 FP8 | 900 GB/s NVLink | NVIDIA datasheet |
| H200 SXM | 141 GB | 4.8 TB/s | 989 (same) | 1,979 FP8 (same) | 900 GB/s NVLink | NVIDIA page |
| B200 | ~192 GB HBM3e (verify) | ~8 TB/s (verify) | ~? | FP4 Tensor Cores | NVLink 5 | NVIDIA announcement Mar 2024 — specs partially unverified |
| GH200 | up to 624 GB combined | varies | same as H100 GPU | same | 900 GB/s C2C + NVLink | NVIDIA page |
| MI300X | 192 GB (verify) | ~5.3 TB/s (verify) | — | FP8 support | Infinity Fabric | AMD page — couldn't verify directly |
| TPU v5p | 95 GB | 2.575 TB/s | 459 | INT8 supported | 3D torus | Google v5p docs |

The pattern: HBM capacity is going up faster than HBM bandwidth, which is going up faster than dense FLOPs. That's exactly the trend you'd expect if memory is the binding constraint for the workloads people care about (inference, RL rollouts). FP4/FP8 are how the FLOPs number keeps growing on the chart even as dense BF16 doesn't.

---

## Memory hierarchy and where it bites

There are four kinds of memory in a typical accelerator setup and they have wildly different bandwidth:

1. **HBM** — on-package high-bandwidth memory. ~3 TB/s on H100, ~5 TB/s on H200, similar order on MI300X. Where weights and activations live.
2. **L2 cache** — on-die SRAM, ~50 MB on H100. ~10 TB/s. Used by kernels for staging.
3. **Shared memory / SMEM** — per-SM scratchpad, ~228 KB per SM on H100. ~30+ TB/s effective. Where FlashAttention-style kernels do their tiling.
4. **Registers** — per-thread, ~256 KB total per SM. The fastest memory, used for the innermost loops.

A kernel that's slow because of "memory bandwidth" almost always means HBM bandwidth, not the other levels. Optimizing a kernel is often about keeping data in SMEM or registers and only touching HBM at the start and end (this is exactly the FlashAttention story below).

### KV cache: the thing that ate your HBM

For an autoregressive LLM, generating a token requires reading the cached keys and values from every previous token at every attention layer. This cache scales as:

```
KV cache size = 2 (K and V) × batch × seq_len × num_layers × num_heads × head_dim × dtype_bytes
```

For a 70B model with, say, 80 layers, 64 heads, and 128-dim heads, at BF16 (2 bytes), the per-token KV memory is:

```
2 × 80 × 64 × 128 × 2 = 2,621,440 bytes/token ≈ 2.5 MB/token
```

So a single sequence with 4096 generated tokens is ~10 GB of KV cache. A batch of 32 such sequences is 320 GB — more than fits in an H100 (80 GB) or even in eight of them combined (640 GB) when you also need to fit the model weights (~140 GB for a 70B model in BF16).

This is the whole reason **PagedAttention** (Lecture 30; Kwon et al. 2023, arXiv:2309.06180) exists. It manages KV cache in fixed-size blocks (like virtual memory pages) rather than contiguous per-sequence buffers, which lets you pack more sequences into the same HBM and avoid fragmentation. For RL rollouts where you want to run as many parallel rollouts as possible, this is essentially mandatory above ~7B model size.

The arithmetic that matters: at long sequences and large batch, **KV cache can easily exceed the size of the weights**. A 70B BF16 model is ~140 GB of weights; 64 parallel sequences at 8K tokens of KV cache is ~640 GB, more than 4x the weight memory. Whoever bought the chip is paying for HBM bandwidth that's mostly being used to shuffle KV around, not to do matmuls.

### Why H200 over H100 for rollouts

Concretely: the H200's 141 GB lets you fit larger KV caches per chip, which means more parallel rollouts, which means higher decode throughput in tokens/sec, which means faster RL iteration. The 4.8 TB/s bandwidth (vs 3.35) also helps decode directly. For a rollout-heavy GRPO run on a 70B model, the H200 is meaningfully better than the H100 even at the same FLOPs. For a training-only workload (compute-bound), the gain is smaller.

### A worked KV-cache example

Take a concrete configuration. Llama-3-70B has 80 layers, 64 query heads, 8 KV heads (it's a grouped-query attention model), and head_dim 128. At BF16:

```
per-token KV = 2 (K, V) × 80 layers × 8 KV heads × 128 dim × 2 bytes
             = 327,680 bytes/token
             ≈ 0.31 MB/token
```

(Note the 8 KV heads — for grouped-query attention the K and V are shared across query-head groups, so they're stored only 8 times per layer, not 64. This is a substantial KV cache saving vs. classical multi-head attention, and is one of the explicit motivations for GQA in models that need to support long context.)

So a single 8K-token sequence carries 8192 × 0.31 MB ≈ 2.5 GB of KV cache. A rollout batch of 32 such sequences is ~80 GB — fills an entire H100 by itself before the weights are loaded. On an H200 (141 GB) you have room for the same 32-sequence batch plus the 140 GB of weights at... wait, you don't. 140 + 80 > 141. You need tensor parallel across at least 2 H200s.

This is the kind of arithmetic that breaks the moment you start scaling. The takeaway: figure out the per-token KV size early, plan for it, and don't be surprised when "fits a 70B model" turns out to mean "fits the weights but not the KV cache for the batch size I wanted."

For models without GQA (or where every layer is a query head), multiply the KV size by the ratio (num_query_heads / num_kv_heads). A model that's "the same parameter count" but uses MHA instead of GQA can have 4–8x larger KV cache, with all the throughput consequences that implies.

---

## Bandwidth-vs-compute regimes, in a picture

A small ASCII map of where common operations sit:

```
                   COMPUTE-BOUND
                   (more FLOPs = faster)
                          │
                          │
                          │  ● large-batch matmul
                          │  ● prefill (B >> 1)
                          │  ● training fwd+bwd
                          │
                          │  ● attention (large seq)
                          │
─────────────────────────●●────────────────────► arithmetic intensity
                          │  ● small-batch matmul
                          │  ● activation funcs
                          │  ● decode (batch=1)
                          │  ● optimizer step (Adam)
                          │  ● all-reduce (NCCL)
                          │
                          │
                   MEMORY-BOUND
                   (more BW = faster)
```

The fact that decode and the optimizer step are both memory-bound is part of why people fold the optimizer state into FSDP shards (Lecture-not-yet-written, ZeRO from Rajbhandari et al. 2019/2020) — the optimizer step is bandwidth-limited so you want to overlap it with compute on other ranks.

For RL: rollouts are decode → memory-bound. Gradient updates are training fwd+bwd → mostly compute-bound for the matmuls, memory-bound for the optimizer step. The mix means both HBM bandwidth and FLOPs matter, but for a typical GRPO run where rollouts are 60–80% of the wallclock, bandwidth dominates.

---

## Kernels worth knowing

A kernel is a piece of GPU code that runs as a single launch. For LLM workloads, attention and matmul kernels dominate runtime, and the choice of kernel implementation can swing throughput by 2–4x.

### FlashAttention

The single most important kernel-engineering result for LLMs. **FlashAttention** (Dao et al. 2022, arXiv:2205.14135) computes attention with O(N) memory rather than O(N²) by tiling the attention matrix and computing softmax incrementally in SMEM. The standard formulation materializes the full N×N attention matrix in HBM, which both blows up memory and adds HBM read/write traffic; FlashAttention never writes the matrix at all.

The result: for long sequences (4K+), FlashAttention is 2–4x faster than naive attention and uses dramatically less memory. The longer the sequence, the bigger the win, because the N² term in the standard kernel scales worst.

There are now three versions:

- **FlashAttention** (Dao, Fu, Ermon, Rudra, Ré 2022, arXiv:2205.14135) — original.
- **FlashAttention-2** (Dao 2023, arXiv:2307.08691) — better parallelism over work partitioning, ~2x faster than v1.
- **FlashAttention-3** (Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao 2024, arXiv:2407.08608) — Hopper-specific (uses async tensor cores and FP8), another ~1.5–2x on H100.

All three are in the `flash-attn` Python package. Using them in PyTorch is roughly this:

```python
import torch
from flash_attn import flash_attn_func

# Q, K, V each of shape [batch, seqlen, num_heads, head_dim], BF16 or FP16.
batch, seq, heads, dim = 4, 8192, 32, 128
q = torch.randn(batch, seq, heads, dim, dtype=torch.bfloat16, device="cuda")
k = torch.randn_like(q)
v = torch.randn_like(q)

# Causal attention, dropout=0. Returns shape [batch, seq, heads, dim].
out = flash_attn_func(q, k, v, causal=True)
```

That's the whole API for the common case. Behind it are several thousand lines of CUDA / CUTLASS. The packaging is good enough that you usually don't think about it once `flash-attn` is installed; PyTorch's `F.scaled_dot_product_attention` will also dispatch to FlashAttention on a compatible backend, so you may already be using it without knowing.

The reason this kernel matters for RL: every rollout token does one attention computation per layer, and every training step does a full attention forward and backward. Long context (8K+) is common in RL with reasoning traces (Lecture 15: R1 chains run thousands of tokens). Without FlashAttention you'd be HBM-bound on attention specifically, on top of the existing decode bandwidth problem. With it, attention becomes a much smaller fraction of decode wallclock and the bottleneck shifts to weight-matrix reads, which is what the memory-bandwidth regime above is about.

### PagedAttention

Covered in Lecture 30; the relevant idea here is that KV cache memory is managed in fixed-size blocks (typically 16 tokens per block) rather than contiguous per-sequence buffers. This lets you pack more sequences into HBM without fragmentation losses, and it's what vLLM is built on. Kwon et al. 2023, arXiv:2309.06180.

For RL, PagedAttention matters because rollouts do a lot of generation in parallel and the KV cache layout determines how many parallel sequences fit. Higher parallelism per GPU means higher rollout throughput.

### Triton

A DSL embedded in Python for writing GPU kernels at roughly numpy-level abstraction, compiling to PTX through MLIR/LLVM. Originally by Tillet, Kung, Cox (MAPL 2019 workshop at PLDI — I tried to verify the publisher page but it returned 403; the paper is widely cited and the citation should be straightforward to find). OpenAI maintains it now ([triton-lang.org](https://triton-lang.org/)).

Why care: a Triton kernel is maybe 100 lines of Python-like code where the equivalent CUDA might be 1000 lines of C++. For custom RL kernels — fused reward computation, custom advantage estimators, packed-sequence layouts — Triton is the right tool. Several papers in 2024 wrote their core kernels in Triton (e.g., quantization kernels, custom attention variants) because the iteration speed is much faster than CUDA.

A Triton kernel skeleton looks like:

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

You launch it with a grid size and let the compiler figure out warps, threads, and memory access patterns. It's not as fast as a hand-tuned CUTLASS kernel on a specific GEMM, but it's good enough for almost everything and dramatically cheaper to write.

### Other kernel libraries

- **CUTLASS** — NVIDIA's open-source GEMM template library. Lower-level than Triton, used to build the fastest matmul kernels (cuBLAS uses it). You usually don't write CUTLASS directly unless you're a kernel engineer.
- **xFormers** — Meta's kernel library, includes attention variants. Largely superseded by FlashAttention and PyTorch's built-in SDPA for most use cases.
- **ThunderKittens** — a more recent (2024) DSL/library out of Stanford (Hazy Research lab) for writing attention-style kernels at a higher level. Less mature than Triton; worth knowing about. I haven't verified a paper for this; treat the citation as "look it up."

For 99% of RL work you don't need to write a kernel. The exceptions are when you have a specific operation that's a hot spot (e.g., a packed-sequence reward computation that doesn't fit any standard kernel) and dropping into Triton can win you a noticeable fraction of throughput.

---

## Quantization for rollouts

Rollouts (decode) are memory-bound. Quantization reduces the bytes per parameter, which directly speeds up decode. The math is straightforward: at 4-bit weights, you read N/2 bytes instead of 2N bytes (vs BF16), so memory traffic drops 4x and (in the bandwidth-bound regime) decode throughput goes up by ~4x. The catch is whether your model still produces good outputs at 4-bit.

Common quantization approaches for inference:

- **FP8** (on H100+, via Hopper Transformer Engine). 1 byte/param. Often essentially free for inference if the model was trained robust to FP8, modest accuracy loss otherwise. The H100 has native FP8 Tensor Cores, so the win is on both axes — less memory traffic *and* more FLOPs.
- **INT8 / INT4 weight-only quantization (PTQ)**. The two main flavors:
  - **GPTQ** (Frantar, Ashkboos, Hoefler, Alistarh 2022, arXiv:2210.17323) — second-order post-training quantization. Solves a per-layer reconstruction problem to find quantized weights that minimize output error.
  - **AWQ** (Lin, Tang, Tang, Yang, Chen, Wang, Xiao, Dang, Gan, Han 2023, arXiv:2306.00978) — activation-aware: protect the 1% of weights that matter most based on activation magnitudes.
- **NF4** (NormalFloat 4) and similar — used in QLoRA-style fine-tuning. Quantizes weights to a non-uniform 4-bit code calibrated to a normal distribution.
- **FP4** (Blackwell-native) — H100 doesn't have it; Blackwell does, with hardware Tensor Core support.

For RL, the typical pattern is **mixed-precision rollouts and training**: keep the trainable policy in BF16 or FP8 for the gradient pass, but run rollouts with the quantized policy (INT4/FP8). Two complications:

1. The rollout policy and training policy now have different precisions, so you have to be careful that the log-probs used in the PPO/GRPO ratio are computed under the right one (usually you take log-probs under the training-precision policy on both sides, even though sampling happened with the quantized one).
2. The quantized policy needs to be updated as the training policy changes. Re-quantizing every step is expensive; people often update the rollout policy every K steps (where K is 4 or 8 or higher) and accept the slight off-policy-ness.

The DeepSeek-R1 paper (Lecture 15; arXiv:2501.12948) doesn't go into rollout-precision details, but more recent RL infrastructure papers and the open-source RLHF stacks (OpenRLHF, verl) implement variations of this.

---

## Interconnect: NVLink, InfiniBand, NCCL

The model and gradient state for a 70B run don't fit on one GPU, so you have multiple GPUs and they need to communicate. Two scales of interconnect:

### Within a node (NVLink)

8 H100s in a typical SXM node are connected by NVLink at 900 GB/s per GPU. That's high enough that intra-node all-reduce for gradient sync is fast — milliseconds for a 70B model's parameters.

For RL, the relevant within-node operations are:

- **Tensor parallel** — shard the matmul weights across GPUs in the node. All-reduce after each layer. Bandwidth-heavy but works at NVLink speeds.
- **Pipeline parallel** — different layers on different GPUs. Less bandwidth, more latency-sensitive.
- **FSDP all-gather/reduce-scatter** — for sharded data parallelism. Each step gathers the weights, runs a forward+backward, then reduce-scatters the gradients.

### Across nodes (InfiniBand or Ethernet)

Between nodes (say, two 8-H100 nodes for a 16-GPU run), the bottleneck is the network. Typical: 400 Gb/s InfiniBand per node, sometimes 800 Gb/s in modern clusters. That's 50 GB/s — much slower than NVLink.

For gradient sync across nodes, NCCL all-reduce time scales with the model size and the slowest network link. For a 70B model in BF16, you're reducing ~140 GB of gradients. At 50 GB/s effective, that's ~3 seconds per all-reduce. Across many iterations this adds up; this is why people pack as much compute as possible into a single high-NVLink node before scaling out.

### RL-specific topology: actor pool vs learner

A common geometry in 2024 RL infrastructure: one group of GPUs runs rollouts (the "actor pool" or "rollout engine"), another runs gradient updates (the "learner"). They communicate twice:

1. **Weights flow from learner to actors** when the learner takes a step. Broadcast operation, size ~weight count × precision. For a 70B BF16 model, ~140 GB to broadcast.
2. **Rollouts flow from actors to learner** as completed sequences. Variable size, much smaller than weights — a batch of completions is maybe a few MB.

The 140 GB weight broadcast is the expensive operation. To avoid doing it every step, you typically update the rollout policy every N steps (N=4 or 16) and accept that the rollouts are slightly off-policy relative to the current learner weights. PPO and GRPO are designed to tolerate some off-policy-ness (that's what the importance ratio in the loss is for — Lecture 06), so this is a deliberate, tunable tradeoff.

Some implementations (verl, OpenRLHF) push this further: the actors and learner share GPU memory and the "broadcast" is actually a memcpy on the same NVLink fabric. Others run actors on a separate cluster entirely, paying the InfiniBand cost. The right choice depends on how much you care about GPU utilization vs. infrastructure simplicity.

### A picture of the geometry

The actor-pool / learner split, sketched:

```
        ┌────────────────────────────────┐
        │     learner (8–32 H100s)       │
        │  ────────────────────────────  │
        │  FSDP/Megatron shards          │
        │  - policy weights              │
        │  - reference policy            │
        │  - optimizer state (Adam)      │
        │  - activations during fwd+bwd  │
        └──────┬─────────────────────┬───┘
               │                     │
   weights to  │                     │  gradients computed locally,
   broadcast   │                     │  not sent (no parameter server)
   every N     ▼                     ▲
   learner    ┌─── NVLink / IB ──────┴───┐
   steps      │                          │
              │   actor pool (8–32 H100s)│
              │  ────────────────────────│
              │  vLLM/SGLang/TensorRT-LLM│
              │  - paged KV cache        │
              │  - parallel rollouts     │
              │  - tokenizer, sampler    │
              └──────────────┬───────────┘
                             │
              rollouts (prompts + completions + log-probs)
                             ▼
                  back to learner for the next gradient step
```

The vertical axis here is roughly the iteration loop; the horizontal is the GPU cluster. In a co-located setup the two boxes share the same NVLink fabric; in a distributed setup they're on separate clusters and connected by InfiniBand. The width of the "weights broadcast" arrow (in bytes) is the main thing that limits how fresh the actors' policy can be — and therefore how off-policy your rollouts are when the learner uses them.

---

## Cost realism for a 70B RLHF/GRPO run

Rough order of magnitude for a 70B-parameter RL run in 2024–2025, based on the configurations described in published recipes (DeepSeek-R1, various OpenRLHF examples). Take these as ballpark, not budget figures.

- **Actor pool**: 8–32 H100s for sampling. Sized to keep rollouts fast enough that the learner isn't waiting.
- **Learner**: 8–32 H100s for gradient updates. Sized to fit the 70B model with optimizer states (~280 GB for Adam in FP32 optimizer states, less with mixed precision) plus activations.
- **Wall clock**: typical RLHF or GRPO runs take multi-day to multi-week. Faster with more compute; not arbitrarily so (some operations are serial across iterations).

I'm not going to invent dollar figures. Cloud H100 pricing varies wildly depending on provider, commitment, and current market — anywhere from a few dollars per GPU-hour to over thirty. A reservation contract changes the math entirely. If you want a real number, get a quote from your cloud provider for the configuration you actually plan to use; anything I write here will be wrong by a factor of 3 by the time you read it.

A practical realism check: if you're learning, **do not run a 70B GRPO experiment from scratch**. The reproducibility window is narrow, the infrastructure overhead is large, and you will not learn the algorithm faster than by doing the toy version on a 1.5B or 7B model. The exercises in `exercises/15-grpo-rlvr/` (planned) use a much smaller setup deliberately.

---

## How to tell which regime you're in

If you don't know whether you're compute-bound or memory-bound, you don't know what to upgrade. A few tools to figure it out.

### `nvidia-smi dmon -s u`

Streams GPU utilization at 1 Hz. The "sm" column is SM (streaming multiprocessor) utilization. If it's pegged near 100%, you're compute-bound; if it's bouncing around 20–60%, you're memory-bound or stalled on something else (data loading, comms, kernel-launch overhead).

Watch out: `nvidia-smi` reports utilization as "fraction of the last sample period during which any SM was active." A kernel that runs at 1% of FLOPs but on every SM will show as 100% util. The number is a coarse signal, not a profiling result.

### `nsys profile` and Nsight Systems

NVIDIA's profiler. Records CUDA kernel launches, NVTX ranges, and CPU activity, then displays them on a timeline. The right tool for "where is wallclock going."

For an RL loop you'd typically wrap a single iteration in NVTX markers (rollout, log-prob, loss, backward, optimizer) and look at the timeline. The bottleneck is whichever block has the largest wallclock contribution. Common findings: rollouts dwarf gradient steps (decode is slow), or the optimizer step has surprising overhead from CPU-side work (Python's allocator, gradient norm computation).

### `nsys` for arithmetic intensity (Nsight Compute)

Nsight Compute (`ncu`) profiles individual kernels and reports achieved compute throughput, achieved memory throughput, and arithmetic intensity. If a kernel sits at 90%+ of memory bandwidth and 5% of compute, it's memory-bound by design (or by accident — sometimes you find a kernel that should be compute-bound but isn't because of a tiling bug).

For LLM workloads, the answers are usually predictable: attention and matmul kernels in decode are memory-bound; attention and matmul in prefill are compute-bound; small elementwise kernels (activations, normalization) are memory-bound. The interesting profiles are the surprises.

### `torch.profiler`

Built into PyTorch. Less detail than nsys but easier to use; outputs a JSON trace viewable in chrome://tracing or in TensorBoard. For figuring out at the Python level which calls are expensive (e.g., is `loss.backward()` 80% of the step? or is it tokenization?), this is the right tool.

### A practical workflow

The cheap diagnostic loop:

1. Run an iteration with `nvidia-smi dmon -s u` open in another terminal. Note SM utilization.
2. Wrap a single iteration in `torch.profiler` and look at the top 5 kernels by time.
3. If something surprising is at the top, profile that kernel with `ncu` to see whether it's bandwidth- or compute-bound.
4. Decide whether to fix it (better kernel, larger batch, different precision) or accept it.

Most RL practitioners don't do this on every run, but doing it once at the start of a project saves a lot of wasted compute later. The most common discovery: a workload that's "supposed to" be compute-bound is actually memory-bound because of a too-small batch size, and the fix is to bump batch size by 4x or 8x, not to upgrade the GPU.

---

## Software stack

A sketch of what's running on the chip, top to bottom:

- **Driver / CUDA toolkit** — NVIDIA's userspace. Mostly invisible.
- **cuBLAS, cuDNN, NCCL** — NVIDIA's libraries for linear algebra, neural net primitives, and collective comms. Used by everything.
- **CUTLASS** — open-source GEMM templates. Used by FlashAttention, by Triton's matmul, etc.
- **Triton** — DSL for kernels.
- **PyTorch** — the framework most RL code is in. `torch.compile` will dispatch to inductor, which generates Triton kernels.
- **FSDP / DeepSpeed / Megatron** — distributed training layers.
- **vLLM / SGLang / TensorRT-LLM** — rollout engines.
- **TRL / OpenRLHF / verl / RLHFlow** — RL-specific orchestration. Glue the above together with PPO/DPO/GRPO loops.

For a Block 3+ lecture on infrastructure specifics, see the (not-yet-existing) lecture on the RL training stack. For now: don't write your own training loop from scratch for a 70B model. Pick one of the orchestration frameworks above and modify it.

---

## A note for classical RL (regime 1, again)

Most of this lecture has been about LLM-RL. To re-anchor: if you're doing Atari, MuJoCo, Procgen, MineRL, or anything where the policy is under ~100M parameters and the environment runs on CPU, **none of this matters to you**. Don't worry about FP8, FlashAttention, NVLink, KV cache, or any of it. Buy a workstation with 16+ cores, a single consumer GPU, plenty of RAM. Use Gymnasium with `AsyncVectorEnv` (or EnvPool if you're doing Atari). Profile with `cProfile` and look at where your CPU time is going. The standard answer is: in the env step, in observation preprocessing, or in Python overhead in the runner.

I'm repeating this because every other section of this lecture is about the LLM regime, and someone running CartPole could easily come away thinking they need an H100. They don't. They need to vectorize the env and stop spawning subprocesses for trivial workloads.

---

## A decision tree for picking hardware

A rough flow for picking compute, depending on what you're running:

```
What are you training?
├── Classical RL (Atari, MuJoCo, Procgen, anything sub-100M policy)
│   └── Workstation with 16+ CPU cores + one consumer GPU.
│       Vectorize the env with Gymnasium (or EnvPool for Atari).
│       Don't even look at H100 prices. Done.
│
└── LLM-RL (PPO/DPO/GRPO on a 7B+ model)
    │
    ├── Just experimenting / sub-7B model?
    │   └── Single H100 (or single 4090 with 24 GB, for very small models).
    │       LoRA the model if it doesn't fit in BF16.
    │       OpenRLHF / TRL run on this comfortably.
    │
    ├── 7B–13B serious training?
    │   └── 8× H100 SXM node. Tensor parallel + FSDP. Fits with margin.
    │       This is the sweet spot for most RL research in 2024–2025.
    │
    ├── 30B–70B training?
    │   └── 8× H100 node minimum; 8× H200 if available (more KV cache headroom).
    │       Consider separating actor pool from learner — saves on weight broadcast.
    │       Multi-node only if you actually need the FLOPs; the InfiniBand cost
    │       is real and you should profile before scaling out.
    │
    └── 100B+ training?
        └── Multi-node H100/H200 with InfiniBand. Or B200 once available.
            At this scale, infrastructure engineering is most of the work;
            the algorithm is the easy part. Don't start here for learning.
```

The single most common mistake at the LLM-RL end: people scale out before they've maxed out a single node. An 8× H100 box is enormously capable. Most research papers I can think of (DeepSeekMath, R1, OpenRLHF examples) describe runs that would fit on 1–4 nodes, not 100. If your run feels too slow, the answer is usually "find the bottleneck" before it's "buy more chips."

---

## What you should be able to do after this

- Given a workload description, say which regime it's in (classical or LLM) and why.
- Given an RL operation (rollout, gradient update, optimizer step), say whether it's memory-bound or compute-bound, and what that implies for hardware choice.
- Estimate KV cache size for a model of given dimensions and a given batch/sequence configuration.
- Know what FlashAttention does and roughly how much it helps and why.
- Know what's in 2024–2025 hardware (H100/H200/B200/MI300X/TPU v5p) at a one-paragraph level, and where to look up the actual specs.
- Resist the urge to over-provision compute for a workload that doesn't need it.

A checkpoint: explain, in 2–3 sentences each, (a) why decode is memory-bound, (b) why prefill is compute-bound, and (c) why this asymmetry is the central fact about LLM-RL hardware. You don't have to derive numbers from scratch; just explain the structure.

---

## References

All arXiv IDs verified against arxiv.org during writing.

**Memory and compute model**

- Hoffmann, Borgeaud, Mensch et al. 2022. "Training Compute-Optimal Large Language Models." DeepMind. arXiv:2203.15556. — The Chinchilla paper. Useful here for its FLOP-accounting framework (the `C ≈ 6ND` formula), not for the scaling law itself.

**Kernels**

- Dao, Fu, Ermon, Rudra, Ré. 2022. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." arXiv:2205.14135. — The original FlashAttention.
- Dao. 2023. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." arXiv:2307.08691.
- Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao. 2024. "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision." arXiv:2407.08608.
- Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica. 2023. "Efficient Memory Management for Large Language Model Serving with PagedAttention." arXiv:2309.06180. — The vLLM paper. Also referenced in Lecture 30.
- Tillet, Kung, Cox. 2019. "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations." MAPL workshop at PLDI 2019. — I couldn't verify the ACM publisher page directly (403), so confirm the venue before citing externally; the paper itself and its acceptance are well-attested.

**Quantization**

- Frantar, Ashkboos, Hoefler, Alistarh. 2022. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." arXiv:2210.17323.
- Lin, Tang, Tang, Yang, Chen, Wang, Xiao, Dang, Gan, Han. 2023. "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." arXiv:2306.00978.

**Hardware sources**

- NVIDIA H100 product page and datasheet — nvidia.com/en-us/data-center/h100/. Source for 80 GB HBM3, 3.35 TB/s, 989 BF16 TFLOPs dense, 900 GB/s NVLink.
- NVIDIA H200 product page — nvidia.com/en-us/data-center/h200/. Source for 141 GB HBM3e, 4.8 TB/s.
- NVIDIA Blackwell announcement (March 2024) — nvidianews.nvidia.com. Source for "208 billion transistors" and FP4 Tensor Cores. Specific spec numbers (HBM cap/BW, FLOPs) not in the announcement I checked; verify against the current product page.
- NVIDIA Grace Hopper Superchip page — nvidia.com/en-us/data-center/grace-hopper-superchip/. Source for 900 GB/s NVLink-C2C and up-to-624 GB combined fast memory.
- Google Cloud TPU v5p docs — cloud.google.com/tpu/docs/v5p-training. Source for 95 GB HBM, 2.575 TB/s, 459 BF16 TFLOPs per chip.
- AMD MI300X product page — amd.com. Not verified directly (the page timed out for me). HBM capacity, bandwidth, and FLOPs should be confirmed against the current AMD datasheet before quoting.

**Vectorized environments (for the classical RL section)**

- Weng, Lin, Huang, Yan, You, Bian, Liu, Zhu, Su. 2022. "EnvPool: A Highly Parallel Reinforcement Learning Environment Execution Engine." arXiv:2206.10558.

**Lectures referenced**

- Lecture 06 — PPO (clipped surrogate objective, importance sampling).
- Lecture 15 — RL with verifiable rewards; GRPO. The KV-cache-shapes-rollout-throughput discussion above is what makes long reasoning chains expensive in practice.
- Lecture 30 — Rollout engines and PagedAttention. (Stable reference assumed; if this lecture isn't written yet, the Kwon et al. 2023 paper above is the primary source.)

---

## Next lecture

The series doesn't have a fixed Lecture 32 yet. A natural sequel would be the RL training stack itself — how OpenRLHF, TRL, verl, and similar orchestrate the actor/learner split — but that's not written. For now, the closest existing material is in the `reference/papers/` notes.
