<!-- status: unreviewed | last-reviewed: never -->

# Lecture 18: Distillation of Reasoning Models

_Unreviewed: no one has checked this end to end. Treat the math, code, and citations as unverified._

**Time**: ~2 h · **Prerequisites**: Lectures 09, 15

---

## Where this fits

[Lecture 15: RL with Verifiable Rewards](./15-rl-verifiable-rewards.md) ended with the DeepSeek-R1 recipe: cold-start SFT on a few thousand high-quality reasoning traces, then GRPO against a verifiable reward, then a final alignment pass. The result is a large model (DeepSeek-R1 runs on a 671B MoE base) that produces long, self-correcting chains of thought and matches or exceeds o1-class performance on math and coding benchmarks.

That leaves an obvious question. Running GRPO on a 671B model is expensive. Can a much smaller model (7B, 14B, 32B) capture most of those reasoning gains without re-running the full RL pipeline?

The 2025 answer from the R1 paper: yes, by training the small model on the large model's chain-of-thought traces via standard supervised fine-tuning. No RL on the student. The DeepSeek-R1 paper (arXiv:2501.12948) releases five distilled models alongside R1 itself: Qwen-2.5-7B-base, Qwen-2.5-14B-base, and Qwen-2.5-32B-base distillations, plus Llama-3-8B-base and Llama-3-70B-base distillations. These are the "R1-distill" series. The smallest (Qwen-7B) scores above many larger non-reasoning models on MATH and AIME; the Qwen-32B distillation approaches R1 on several benchmarks while running at a fraction of the cost.

This lecture covers how the recipe works, why it works as well as it does, its relationship to earlier self-improvement methods, how it differs from classical knowledge distillation, and where it falls short.

There's a broader pattern worth naming before going further. Across the series, we've seen two main strategies for getting better behavior from a language model after pre-training: optimizing against a reward signal (RL: Lectures 06, 09, 10, 15), and fine-tuning on examples of desired behavior (SFT, DPO: Lectures 11, 17). Distillation is the second strategy, but with a much stronger source of examples: a teacher that was itself trained by the first strategy.

You're collapsing the expensive RL training of the large model into a cheap SFT of the small model by using the large model's outputs as demonstrations. The value of RL training, which is that it found the distribution of long correct reasoning chains, gets transferred to the small model via behavioral cloning. What makes this work is that the verifiable checker lets you filter the teacher's outputs to demonstrably correct examples, so the student isn't just learning to imitate the teacher blindly, but learning to imitate the teacher's correct reasoning behavior specifically.

This combination (RL training for discovery, SFT training for transfer) is likely the general pattern for deploying expensive reasoning capability at scale. Run the hard search once; transfer what it found many times.

---

## The R1-distill recipe

The procedure described in DeepSeek-AI 2025 (arXiv:2501.12948) has three steps.

### Step 1: corpus generation

Use the trained R1 model (the teacher) to generate responses to a curated set of reasoning-heavy prompts: math problems, coding tasks, logic puzzles. For each prompt, generate multiple completions at some temperature. Each completion includes a `<think>` block of variable length followed by a final answer.

The prompt set matters. The R1 paper describes using around 800k samples drawn from open math and code datasets, with quality filtering to remove near-duplicates, trivial problems, and problems where the answer can't be verified. The curation is intentional: you want problems hard enough that the teacher has to actually reason (not just pattern-match), and you want a verifiable answer for the filtering step.

Temperature during generation is typically moderate (around 0.7). Lower temperature gives more consistent chains but less diversity; higher temperature gives more varied chains, some of which will be more creative but more will be wrong. The filtering step handles incorrect completions, so diversity is generally preferred.

### Step 2: filtering

Run each (question, teacher completion) pair through the same verifiable checker used during R1's RL training. Keep only completions where the extracted final answer matches the known correct answer. Discard the rest.

This is the step that distinguishes the distillation corpus from a generic SFT dataset. The checker provides a quality guarantee that is hard to fake: a chain that ends with the wrong answer is discarded, regardless of how plausible or well-structured it looks. If the teacher hallucinated a step that happened to produce a wrong final answer, that trace is removed. If the teacher's chain was convoluted but arrived at the correct answer, it stays.

For math, the checker normalizes the extracted answer and compares it to the reference. For code, the checker runs the extracted code through a sandboxed test harness and checks pass rate. The verifier is identical to the one used during RL training, which means the filtering criteria are consistent with what the teacher was itself optimized against.

The filtering rate varies by problem difficulty. On harder problems, the teacher's success rate may be 20–40%, so you generate 3–5x more traces than you keep. This is an inference compute cost, but it's paid once during corpus construction, not during student training.

### Step 3: SFT on the student

Train the smaller base model on the filtered corpus using standard next-token cross-entropy. No RL. No KL penalty relative to a reference policy. No reward model. Plain supervised fine-tuning, with the same loss and optimizer you'd use for any instruction fine-tuning.

A pseudocode sketch of the full pipeline:

```python
corpus = []

for prompt in reasoning_prompts:
    # Sample K completions from the teacher at moderate temperature.
    # In practice the teacher is a very large model; generation at scale
    # is the dominant compute cost of the distillation pipeline.
    traces = teacher.generate(prompt, n=K, temperature=0.7)

    # Keep only traces where the verifiable checker confirms the answer.
    # The checker is the same one used during R1's RL training.
    correct_traces = [t for t in traces if checker(prompt, t.answer)]

    # Each correct trace is a full (question, <think>...</think>\nAnswer: ...) string.
    corpus.extend((prompt, t.full_text) for t in correct_traces)

print(f"Retained {len(corpus)} traces from {len(reasoning_prompts) * K} generated")

# Standard SFT: train the student to predict each token in the teacher's
# correct chain, conditioned on the prompt.
# Hyperparameters are ordinary SFT choices; nothing unusual here.
sft(
    model=student,
    data=corpus,
    lr=1e-5,
    batch_size=64,
    epochs=2,
    loss="cross_entropy",  # next-token, no special weighting
)
```

Several things are hand-waved in this sketch. The teacher is enormous; running it for inference at scale is the dominant compute cost of the pipeline. The prompt curation is real work; the R1 paper treats dataset assembly as a substantial engineering effort. The SFT loop is otherwise unremarkable: cross-entropy on the student's output tokens conditioned on the prompt and the teacher's full chain, including the `<think>` block. The student trains to generate both the reasoning steps and the final answer.

### What the student learns

After SFT, the student doesn't just produce the right format. It produces the right reasoning structure for each problem class: identifying what the problem is asking, breaking it into steps, checking intermediate results, backtracking explicitly when an approach fails, restating conclusions in a verified form before committing to an answer. These behaviors appear because the teacher's correct chains consistently exhibit them, and the student sees enough examples to internalize the pattern.

The student also inherits the teacher's chain length behavior. On easy problems the teacher produces short chains; on hard problems it produces long ones. The student, after distillation, does the same; it allocates more tokens to harder problems, not because it was told to, but because the training examples demonstrate it.

### Reported results from the R1 paper

The R1 paper reports benchmark scores for the distilled models on AIME 2024, MATH-500, and Codeforces. The key finding: the distilled models significantly outperform non-reasoning models of similar or larger size.

For MATH-500 (reported in the paper): the Qwen-7B-R1-distill model outperforms QwQ-32B-Preview (a 32B reasoning model) and GPT-4o on this benchmark. The Qwen-32B-R1-distill model approaches R1 itself on MATH-500 while running at roughly 1/20 the parameter count.

For AIME 2024: the distilled Llama-70B outperforms o1-mini on AIME pass@1. The distilled Qwen-32B also outperforms o1-mini on this benchmark.

These numbers should be treated with the standard caution: benchmark results depend on evaluation setup (number of samples, temperature, answer extraction heuristics), and the paper uses greedy decoding in some comparisons and sampling in others. The qualitative conclusion, that distillation from R1 transfers most of R1's reasoning capability at much smaller scale, holds across the evaluation setups in the paper; the specific rank ordering against other models may shift with evaluation details.

The paper also reports that RL training directly on the small models (without distillation) produces worse AIME and MATH scores than distillation, for the same small-model bases. This is the comparison that makes the distillation approach practically interesting: if RL on small models worked as well, you could skip the large teacher entirely.

### A concrete illustration of the filter

Suppose the teacher is generating solutions to: "Find all integer solutions to x² − 5x + 6 = 0."

The teacher might produce three completions. One works correctly:

```
<think>
Factor the quadratic: x² − 5x + 6 = (x − 2)(x − 3).
Setting each factor to zero: x = 2 or x = 3.
Check: 4 − 10 + 6 = 0 ✓, 9 − 15 + 6 = 0 ✓.
</think>
Answer: x = 2 and x = 3
```

A second completion tries the quadratic formula but makes an arithmetic error mid-chain and arrives at x = 1 and x = 4. A third completion produces the correct approach but formats the answer as a decimal approximation ("x ≈ 2.0 and x ≈ 3.0") and the checker's normalization step marks it as a mismatch against the integer reference.

The filter keeps the first completion. The second fails the checker outright. The third is a judgment call: depending on checker implementation, it might pass (if the checker recognizes 2.0 == 2) or fail (if it expects exact integer format). This kind of edge case in the checker is why checker design matters. An overly strict checker rejects valid solutions; an overly loose checker accepts wrong ones.

What the student trains on from this example is the first completion: the factoring approach, the verification step, and the specific answer format. The student doesn't see the arithmetic-error chain or the decimal chain. If the factoring approach is consistently present in correct chains for this problem type (it is, because it's the cleanest solution), the student will learn to use it.

---

## Why it works

SFT is effective when the training targets are high quality and the student has enough model capacity to imitate them. Both conditions hold here in ways worth being precise about.

### The verifier provides a quality guarantee

Standard instruction-tuning datasets often contain plausible-but-wrong examples. Human annotators make mistakes. LLM-generated datasets contain hallucinations that look convincing. When you train on these, the student learns to imitate confident-sounding outputs regardless of correctness.

The verifiable checker eliminates this problem for the reasoning domain. Any chain that reached the wrong answer is excluded, regardless of how well-constructed it looks. The student trains on demonstrably correct reasoning, not just confident-sounding reasoning. This is a stronger training signal than generic SFT because the targets have been certified.

### Reasoning has learnable style

The long chain-of-thought format (breaking a problem into explicit steps, noting when an intermediate result seems wrong, restating the original constraint before committing to an answer) is a consistent style across the teacher's correct chains. The student sees this pattern at sufficient scale to internalize it. After SFT, it produces the same format and the same self-correction idioms, even on problems it wasn't directly trained on.

"Style" here shouldn't be read as superficial. The student isn't merely learning to format chains the same way; it's learning the order of operations the teacher uses on each problem type, because the chains it trains on are the chains that reliably reached correct answers. The format and the reasoning content are coupled.

### The student has existing capacity

A 7B or 14B model trained on text at scale has already internalized substantial arithmetic, algebraic structure, code syntax, and logical inference from pre-training. Distillation doesn't build reasoning from nothing; it shapes when and how the model deploys those existing capabilities. SFT on correct reasoning chains pushes the model to deploy them explicitly, in a structured format, at the right points in a chain.

This is why capacity matters. If the student were too small (say, 1B parameters on a narrow pre-training corpus), it might not have enough latent capability to imitate the teacher's chains even with correct training examples. The R1-distill series uses models with pre-training quality sufficient to absorb the training signal.

### What SFT cannot do

This framing also clarifies the limits. SFT can teach the student to produce the same chains the teacher produces. It cannot teach the student to verify those chains independently. The student learns to generate the self-correction idioms ("wait, let me reconsider this step") because the teacher's correct chains contain them, but the student doesn't learn when to generate them from first principles. It's mimicking the verification behavior, not independently discovering when an intermediate step is wrong.

In practice this means the student's self-correction is less reliable than the teacher's on novel problem types that weren't in the training distribution. The student produces the format of self-correction but may do it at the wrong times, or not at all, when the problem structure is sufficiently different from what it trained on. The teacher, trained via RL on verifiable rewards, learned when to self-correct because doing so actually improved the correctness reward. The student learned when to self-correct because the training examples showed it, and those examples cover the training distribution, not all possible inputs.

This gap between "can produce the format" and "independently learned the behavior" is part of why a short RL pass on top of a distilled model can still help. RL on the distilled student gives it signal about whether its self-correction is actually working on new problems, which SFT alone never provided.

---

## The relationship to STaR

The R1-distill recipe is a cross-model version of a loop that was first described as self-distillation.

STaR (Self-Taught Reasoner, Zelikman et al. 2022, arXiv:2203.14465) uses the model being trained as both teacher and student:

1. Sample chains from the current model on a labeled dataset.
2. Keep chains where the final answer is correct.
3. SFT on the correct chains.
4. Repeat.

The model improves iteratively because each SFT round trains it on harder problems it couldn't solve in earlier rounds. The loop is EM-flavored: the E step samples chains from the current policy and filters to correct ones, the M step trains on those chains. The verifiable checker is what makes the bootstrapping trustworthy; at each step you know the chains you're training on are correct.

STaR also includes a "rationalization" fallback: for problems the model fails on completely, generate a chain conditioned on the known correct answer (i.e., give the model the answer and ask it to explain how to get there). This prevents the training set from being entirely composed of easy problems the model already gets right. Without rationalization, STaR can stall if the model's baseline success rate on hard problems is near zero.

ReST (Reinforced Self-Training, Gulcehre et al. 2023, arXiv:2308.08998) is an offline version of the same idea. Generate a large batch of completions from the current policy all at once, filter by reward threshold, train on what passes, repeat with the updated model. The key distinction from STaR is that all generation happens before any training in each round: no interleaving of generation and training within a round. This is more compute-efficient because generation can be parallelized across hardware, and it's arguably less susceptible to within-round reward hacking because the reward is applied at filter time, not as a live training signal.

R1-distill sits in this lineage but differs in one critical way: the teacher and student are different models, with the teacher being much larger and already RL-trained to produce strong reasoning. The loop runs exactly once rather than iteratively. The EM interpretation still applies (filter correct chains, train on them), but the E step (generating chains) is done by a model far stronger than the student, which means the filtered corpus covers harder problems than the student could generate for itself.

The practical consequence: a single round of cross-model distillation from R1 gets a 7B model to higher accuracy than multiple rounds of self-distillation (STaR) starting from the same 7B base. The teacher's greater capability translates directly into a harder and more useful training corpus.

The lineage: STaR (2022) → ReST (2023) → R1-distill (2025). The core insight, train on verifiably correct chains, is the same throughout. What changed is scale and cross-model transfer.

A brief comparison:

| | STaR | ReST | R1-distill |
|---|---|---|---|
| Teacher | Self (current model) | Self (current model) | Separate large model |
| Loop | Iterative | Iterative | Single pass |
| Data freshness | Fresh each round | Fresh each round | One-time corpus |
| Hard problem coverage | Limited by self-success rate | Limited by self-success rate | Set by teacher capability |
| Rationalization fallback | Yes | No | Not needed |

The rationalization fallback in STaR, generating a chain conditioned on the known answer for problems the model fails, is a patch for the fact that the model can't generate correct chains on its own for hard problems. R1-distill doesn't need this patch because the teacher handles hard problems directly.

The EM framing is more than an analogy here. In expectation-maximization proper, the E step computes a posterior over latent variables given observed data and the current parameters, and the M step maximizes expected log-likelihood under that posterior. The "latent variable" in the reasoning distillation setting is the chain of reasoning, unobserved during training (we only see the question and answer label), marginalized over by the E step (generating candidate chains and filtering to plausible ones), and then treated as observed in the M step (SFT on the filtered chains). STaR and R1-distill both follow this structure; what differs is what distribution generates the E step candidates (the current model itself in STaR; a much stronger teacher in R1-distill).

---

## Why not just run RL on the small model directly?

The R1 paper explicitly tried this, and the result is one of the more informative findings in the paper. The authors ran GRPO directly on the smaller base models, the same Qwen-2.5 and Llama-3 bases used for distillation, rather than distilling from R1. The RL-trained small models produced worse results than the distilled models.

This is somewhat unexpected. GRPO on verifiable rewards is the algorithm that made R1 as strong as it is at the large scale. Running it at small scale on the same reward signal should, in principle, discover similar reasoning behaviors. In practice it doesn't, or at least not to the same degree.

The explanation the paper offers is consistent with the general interpretation from Lecture 15: RL amplifies what the base model already produces; it doesn't create capability from nothing. A 7B model's pre-training gives it weaker priors over long, self-correcting reasoning chains than a 671B model's pre-training. The RL search starts from a worse initialization and the gradient signal from binary correct/incorrect rewards is sparse; the small model succeeds infrequently enough on hard problems that there aren't many positive examples to learn from.

The distillation corpus solves this directly. It hands the small model thousands of correct, long reasoning chains: positive examples the small model never would have produced on its own. The SFT on those chains teaches the student what successful reasoning looks like at the structural level, which then becomes a better initialization for any subsequent RL pass.

The practical upshot: if you have a strong teacher and want reasoning capability in a smaller model, distill first. RL directly on the small model is a less efficient path to the same place, and it doesn't get as far.

A secondary implication: you don't need to run RL separately for each model size. Run it once at the largest scale you can afford, then distill downward to whatever sizes you need. The expensive RL training is a one-time cost; the distillation is inference + SFT, which is substantially cheaper.

This changes the economics of reasoning model deployment. Before R1, getting reasoning capability into a 7B model would have required running GRPO on a 7B model, and the evidence from the R1 paper suggests that doesn't work well. After R1, you run GRPO once on a 671B model, generate a corpus from it, and SFT five different student sizes from the same corpus. The RL compute scales once with the teacher; the rest is parallelizable and cheap per model size.

There's also a compounding benefit: as stronger teachers become available (via more RL training, more pre-training compute, or better reward functions), you can re-distill to get better students without changing the student architecture or training procedure at all. The student training pipeline is fixed; only the corpus changes when the teacher improves.

---

## Relationship to classical knowledge distillation

The canonical reference on knowledge distillation is Hinton, Vinyals, and Dean 2015 (arXiv:1503.02531). Their setup is different enough from R1-distill that it's worth being precise about the distinction.

In Hinton et al., the teacher's knowledge is transferred via soft targets: the full probability distribution over output classes, not just the argmax. Training on soft targets is better than training on one-hot labels because the teacher's distribution encodes structure that hard labels don't. If a teacher assigns 2% probability to class B when the correct answer is class A, that probability says something about how similar A and B are in the teacher's representation. A student trained to match this distribution (at some temperature that softens the distribution further) absorbs that similarity structure, not just the correct class membership.

The paper also introduces temperature scaling as a tool for this: at inference temperature T=1, the distribution is already peaked; at higher T the distribution softens and more inter-class similarity information is visible. The student trains at high T to extract the structure, then at T=1 to make predictions.

R1-distill doesn't use soft targets or logit-level transfer at all. It uses the teacher's output text, the full chain-of-thought plus answer, as a training sequence, and trains the student with cross-entropy against those token sequences. This is behavioral cloning on the teacher's outputs. The student learns what sequence of tokens to generate, not how the teacher's probability distribution looks over all alternatives at each step.

The two approaches transfer different things:

- Logit distillation (Hinton et al.): transfers the teacher's uncertainty structure across all alternatives at each prediction step. More information per example; requires access to the teacher's logits (not just sampled text).
- Behavioral cloning on chains (R1-distill): transfers the teacher's decision sequence on verified examples. The student learns the chain structure from text alone; no access to the teacher's internal probability distribution needed.

For reasoning in verifiable domains, the behavioral approach is both simpler to implement and apparently sufficient. Soft targets for token-level distillation in long reasoning chains would require storing the teacher's full vocabulary distribution at every token position across millions of training examples, an enormous storage cost. Sampling chains and filtering keeps the corpus tractable.

There's also an argument that soft targets add noise in this setting. If the teacher is uncertain about which next reasoning step to take (reflected in high-entropy token distributions mid-chain), you may not want to transfer that uncertainty to the student. The filtering step already ensures that the chains you train on are correct; you want the student to be confident on correct chains, not to imitate the teacher's mid-chain uncertainty.

Chain-of-thought distillation as an explicit technique predates R1. Hsieh et al. 2023 (arXiv:2305.02301, "Distilling Step-by-Step!") trained smaller models on reasoning rationales generated by a large model, showing that rationale-trained students outperform those trained on direct answer labels at the same size. They found this held even with fewer training examples, because the rationale provides information about which intermediate steps the teacher used, not just what the final answer was. This matches the intuition that behavioral cloning on chains transfers structural information that labels-only training doesn't.

Magister et al. 2022 (arXiv:2212.08410, "Teaching Small Language Models to Reason") found a similar result: SFT on chain-of-thought solutions from GPT-3 improved smaller model reasoning more than SFT on answers alone across several benchmarks. They also noted that training on chains from a model slightly stronger than the student (rather than much stronger) sometimes worked as well or better, suggesting that chain complexity relative to student capacity matters: chains that are too long or too complex for the student to absorb may be less useful than somewhat shorter chains the student can imitate reliably.

R1-distill sits in this line of work but adds the verifiable-checker filter more explicitly than the earlier papers. Hsieh et al. filtered by accuracy (used correct chains for SFT, incorrect ones for a secondary training signal), and Magister et al. sometimes trained on all generated chains regardless of correctness. The R1 paper treats filtering as central: only correct chains are used, and the checker is the same one used during RL training, ensuring consistency between the RL phase and the distillation phase.

### When would you use logit distillation instead?

Logit distillation (Hinton et al.) is a better fit when:

- You have access to the teacher's logits (not just its sampled text). Many commercial APIs don't expose logits; open-weight models do.
- The task has a small output space. Classification over a few hundred classes benefits from soft targets because the per-class probabilities are meaningful and transferable. For language modeling over a 128k-token vocabulary, storing soft targets is usually not practical.
- The teacher's uncertainty is informative. In classification, a teacher that puts 40% on class A and 40% on class B is signaling something useful: ambiguity between those two classes. In a long reasoning chain, the teacher's uncertainty about the next token mid-chain is mostly noise, not signal.

Behavioral cloning on chains is a better fit when:

- You only have access to sampled outputs (the typical case for API-accessed models).
- The output space is large (language generation) and storing per-token distributions is impractical.
- You have a verifier that can certify which sampled outputs are correct. The verifier turns noisy samples into a clean training set.

These conditions describe the R1-distill setup almost exactly.

---

## Limits

The distilled models inherit the teacher's blind spots directly. If R1 systematically fails on a class of problems, the distillation corpus won't contain correct chains for those problems. The student doesn't learn to handle them either. Teacher coverage gaps become student coverage gaps. This is a structural limitation: the student can only learn what the teacher can demonstrate correctly.

The student can't exceed the teacher. RL training can, in principle, find solutions the teacher never demonstrated; it explores the space of possible chains and gets feedback on whether they work. SFT on the teacher's outputs cannot; it learns to imitate what's in the corpus and nothing more. If the teacher has a systematic error or a domain where it underperforms, no amount of SFT on that teacher's outputs will produce a student that surpasses it.

The approach depends on having a verifiable domain. The filtering step is what makes the corpus trustworthy, and the filtering step requires a checker. For open-ended writing, complex multi-step planning, or tasks requiring subjective judgment, you can't cleanly filter the corpus. An LLM judge as a soft filter reintroduces the biases and miscalibration that the verifiable checker was eliminating. The strength of R1-distill is closely tied to math and code, where reliable checkers exist and where "correct" has an unambiguous meaning.

The filtered corpus is smaller than the raw corpus. If the teacher gets 30% of hard problems correct, you generate roughly 3x more traces than you keep. Inference compute for the teacher scales with total traces generated, not with traces that pass the filter. For very hard problems (teacher success rate 5-10%), the sampling cost becomes a real constraint.

Problem difficulty calibration matters. If the reasoning prompts are too easy, the teacher's chains are short; the student doesn't see the extended reasoning, backtracking, and self-checking that characterize R1's behavior on hard problems. If the prompts are too hard, the teacher's success rate is low and you keep very few chains per problem. The useful training data is in the middle: problems where the teacher succeeds often enough to provide coverage but hard enough that the chains show non-trivial reasoning.

The student learns a static snapshot. Distillation captures the teacher's capability at corpus generation time. If the teacher is later improved, the distillation has to be re-run. There's no incremental update mechanism.

**Hallucination inheritance.** When the teacher produces a correct answer via a chain that contains a false intermediate claim (a numerical hallucination that cancels out, or an incorrect algebraic manipulation that happens to reach the right result), that chain passes the filter and enters the training set. The student learns to produce correct final answers via chains that include the same false intermediate claim. This is harder to detect than a flat wrong answer because the checker only checks the endpoint, not the path. Chains that reach the right answer via wrong intermediate steps are a known failure mode of outcome-supervised training generally (see the Uesato et al. 2022 discussion in Lecture 15), and they transfer into the distillation corpus if the teacher produces them.

**Style overfitting.** If the teacher has a strong preference for one reasoning approach on a problem class (say, always using the quadratic formula rather than factoring), the student will inherit that preference even when the alternative would be shorter and clearer. The teacher's stylistic preferences, not just its capabilities, become the student's defaults. This is usually fine, but it means the student's chains can be longer than necessary for problems where the teacher's preferred approach is verbose.

---

## Practical training notes

The SFT step is standard, but a few choices are specific to reasoning chain distillation.

**What to mask.** A common question is whether to apply the SFT loss to the entire sequence (prompt + chain + answer) or only to the model's completion (chain + answer). Masking out the prompt tokens from the loss is standard practice; you don't want the model to optimize for predicting the question, only for predicting the reasoning and answer. Whether to include the `<think>` block in the loss or only the final answer tokens is a design choice. Including the full chain in the loss is more common and is what the R1 paper describes; it gives the model more gradient signal per example and teaches it to produce the intermediate reasoning, not just the final answer.

**Chain length and loss weighting.** Long chains contribute more tokens to the loss than short chains. If hard problems have consistently long correct chains and easy problems have short ones, the SFT loss is dominated by hard-problem tokens, which may be desirable (hard problems contribute more to the model's reasoning improvement) or may cause the model to weight hard-problem style disproportionately. Some implementations normalize by sequence length when computing the batch loss; others don't. The R1 paper doesn't describe this in detail; it's an implementation choice that can affect which problem types the distillation optimizes most.

**Number of epochs.** Reasoning chain SFT tends to use fewer epochs than general instruction fine-tuning. The chains are long, so one epoch covers many tokens. Two epochs is a common choice; more than three risks overfitting to the specific chains in the corpus rather than the reasoning patterns they demonstrate. On small training sets (fewer than 50k examples), overfitting can appear quickly; the model starts reproducing specific chains verbatim rather than generalizing from them.

**Evaluation during training.** Unlike RL training, SFT gives you no signal about whether the student's reasoning is actually correct; cross-entropy loss going down doesn't tell you whether the model is learning to reason or just memorizing tokens. Running the verifiable checker on held-out problems throughout training (say, every 500 steps) gives you an independent correctness signal. If held-out accuracy plateaus while training loss is still dropping, you're likely overfitting.

**Cold-start distillation vs. fine-tuning a pretrained model.** The R1-distill models initialize from base pre-trained models (Qwen-2.5-base, Llama-3-base) rather than from instruction-tuned checkpoints. Base models are generally a better starting point for reasoning distillation because they don't already have an instruction-following style that might conflict with the teacher's chain-of-thought format. If you start from an instruction-tuned checkpoint, the existing RLHF alignment can interfere with the chain format, since RLHF training may have suppressed long internal monologue in favor of concise responses.

**When to stop.** Unlike RL training, SFT loss doesn't have a natural stopping criterion tied to task performance. A common heuristic: checkpoint every 500 steps, evaluate on a held-out problem set with the verifiable checker, and stop when held-out accuracy peaks (not when training loss bottoms out). If you don't have a held-out verifiable set, use perplexity on a held-out portion of the training corpus as a proxy, but be aware that lower perplexity on the training distribution doesn't always translate to higher accuracy on novel problems. The safest approach is a held-out problem set with a checker; everything else is a proxy.

---

## When to use this

Distillation from a strong teacher is the right approach when you need reasoning capability in a smaller model and have access to a teacher that can reliably demonstrate that capability on a verifiable domain.

Running RL directly on the student is the right approach when you don't have a strong teacher (or can't afford teacher inference at scale), when you want to push past the teacher's capability ceiling, or when the domain doesn't have a verifier that would let you filter distillation traces.

In practice, the recipe that emerged in 2024–2025 for small reasoning models combines both. Distill from a large teacher to get most of the reasoning behavior quickly, then run a short RL pass on top to push the student further on the specific tasks you care about. The distillation step does the heavy lifting; the RL step handles the remaining gap. The R1 paper describes this sequencing implicitly in its pipeline: distill first, then refine. Neither alone is as effective as both in sequence.

One practical constraint: you need access to the teacher's outputs, or a strong open-weight teacher. For practitioners without access to frontier model APIs or the compute to run a 671B model, the distillation approach is gated on having something to distill from. Open-weight reasoning models (R1 itself, Qwen-QwQ) make this feasible for more practitioners than it was before 2025.

For domain-specific applications: if you have a proprietary verifiable domain (specialized math, domain-specific code with test suites), you can generate a custom distillation corpus from a general strong teacher and then fine-tune on it. The student can pick up the domain-specific reasoning style without the teacher needing to be pre-specialized for that domain, as long as the checker can verify correctness.

A rough decision rule:

```
Do you have a strong teacher whose outputs you can sample?
├── No  → RL on the student directly, if you have a verifier.
│         No verifier? → SFT on curated human-written chains.
└── Yes → Is the domain verifiable (math, code, formal proofs)?
    ├── No  → Distill with an LLM judge as soft filter (weaker guarantee).
    │         Quality depends on judge reliability.
    └── Yes → Distill using the verifiable checker as your filter.
              Then optionally: short RL pass on top of the distilled model.
```

[Lecture 09: Reward Modeling](./09-reward-modeling.md) covers how to build or borrow a reward model for the RL pass on top. For the SFT machinery, any standard fine-tuning tutorial applies; there's nothing in the student training loop that's specific to reasoning beyond the corpus content.

A few concrete scenarios where the decision isn't obvious:

**You want a reasoning model for internal tooling (e.g., automated code review) and your codebase has a test suite.** The test suite is your checker. Sample from an open-weight reasoning teacher (R1, QwQ), filter to traces that pass the tests, SFT on a smaller model you can deploy on cheaper hardware. This is R1-distill applied to a proprietary domain without any modification to the algorithm.

**You want to improve a model on multi-step word problems and you don't have a symbolic verifier.** You can write a simple Python function that extracts the final number from the model's output and compares it to the labeled answer. This is good enough for most arithmetic word problem datasets. If you can't write even this, you're back to LLM judging, which is workable but weaker.

**You have a strong teacher but no ground-truth answer labels (e.g., the teacher is solving open-ended scientific questions).** Distillation without a verifier requires trusting the teacher's output quality without an independent check. You're training the student on chains the teacher believes are correct, not chains that are certified correct. This is riskier than the R1-distill setup and closer to vanilla instruction fine-tuning on model-generated data; the gain depends on the teacher's calibration, not on a verifiable correctness signal.

---

## Exercises

These don't require an `exercises/` directory; they're things to try locally.

**Self-distillation loop on a math toy task.** Take a small language model (a 1B model works) and a set of arithmetic problems with verifiable answers. Implement one round of STaR: generate chains from the model, filter to correct ones, run SFT, repeat. Measure accuracy before and after each round. How many rounds until improvement plateaus? What fraction of generated chains are correct at each round, and how does that fraction change over rounds? If you implement the rationalization fallback (conditioning on the known answer for hard problems), does it meaningfully change the convergence rate?

**Correct traces vs. all traces.** For the same toy task, generate chains from a small teacher (or from a GPT-2-class model that sometimes gets arithmetic right). Train two students: one on correct-only traces, one on all traces regardless of correctness. Compare accuracy after SFT, and also compare the distribution of chain lengths. Do students trained on all traces (which include failed reasoning attempts) produce longer or shorter chains than those trained on correct-only traces? The gap in accuracy between the two conditions is a direct empirical measure of how much the filtering step matters.

**Chain length preservation.** After distillation, measure the average token length of the student's chains versus the teacher's chains on the same set of problems. Does the student produce shorter or longer chains? Does chain length correlate with accuracy on the student the same way it does on the teacher? A student that produces correct short chains where the teacher produced long ones may be memorizing solutions rather than reasoning through them; a student that produces long chains even where the teacher was short may be failing to generalize the teacher's difficulty-adaptive length behavior.

**Capacity threshold.** Try distilling from a moderate teacher (say, a 7B model) into students of different sizes (1B, 3B, 7B). At what student size does held-out accuracy start to degrade noticeably? This gives an empirical sense of where "the student doesn't have the capacity to absorb the teacher's chains" starts to become real. Compare this degradation against the degradation from simply reducing the training data volume. Is the accuracy drop from insufficient model size similar to the drop from using fewer training examples?

**Checker sensitivity.** Take a fixed distillation corpus and vary the checker's strictness: (a) exact match on the final answer, (b) within 1% relative tolerance, (c) accept any numeric answer that rounds to the correct integer. Train three students on the three filtered corpora and compare accuracy on a held-out test set evaluated with exact match. This tests how the checker's design propagates into student quality. A loose checker keeps more (possibly wrong) chains; a strict checker keeps fewer but more precisely correct ones.

---

## References

**DeepSeek-R1 (primary source for this lecture)**

- DeepSeek-AI. 2025. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv:2501.12948. Primary source for the R1-distill recipe, the distilled model series (Qwen-7B through Qwen-32B, Llama-8B, Llama-70B), and the finding that RL on small models directly produced weaker results than distillation from R1. The distillation section is §3 of the paper.

**Classical knowledge distillation**

- Hinton, Vinyals, Dean. 2015. "Distilling the Knowledge in a Neural Network." arXiv:1503.02531. Introduces soft targets and temperature-scaled logit distillation. The approach transfers the teacher's uncertainty structure across all output classes; this is distinct from the behavioral cloning on sampled chains that R1-distill uses.

**Chain-of-thought distillation**

- Hsieh, Li, Yao, Ruan, Raj, Chen, Perot. 2023. "Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes." arXiv:2305.02301. Trains small models on reasoning rationales from a large model; rationale-trained students outperform label-trained students at matched size. Verified.
- Magister, Engelbrecht, Mallick, Kruger, Kreutzer. 2022. "Teaching Small Language Models to Reason." arXiv:2212.08410. SFT on chain-of-thought solutions from GPT-3 improves small-model reasoning more than SFT on direct answers; also examines the effect of chain quality and teacher-student size mismatch. Verified.

**STaR and ReST (self-distillation lineage)**

- Zelikman, Wu, Mu, Goodman. 2022. "STaR: Bootstrapping Reasoning With Reasoning." NeurIPS 2022. arXiv:2203.14465. Self-distillation loop: sample chains from the current model, filter correct ones, SFT, repeat. Introduces the rationalization fallback for hard problems. Verified.
- Gulcehre, Le Paine, Srinivasan et al. 2023. "Reinforced Self-Training (ReST) for Language Modeling." Google DeepMind. arXiv:2308.08998. Offline version of the same loop: generate a large batch, filter by reward threshold, train on what passes, repeat. Verified.

**Process supervision (related background)**

- Uesato, Kushman, Kumar et al. 2022. "Solving math word problems with process- and outcome-based feedback." arXiv:2211.14275. Covers how outcome-supervised chains can reach correct answers via incorrect intermediate steps, a source of hallucination inheritance in distillation corpora. See Lecture 15 for fuller discussion. Verified.

**Reading order suggestion**: start with §3 of the R1 paper (arXiv:2501.12948) for the distillation recipe and benchmark results, then read STaR (arXiv:2203.14465) for the EM-flavored self-distillation perspective, then Hsieh et al. (arXiv:2305.02301) for the earlier cross-model chain distillation results. Hinton et al. (arXiv:1503.02531) is worth reading for the classical logit distillation framing even though R1-distill doesn't use it; understanding what it does differently clarifies why the behavioral cloning approach is the natural choice in the language reasoning setting.

---

## Next lecture

[Lecture 19: Offline RL](./19-offline-rl.md): being written in parallel. Target filename `19-offline-rl.md`.
