<!-- status: unreviewed | last-reviewed: never -->

# Lecture 26: RL for mathematical reasoning

_Unreviewed: no one has checked this end to end. Treat the math, code, and citations as unverified._

**Time**: ~3 h · **Prerequisites**: Lectures 12, 15

---

## Where this fits

[Lecture 15](./15-rl-verifiable-rewards.md) covered the general shape of RL with verifiable rewards: a checker replaces a learned reward model, GRPO replaces PPO's critic, and the binary correctness signal turns out to be enough to shape long-horizon generation behavior. That lecture used math as the running example, but it framed RLVR as a general technique.

This lecture is narrower. It's about the specific systems that pushed math reasoning past 90% on the MATH benchmark and into double-digit AIME pass rates during 2024 and 2025: what their training data looked like, what their reward functions actually computed, what the loss curves did, and which failure modes are specific to the math setting.

Math is the cleanest domain for RLVR, which is why most of the public work happened here first. Answers are short. Verification is mechanical. Difficulty graduates smoothly from grade-school arithmetic up to olympiad problems. Datasets exist with verified answers at every level. None of that holds for open-ended writing, agentic tasks, or even most code generation, where verifier quality varies wildly. So the math-reasoning systems are where the recipe is most worked out, and they're the place to look if you want to understand what RLVR can do when nothing else is in the way.

The story compresses into a few moves: prompting (chain-of-thought) bought a lot. Self-training (STaR) bought more. Process supervision (PRM800K) bought additional reliability on harder problems. GRPO with rule-based rewards (DeepSeekMath, then R1) bought the jump from "good" to "competitive with human olympiad performers on some subsets." Tool use (PAL, then formal provers) bought a different axis of improvement by changing what the model has to do unaided.

This lecture goes through those moves in order. The algorithmic content for the RL parts is in Lecture 15; what's here is what's specific to math.

---

## The benchmarks that drove this

Benchmark choice shaped the whole research program. The systems described later were optimized against these specific datasets, and a lot of the design choices only make sense in light of what each benchmark rewards.

### GSM8K

Cobbe et al. 2021 (arXiv:2110.14168). 8.5K grade-school math word problems, each with a step-by-step worked solution and a single numeric answer. Problems are linguistically diverse but mathematically simple: the typical solution takes 2–8 steps of basic arithmetic.

GSM8K was the benchmark that made chain-of-thought prompting look impressive (Wei et al. 2022, arXiv:2201.11903, hit 56.9% with a 540B model). It was also the benchmark where outcome-based verifier training was first studied at scale, in the original paper by Cobbe et al. The verifier training described there is a precursor to the RLVR work that came later.

By 2024 GSM8K is essentially saturated. Frontier models score in the high 90s. It's still a useful smoke test (a model that can't crack GSM8K isn't going to do well anywhere else), but it doesn't differentiate between strong reasoning models any more.

### MATH

Hendrycks et al. 2021 (arXiv:2103.03874). 12,500 problems drawn from US high-school competition mathematics (AMC, AIME, and similar), each labeled with a difficulty level (1–5) and one of seven subject categories (algebra, geometry, number theory, counting and probability, intermediate algebra, prealgebra, precalculus). Each problem has a worked solution and a final answer in `\boxed{...}` form.

MATH is where the 2024–2025 jump showed up most clearly. The Hendrycks paper itself reported that scaling alone wasn't getting models past around 7% pass rate at the time of publication, and concluded that "new algorithmic advancements" would be needed. That conclusion turned out to be right: chain-of-thought, then self-consistency, then process supervision, then GRPO, each moved the number up substantially. DeepSeek-R1 reports around 97% pass@1 on MATH-500 (the common 500-problem evaluation subset), which would have been surprising at the time of the original benchmark paper.

The reason MATH is harder than GSM8K isn't just the math content: it's the answer space. GSM8K answers are integers (small ones, usually). MATH answers can be expressions, fractions, surds, ordered tuples, intervals, sets. The verifier has to handle equivalent forms (`1/2` and `0.5`, `\sqrt{2}/2` and `\frac{1}{\sqrt{2}}`), units, and formatting variations. The answer-extraction heuristic and the equivalence checker do a lot of work, and weaknesses in either show up directly as inflated or deflated pass rates.

### AIME

American Invitational Mathematics Examination. A 15-problem competition for top US high-school math students, with integer answers in 0–999. No public arXiv paper for the benchmark itself: it's just the competition problems, scraped or transcribed from past years.

AIME 2024 became the de facto "frontier reasoning" benchmark during 2024–2025 because problems were recent enough to not be in pre-training corpora at scale, hard enough that even strong models scored well below 100%, and short enough (integer answers) that verification is trivial.

The number reported is usually pass@1 averaged across problems, with multiple samples per problem (often 8 or 32) to reduce variance. DeepSeek-R1 reported around 79.8% on AIME 2024 (pass@1, with high sample counts). OpenAI o1 reported similar numbers. Before reasoning models, frontier non-reasoning models were typically below 20%.

Because there are only 15 problems per year, pass@1 numbers on AIME jump around. A single problem flipping correct/incorrect is 6.7% on the score. Treat AIME numbers as noisy; report multiple-sample variance if you can.

The other thing to watch for on AIME: contamination. Older years (pre-2023) have been on the public web for a long time, and worked solutions appear in many tutoring sites and pre-training crawls. Reported AIME numbers on those years are inflated to the extent that the model has effectively memorized the answer. AIME 2024 became standard partly because it was recent enough to plausibly be outside pre-training cutoffs for most models published in late 2024 and 2025. By the time you read this, AIME 2024 may itself be contaminated for newer models; switch to the most recent year's problems for the cleanest signal.

### MMLU-STEM

A subset of MMLU (Hendrycks et al. 2020, arXiv:2009.03300) covering science, technology, engineering, and math questions across 57 subjects total. Multiple choice, not free-form reasoning. Less commonly used for reasoning-specific evaluation but cited alongside MATH and GSM8K when comparing general models. Not a primary driver of the reasoning-model recipe.

### OlympiadBench

He et al. 2024 (arXiv:2402.14008). 8,476 olympiad-level problems in math and physics, bilingual (Chinese and English), with expert step-level annotations. Harder than MATH on average, with more problems requiring extended multi-step reasoning. Frontier reasoning models still leave substantial headroom here.

### MiniF2F

Zheng, Han, Polu 2021 (arXiv:2109.00110). 488 formal-math problem statements (in Lean, Coq, Isabelle, Metamath, HOL Light) drawn from AIME, AMC, IMO, and academic coursework. Cross-system: the same theorem is stated in multiple proof assistants, so results are comparable across systems.

MiniF2F is the standard formal-reasoning benchmark. Unlike the natural-language benchmarks above, the verifier here is a proof assistant: the model has to produce a proof that the assistant accepts, not just a correct final answer. This changes the reward structure dramatically (more on this in the tool-use section).

### The benchmark hierarchy

Roughly, ordered by current difficulty for frontier reasoning models:

```
GSM8K            (saturated, >97% for strong models)
MMLU-STEM        (saturated for top models, but multi-domain)
MATH             (high 90s for top reasoning models; was sub-50% in 2022)
OlympiadBench    (substantial headroom, even for o1 and R1)
AIME             (50-80% for top reasoning models, noisy)
MiniF2F          (formal proofs; rapid progress but headroom remains)
```

When someone says a system "does well on math," ask which benchmark. GSM8K saturating tells you very little about whether a model can do AIME, and AIME numbers can flip dramatically with sample count and prompt format.

---

## What came before RL: prompting and self-consistency

Chain-of-thought prompting (Wei et al. 2022, arXiv:2201.11903) is the pre-RL baseline that made the rest of this work possible. The technique itself is just adding "let's think step by step" or a few worked examples to the prompt and letting the model generate intermediate reasoning before its final answer. On GSM8K, this moved a 540B model from around 18% (direct prompting) to around 57% (few-shot CoT prompting). On MATH, the gains were smaller but still meaningful.

What CoT does mechanically: it conditions the answer token on a long sequence of reasoning tokens, which lets the model use intermediate computation rather than producing the answer in one forward pass. The model that produces "3 + 4 + 5 = 12" before the answer is doing work that the model producing only "12" can't do.

CoT doesn't require any training. It's a prompting technique. But its existence is what made RLVR worth trying: if the model can already produce correct chains some fraction of the time, you have positive examples to filter and train on. Without CoT-capable base models, the reward signal from a math verifier would be nearly always zero, and there would be nothing for GRPO to amplify.

Self-consistency (Wang et al. 2022, arXiv:2203.11171) is the natural complement. Sample many chains from the model, extract the final answer from each, take the majority answer. Performance scales with sample count up to some plateau. On GSM8K, self-consistency at K=40 added roughly 18 points on top of greedy CoT.

The mechanism is simple: if the model produces the correct answer more often than any single incorrect answer, the majority vote is more reliable than any single sample. This requires the model's distribution over answers to have the correct answer as the mode, which holds for problems where the model has "real" capability (it can derive the answer multiple ways), and fails for problems where the model's failures cluster on the same wrong answer.

Self-consistency is the cheapest version of inference-time compute scaling. Lecture 15 covers more sophisticated variants (best-of-N with a reward model, beam search with a PRM). The connection to RLVR: any technique that improves single-sample pass rate also improves majority-vote pass rate, so RL training and inference-time scaling are complementary, not alternatives.

### Program-aided language models

PAL (Gao et al. 2022, arXiv:2211.10435) takes the chain-of-thought idea and offloads the arithmetic to a Python interpreter. The model generates code that computes the answer rather than producing the answer in natural language. The interpreter runs the code and returns the result.

On GSM8K, PAL with a 540B Codex-class model hit 72%, substantially better than CoT alone on the same model. The improvement is mostly on problems that require multi-step arithmetic, where the model knows the right operations but is unreliable at executing them token-by-token. Offloading the arithmetic to a deterministic interpreter removes that failure mode entirely.

PAL is a precursor to the tool-use story later in this lecture. The reward structure changes once you have a tool in the loop: the model is no longer being credited for arithmetic, only for getting the problem framing right and calling the right tool. That has implications for what RL training shapes.

---

## STaR: self-training from correct chains

STaR (Self-Taught Reasoner, Zelikman et al. 2022, arXiv:2203.14465) was the first system to convert CoT-style prompting into a training signal at scale. The pipeline:

1. Sample chains of thought from the current model on a labeled math dataset.
2. For each problem, check whether the final answer matches the labeled correct answer.
3. Keep the (problem, chain, answer) triples where the answer was correct.
4. SFT the model on those triples: train it to produce the chain it produced when it got the right answer.
5. Repeat.

For problems where the model fails on every attempt, STaR adds a "rationalization" step: condition the model on the correct answer and ask it to generate a chain that leads to it. The rationalized chain isn't necessarily faithful to how the model "would have" solved the problem, but it provides a positive training example for problems the model otherwise can't get any signal from.

STaR is interpretable as expectation-maximization. The E step samples chains from the current policy and filters to ones consistent with the correct answer. The M step updates the policy to make those chains more likely under the model. The "latent variable" is the reasoning chain, which is unobserved (only the answer is labeled) and approximated by filtered samples.

What STaR establishes for everything later:

- A verifiable checker turns sampled model outputs into a clean training set. The checker is what separates "self-training" from "training on whatever the model produced."
- Iteration matters: as the model improves, it succeeds on harder problems, which feed back into the training set. The corpus difficulty grows with the model's capability.
- A bootstrapping problem exists: if the model can't solve any problems in a class at all, no amount of self-training on that class will fix it. The rationalization step is one patch; using a stronger teacher (the distillation approach from [Lecture 18](./18-distillation-reasoning.md)) is another.

STaR isn't an RL method: there's no gradient flowing through a reward signal, just an SFT loss on filtered chains. But the filter-and-train loop is the same template that GRPO formalizes with policy gradients. Practically, STaR is cheaper than RL and works fine when the base model has enough latent capability to solve problems often enough that filtering yields a usable corpus.

---

## Process supervision: PRM800K

Outcome supervision tells the model whether its final answer was right. Process supervision tells the model whether each intermediate step was right. Lightman et al. 2023 (arXiv:2305.20050) released PRM800K, 800,000 step-level human-feedback labels on MATH solutions, and trained a process reward model on top.

The dataset construction: take chains of thought generated by a strong model on MATH problems, segment them into steps (roughly sentence-level), have human annotators label each step as positive (correct and useful), neutral (true but not progressing), or negative (incorrect). Train a reward model to predict step labels given the prefix up to that step.

The PRM is then used for best-of-N selection at inference time: generate N chains, score each step with the PRM, sum or aggregate per-chain, return the chain with the highest aggregate score. Lightman et al. showed this outperforms outcome-based best-of-N (where the reward model only sees the final answer) at matched N. The process reward model is more reliable because it can identify and downweight chains that arrived at correct answers via flawed reasoning.

The process reward model can also be used for training, not just inference. The gradient signal is much denser than an outcome reward: every step gets a score, instead of one number per chain. This denser signal helps the model learn what good intermediate steps look like, not just what correct final answers look like. The Uesato et al. 2022 paper (arXiv:2211.14275) had run an earlier comparison of process vs. outcome supervision on GSM8K and found them comparable in final-answer accuracy but different in reasoning-error rate; the Lightman PRM800K work is the version that scaled the approach and showed clearer wins on MATH.

Why PRMs haven't completely displaced outcome supervision: cost. 800K step-level human labels is expensive. Most current systems use outcome supervision (DeepSeek-R1 uses outcome rewards in its RL stage, not a PRM, by the description in arXiv:2501.12948). PRMs show up more often at inference time, where they're used for best-of-N selection over chains generated by a model trained with outcome supervision. The training pipeline ends up being: outcome RL → PRM-based inference search.

There's also a quieter issue with PRMs as training rewards: the per-step credit can be miscalibrated in ways that bias the policy. A PRM that over-rewards verbose intermediate steps pushes the policy toward verbose chains regardless of whether they're more correct. A PRM that under-rewards skipping clear-but-redundant steps pushes the policy toward including those steps even when they don't add information. These miscalibrations are real-world phenomena reported across PRM-trained systems, and they're hard to detect because the per-step rewards aren't always inspectable in detail after training.

A pragmatic middle position several recent systems take: outcome rewards for RL training (simpler, less calibration-sensitive), PRM for inference-time search (where miscalibration matters less because you're ranking generated chains, not training against the scores). This sidesteps both the data cost of PRM training and the calibration risk while still capturing the inference-time benefits.

[Lecture 15](./15-rl-verifiable-rewards.md) goes into the process-vs-outcome tradeoff in more detail. The short version: process supervision helps most when correct answers can be reached via incorrect reasoning, which is more common on harder problems with longer chains.

---

## DeepSeekMath and GRPO

DeepSeekMath (Shao et al. 2024, arXiv:2402.03300) introduced GRPO and applied it to math reasoning. The algorithmic mechanics are in Lecture 15; what's worth surfacing here is what the paper did specifically with math.

The setup:

1. Pre-train a base model (DeepSeek-LLM 7B in the paper) on a large math-heavy corpus. The corpus construction was a substantial part of the work: they filtered Common Crawl for math content using a classifier trained on known-good math text, then iteratively refined.
2. SFT on a math instruction-tuning dataset.
3. Run GRPO with rule-based rewards. The reward is binary: 1 if the extracted final answer matches the labeled correct answer (after normalization), 0 otherwise.

The result was a 7B model that scored 51.7% on MATH and 88.2% on GSM8K, putting it above much larger models that hadn't been math-specialized at the time.

The two key engineering details the paper emphasized:

- Pre-training corpus matters more than the RL algorithm. Most of the gain over a comparable base model came from the math-heavy pre-training. The RL step sharpens what's already in the base model; it doesn't conjure capability.
- The verifier needs to handle equivalent forms. The reward extraction step uses a math-specific parser that normalizes fractions, decimal representations, ordered tuples, and other formats. A naive string-match verifier would reject correct answers in alternate forms, which artificially deflates the reward signal.

DeepSeekMath is the algorithmic precursor to R1. The R1 paper (arXiv:2501.12948) uses the same GRPO machinery and the same kind of rule-based reward, applied at a much larger scale and with the cold-start SFT modification described next.

---

## DeepSeek-R1: the full recipe

The R1 paper (DeepSeek-AI 2025, arXiv:2501.12948) describes two variants:

- **R1-Zero**: GRPO applied directly to a base model, no SFT warm-up. Just RL on a base language model with a rule-based reward.
- **R1**: cold-start SFT on a small dataset of high-quality reasoning traces, then GRPO, then a final alignment pass.

Lecture 15 covers the full pipeline. Here, the focus is on the math-specific details: what was in the training data, what the reward function actually computed, what the loss and response-length curves did, and what the "aha moment" looked like.

### Training data

The math portion of the RL training data was a mix of public math problem corpora with verified answers. The R1 paper doesn't give a full breakdown, but the typical sources for this kind of training are:

- MATH (Hendrycks et al. 2021, arXiv:2103.03874) training split, with the answer labels.
- GSM8K training split.
- AIME problems from earlier years (1980s–2010s, with 2024 held out for evaluation).
- Olympiad problems with verified answers (similar to OlympiadBench, but pulled from various competition archives).
- Synthetic problems generated by a teacher model and filtered by another model or a symbolic solver.

The exact mix isn't documented in the paper. What matters for training stability is that the problems have a difficulty distribution that gives the model some signal: too easy and group-relative advantages collapse to zero (everyone gets r=1), too hard and they collapse the other way (everyone gets r=0). The training curriculum is implicitly shaped by which problems land in the "10-80% success rate" zone for the current policy.

Code and other domains were mixed in alongside the math problems. Both contribute verifiable reward signal, and the policy improvement on one transfers somewhat to the other through shared reasoning patterns (read-the-problem-carefully, plan-then-execute, check-your-work). The total RL training set was described as substantial: hundreds of thousands of problems with verified answers, across math, code, and adjacent domains.

### Reward function

For math problems, the reward has two parts.

**Correctness reward.** Extract the final answer from the model's output (between `\boxed{...}` or after "Answer:"), normalize it, compare to the labeled answer. Returns 1 on match, 0 otherwise. The normalization handles:

- Whitespace and punctuation.
- Equivalent numeric forms (`1/2`, `0.5`, `0.500`).
- Equivalent symbolic forms (`\frac{1}{2}`, `1/2`, `0.5`).
- Equivalent unit representations where appropriate.
- Set and tuple orderings where the problem allows.

**Format reward.** Check that the response uses the expected structure: a `<think>...</think>` block containing reasoning, followed by a final answer. Returns a small positive value (around 0.1) if the format is correct, zero otherwise.

Total reward = correctness + format, treated as a scalar by GRPO. The format reward provides dense early-training signal (the model learns to produce structured output within hundreds of steps); the correctness reward provides the long-term learning signal that shapes reasoning.

This is the math version of the general RLVR reward function from [Lecture 15](./15-rl-verifiable-rewards.md). Code uses a similar structure with `pass_rate(tests)` substituted for `numeric_match(answer)`.

### What the loss looks like

The training dynamics reported in the R1 paper, as far as is publicly described:

- **Format reward** rises quickly (within a few hundred steps) and plateaus near maximum. The model learns the output structure long before it learns to solve harder problems.
- **Correctness reward** rises more gradually over thousands of steps. The shape is concave: fast early improvement on the easier problems in the distribution, slower improvement on the harder ones.
- **Response length** grows substantially over training. The R1 paper highlights this: average tokens per response increases from a few hundred at the start of RL to many thousands by the end. The growth isn't explicitly rewarded: it emerges because longer chains are more often correct on hard problems, so the policy learns to extend chains where extension helps.
- **KL divergence to the reference policy** grows steadily. With `kl_beta` around 0.04 (typical), the KL plateaus rather than exploding, but it doesn't return to zero: the policy has genuinely shifted away from the reference distribution.

The response-length growth deserves emphasis. It's one of the more interesting features of R1-style training: the model spontaneously learns to allocate more tokens to harder problems. Looking at a curve of average response length vs. training step, the growth is roughly monotonic and substantial: often 5–10x increase from start to end of RL training. This is what people mean by "test-time compute emerging from RL": the model itself learns to spend more inference compute where it helps, without being explicitly told to.

### The "aha moment"

The R1 paper describes a qualitative phenomenon: at some training step (the paper shows it at around step 8000 of R1-Zero training), the model begins producing explicit reflection in its chains, phrases like "wait, let me reconsider" or "I made an error above; let me try again." These weren't in the training data (R1-Zero starts from a base model with no SFT), and they weren't explicitly rewarded. They emerged because backtracking from incorrect intermediate steps is instrumentally useful for getting the correctness reward.

The "aha moment" framing is somewhat romanticized: the underlying mechanism is just that the policy gradient pushes toward behaviors that correlate with reward, and explicit self-correction correlates with reward on harder problems where the first attempt is more often wrong. But the emergence is genuine: the model develops self-reflection behaviors without any direct training signal for them.

This is the strongest piece of evidence for the "RL amplifies latent capability" interpretation discussed in Lecture 15. The base model could already produce self-reflective text (it was in the pre-training corpus somewhere), but rarely. RL made it systematic.

A subtlety worth being explicit about: the "aha moment" is reported as a step-localized phenomenon in the paper's figures, but it's not a phase transition in the mathematical sense. The frequency of self-reflective phrases in the model's outputs grows continuously over training; what makes the moment notable is that it crosses a threshold of being visible and consistent rather than sporadic. Models earlier in training also produce these phrases occasionally, just not reliably enough that you'd say the model "uses self-reflection." The paper's framing is partly narrative; the underlying dynamics are gradient-driven and gradual. Don't expect a clean spike when you reproduce this in your own training runs; expect a slow climb that becomes visible once it's above some inspection threshold.

### R1-Zero failure modes

R1-Zero (the base-model + RL only version) developed problems that R1 (with cold-start SFT) doesn't have:

- **Language mixing.** Without an SFT pass enforcing consistent output language, the model would switch between English and Chinese mid-chain. The format reward doesn't penalize this; the correctness reward doesn't either, as long as the final answer is right.
- **Low readability.** Without legibility supervision, R1-Zero's chains can be terse, jargony, or structurally idiosyncratic. They still reach correct answers but aren't easy for humans to follow.
- **Style drift.** Various stylistic features that depend on having been trained on human-curated text show up only weakly in R1-Zero.

The cold-start SFT in R1 (a few thousand high-quality reasoning traces, used for a light SFT pass before RL begins) addresses these issues by giving the policy a strong prior on what legible reasoning looks like. The RL stage then improves the reasoning capability on top of an already-legible base.

### A practical detail: response truncation

A real engineering issue in long-response RL training: the response length growth interacts with the model's max context. If the model wants to produce a 32K-token chain but the training infrastructure truncates at 8K, the truncated response gets scored as a failure (no answer found), which gives a misleading reward signal. The R1 paper increased the max generation length during training to accommodate the growth. If your max length is too small relative to what the model would produce, training stalls or regresses on hard problems.

---

## Other systems: o1, QwQ, and what's not public

OpenAI's o1 (announced September 2024) was the first widely-publicized reasoning model in this line of work. The official source is OpenAI's blog post ("Learning to reason with LLMs," September 2024); there's no full technical paper.

What's documented publicly:

- o1 is trained with RL on chain-of-thought.
- Performance on math and coding benchmarks scales with inference token budget: longer reasoning traces score higher, up to a plateau.
- The training compute and the inference compute are described as two separable levers.

What isn't documented publicly: the specific RL algorithm, whether a learned PRM is used for inference search, the reward function structure, how inference compute is allocated. Anything beyond the blog post is speculation. Treat claims about o1's internals from secondary sources with skepticism: the official description is intentionally high-level, and the people who know aren't writing public papers.

Qwen's QwQ ("Qwen with Questions," released as a preview by Alibaba's Qwen team in late 2024) is another reasoning model in the same lineage. As of writing, there's no full technical paper for QwQ either: it's been described in blog posts and release notes. The behavior is similar to R1-Zero and o1: long structured reasoning chains, scaling with inference compute, RL training on verifiable rewards.

The pattern: OpenAI announced o1 in September 2024, DeepSeek released R1 in January 2025 with a full open paper and open weights, and several other groups followed with similar systems. R1 is the system you can study in depth because the paper, weights, and (subsequently) reproductions are all available. The others share a recipe family with R1 but have to be inferred from external behavior and partial disclosures.

For learning purposes: read the R1 paper carefully, treat o1 and QwQ as confirmation that the recipe generalizes across labs, and don't trust unsourced claims about how the closed-source systems work internally.

---

## A code sketch: the math RLVR rollout loop

The full GRPO update is in [Lecture 15](./15-rl-verifiable-rewards.md). What's specific to math is the rollout-and-verify loop: sampling K completions per problem, extracting and normalizing answers, comparing to gold, producing the rewards tensor that GRPO consumes.

This is a simplified version of the math verifier and rollout loop. It's not optimized; it's meant to make the structure clear.

```python
import re
import torch
from fractions import Fraction
from typing import Optional


# ── Answer extraction ──────────────────────────────────────────────────────

ANSWER_PATTERNS = [
    r"\\boxed\{([^{}]+)\}",        # \boxed{42} or \boxed{1/2}
    r"Answer\s*[:=]\s*([^\n]+)",   # Answer: 42
    r"answer is\s*([^\.\n]+)",     # ...answer is 42.
]


def extract_answer(response: str) -> Optional[str]:
    """
    Pull the final answer out of a model response.

    Tries each pattern in order; returns the last match (a model often
    writes intermediate "Answer: X" lines, and the final one is the
    committed answer).
    """
    for pattern in ANSWER_PATTERNS:
        matches = re.findall(pattern, response)
        if matches:
            return matches[-1].strip()
    return None


# ── Answer normalization ───────────────────────────────────────────────────

def _strip_units(s: str) -> str:
    """Remove trailing units, common formatting."""
    s = s.replace("$", "").replace("%", "").strip()
    s = re.sub(r"\s+(degrees?|°|cm|m|km|kg|g)$", "", s, flags=re.I)
    return s


def _normalize_fraction(s: str) -> Optional[Fraction]:
    """
    Try to parse s as a fraction.
    Handles: '1/2', '0.5', '-3/4', '\\frac{1}{2}'.
    Returns None if parsing fails.
    """
    s = s.replace(" ", "")
    # LaTeX \frac{a}{b}
    frac_match = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if frac_match:
        num, den = int(frac_match.group(1)), int(frac_match.group(2))
        return Fraction(num, den)
    # plain a/b
    if re.fullmatch(r"-?\d+/-?\d+", s):
        num, den = s.split("/")
        return Fraction(int(num), int(den))
    # decimal
    try:
        return Fraction(s).limit_denominator(10**6)
    except (ValueError, ZeroDivisionError):
        return None


def normalize_answer(s: str) -> str:
    """
    Normalize an answer string so equivalent forms compare equal.
    Returns a canonical string representation.
    """
    s = _strip_units(s)
    # Try fraction normalization first: handles most numeric cases.
    f = _normalize_fraction(s)
    if f is not None:
        if f.denominator == 1:
            return str(f.numerator)
        return f"{f.numerator}/{f.denominator}"
    # Fall back to lowercased, whitespace-stripped string.
    return s.lower().replace(" ", "")


def numeric_equal(a: str, b: str) -> bool:
    """Compare two answer strings under normalization."""
    return normalize_answer(a) == normalize_answer(b)


# ── Format check ───────────────────────────────────────────────────────────

THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def has_valid_format(response: str) -> bool:
    """
    Check that the response has the expected structure:
    a <think>...</think> block followed by an answer.
    """
    if not THINK_PATTERN.search(response):
        return False
    if extract_answer(response) is None:
        return False
    return True


# ── Reward function ────────────────────────────────────────────────────────

def math_reward(response: str, gold_answer: str) -> float:
    """
    Compute the RLVR reward for one math response.

    Reward = format_bonus + correctness.
    Format bonus is small (0.1) so it dominates only early in training.
    """
    format_bonus = 0.1 if has_valid_format(response) else 0.0

    extracted = extract_answer(response)
    if extracted is None:
        return format_bonus

    correctness = 1.0 if numeric_equal(extracted, gold_answer) else 0.0
    return format_bonus + correctness


# ── Rollout-and-verify loop ────────────────────────────────────────────────

def rollout_and_score(
    policy,                   # model with a .generate(prompt, n, temperature) method
    problems,                 # list of (prompt, gold_answer) tuples
    K: int = 16,              # samples per problem
    temperature: float = 0.7,
) -> tuple[list[list[str]], torch.Tensor]:
    """
    For each problem, sample K completions and score each one.

    Returns:
        completions: list of length B; each element is a list of K response strings.
        rewards:     tensor of shape [B, K] with float rewards.
    """
    B = len(problems)
    completions: list[list[str]] = []
    rewards = torch.zeros(B, K)

    for b, (prompt, gold) in enumerate(problems):
        responses = policy.generate(prompt, n=K, temperature=temperature)
        completions.append(responses)
        for i, response in enumerate(responses):
            rewards[b, i] = math_reward(response, gold)

    return completions, rewards


# ── Example use ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Fake policy that returns canned responses for demonstration.
    class FakePolicy:
        def generate(self, prompt, n, temperature):
            # In reality this would forward-pass the model; here we
            # just return varied strings so the scoring shows variety.
            return [
                f"<think>Let me think.</think>\nAnswer: 12",
                f"<think>Computing.</think>\nAnswer: \\boxed{{1/2}}",
                f"<think>Wrong answer here.</think>\nAnswer: 99",
                f"No format here, just 12.",
            ][:n]

    problems = [
        ("What is 3 * 4?", "12"),
        ("What is 1/4 + 1/4?", "1/2"),
    ]
    policy = FakePolicy()
    completions, rewards = rollout_and_score(policy, problems, K=4)

    for b, (prompt, gold) in enumerate(problems):
        print(f"\nProblem: {prompt} (gold: {gold})")
        for i in range(rewards.shape[1]):
            print(f"  K={i}: reward={rewards[b, i].item():.2f}  | response: {completions[b][i][:60]!r}")
```

The rewards tensor returned by `rollout_and_score` goes straight into `compute_group_advantages` from Lecture 15, which produces the group-relative advantages that GRPO's loss uses. The rest of the training loop is identical to the general RLVR setup.

Things worth noticing in the verifier code:

- The answer extractor returns the *last* matched answer, not the first. Models often write intermediate "the answer might be X" phrases before committing; the final answer is what counts.
- The normalization handles LaTeX `\boxed{}` and `\frac{}{}` because MATH labels them this way, and a naive string match would reject `\frac{1}{2}` against `1/2`.
- The format check is generous on purpose. A strict check (e.g., requiring specific whitespace around `<think>` tags) gives more false negatives, which makes the format reward noisier without much benefit.
- The fraction normalization uses `Fraction.limit_denominator(10**6)` to handle decimal approximations. Without this, `0.333` would fail to compare equal to `1/3`. With it, `0.333` becomes `333/1000` after parsing, which is still not `1/3`, so the comparison falls through to string match. This is a real edge case in MATH verifiers; some implementations accept "close enough" decimal answers, others require exact rational equality. Decide which based on the dataset.

The verifier is the most failure-prone piece of an RLVR math system. When the reward isn't behaving as expected, the bug is more often in the verifier than in the RL algorithm.

---

## Math-specific failure modes

Lecture 15 covers reward hacking in general. This section covers what shows up specifically when the verifier is a math answer checker.

### Spurious correlations in the answer distribution

If the training set has systematic patterns in the answer distribution (say, problems whose correct answer is a small positive integer between 1 and 20 are overrepresented), the model can learn to bias its guesses toward that range and pick up reward without doing the underlying reasoning. The model isn't solving the problem; it's pattern-matching on superficial features that correlate with answer range.

This is detectable: hold out a test set where the answer distribution doesn't match training, and measure whether the model's accuracy drops. If it does, the model was surface-matching. GSM8K is somewhat susceptible to this because the answers cluster on small integers; AIME (where answers are integers 0–999, uniformly distributed) is less susceptible.

### Memorization of problem types

A more subtle version: the model memorizes the answer pattern for specific problem types rather than learning to derive them. "Find the area of a triangle with sides 3, 4, 5" always has the same answer (6), and the model can learn to map the surface features to the answer without doing the Heron's formula computation.

This is hard to detect on standard benchmarks because the training data and the test data often share problem types. A model that memorizes "the area of a 3-4-5 triangle is 6" looks just as good on a test of 3-4-5 triangle problems as a model that derives it. To detect memorization, you need novel problem types not present in any standard math corpus, which is part of why AIME 2024 became the de facto frontier benchmark in 2025: the problems were post-cutoff for most pre-training corpora, so the test set was less likely to overlap with training.

The general principle: be skeptical of benchmark numbers when the model's training data contains close paraphrases or templates of the benchmark problems. The RL training stage doesn't create memorization (the gradients are usually weak enough that it can't memorize 800K problems verbatim), but the pre-training and SFT stages can.

A useful test for memorization on a specific benchmark: take the problems and perturb the numbers (or the variables, or the phrasing) without changing the underlying mathematical structure. A 3-4-5 triangle area problem becomes a 5-12-13 triangle area problem. A "find x such that 2x + 3 = 11" problem becomes "find y such that 3y + 4 = 13." If the model's accuracy drops substantially on the perturbed version while staying high on the original, memorization is the most likely explanation. If accuracy stays roughly the same, the model is doing more general work. Some recent benchmark papers (post-2024) bake this kind of perturbation in by construction, generating multiple variants of each problem; older benchmarks don't, which is why perturbation testing is a standard sanity check now.

### Verifier weaknesses

The math verifier is a stack of normalization rules and equivalence checks. Every weakness in that stack is a place where the reward signal disagrees with "actually correct":

- **Numerically equal but textually different forms.** `1/2 == 0.5 == 2/4`, but a naive checker might reject 2/4. Most production checkers reduce fractions; many don't handle decimal-to-fraction equivalence.
- **Symbolically equal forms.** `\sqrt{2}` vs `2^{1/2}` vs `1.414`. Symbolic equivalence requires a CAS (computer algebra system) to check reliably.
- **Set and tuple representations.** If the answer is the set `{1, 2, 3}`, the model might write `1, 2, 3` or `[1, 2, 3]` or `(1, 2, 3)` or `{3, 2, 1}`. The checker has to decide which of these count.
- **Units.** "5 meters" vs "5 m" vs "5": depends on whether the question specifies units.
- **Interval notation.** `(0, 1)` (open interval) vs `0 < x < 1` vs `]0, 1[` (continental European convention).

Every one of these is a place where a correct chain gets zero reward because the answer extractor or normalizer didn't handle the form. The model's policy can't tell whether it's being penalized for reasoning errors or verifier mismatches: they look the same in the reward signal.

The practical mitigation: log a sample of "incorrect" responses periodically and check by hand whether they're verifier failures rather than reasoning failures. If 10% of zero-reward responses turn out to be correct answers in unexpected forms, fix the verifier: that's a meaningful chunk of training signal you're throwing away.

### Length hacking

If something in the pipeline rewards length (a length bonus, or a correlation between length and correctness that the model latches onto), the model can produce arbitrarily long chains without adding reasoning. R1-Zero in particular sometimes produces very long chains with substantial padding, repetition, or redundant verification steps.

This isn't always bad: longer chains often are better, as the response-length growth in R1 training shows. But the failure mode to watch for is unproductive length: chains that grow because length itself was implicitly rewarded, not because longer reasoning was solving harder problems. Monitor average chain length vs. average correctness over training. If length keeps growing while correctness has plateaued, the policy may be over-investing in length.

### Reward signal from leakage in the problem statement

Some math problems contain hints in the problem statement that the model can pattern-match without doing the math. "Find the largest prime less than 100. Hint: it's 97" is an extreme example; subtler versions appear in problems where the answer is implicit in the setup. A model trained against a verifier on these problems learns to mine the problem statement rather than reason from first principles. Filter your training set for problems whose answer is too easily extractable from the prompt: usually by running a zero-CoT baseline (just predicting the answer from the prompt directly) and excluding problems where the baseline does well.

---

## Tool use: code execution and formal provers

Math reasoning splits cleanly into two regimes based on what the model has to do unaided.

**Pure-LLM math** (the GSM8K/MATH/AIME setting described above): the model produces natural-language reasoning and a final answer, and the only external check is on the final answer. Everything else (arithmetic, algebra, symbolic manipulation) happens inside the model's chain of thought, in natural language tokens. The model is doing math from scratch each time, with no calculator and no proof assistant to verify intermediate steps.

**Tool-augmented math**: the model can call an external tool to handle some part of the work. Common tools include a Python interpreter (PAL-style), a CAS, a formal proof assistant (Lean, Coq, Isabelle, Metamath), or a calculator. The model produces code or proof terms; the tool executes them; the result is fed back into the model's context.

The shift from pure-LLM to tool-augmented changes what the model needs to do and what the reward shapes:

- **Arithmetic offload.** Once the model can call a Python interpreter, it no longer needs to do arithmetic in natural-language tokens. The reward stops shaping arithmetic skill and starts shaping "figure out what to compute and call the right function." This is what PAL (Gao et al. 2022, arXiv:2211.10435) demonstrated: a model that calls Python beats a same-sized model doing arithmetic in tokens on math benchmarks where arithmetic is a bottleneck.
- **Symbolic offload.** With a CAS, the model can offload algebraic manipulation, integration, factorization. The model decides what symbolic operation to apply; the CAS executes it.
- **Proof checking offload.** For formal math, the proof assistant is the ground truth. The model proposes proof steps; the assistant accepts or rejects each one. The reward signal is per-step (does this tactic close this goal or produce valid subgoals), which is denser than the outcome-only reward in pure-LLM math.

The tool-augmented setting changes the reward structure in important ways. In pure-LLM math, the reward is sparse: one number per chain, only on the final answer. In tool-augmented math, the tool calls produce intermediate signals: a Python interpreter that errors out tells you the code was wrong; a proof tactic that closes a goal tells you the step was correct. These intermediate signals can be used as part of the reward, giving GRPO a denser learning signal than pure outcome supervision.

### Formal proofs: AlphaProof and the IMO 2024 result

DeepMind announced in July 2024 that a combination of two systems (AlphaProof and AlphaGeometry 2) achieved a silver-medal-level performance on the 2024 International Mathematical Olympiad, solving 4 of 6 problems. AlphaProof was the system for the algebraic and number-theoretic problems, and it used the Lean proof assistant as its verifier.

There is, as of writing, no full technical paper for AlphaProof. The announcement was a blog post; a paper was suggested as forthcoming. Treat this section accordingly: the high-level approach is publicly stated, the details are not.

What's publicly described:

- AlphaProof translates natural-language IMO problems into Lean problem statements (a substantial step itself, requiring careful formalization).
- It then uses an RL system to search the space of Lean proofs, similar in flavor to AlphaZero's search but applied to proof construction rather than game play.
- The reward signal is binary at the proof level (the Lean compiler accepts or rejects), with a denser per-step signal for individual tactic applications.

The architecture is reported to involve a learned policy and value function with Monte Carlo tree search over proof states, in the AlphaZero lineage. The training data is generated by self-play on a large library of Lean problems, with the proof assistant as the verifier at every step.

This is the formal-math analog of the natural-language reasoning systems described earlier. The verifier is stricter (Lean accepts only correct proofs, with no ambiguity) and the reward signal is denser (per-tactic, not just per-proof), but the high-level RL recipe is similar to RLVR on math.

What earlier work this builds on:

- GPT-f (Polu and Sutskever 2020, arXiv:2009.03393) was an early system applying transformer-based language models to formal theorem proving in Metamath, with reinforcement learning over proof search.
- HyperTree Proof Search (HTPS, Lample et al. 2022, arXiv:2205.11491) extended this to Lean and Metamath with explicit tree search over proof tactics.
- "Draft, Sketch, and Prove" (Jiang et al. 2022, arXiv:2210.12283) connected informal natural-language proof sketches to formal proof construction in Isabelle.
- MiniF2F (Zheng, Han, Polu 2021, arXiv:2109.00110) provided the cross-system benchmark these methods are evaluated against.

AlphaProof builds on this line of work: RL on top of a Lean-based proof search, scaled up substantially and combined with strong autoformalization (translating natural-language statements into formal Lean). The IMO 2024 result was widely treated as a milestone, but the absence of a full paper means the specifics aren't reproducible from public information yet.

### When to use tools in RL training

The decision of whether to allow tool use during RL training is partly about what behavior you want to shape and partly about what infrastructure you can support.

If you want the model to do math in pure natural language (good for inspectability, simplicity of deployment), don't add tools. The model has to internalize arithmetic and symbolic manipulation. R1-style training is in this mode.

If you want maximum performance on a math benchmark and don't mind the deployment complexity, allow Python execution. PAL-style systems with RL training can hit higher accuracy on the same benchmark with less per-step inference cost (no need to spell out arithmetic in tokens), at the cost of needing a sandboxed code execution environment in both training and deployment.

If you want formal correctness guarantees (no plausible-but-wrong reasoning), use a proof assistant. The verifier is exact, and the reward signal is denser. The cost is that everything has to be formalizable, which excludes most natural-language problem statements unless you also build a translation layer.

The tradeoffs aren't unique to math: they show up wherever RL meets a domain with tool support. But math is where the design space has been most thoroughly explored.

One under-discussed cost of tool-augmented RL: the rollout pipeline gets slower and harder to parallelize. Pure-LLM rollouts are dominated by GPU forward passes, which batch cleanly. Tool-augmented rollouts have a tool-call step where the GPU sits idle waiting for an external process (a Python sandbox spinning up, a Lean compiler checking a proof). At K=16 rollouts per problem and a batch of hundreds of problems, the cumulative tool-call latency can be the bottleneck. Practical setups handle this with asynchronous evaluation pipelines, pre-warmed tool processes, and batching tool calls across rollouts where possible. None of this is conceptually hard, but it's a non-trivial engineering layer that pure-LLM RLVR doesn't need.

---

## Practical training dynamics

Most of what's in Lecture 15's "practical training dynamics" section applies here. What's worth adding specifically for math:

- **Curriculum on difficulty.** If your training set is mostly easy problems (GSM8K-style), the model saturates quickly and the gradient signal goes to zero. If it's mostly hard problems (AIME-style), pass rates are too low to give signal on most problems. A mix that lands the model in the 20-70% pass rate range per problem is what gives the most useful gradient. In practice this is handled by including problems across difficulty levels and letting the implicit curriculum shape itself: easier problems contribute signal early in training, harder problems contribute signal later.
- **Length stability.** Watch the average response length over training. R1-style growth is healthy (longer chains, higher accuracy). Sudden spikes in length without corresponding accuracy improvements are usually length hacking: investigate.
- **Verifier sampling.** Periodically pull random "incorrect" responses and check them by hand. If a non-trivial fraction are actually correct answers in forms the verifier rejected, fix the verifier. This pays for itself: a verifier that accepts 5% more correct answers is equivalent to 5% more positive training examples.
- **Reward-distribution per problem.** Log the distribution of rewards within each problem group across training. Healthy distributions have a mix of high and low rewards (which gives advantages with non-zero variance). All-zero or all-one groups produce no learning signal. If most groups are degenerate, your problem difficulty is mismatched to your model's capability.

The Lecture 15 hyperparameter table is a reasonable starting point for math RLVR (K=4-16, clip_eps=0.1-0.3, kl_beta=0.01-0.1, lr around 1e-6). The one parameter worth tuning more carefully for math is `K`: math is cheap to verify (a regex match, not a code execution), so larger K is more feasible than for code. K=16-32 is reasonable when the verifier is fast.

---

## When to use this

The full math-reasoning recipe is worth using when:

- You want frontier-class math performance from an open-weight or self-trained model.
- You have access to a verifiable problem corpus with reliable answer labels.
- You can run inference on a large base model at scale (for either RL rollouts or teacher-generation in a distillation setup).
- The math domain is "in-distribution" for the kind of math you care about: competition-style problems, with short numeric or symbolic answers. The recipe transfers less directly to research-level mathematics where the "answer" is a proof or a construction.

The cheaper paths that may be sufficient:

- **CoT prompting + self-consistency** on a strong base model gets surprisingly close to RL-trained models on easier benchmarks like GSM8K. If your needs are bounded to that level, you might not need RL at all.
- **STaR on a smaller model** can get meaningful gains on medium-difficulty problems without the GRPO infrastructure.
- **Distillation from a strong reasoning teacher** (see [Lecture 18](./18-distillation-reasoning.md)) gets a small model most of the way without running RL on the student. If R1 or similar is available as a teacher, this is the cheapest path to a reasoning-capable small model.

The recipe is not the right approach when:

- The "math" you care about is open-ended (research-style derivations, novel constructions). The verifier can't check these reliably; the reward signal degrades to noise.
- You don't have answer labels. RLVR needs a checker; without it, you're back to preference-based methods or LLM-judge feedback, which are weaker.
- Your base model has no math capability at all. RL amplifies what's there; it doesn't conjure capability from nothing. A model with a pre-training corpus that includes no math text is not going to learn math from RL on math problems.

A practical decision tree for picking among the options:

```
Do you need frontier-class performance on competition math (AIME, OlympiadBench)?
├── No: GSM8K / MATH-500 level is enough?
│    ├── Have a strong base model? → CoT prompting + self-consistency may suffice.
│    └── Smaller base model? → STaR or distill from an open reasoning teacher.
└── Yes
     ├── Have GPU budget for RL at scale + a strong base?
     │    └── Run the R1-style recipe: cold-start SFT → GRPO → final alignment.
     └── Have access to a strong open reasoning teacher (R1, QwQ)?
          └── Distill chains → SFT student → optionally short RL pass on top.
```

The path that's clearly dominated: running GRPO from scratch on a small model without distillation from a strong teacher first. The R1 paper showed this produces worse results than distilling from a larger RL-trained teacher. If you have the option to distill, do that before reaching for direct small-model RL.

---

## Exercises

- **Build a MATH verifier.** Take the MATH test set, pick a 100-problem subset, and write an answer extractor + equivalence checker for it. Compare your verifier's accept rate on known-correct human solutions to a baseline (string-match only). The gap is the fraction of correct answers your verifier handles that string-match doesn't. Aim for >95% accept rate on labeled correct solutions.
- **Run STaR on a small math model.** Use a 1B-class model and the GSM8K training set. Sample chains, filter to correct ones, SFT, repeat. Measure pass@1 on GSM8K test after each round. Plot the curve. Notice where it plateaus: usually after 2–4 rounds, depending on the model's starting capability.
- **Self-consistency curve.** Take a CoT-prompted model on GSM8K. Sample K chains per problem for K in {1, 2, 4, 8, 16, 32, 64}. Plot majority-vote pass rate vs K. Where does the curve flatten? Compare to single-sample greedy pass rate: the gap is your inference-time compute benefit.
- **Verifier-failure audit.** On the same GSM8K runs, pull random "incorrect" responses and check by hand whether they're verifier failures or reasoning failures. Estimate the fraction of each. This calibrates how much faith to put in benchmark numbers.
- **Length vs. accuracy correlation.** On a model that produces variable-length chains (any CoT model will do), compute the correlation between chain length and correctness across a benchmark. Is the correlation positive (longer chains do better) or negative (correct answers can be reached quickly)? The R1 training dynamics suggest positive on hard problems, near-zero on easy ones. Reproduce this on your benchmark.
- **Answer normalization sensitivity.** Take a MATH verifier and intentionally weaken the normalization (e.g., remove fraction reduction, remove LaTeX `\frac{}{}` handling). Measure how much the apparent pass rate of a fixed model drops. The drop is a lower bound on how much verifier engineering matters.

---

## Debugging checklist

These are the math-specific items worth checking when something's off in an RLVR-on-math training run. The general RLVR debugging items are in [Lecture 15](./15-rl-verifiable-rewards.md).

**Training reward not increasing on math but format reward is fine.** The model is producing well-formatted output but not solving problems. Check whether the verifier is accepting equivalent-form answers: pull a sample of "incorrect" responses by hand. If half of them are correct in unexpected forms, fix the verifier first, then resume training.

**Accuracy plateaus at low level (10-20%) on a benchmark.** Either the base model lacks the math capability to bootstrap (pre-train on math more, or use a stronger base), or the problems are too hard for the current policy and group-relative advantages are mostly zero. Look at the per-problem pass-rate distribution: if most problems have all-zero rewards, lower the problem difficulty or increase K to get some variance in the groups.

**Accuracy improves on training problems but degrades on held-out.** Reward hacking on the verifier, or memorization of problem types from the training set. Verify with a held-out set whose answer distribution and problem types differ from training. If degradation correlates with specific problem types, the model is exploiting type-specific patterns.

**Response length grows but accuracy doesn't.** Length hacking. The model is investing in chain length without that length being productive. Check whether there's any implicit length reward in your pipeline (e.g., longer chains have higher chance of containing the answer string by accident, which gives format reward without correctness reward). Cap response length or add a length penalty.

**Verifier accepts wrong answers occasionally.** The model can find adversarial inputs that fool the normalizer (e.g., answers that normalize to the gold answer despite being wrong). Audit "correct" responses periodically; if the accept rate on intentional wrong answers is non-trivial, tighten the verifier.

**Pass rate on AIME drops between training checkpoints.** AIME's small problem count (15) means single-problem flips dominate the pass rate. Average over more samples per problem (K=8 or K=32) before reporting. Also report pass@8 or pass@32 alongside pass@1 to give a less noisy signal.

**KL to reference model spikes during math training.** Usually this means the policy is rapidly shifting away from the base model's distribution to handle a specific problem type. Lower the learning rate or raise `kl_beta`. If it persists, the model may be entering a mode where it produces math-specific outputs that aren't well-supported by the base model, which can be fine, but is worth watching.

---

## References

All arXiv IDs verified against arxiv.org.

**Benchmarks**

- Cobbe, Kosaraju, Bavarian et al. 2021. "Training Verifiers to Solve Math Word Problems." OpenAI. arXiv:2110.14168. Introduces GSM8K (8.5K grade-school word problems) and the verifier-training approach that anticipates RLVR.
- Hendrycks, Burns, Kadavath et al. 2021. "Measuring Mathematical Problem Solving With the MATH Dataset." arXiv:2103.03874. Introduces MATH (12,500 competition problems). The paper's conclusion that "scaling is not currently solving MATH" is what RLVR-on-math went on to disprove.
- Hendrycks, Burns, Basart et al. 2020. "Measuring Massive Multitask Language Understanding." arXiv:2009.03300. MMLU; the MMLU-STEM subset is occasionally used for reasoning-model comparison.
- He, Luo, Bai et al. 2024. "OlympiadBench." arXiv:2402.14008. 8,476 olympiad-level math and physics problems, bilingual.
- Zheng, Han, Polu. 2021. "MiniF2F: a cross-system benchmark for formal Olympiad-level mathematics." arXiv:2109.00110. Cross-system formal-math benchmark.

**Prompting baselines (pre-RL)**

- Wei, Wang, Schuurmans et al. 2022. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." arXiv:2201.11903. Introduces CoT prompting; on GSM8K, 540B model jumps from ~18% to ~57% with few-shot CoT.
- Wang, Wei, Schuurmans et al. 2022. "Self-Consistency Improves Chain of Thought Reasoning in Language Models." arXiv:2203.11171. Sample many CoT chains, take majority vote; adds ~18 points on GSM8K at K=40.
- Gao, Madaan, Zhou et al. 2022. "PAL: Program-aided Language Models." arXiv:2211.10435. Offload arithmetic to a Python interpreter; on GSM8K hits 72% with Codex-class model.

**STaR and self-training**

- Zelikman, Wu, Mu, Goodman. 2022. "STaR: Bootstrapping Reasoning With Reasoning." NeurIPS 2022. arXiv:2203.14465. Sample chains, filter correct, SFT, repeat. Includes rationalization fallback for hard problems.
- Gulcehre, Le Paine, Srinivasan et al. 2023. "Reinforced Self-Training (ReST) for Language Modeling." Google DeepMind. arXiv:2308.08998. Offline version of the same loop; relevant as a cheaper alternative to GRPO.

**Process supervision**

- Uesato, Kushman, Kumar et al. 2022. "Solving math word problems with process- and outcome-based feedback." DeepMind. arXiv:2211.14275. First systematic comparison of process vs. outcome supervision on GSM8K.
- Lightman, Kosaraju, Burda et al. 2023. "Let's Verify Step by Step." OpenAI. arXiv:2305.20050. Releases PRM800K (800K step-level human labels on MATH); process reward model outperforms outcome RM at matched scale.

**GRPO and DeepSeek-R1**

- Shao, Wang, Zhu et al. 2024. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." DeepSeek-AI. arXiv:2402.03300. Introduces GRPO; applies it to a math-pre-trained 7B model and hits MATH 51.7%.
- DeepSeek-AI. 2025. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv:2501.12948. R1-Zero (RL on base model) and R1 (cold-start SFT + RL + alignment). The primary source for the modern math-reasoning recipe at scale.

**Formal proofs and tool-augmented math**

- Polu, Sutskever. 2020. "Generative Language Modeling for Automated Theorem Proving." arXiv:2009.03393. GPT-f; transformer-based theorem proving in Metamath. An early proof-search system.
- Jiang, Welleck, Zhou et al. 2022. "Draft, Sketch, and Prove: Guiding Formal Theorem Provers with Informal Proofs." arXiv:2210.12283. Connects informal natural-language sketches to formal proof construction.
- DeepMind. 2024. "AI achieves silver-medal standard solving International Mathematical Olympiad problems." Blog post, July 2024. AlphaProof + AlphaGeometry 2 solve 4 of 6 IMO 2024 problems. No full technical paper available as of writing; treat specifics as not yet documented.

**Publicly announced systems without full papers**

- OpenAI. September 2024. "Learning to reason with LLMs." Blog post. Describes o1's training-and-inference-compute split at a high level. No technical paper. Claims about o1's specific algorithm or reward function not in this blog post are speculative.
- Qwen Team (Alibaba). Late 2024. "QwQ: Reflect Deeply on the Boundaries of the Unknown." Blog/preview release. Qwen's reasoning model preview. No full technical paper. Behavior consistent with the R1/o1 recipe family.

**Reading order suggestion**

If you're reading the primary sources end-to-end: start with the Hendrycks MATH paper (arXiv:2103.03874) for the benchmark context, then Wei's CoT paper (arXiv:2201.11903) for the prompting baseline. STaR (arXiv:2203.14465) is short and worth reading in full; it's the cleanest of the early self-training papers. Then DeepSeekMath (arXiv:2402.03300) for GRPO and the math-pre-training emphasis, then the R1 paper (arXiv:2501.12948) for the full recipe. The Lightman PRM800K paper (arXiv:2305.20050) is the reference for process supervision and is the clearest source on what step-level labels look like. The IMO 2024 blog post and the o1 blog post are short reads but lack technical depth: read them for context, not for implementation details.

---

## Next lecture

This is part of the late-stage curriculum covering specific application domains. The neighboring lectures cover other RL-for-LLM specializations: see `notes/README.md` for the current index.
