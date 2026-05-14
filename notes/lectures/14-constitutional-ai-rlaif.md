<!-- status: unreviewed | last-reviewed: never -->

# Lecture 14: Constitutional AI, RLAIF, and Self-Improvement

_Unreviewed — no one has checked this end to end. Treat the math, code, and citations as unverified._

**Time**: ~2-3 h · **Prerequisites**: Lectures 09–11

---

## Where this fits

[Lecture 09](./09-reward-modeling.md) showed how to train a reward model from human preference pairs. [Lecture 10](./10-ppo-for-llms.md) showed how to use that reward model inside PPO to fine-tune a language model. Both assumed a dataset of human-labeled comparisons — annotators reading two responses and picking the better one.

That label collection is the bottleneck. At the scale of InstructGPT (~40k comparisons), it's manageable. At the scale needed to align a model across many tasks, topics, and languages simultaneously, it gets expensive and slow. Human annotators also introduce inconsistencies: different people apply different standards, and calibrating them takes time.

The core idea of this lecture is to replace the human labeler with an LLM. The rest of the RLHF pipeline stays the same — you still train a reward model on comparison pairs, still run PPO or DPO against it. The question is just where the comparisons come from. If a sufficiently capable model can produce labels that agree with human judgments at roughly the same rate that two humans agree with each other, you can scale label generation by running more API calls instead of hiring more annotators.

This idea takes three main forms covered here: LLM-as-judge for pairwise preference labeling, RLAIF (Reinforcement Learning from AI Feedback) as a drop-in substitute for RLHF, and Constitutional AI, which combines AI-feedback with an explicit set of written principles and adds a supervised critique-revision phase before the RL stage.

After those three, two later ideas extend the concept further: self-rewarding language models (the policy judges its own outputs and trains on that judgment) and SPIN (the policy generates synthetic negatives from its own previous weights to create preference signal without any external judge at all).

---

## LLM as judge

The simplest form of AI feedback: given a prompt and two responses, ask a capable model which response is better. The output is a preference label, the same shape as a human label.

```python
import random
import anthropic

client = anthropic.Anthropic()

JUDGE_PROMPT = """You are evaluating two responses to a user prompt.
Decide which response is better. Consider helpfulness, accuracy, and clarity.

[User prompt]
{prompt}

[Response A]
{response_a}

[Response B]
{response_b}

Which response is better? Reply with exactly one letter: A or B."""


def llm_judge_preference(
    prompt: str, response_a: str, response_b: str
) -> int:
    """
    Ask a judge model to pick the better of two responses.

    Returns 0 if response_a is preferred, 1 if response_b is preferred.
    Returns -1 if the judge's output is unparseable.

    NOTE: response order is passed as-is here. To fight position bias,
    randomize which response appears as A vs B before calling this,
    and flip the label back afterward. See llm_judge_debiased() below.
    """
    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=10,
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    prompt=prompt,
                    response_a=response_a,
                    response_b=response_b,
                ),
            }
        ],
    )
    verdict = message.content[0].text.strip().upper()
    if verdict == "A":
        return 0
    elif verdict == "B":
        return 1
    return -1


def llm_judge_debiased(
    prompt: str, response_a: str, response_b: str
) -> int:
    """
    Run the judge twice with swapped order; return a preference only when
    both orderings agree. Reduces position bias at the cost of 2x calls.

    Returns 0 (prefer a), 1 (prefer b), or -1 (inconsistent / tie).
    """
    # First call: a then b
    label_forward = llm_judge_preference(prompt, response_a, response_b)

    # Second call: b then a — flip the returned label
    label_swapped_raw = llm_judge_preference(prompt, response_b, response_a)
    label_swapped = (
        1 - label_swapped_raw if label_swapped_raw != -1 else -1
    )

    if label_forward == label_swapped and label_forward != -1:
        return label_forward
    return -1  # inconsistent — treat as unusable
```

Zheng et al. (2023, arXiv:2306.05685) studied this setup in detail with MT-Bench and Chatbot Arena. The headline finding: GPT-4 as a judge agrees with human preferences at over 80%, which is about the same as the inter-human agreement rate. That makes AI labels statistically comparable to human labels, at least for well-specified tasks.

The paper also catalogs the failure modes. **Position bias**: models favor the response in the first position (or sometimes the second) regardless of content. **Verbosity bias**: longer responses get rated higher even when the extra length adds nothing. **Self-preference bias**: a model used as judge tends to rate responses that match its own style more favorably. Each of these systematically distorts the training signal if left uncorrected. The order-randomization trick in `llm_judge_debiased` above handles position bias. Verbosity bias is harder — you can instruct the judge to ignore length, but it doesn't fully remove the effect. Self-preference bias matters most when the judge and the policy being trained are the same or closely related model family.

---

## RLAIF: swapping the labeler

Lee et al. (2023, arXiv:2309.00267) ran the clearest head-to-head comparison. They took the standard RLHF pipeline — collect comparisons, train a reward model, run RL — and replaced the human comparison step with an LLM generating those comparisons. Everything else was identical.

The result: on summarization, helpful dialogue, and harmless dialogue, RLAIF achieves performance comparable to RLHF by human evaluation. Human raters, when shown outputs from the RLAIF model and the RLHF model without knowing which was which, preferred them at roughly equal rates. This is a strong result because it means AI feedback is not obviously worse — at least on these tasks, with a capable enough labeler.

The paper also introduced "direct RLAIF" (d-RLAIF), which skips reward model training entirely. Instead of using the AI labels to train a separate reward model and then running PPO, d-RLAIF queries the judge model in real time during RL training and uses its output directly as the scalar reward. This is more expensive per training step but avoids reward model overfitting. It performed at least as well as canonical RLAIF in their experiments.

An interesting side result: RLAIF can work even when the judge model is the same size as the policy, or even the same checkpoint. You don't need a much larger "oracle" model. The policy's current version can meaningfully evaluate its own outputs, at least to produce signal that is better than random. This matters because it suggests AI self-improvement isn't completely circular — there's genuine signal even from a weak self-evaluator.

The plumbing for RLAIF is exactly what you've seen in Lectures 09 and 10 — you just swap the data source:

```
# RLHF
human_labeler(prompt, response_a, response_b) -> {chosen, rejected}

# RLAIF
llm_judge(prompt, response_a, response_b) -> {chosen, rejected}

# Rest of the pipeline is identical:
reward_model = train_on_comparisons(comparison_dataset)
policy = ppo(policy_init, reward_model)
```

---

## Constitutional AI

Constitutional AI (Bai et al. 2022, arXiv:2212.08073) is Anthropic's version of RLAIF, extended with two things: an explicit written "constitution" of principles, and a supervised fine-tuning phase before the RL stage.

The constitution is a list of principles in plain text — something like "choose the response that is least likely to cause real-world harm", "choose the response that is most honest and avoids deception", "prefer responses that respect human autonomy". In the paper there are around 16 principles; different subsets are sampled for different steps. The constitution is not a complex formal system — it's a prompt engineering artifact, a list of sentences that the model is asked to apply when making judgments.

CAI runs in two phases.

### Phase 1: supervised critique and revision (SL-CAI)

Start with a raw model that can follow instructions but has not been specifically aligned for harmlessness. Give it a prompt that might elicit a harmful response — a "red teaming" prompt. Let it generate a response. Then run a critique-revision loop:

```python
import random
import anthropic

client = anthropic.Anthropic()

CONSTITUTION_PRINCIPLES = [
    (
        "Identify specific ways in which the assistant's last response is harmful, "
        "unethical, racist, sexist, toxic, dangerous, or illegal."
    ),
    (
        "Identify specific ways in which the assistant's response is dishonest, "
        "unhelpful, or not in the interest of the user."
    ),
    (
        "Identify specific ways in which the assistant's response might be objectionable "
        "based on its potential to demean or disrespect any group of people."
    ),
    (
        "Think carefully about whether the human's request succeeded in making you produce "
        "content that is harmful, unethical, racist, sexist, toxic, dangerous, or illegal. "
        "Identify specific harms."
    ),
]

CRITIQUE_TEMPLATE = """{conversation}

Critique the assistant's last response using the following principle:
{principle}

Critique:"""

REVISION_TEMPLATE = """{conversation}

Critique: {critique}

Using the critique above, revise the assistant's last response to make it more "
"helpful, harmless, and honest. Revised response:"""


def cai_critique_revise(
    prompt: str,
    initial_response: str,
    num_rounds: int = 1,
) -> str:
    """
    Run one or more rounds of CAI critique-then-revise.

    Each round:
      1. Sample a constitutional principle.
      2. Ask the model to critique the current response against that principle.
      3. Ask the model to revise the response based on the critique.
      4. The revised response becomes the input to the next round.

    Returns the final revised response.
    """
    current_response = initial_response

    for round_idx in range(num_rounds):
        principle = random.choice(CONSTITUTION_PRINCIPLES)

        # Format conversation so far
        conversation = (
            f"Human: {prompt}\n\nAssistant: {current_response}"
        )

        # Step 1: critique
        critique_prompt = CRITIQUE_TEMPLATE.format(
            conversation=conversation,
            principle=principle,
        )
        critique_msg = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=512,
            messages=[{"role": "user", "content": critique_prompt}],
        )
        critique = critique_msg.content[0].text.strip()

        # Step 2: revise
        revision_prompt = REVISION_TEMPLATE.format(
            conversation=conversation,
            critique=critique,
        )
        revision_msg = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=512,
            messages=[{"role": "user", "content": revision_prompt}],
        )
        current_response = revision_msg.content[0].text.strip()

        print(f"Round {round_idx + 1} principle: {principle[:60]}...")
        print(f"Critique (truncated): {critique[:120]}...")
        print(f"Revised response (truncated): {current_response[:120]}...\n")

    return current_response
```

You run this over many prompts. The output is a dataset of (prompt, revised_response) pairs. Fine-tuning on this dataset produces SL-CAI — a model that has already internalized some of the constitutional principles via supervised learning before any RL happens.

The intuition is that the critique-revise loop steers the model toward less harmful outputs in a sample-efficient way. Rather than waiting for RL to slowly shift the distribution, you're directly generating examples of the model applying the principles and then training on those examples. It's a form of distillation from the model's own reasoning.

### Phase 2: RL with AI feedback (RLAIF)

Phase 2 is standard RLAIF, but the comparison criterion comes from the constitution rather than a generic "which is better" prompt.

For each prompt, generate two responses (from SL-CAI). Sample a principle from the constitution. Ask the model: "Which response better satisfies this principle?" This gives a preference pair. Repeat over many prompts to build a preference dataset. Train a preference model on that dataset — this is the "Constitutional AI Preference Model" (CAPM). Run PPO against the CAPM.

```python
COMPARISON_TEMPLATE = """Below are two responses to the following prompt:

Prompt: {prompt}

Response A:
{response_a}

Response B:
{response_b}

Constitutional principle: {principle}

Which response better satisfies the above principle? Answer with "A" or "B"."""


def cai_preference_label(
    prompt: str,
    response_a: str,
    response_b: str,
) -> dict:
    """
    Generate a CAI preference label for a pair of responses.

    Samples a random principle from the constitution and asks the model
    to pick the response that better satisfies it.

    Returns a dict with chosen, rejected, and the principle used.
    """
    principle = random.choice(CONSTITUTION_PRINCIPLES)

    comparison_prompt = COMPARISON_TEMPLATE.format(
        prompt=prompt,
        response_a=response_a,
        response_b=response_b,
        principle=principle,
    )

    msg = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=10,
        messages=[{"role": "user", "content": comparison_prompt}],
    )
    verdict = msg.content[0].text.strip().upper()

    if verdict == "A":
        chosen, rejected = response_a, response_b
    elif verdict == "B":
        chosen, rejected = response_b, response_a
    else:
        return None  # unparseable — skip

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "principle": principle,
    }
```

One design detail worth noting: sampling the principle randomly rather than using the same one every time means the preference model is trained against a distribution of principles rather than being overfit to any single one. The constitution as a whole defines a region of behavior, not a specific target.

### What "constitutional" actually means here

The paper's name suggests something formal, but the constitution is just a list of text strings. There's no formal logic, no theorem prover, no syntax checker. The model's behavior changes because it has been trained to apply these strings as evaluation criteria — both during the revision step and during preference labeling. The constitution's authority comes entirely from the model's pre-training knowledge of what these English sentences mean.

This makes it both flexible and fragile. Flexible because you can write principles for almost any domain in plain English. Fragile because the model's interpretation of the principles depends on its prior training, and because subtle wording changes can have large behavioral effects. Getting the constitution right is an empirical process.

Anthropic followed up this work with "Collective Constitutional AI" (Huang et al. 2024, arXiv:2406.07814, published at FAccT 2024), where the principles were sourced from a public deliberation process involving around 1,000 Americans rather than being written by researchers. The resulting model showed lower bias on several demographic dimensions while maintaining comparable performance on helpfulness benchmarks — though note this paper appeared in 2024, after the original CAI work.

---

## Self-rewarding language models

Yuan et al. (2024, arXiv:2401.10020) push the self-improvement idea to its limit: the same model acts as both policy and judge.

The setup: start from a supervised fine-tuned model. At each training iteration, generate several candidate responses to each prompt. Use the model itself (via LLM-as-judge prompting, scoring each response on a 0–5 scale) to identify the best and worst responses. Build (chosen, rejected) pairs from that comparison. Run DPO. Repeat.

```
Iteration t:
  model_t generates candidate responses to prompts
  model_t scores its own responses (LLM-as-judge)
  best/worst pairs -> DPO training data
  DPO produces model_{t+1}
  
Iteration t+1:
  model_{t+1} generates better responses (policy improved)
  model_{t+1} also judges better (judge improved alongside policy)
  ...
```

The key observation from the paper: the judge improves alongside the policy. As the model gets better at generating responses, it also gets better at scoring them. This contrasts with RLHF, where the reward model is trained once and frozen — so there's a ceiling on how good the preference signal can be. In the self-rewarding setup, the ceiling rises with each iteration.

Fine-tuning Llama 2 70B through three iterations of this approach produced a model that outperformed several larger systems (including Claude 2 and GPT-4 0613) on AlpacaEval 2.0. The caveat: AlpacaEval uses an LLM judge, so a model that has learned to produce outputs that look good to an LLM judge will score well there regardless of whether those outputs are actually better by human standards. This is the central tension in all self-improvement schemes.

---

## Self-play fine-tuning (SPIN)

Chen et al. (2024, arXiv:2401.01335) take a different route. They don't need an explicit judge at all.

The idea, called SPIN (Self-Play fIne-tuNing), treats the supervised fine-tuning data as the target distribution and the model's own generations as the "rejected" side. At each iteration:

1. Take a prompt from the SFT dataset.
2. The current model generates a response (the "weak player" / negative).
3. The ground-truth SFT response is the positive.
4. Train with DPO using (SFT response, model generation) as (chosen, rejected).
5. Repeat with the updated model.

```
# SPIN iteration schematic

for iteration in range(num_iterations):
    preference_pairs = []
    
    for prompt, sft_response in sft_dataset:
        # Model plays against itself
        model_generation = current_model.generate(prompt)
        
        # SFT data is "real", model output is "fake"
        preference_pairs.append({
            "prompt": prompt,
            "chosen": sft_response,       # from data
            "rejected": model_generation,  # from current model
        })
    
    # DPO on these pairs
    current_model = dpo_update(current_model, preference_pairs)
    
# Each iteration: model gets better at distinguishing
# its own outputs from the SFT reference distribution.
# The training signal weakens as the model improves.
```

The game-theoretic framing: the "main player" (current model) tries to match the SFT data distribution; the "opponent" (same model from the previous iteration) generates the negatives. As the main player improves, the opponent's generations become harder to distinguish from real data — the game gets harder. This mirrors self-play in games like chess or Go, where playing against stronger opponents drives improvement.

The theoretical result: the global optimum is achieved exactly when the model's distribution matches the SFT training distribution. The practical implication is that SPIN can only improve up to the quality ceiling of the SFT data. It's not creating new knowledge; it's more efficiently extracting the behavior implicit in the existing supervised examples.

Empirically, SPIN improved a Mistral 7B model on the HuggingFace Open LLM Leaderboard by several percentage points across three iterations, and matched performance of DPO models that used additional GPT-4 preference data. That's notable: no judge model, no human labels, no external preference signal — just the SFT data and the model's own generations.

The limitation is exactly the SFT ceiling. If the SFT data has quality problems, SPIN converges toward those problems. It also converges eventually — once the model's generations are indistinguishable from the SFT data, there's no more gradient signal.

---

## When to use this over plain RLHF

Human labels remain the gold standard when you can get them. AI feedback has two practical advantages and several risks.

**When AI feedback makes sense:**

Human labels are the bottleneck. If label collection is slowing down your training cycle and you have a capable model available as a judge, RLAIF can produce comparable results faster. Lee et al. estimated AI preference labeling at over 10x cheaper than human labeling in their setup.

You have a much stronger model available to be the judge. A GPT-4-class judge labeling outputs from a smaller model will produce cleaner signal than the smaller model judging itself. The quality gap matters.

Your task has clear, articulable criteria. Constitutional principles work best when the criteria for "good" can be written as plain English sentences that a language model will interpret consistently. Open-ended tasks like "write me a poem I'll love" are harder to specify this way.

**The risks:**

The judge's biases define "good". If the judge model has stylistic preferences — longer responses, specific sentence structures, particular political framings — those preferences get baked into the training signal and then into the policy. This is not hypothetical; Zheng et al. document several such biases, and they're not fully fixable by instruction. You're not escaping human biases; you're replacing diverse human biases with the systematic biases of one model.

Self-improvement can collapse into stylistic priors. When the model judges its own outputs and trains on that judgment, it can reinforce whatever tendencies already exist in the model, even if those tendencies don't correlate with actual quality. The feedback loop amplifies the model's existing preferences, and there's no external check. Yuan et al. note that AlpacaEval — itself judged by an LLM — may be rewarding LLM-appeal rather than human-appeal.

The constitution is hard to get right. Principle wording has large behavioral effects that are difficult to predict without empirical testing. Principles can conflict; the model's resolution of those conflicts is implicit and not guaranteed to match your intent. And there's no formal verification that the model is actually applying the principles you wrote versus applying its own interpretation of them.

---

## Exercises

These don't require a large model or GPU. A local model (Llama 3.1 8B, Mistral 7B via Ollama, or any API access) is enough.

1. **Implement the critique-revise loop.** Pick ten prompts that elicit unhelpful or borderline responses from a model. Run `cai_critique_revise` with one round and measure whether the revisions are actually better. Write two or three principles yourself and test whether different principles produce consistently different revisions on the same prompt.

2. **Measure position bias in an LLM judge.** Take 50 prompt-response pairs where you have ground truth about which response is better (e.g., a small human-annotated set, or synthetic pairs where one response is clearly worse). Run `llm_judge_preference` twice for each pair — once with (a, b) order, once with (b, a) order. Compute the rate at which the model changes its answer between the two orderings. Report what fraction of cases are inconsistent.

3. **Compare DPO on human vs. AI labels.** Take a small preference dataset (Anthropic's HH-RLHF is available on HuggingFace). Train one DPO model on the human labels. Then re-label the same prompts using an LLM judge and train a second DPO model. Evaluate both on a held-out set — either by win rate against a baseline or by another judge. Does the performance gap between human-labeled and AI-labeled DPO match what Lee et al. found?

4. **Run one iteration of SPIN.** Take a small SFT dataset (100–500 examples). Generate responses with a local model. Build (SFT response, model generation) preference pairs. Run DPO for one epoch. Sample a few prompts and compare outputs before and after. What changed? Did the model improve, or did it just drift?

---

## References

**Zheng et al. (2023)** — "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." NeurIPS 2023 (Datasets and Benchmarks Track). arXiv:2306.05685. _Verified._
Introduces MT-Bench and Chatbot Arena; documents position bias, verbosity bias, and self-preference bias in LLM judges; shows GPT-4 achieves >80% agreement with human preferences.

**Lee et al. (2023)** — "RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback." arXiv:2309.00267. _Verified._
Direct comparison of RLAIF and RLHF on summarization and dialogue; shows comparable performance; introduces d-RLAIF (direct AI feedback without a reward model).

**Bai et al. (2022)** — "Constitutional AI: Harmlessness from AI Feedback." Anthropic. arXiv:2212.08073. _Verified._
Introduces the two-phase CAI pipeline: SL-CAI (critique-revise fine-tuning) and RLAIF with constitutional principles as preference criteria.

**Yuan et al. (2024)** — "Self-Rewarding Language Models." Meta / NYU. arXiv:2401.10020. _Verified._
The model acts as its own judge, generates preference pairs, trains with DPO, and iterates. Shows that both policy quality and judge quality improve across iterations.

**Chen et al. (2024)** — "Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models" (SPIN). UCLA. arXiv:2401.01335. _Verified._
Uses the gap between SFT data and model generations as a preference signal; no external judge required; converges when model distribution matches SFT data distribution.

**Huang et al. (2024)** — "Collective Constitutional AI: Aligning a Language Model with Public Input." Anthropic / Collective Intelligence Project. FAccT 2024. arXiv:2406.07814. _Verified (appeared 2024, not 2023 — the Anthropic blog post was 2023 but the arXiv paper is 2024)._
Crowdsources constitutional principles from ~1,000 Americans via online deliberation; the resulting model shows lower bias on several demographic dimensions.

---

## Next lecture

[Lecture 15: RL with Verifiable Rewards](./15-rl-verifiable-rewards.md)
