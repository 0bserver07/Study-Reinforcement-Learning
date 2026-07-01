<!-- status: unreviewed | last-reviewed: never -->

# RL-for-LLM training loops: 2024/2025 map

_Unreviewed: no one has checked this end to end. Treat the math and citations as unverified._

A map of the major RL-for-LLM training loops as practiced through 2024 and into 2025. One ASCII diagram per loop, one paragraph of annotation, plus a cross-cutting comparison at the end. The loops differ less in their RL algorithms than in where the preference signal comes from and how often it's refreshed.

Read this alongside `notes/lectures/10-ppo-for-llms.md` (the classical RLHF pipeline), `notes/lectures/14-constitutional-ai-rlaif.md` (CAI / RLAIF / self-rewarding), `notes/lectures/15-rl-verifiable-rewards.md` (GRPO and the R1 recipe), `notes/lectures/17-online-iterative-preference.md` (iterative and online DPO), and `notes/lectures/18-distillation-reasoning.md` (R1-distill).

Conventions used in the diagrams below:

```
[ box ]    a stored artifact: a dataset, a model checkpoint, a corpus
( oval )   a process: sampling, training, scoring, filtering
==>        bulk data flow (a dataset or corpus)
-->        per-step flow (a batch, a rollout, a single label)
. . .>     feedback edge: data path that closes a loop across rounds
```

---

## 1. Pretrain to SFT (context)

```
                 [ web + books + code corpus ]
                              ==>
                ( next-token loss, AdamW, many trillions of tokens )
                              -->
                       [ base LM checkpoint ]
                              |
                              v
                 [ (prompt, demonstration) pairs ]
                              ==>
                  ( supervised fine-tuning: cross-entropy
                    on completion tokens, prompt masked )
                              -->
                       [ SFT model checkpoint ]
                              |
                              v
                ---- everything below starts here ----
```

This loop is where RL is not. Pre-training builds the base model from raw text via next-token prediction; SFT trains it on a curated set of (prompt, response) pairs so it learns the instruction-following format and produces something other than text continuation. SFT is the launchpad for every RL loop in this document: almost all of them use the SFT checkpoint either as the policy initialization, the KL reference, or both. The one exception is R1-Zero (loop 7), which runs RL directly on the base model. The InstructGPT paper used roughly 13k SFT demonstrations (Ouyang et al. 2022, arXiv:2203.02155); modern open-weight pipelines use anywhere from tens of thousands to a few million SFT examples, often filtered or rewritten by a stronger model. Skipping SFT and going straight to RL produces incoherent output because the base model has no prior on what assistant-shaped text looks like (Ouyang et al. 2022).

Required: a base model, a curated SFT corpus, an Adam-class optimizer.

---

## 2. SFT to reward model to PPO-RLHF (the InstructGPT loop)

```
              [ SFT model ]                  [ SFT model (frozen) ]
                  |                                   |
                  | clone                             | freeze
                  v                                   v
            [ policy pi_theta ]                 [ pi_ref ]
                  |                                   ^
                  | sample y ~ pi(.|x)                | KL(pi || pi_ref) per token
                  v                                   |
                                                      |
   [ prompt x ] ----------+                           |
                          |                           |
   ( generation, batch )  v                           |
                  +------------ y ------------+       |
                  |                           |       |
                  v                           v       |
       [ reward model r_phi ] <--- preference data    |
                  |                                   |
                  | scalar r_RM(x, y) at end-of-seq   |
                  v                                   |
   reward = r_RM(x, y) - beta * KL(pi || pi_ref)      |
                  |                                   |
                  v                                   |
           ( GAE + PPO clipped objective +            |
             value-head MSE, optimizer step ) --------+
                  |
                  v
            [ RLHF model ]

  --- offline (one-time) ---
  [ pairs (x, y_w, y_l) ] ==> ( Bradley-Terry MLE on r_phi ) --> [ r_phi ]
```

This is the canonical three-stage InstructGPT pipeline (Ouyang et al. 2022, arXiv:2203.02155; earlier on summarization in Stiennon et al. 2020, arXiv:2009.01325). Stage 1 is the SFT above. Stage 2 trains a scalar reward model `r_phi(x, y)` on human-labeled preference pairs using a Bradley-Terry log-loss; the reward model is usually initialized from the SFT model with a scalar head bolted on. Stage 3 runs PPO (Schulman et al. 2017, arXiv:1707.06347): the policy generates completions for a batch of prompts, the RM scores each completion, a per-token KL penalty to the frozen `pi_ref` (the SFT model) is subtracted from the reward, GAE computes per-token advantages using a separately trained value head, and the PPO clipped surrogate objective updates the policy. Memory footprint: four model-sized objects in play: policy (training), reference (frozen, for KL), reward model (frozen, for scoring), and a critic / value head (training). The KL term is what keeps the policy from gaming the RM and emitting reward-hacking gibberish (Bai et al. 2022, arXiv:2204.05862, document this empirically); typical beta around 0.1. This loop is the algorithmic baseline against which every later loop is a simplification or a specialization.

Required: SFT model, frozen reference, scalar RM (trained on ~tens of thousands of human preference pairs), value head, PPO trainer with GAE.

---

## 3. SFT to offline DPO (the simplest preference loop)

```
   --- offline data collection (done once, often by humans) ---

   [ prompt x ] --> ( policy or annotator generates two responses )
                    --> y_a, y_b
                    --> ( human or stronger LLM judge picks chosen / rejected )
                    --> [ (x, y_w, y_l) ]
                  ==>
   [ preference dataset D ]   (the dataset is now static)

   --- training (no generation in the loop) ---

   [ SFT model ]                   [ SFT model (frozen) ]
       |                                   |
       v                                   v
   [ pi_theta ]                         [ pi_ref ]
       |                                   |
       |                                   | log pi_ref(y_w|x), log pi_ref(y_l|x)
       |                                   |  (cached or recomputed once)
       |                                   |
       v                                   v
              +-----------------------+
              |                       |
              v                       v
   for batch in D:
       compute log pi_theta(y_w|x), log pi_theta(y_l|x)
       L = -E[ log sigma( beta * ( (log pi_theta(y_w|x) - log pi_ref(y_w|x))
                                  -(log pi_theta(y_l|x) - log pi_ref(y_l|x)) ) ) ]
       optimizer step on pi_theta
              |
              v
       [ DPO-tuned model ]
```

DPO (Rafailov et al. 2023, arXiv:2305.18290) reparameterizes the RLHF objective so that the reward model becomes implicit in the log-probability ratio between the policy and the reference. The training loop is then ordinary supervised learning on preference triples: no rollouts, no reward model to maintain, no value head, no PPO clipping. The dataset `D` is collected once from some data-generating policy (could be the SFT model itself, a stronger model, or human-written A/B pairs). Once `D` is fixed, the training loop is just batched forward passes through `pi_theta` and `pi_ref` plus an Adam step on the sigmoid-cross-entropy loss. This is the cheapest loop to set up and the easiest to debug, which is why most open-weight alignment recipes started here (Zephyr, Tunstall et al. 2023, arXiv:2310.16944, is the canonical small-scale example). The cost is the data-distribution gap: after a few hundred steps `pi_theta` has moved away from the policy that generated `D`, and the implicit reward can extrapolate badly into regions where `D` has no coverage. Xu et al. 2024 (arXiv:2404.10719) ran a careful comparison and found well-tuned PPO consistently beats offline DPO on dialogue and code; the gap is structural, not a tuning issue. The fix, keeping the data on-policy by regenerating it, is loop 4.

Required: SFT model (used as both starting point and reference), a static preference dataset, an Adam-class optimizer. No reward model. No rollouts. No critic.

---

## 4. SFT to online iterative DPO (refresh data each round)

```
   round r = 0, 1, 2, ...

   [ pi_r ]  (initialized at r=0 from SFT model)
      |
      | for each prompt x in P:
      |     sample K responses y_1..y_K ~ pi_r(.|x)
      v
   [ rollouts {(x, y_1..y_K)} ]
      |
      | for each (x, y_i):
      |     score s_i = scorer(x, y_i)
      |       scorer is one of:
      |         - learned reward model r_phi
      |         - LLM judge (pairwise or pointwise)
      |         - verifiable checker (correct? 0/1)
      v
   [ scored rollouts ]
      |
      | pair selection (best-vs-worst, contrastive,
      |                 or threshold-based)
      v
   [ preference pairs D_r ]
      |
      | DPO loss (loop 3) against the *frozen* SFT pi_ref
      | for ~100-500 steps
      v
   [ pi_{r+1} ] . . . . . . . . . . . . . . feeds next round . . .

   reference pi_ref stays frozen at the SFT model across all rounds
```

Iterative DPO closes the data-distribution loop that offline DPO leaves open. At the start of round `r` the current policy generates fresh responses, those responses are scored, pairs are built, DPO runs for a few hundred steps, and the updated policy becomes the rollout policy for round `r+1`. This is what most production alignment pipelines look like in 2024-2025 (Llama 3, Dubey et al. 2024, arXiv:2407.21783, describes multiple rounds of this; the exact number and data volumes aren't fully public). Self-Rewarding Language Models (Yuan et al. 2024, arXiv:2401.10020) use the same loop with the policy itself as judge: both the policy and the judge improve together across rounds. Iterative Reasoning Preference Optimization (Pang et al. 2024, arXiv:2404.19733, NeurIPS 2024) applies it to chain-of-thought with a correctness-based scorer and a DPO+NLL objective; two rounds from Llama-2-70B-Chat take GSM8K from 55.6% to 81.6%. RSO (Liu et al. 2023, arXiv:2309.06657, ICLR 2024) gives the formal motivation: the MLE of the optimal RLHF policy needs preference pairs sampled from that optimal policy, and rejection sampling on the current policy is a lighter-weight approximation. Online DPO (Guo et al. 2024, arXiv:2402.04792, "OAIF") shrinks the round length toward zero: every step samples two on-policy responses, gets a fresh label from an LLM judge, and runs one DPO step. The reference policy stays pinned to the original SFT model throughout: updating it would relax the KL anchor and let the policy drift further per round. Two to four rounds is what most public reports use; more risks compounding scorer biases and you should monitor a held-out metric (Gao et al. 2022, arXiv:2210.10760).

Required: SFT model, a scorer (RM or judge or checker), a generation engine for the policy, the offline DPO trainer from loop 3. The scorer's cost is now in the inner loop, so vLLM-style fast inference matters.

---

## 5. SFT to GRPO with verifiable reward (the DeepSeek-R1 / RLVR loop)

```
   --- cold-start SFT (a few thousand reasoning traces) ---

   [ base model ]
       |
       | SFT on (prompt, <think>...</think> + answer) examples
       v
   [ cold-start checkpoint pi_0 ]
       |
       v
   --- GRPO loop (repeats for many steps) ---

   [ pi_theta ] ----------------+
       |                        |
       | for each prompt x in batch:                          [ pi_ref ]
       |     sample K completions y_1..y_K ~ pi_theta(.|x)     (frozen, == pi_0)
       |                        |                                  ^
       v                        v                                  |
   [ rollouts ]                                                    |
       |                                                           |
       | for each (x, y_i):                                        |
       |     r_i = format_reward(y_i) + correctness_reward(y_i)    |
       |        where correctness comes from a deterministic       |
       |        checker (symbolic math, regex on answer,           |
       |        sandboxed test suite for code, proof assistant)    |
       v                                                           |
   [ rewards r_1..r_K per prompt ]                                 |
       |                                                           |
       | A_i = (r_i - mean(r)) / (std(r) + eps)                    |
       |  group-relative advantage; no learned critic              |
       v                                                           |
   [ advantages A_1..A_K ]                                         |
       |                                                           |
       | L_CLIP = -E[ min(ratio_i * A_i,                           |
       |                  clip(ratio_i, 1-eps, 1+eps) * A_i) ]     |
       | L_KL   =  beta * E[ log pi_theta - log pi_ref ] <---------+
       | L      =  L_CLIP + L_KL
       | optimizer step on pi_theta
       v
   [ pi_theta (updated) ]
```

GRPO (Shao et al. 2024, arXiv:2402.03300, the DeepSeekMath paper) is PPO without a learned critic: the advantage baseline for each completion is the within-group mean reward across K samples drawn from the same prompt, normalized by the group's standard deviation. No value head, no GAE, no per-token credit assignment: the whole completion is credited or debited as a unit. The reward function is rule-based and computed by a checker that the model cannot fool by producing plausible-looking text: exact-match on a normalized numeric answer for math, pass rate on a private test suite for code, formal verification for proofs. A small format reward (typically 0.1) is added so the model learns the `<think>...</think>` scaffold early before correctness signal becomes reliable. Group sizes K of 4-16 are typical; larger K cuts advantage variance but multiplies forward-pass cost. The R1 recipe (DeepSeek-AI 2025, arXiv:2501.12948) wraps GRPO in: cold-start SFT on a few thousand high-quality reasoning traces (so the policy starts with a legible format), the GRPO RL stage, then rejection sampling and a second SFT pass on a broader task mix, then a final RLHF stage for general helpfulness. The key empirical observations from that paper: chain length grows over training without an explicit length reward (longer chains turn out to be instrumentally useful on hard problems), and the policy spontaneously learns to backtrack and self-check. Failure modes are loud: entropy collapse, KL overrun if beta is too low, reward hacking on weak verifiers, and "baseline collapse" when every completion scores identically (group std is zero and the gradient vanishes). See `notes/lectures/15-rl-verifiable-rewards.md` for the full implementation and debugging notes.

Required: cold-start SFT checkpoint, frozen reference, a deterministic checker, a generation engine that can produce K rollouts per prompt cheaply (vLLM or similar). No reward model, no critic, no human labels at the RL stage itself.

---

## 6. Constitutional AI / RLAIF (the model critiques its own outputs)

```
   --- Phase 1: supervised critique-and-revise (SL-CAI) ---

   [ red-teaming prompts P ]
        |
        v
   ( pi_helpful generates response r_0 )           [ constitution: list of
        |                                            ~16 plain-English principles ]
        v                                                       |
   ( judge_model critiques r_0 using principle p_i ~ P )<-------+
        |
        | critique text
        v
   ( judge_model revises r_0 given the critique
     --> r_1, possibly repeat for N rounds )
        |
        v
   [ corpus C_sft = {(prompt, r_N)} ]
        |
        | standard SFT cross-entropy on the revised pairs
        v
   [ SL-CAI model ]
        |
        v

   --- Phase 2: RLAIF with constitutional preferences ---

   [ SL-CAI model = pi_theta ]
        |
        | for each prompt x:
        |     sample two responses y_a, y_b ~ pi_theta(.|x)
        |     sample principle p ~ constitution
        |     ask judge_model: which response better satisfies p?
        v
   [ preference pairs (x, chosen, rejected) ]
        |
        | train CAPM (constitutional AI preference model)
        | with Bradley-Terry loss
        v
   [ r_CAPM ]
        |
        v
   ( PPO or DPO loop, exactly like loop 2 or loop 3,
     but with r_CAPM as the scoring function and judge labels
     instead of human labels )
        |
        v
   [ CAI model ]
```

Constitutional AI (Bai et al. 2022, arXiv:2212.08073) replaces human preference labels with labels generated by an LLM applying a written set of principles. The "constitution" is a list of plain-English sentences (the paper uses around 16; different subsets get sampled at different steps): not formal logic, just prompt-engineering artifacts whose interpretation is grounded in the judge model's pre-training. Phase 1 (SL-CAI) is the critique-and-revise loop: the model generates a response, critiques it against a sampled principle, then revises. Doing this over many prompts gives a corpus of revised pairs to SFT on, baking some of the constitutional behavior in before any RL runs. Phase 2 is plain RLAIF: two on-policy responses get compared by the judge against a sampled principle, the pair labels train a constitutional preference model, and PPO (or DPO) runs against that preference model. The plumbing from loop 2 (PPO-RLHF) is reused unchanged; only the source of preference labels differs. Lee et al. 2023 (arXiv:2309.00267) ran the cleanest head-to-head: RLAIF achieves performance comparable to RLHF by human evaluation on summarization and dialogue, and "direct RLAIF" (d-RLAIF) skips the preference model entirely and uses the judge's score as the live reward during PPO. The dominant failure modes come from the judge: position bias, verbosity bias, self-preference bias (Zheng et al. 2023, arXiv:2306.05685). Self-Rewarding Language Models (Yuan et al. 2024, arXiv:2401.10020) push this further by making the policy be its own judge; SPIN (Chen et al. 2024, arXiv:2401.01335) removes the judge entirely by treating SFT data as the positive and the model's own generations as the negative, training with DPO. Collective Constitutional AI (Huang et al. 2024, arXiv:2406.07814, FAccT 2024) sources the principles from a public deliberation process instead of researchers writing them.

Required: helpful-only SFT model, a judge model (often a stronger frontier model, often the same model), the constitution as a text file, the same RM / PPO / DPO machinery as loops 2 or 3. Optionally a curated set of red-teaming prompts for Phase 1.

---

## 7. R1-Zero (RL directly from a base model, no SFT warm-up)

```
   [ base LM ] = pi_theta
        |                                       [ pi_ref = base LM (frozen) ]
        |                                                   ^
        | exactly the GRPO loop from loop 5,                |
        | but the starting point is the raw base model,     |
        | not a cold-start SFT checkpoint                   |
        |                                                   |
        | for each prompt x in batch:                       |
        |     sample K completions y_1..y_K ~ pi_theta      |
        |     r_i = format_reward + correctness_reward      |
        |     A_i = (r_i - group_mean) / group_std          |
        v                                                   |
   ( PPO-clipped objective + KL to pi_ref ) <---------------+
        |
        v
   [ pi_theta updated ]

   observed behaviors that emerge without being trained for:
     - increasing chain length on harder problems
     - backtracking ("wait, let me reconsider that step")
     - self-verification before committing to an answer
   observed pathologies:
     - language mixing mid-chain (no SFT prior on legible style)
     - low readability of the <think> block
     - occasional repetition / circular reasoning
```

R1-Zero (DeepSeek-AI 2025, arXiv:2501.12948) runs the GRPO loop from #5 directly on a base model with no SFT warm-up. The reward is the same rule-based combination of format and correctness. The motivating finding from the paper: emergent reasoning behaviors (extended chains, backtracking, self-checking) show up even without curated demonstrations of those behaviors. They aren't explicitly rewarded; they appear because they're instrumentally useful for getting the correctness signal. The downside is that without an SFT prior on what a legible output looks like, the model develops idiosyncratic failure modes: language mixing, hard-to-read traces, occasional circular reasoning. This is the loop that demonstrated "RL alone can elicit reasoning," but the practical deployment version (R1 itself) puts a small SFT pass in front to clean up the format, then runs the same GRPO. The standard interpretation in the field is that RL is amplifying reasoning behaviors that were already latent in the base model's pre-training; a base model that has seen essentially no mathematical reasoning during pre-training won't suddenly reason well after GRPO. The ceiling is set by pre-training; RL just finds and sharpens whatever was already in there.

Required: base model (good enough for some completions to be correct on the training problems), frozen reference (the same base model), checker, GRPO trainer. No SFT data, no preference data, no human labels at any stage. The reward signal is entirely rule-based.

---

## 8. R1-distill (large RL-trained teacher hands traces to a small student)

```
   --- corpus generation (done once, expensive) ---

   [ teacher: large RL-trained reasoning model, e.g. R1 ]
                              |
                              | for each prompt x in reasoning prompt set P:
                              |     generate K completions at temperature ~0.7
                              v
   [ raw traces {(x, <think>...</think> + answer)_i} ]
                              |
                              | filter by the verifiable checker
                              | (same checker that was used during R1's RL training)
                              v
   [ filtered corpus C_distill ]
     ~ only correct traces, full chain-of-thought + answer kept

   --- student SFT (cheap, standard cross-entropy) ---

   [ student base model, e.g. Qwen-2.5-7B-base, Llama-3-8B-base ]
                              |
                              | standard next-token cross-entropy on C_distill
                              | mask prompt tokens from the loss
                              | include the full <think> block in the loss
                              | typical 2 epochs, lr ~1e-5
                              v
   [ student distilled model ]
                              |
                              | optionally:
                              v
                ( short GRPO pass to push past the
                  teacher's coverage gaps - see loop 5 )
```

R1-distill (DeepSeek-AI 2025, arXiv:2501.12948, section 3) is the recipe for compressing R1's reasoning capability into much smaller models without re-running the RL pipeline. The teacher generates many completions on a curated reasoning prompt set (the paper describes around 800k samples drawn from open math and code datasets), the verifiable checker keeps only the correct ones, and the student trains on the filtered corpus with ordinary SFT: no RL on the student, no reward model, no KL penalty. The filter is what distinguishes this from generic SFT-on-model-outputs: a checker that the teacher's chain successfully passed gives a stronger quality guarantee than "looks plausible." The released distilled series (Qwen-2.5-7B/14B/32B and Llama-3-8B/70B distillations) report Qwen-32B-R1-distill approaching R1 itself on MATH-500 at roughly 1/20 the parameter count, and distilled Llama-70B outperforming o1-mini on AIME 2024. The lineage: STaR (Zelikman et al. 2022, arXiv:2203.14465) and ReST (Gulcehre et al. 2023, arXiv:2308.08998) used the same sample-filter-SFT pattern but with the model distilling itself; the R1 paper's contribution is showing that cross-model distillation from a much stronger RL-trained teacher beats iterative self-distillation at small scales. Related cross-model chain-of-thought distillation was studied earlier in Hsieh et al. 2023 (arXiv:2305.02301) and Magister et al. 2022 (arXiv:2212.08410). Classical logit distillation (Hinton et al. 2015, arXiv:1503.02531) transfers the teacher's full output distribution at each token. That's a different technique and not what R1-distill does; R1-distill is behavioral cloning on sampled, filtered text. The R1 paper also reports that running GRPO directly on the small student is worse than distillation, which is the empirical finding that makes this pipeline interesting: at small scale, distill first, RL second.

Required: a strong teacher model accessible for bulk inference, a curated reasoning prompt set, a checker, a student base model, a standard SFT trainer. No live RL during student training. The dominant cost is teacher inference at corpus-generation time.

---

## 9. Agentic / tool-use RL (multi-turn rollouts with environment feedback)

```
   [ policy pi_theta ]
        |
        |  start of episode: prompt x = task description, available tools
        v
   +----------------------------------------------------------------------+
   |                                                                      |
   |  turn t = 0, 1, 2, ... until done:                                   |
   |                                                                      |
   |    pi_theta generates a_t                                            |
   |      = "Thought: ... \n Action: tool_name[args]"   (ReAct format)    |
   |          |                                                           |
   |          v                                                           |
   |    [ environment / tool sandbox ]                                    |
   |          | execute the action (run code, shell command, search,      |
   |          |   click in browser, edit a file, query an API)            |
   |          v                                                           |
   |    observation o_t = stdout / page contents / file diff / API json   |
   |          |                                                           |
   |          v                                                           |
   |    pi_theta sees x + (a_0, o_0) + ... + (a_t, o_t)                   |
   |    proceeds to turn t+1                                              |
   |                                                                      |
   +----------------------------------------------------------------------+
                              |
                              | end of episode:
                              | final reward r = verifier(trajectory)
                              |   e.g. hidden test suite passes? (SWE-bench)
                              |        task completed? (browser benchmark)
                              |        unit tests green? (code agent)
                              v
   [ trajectory + scalar reward ]
                              |
                              v
   ( PPO or GRPO update: treat the whole multi-turn trajectory
     as one rollout; compute per-token log-probs for the
     policy's *generated* tokens only - observation tokens
     get masked out of both the loss and the KL penalty )
                              |
                              v
   [ pi_theta updated ]
```

Agentic RL extends the verifiable-reward loop (#5) to multi-turn rollouts where the model interacts with a real environment between generations. The ReAct scaffold (Yao et al. 2022/2023, arXiv:2210.03629) is the standard interface: the policy emits alternating thought / action / observation tokens, where the action is parsed and dispatched to a tool (Python interpreter, shell, browser, search API, code editor). Each tool returns an observation that gets appended to the context, and the policy generates the next thought / action. The episode ends when the model emits a "done" signal or hits a step limit; the reward is computed once on the full trajectory by a deterministic verifier: hidden test suite for code (SWE-bench: Jimenez et al. 2023, arXiv:2310.06770, ICLR 2024), task-completion check for browsers, success criteria for an API call sequence. The RL training itself is structurally close to GRPO or PPO from earlier loops, but with a critical implementation detail: observation tokens (which the model received from the environment, not generated) must be masked out of the policy loss and the KL computation. The policy is not being trained to predict the tool outputs. WebGPT (Nakano et al. 2021, arXiv:2112.09332) is one of the early examples: RLHF on a browser agent where rewards came from human raters comparing answers. Toolformer (Schick et al. 2023, arXiv:2302.04761) is the contrast case: it uses self-supervised learning rather than RL to teach a model to call APIs, by checking which API calls improve next-token prediction on surrounding text. SWE-RL (Wei et al. 2025, arXiv:2502.18449) is a current open recipe: RL on Llama 3 against public software-evolution data (real issues, real diffs, real tests), reaching 41.0% on SWE-bench Verified at 70B. The hard parts here are infrastructure: the rollout loop now includes real tool execution (which is slow, can fail, has side effects, may need sandboxing), and a single trajectory can take minutes rather than seconds.

Required: a tool sandbox (Docker / VM / browser harness, depending on the environment), a verifier that scores trajectories, an RL trainer that handles variable-length multi-turn rollouts and masks observation tokens correctly, plus all the GRPO / PPO machinery from earlier loops. SFT pre-training on tool-use traces is usually the warm-up: pure RL from a base model on multi-turn agentic tasks works much worse than on single-turn math.

---

## What differs across the loops

A reader who wants to know which loops need which ingredient can use this table directly. "Y" means required, "N" means not required, "opt" means used by some implementations but not core.

| Loop | Reward model | Verifiable reward | Preference data | Starts from base | Human labels | On-policy data |
|---|---|---|---|---|---|---|
| 1. Pretrain + SFT | N | N | N | Y | Y (demos) | Y |
| 2. PPO-RLHF | Y | N | Y | N | Y (pairs) | Y |
| 3. Offline DPO | N | N | Y | N | Y or AI | N |
| 4. Iterative / online DPO | opt | opt | Y | N | opt | Y |
| 5. GRPO + verifiable | N | Y | N | opt (cold-start SFT) | N | Y |
| 6. CAI / RLAIF | Y (CAPM) | N | Y (AI-labeled) | N | opt | Y |
| 7. R1-Zero | N | Y | N | Y | N | Y |
| 8. R1-distill (student SFT) | N | Y (used at filter time) | N | Y (student) | N | N (student is SFT on a static corpus) |
| 9. Agentic / tool-use RL | opt | Y (trajectory verifier) | opt | N | opt (for SFT warm-up) | Y |

Some notes on the table:

- **Reward model**: Y means a learned scalar `r_phi` is in the inner loop. Loop 6 trains a constitutional preference model (CAPM) that is structurally a reward model.
- **Verifiable reward**: Y means a deterministic checker (symbolic solver, regex, test suite, proof assistant, trajectory verifier) supplies the reward. Loop 8 uses the checker only at corpus-filter time, not during gradient steps on the student.
- **Preference data**: Y means the loop consumes (chosen, rejected) pairs at some point.
- **Starts from base**: whether the loop will work without SFT first. The honest answer for most of these is "technically yes, but you'll regret it." R1-Zero is the headline counterexample where running from base produces useful results.
- **Human labels**: Y means humans are labeling data somewhere in the pipeline (demos for SFT, comparisons for the RM, end-of-episode evaluations for agents). RLAIF / CAI explicitly remove human labels in favor of AI labels but typically still use humans for a small calibration set; the table reads "opt" for that.
- **On-policy data**: Y means the gradient steps use data sampled from a recent version of the policy. Offline DPO is the only loop where this is structurally N. Iterative DPO closes the gap; online DPO closes it further.

A second cut, sorted by what the loops are best at:

- **Plain instruction following on subjective tasks (helpfulness, tone, style)**: loops 2, 3, 4, 6. Use loops 4 or 6 if you have judge access. Use 3 if you only have a static dataset.
- **Reasoning on verifiable domains (math, code, formal proofs)**: loops 5, 7, 8, 9. Use 5 with a cold-start SFT if you're training the model. Use 8 if you're targeting a smaller deployment. Use 9 if the task is multi-turn / tool-using.
- **No human labels at all**: loops 5, 7. SPIN (within loop 6) is the no-judge no-RM version that still needs SFT data.
- **No reward model at all**: loops 3, 4 (DPO-only versions), 5, 7, 8. PPO is the only common loop that strictly requires a learned RM.
- **No SFT prerequisite**: loops 1 (it is the SFT step), 7 (deliberately skips it).

A third cut, by stability:

- Most stable to train, least sensitive to hyperparameters: 3 (offline DPO). Boring loop, boring failure modes.
- Most sensitive: 2 (PPO-RLHF). Four interacting models, many hyperparameters, reward-hacking risk if beta drifts.
- Newest and least understood: 9 (agentic). Multi-turn credit assignment, expensive rollouts, environment non-determinism.

---

## Minimal infrastructure required, per loop

Rough sketches; assumes single-node-to-small-cluster scale. Production training of a frontier model is a different exercise. "Judge call" assumes API-accessed; running a judge locally is also fine and changes the cost model.

**Loop 1: Pretrain to SFT**
- 1+ training GPUs (more for larger models)
- a curated SFT corpus (~10k to ~1M examples, depending on scope)
- HuggingFace `Trainer` or equivalent
- no inference engine needed at training time

**Loop 2: PPO-RLHF**
- training GPUs that fit four models simultaneously: policy (training), frozen reference, reward model, value head. For a 7B policy on bf16, this is roughly 4 x ~14GB plus optimizer state: call it 2 x 80GB cards minimum, more for headroom.
- a generation engine (vLLM works) for the rollout phase, co-located or separate
- a pre-trained scalar reward model (training the RM is a separate step that needs a human preference dataset, typically ~tens of thousands of pairs)
- monitoring: KL to reference, reward distribution, value-loss / policy-loss ratio, sample completions every step

**Loop 3: Offline DPO**
- training GPUs that fit two models: policy and frozen reference. Roughly half of loop 2's footprint.
- a static preference dataset (HH-RLHF, UltraFeedback, or a custom one)
- no rollout machine. no reward model. no value head.
- the cheapest loop on this list

**Loop 4: Iterative / online DPO**
- training GPUs (same as loop 3: policy + reference)
- a generation engine for rollouts (vLLM cluster or local)
- a scorer: either a learned RM (then GPUs to host it), or an LLM judge (then API access or a separate model serving), or a verifiable checker (then a sandbox / solver)
- 2-4 rounds of: generate K rollouts per prompt, score, build pairs, DPO for ~100-500 steps. Per-round generation cost dominates if K is large or the prompt set is big.

**Loop 5: GRPO with verifiable reward**
- training GPUs for policy + reference (similar footprint to loop 3, plus generation buffer)
- a fast generation engine for K rollouts per prompt (K typically 4-16). vLLM is the standard.
- a checker: symbolic math solver, sandboxed code executor, proof assistant. The checker latency multiplies by `batch_size * K` per step.
- monitoring: KL, advantage std (zero advantage std = no learning signal), entropy, clip fraction, reward histograms

**Loop 6: CAI / RLAIF**
- everything from loop 2 (if running PPO) or loop 3/4 (if running DPO)
- a judge model (the same model, a stronger model, or a frontier API)
- the constitution as a text file plus prompt templates for critique / revise / compare
- one pass of corpus generation for Phase 1 SL-CAI (expensive if the judge is a large model; many critique-revise calls per prompt)

**Loop 7: R1-Zero**
- training GPUs for base model + frozen reference (the reference is just the unchanged base model)
- a fast generation engine
- a checker
- the same infra as loop 5, minus the cold-start SFT step. Less compute up front, more burned during RL because the policy starts further from any useful output.

**Loop 8: R1-distill (student SFT)**
- one large teacher inference cluster, used once: enough vLLM throughput to generate ~100k-1M traces in a reasonable time
- a checker to filter
- training GPUs for the student (a 7B-32B SFT job, ordinary fine-tuning footprint)
- no live RL infrastructure. no reward model. no critic.
- the dominant cost is teacher inference at corpus-generation time; student training is comparatively cheap

**Loop 9: Agentic / tool-use RL**
- training GPUs for policy + reference (typical RL footprint)
- a tool sandbox: Docker per rollout for code agents, headless browser pool for web agents, file-system snapshots for editor agents
- a trajectory verifier: hidden test suite for code, success criterion for browsers, task-completion judge for general tasks
- an orchestrator that can run many concurrent rollouts (each rollout is slow because of environment latency)
- careful logging: trajectory-level rewards are sparse, debugging requires reconstructing what the agent saw at each turn

---

## A 30-second cheat sheet

If you only remember six things from this document:

1. **PPO-RLHF (loop 2)** is the algorithmic baseline: SFT, RM, PPO with a KL penalty to the reference. Every other loop is a simplification or a specialization of this.
2. **DPO (loop 3)** removes the RM and the PPO loop. It's offline and supervised. It underperforms PPO when the data goes stale relative to the policy.
3. **Iterative DPO (loop 4)** fixes DPO's data-staleness problem by regenerating preference data each round. This is the default production recipe in 2024-2025.
4. **GRPO (loop 5)** removes the critic from PPO by using the within-group mean reward as the baseline. It pairs naturally with verifiable rewards (math, code, proofs) where a checker can replace a learned RM.
5. **R1-Zero (loop 7)** is the existence proof that pure RL on a base model elicits reasoning behaviors. R1-distill (loop 8) is the practical recipe for getting those behaviors into smaller models cheaply.
6. **The reward source is what differs most across loops**: learned RM (2, 6), DPO's implicit RM (3, 4), verifiable checker (5, 7, 8 at filter time, 9 at trajectory level), or a judge LLM (4, 6). The RL machinery (policy gradient, clipping, KL) is largely shared.

---

## References

All arXiv IDs were copied from the lecture sources that this cheat sheet draws on (`notes/lectures/10`, `14`, `15`, `17`, `18`, plus `11-dpo.md` and `16-agentic-rl.md` for cross-references). Status: each ID's month / year encoding is internally consistent with the cited year, but a full second-pass verification against arxiv.org has not been done from this document. Treat as unverified per the file header.

**Foundational RLHF**
- Stiennon et al. 2020. "Learning to summarize from human feedback." arXiv:2009.01325.
- Ouyang et al. 2022. "Training language models to follow instructions with human feedback" (InstructGPT). arXiv:2203.02155.
- Bai et al. 2022. "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback." arXiv:2204.05862.

**PPO algorithm**
- Schulman et al. 2017. "Proximal Policy Optimization Algorithms." arXiv:1707.06347.

**DPO and variants**
- Rafailov et al. 2023. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." arXiv:2305.18290.
- Tunstall et al. 2023. "Zephyr: Direct Distillation of LM Alignment." arXiv:2310.16944.
- Xu et al. 2024. "Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study." arXiv:2404.10719.

**Iterative / online preference optimization**
- Liu et al. 2023. "Statistical Rejection Sampling Improves Preference Optimization" (RSO). arXiv:2309.06657. ICLR 2024.
- Yuan et al. 2024. "Self-Rewarding Language Models." arXiv:2401.10020.
- Guo et al. 2024. "Direct Language Model Alignment from Online AI Feedback" (OAIF). arXiv:2402.04792.
- Pang et al. 2024. "Iterative Reasoning Preference Optimization." arXiv:2404.19733. NeurIPS 2024.
- Dubey et al. 2024. "The Llama 3 Herd of Models." arXiv:2407.21783.

**LLM judges, RLAIF, Constitutional AI**
- Bai et al. 2022. "Constitutional AI: Harmlessness from AI Feedback." arXiv:2212.08073.
- Zheng et al. 2023. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." arXiv:2306.05685. NeurIPS 2023.
- Lee et al. 2023. "RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback." arXiv:2309.00267.
- Chen et al. 2024. "Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models" (SPIN). arXiv:2401.01335.
- Huang et al. 2024. "Collective Constitutional AI: Aligning a Language Model with Public Input." arXiv:2406.07814. FAccT 2024.

**Reward overoptimization**
- Gao, Schulman, Hilton 2022. "Scaling Laws for Reward Model Overoptimization." arXiv:2210.10760. ICML 2023.

**GRPO, RLVR, R1**
- Shao et al. 2024. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." arXiv:2402.03300. (Introduces GRPO.)
- DeepSeek-AI 2025. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv:2501.12948. (R1, R1-Zero, R1-distill series.)
- Uesato et al. 2022. "Solving math word problems with process- and outcome-based feedback." arXiv:2211.14275.
- Lightman et al. 2023. "Let's Verify Step by Step." arXiv:2305.20050.

**Self-distillation lineage and chain-of-thought distillation**
- Zelikman et al. 2022. "STaR: Bootstrapping Reasoning With Reasoning." arXiv:2203.14465. NeurIPS 2022.
- Magister et al. 2022. "Teaching Small Language Models to Reason." arXiv:2212.08410.
- Hinton, Vinyals, Dean 2015. "Distilling the Knowledge in a Neural Network." arXiv:1503.02531.
- Hsieh et al. 2023. "Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes." arXiv:2305.02301.
- Gulcehre et al. 2023. "Reinforced Self-Training (ReST) for Language Modeling." arXiv:2308.08998.

**Agentic / tool-use RL**
- Nakano et al. 2021. "WebGPT: Browser-assisted question-answering with human feedback." arXiv:2112.09332.
- Yao et al. 2022/2023. "ReAct: Synergizing Reasoning and Acting in Language Models." arXiv:2210.03629.
- Schick et al. 2023. "Toolformer: Language Models Can Teach Themselves to Use Tools." arXiv:2302.04761.
- Jimenez et al. 2023. "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" arXiv:2310.06770. ICLR 2024.
- Wei et al. 2025. "SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution." arXiv:2502.18449.

---

## See also

- `notes/cheat-sheets/RL-Quick-Reference.md`: classical RL algorithms (DQN, A2C, PPO, SAC) and when to use which.
- `notes/cheat-sheets/RL-Math-Formulas.md`: the underlying math (Bellman, policy gradient, KL, GAE).
- `notes/lectures/10-ppo-for-llms.md`: full derivation and implementation of loop 2.
- `notes/lectures/11-dpo.md`: derivation of loop 3.
- `notes/lectures/14-constitutional-ai-rlaif.md`: loop 6 with code.
- `notes/lectures/15-rl-verifiable-rewards.md`: loops 5 and 7 with code.
- `notes/lectures/16-agentic-rl.md`: loop 9, environment design and verifier choices.
- `notes/lectures/17-online-iterative-preference.md`: loop 4 with code.
- `notes/lectures/18-distillation-reasoning.md`: loop 8 with code.
