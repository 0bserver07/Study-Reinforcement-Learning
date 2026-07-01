<!-- status: unreviewed | last-reviewed: never -->

# Reference: agentic RL — tool use, computer use, multi-turn

Reading list for agents that act in a loop — read files, run code, click buttons, call APIs — and the training methods that fit them. The defining shift from earlier alignment work is that the model is no longer scored on a single output; it produces a sequence of actions against an environment that evolves in response, and the reward arrives at the end (or, when you can engineer it, partway through).

The course covers the conceptual ground in [lecture 16: agentic RL](../../../notes/lectures/16-agentic-rl.md) and the credit-assignment problem in [lecture 25: long-horizon credit](../../../notes/lectures/25-long-horizon-credit.md). [Lecture 24: computer-use agents](../../../notes/lectures/24-computer-use-agents.md) covers the screen-pixel-and-mouse variant. This README is for going deeper than the lectures.

A note on scope: a lot of the strongest results in this space (2024–2025) come from industry reports and product launches with no published training details. Where that's the case the entry says so. Don't read "no public paper" as "doesn't exist" — read it as "you cannot verify how it was built."

## Where to start

Three papers, in this order, for someone new to the space:

1. **Yao et al. 2022 — ReAct** ([arXiv:2210.03629](https://arxiv.org/abs/2210.03629)). The scaffold that everything else is built on: think, act, observe, repeat. Originally a prompting trick on a frozen model. Read it for the interface, not the results — the loop structure is what got adopted.
2. **Jimenez et al. 2023 — SWE-bench** ([arXiv:2310.06770](https://arxiv.org/abs/2310.06770)). The benchmark that turned "resolve a real GitHub issue" into a checkable agentic-RL target. Read this and you'll understand why the field's reward function has settled on "hidden tests pass."
3. **Wei et al. 2025 — SWE-RL** ([arXiv:2502.18449](https://arxiv.org/abs/2502.18449)). The most concrete published RL training pipeline for an agentic coding model. Reaches 41.0% on SWE-bench Verified with a Llama 3 70B. After this you'll see why the proprietary systems are doing roughly the same thing at larger scale.

Once those three are in your head, branch into whichever section below matches your interest.

---

## 1. Tool use and reasoning frameworks

These define how an agent is structured to act in the world. None of them are RL papers in the strict sense — they're about the scaffold, the interface, and the prompting conventions that RL later optimizes on top of. Treat them as the substrate: when you train an agent with RL, you're tuning a policy that lives inside one of these scaffolds. The choice of scaffold determines the action space and the trajectory format your gradients flow through.

**ReAct** — Yao et al., 2022. [arXiv:2210.03629](https://arxiv.org/abs/2210.03629).
The canonical think/act/observe loop. Each step the model emits a `Thought:` (free-form reasoning), then an `Action:` (a tool call), and the environment fills in the `Observation:`. Introduced as in-context prompting on a frozen model; later adopted as the default interface for RL fine-tuning. The contribution is the format and the demonstration that interleaving reasoning and acting beats acting alone — not any specific training method. See [lecture 16, "The scaffold"](../../../notes/lectures/16-agentic-rl.md) for how RL plugs into this format.

**Toolformer** — Schick et al., 2023. [arXiv:2302.04761](https://arxiv.org/abs/2302.04761).
Self-supervised tool use, not RL. The model generates candidate API calls inline in text, executes them, and keeps the calls that improve next-token prediction on the surrounding tokens. Result: a model that can call a calculator, search engine, translator, and calendar without any explicit reward. Useful here as a contrast — tool use as a capability can emerge from supervised learning, but you only get task-completion behaviour if you reward task completion. Toolformer answers "how do I teach the model the API exists." Agentic RL answers "how do I teach the model to use the API to actually finish the task."

**Reflexion** — Shinn et al., 2023. [arXiv:2303.11366](https://arxiv.org/abs/2303.11366).
"Verbal reinforcement learning." After a failed attempt, the agent generates a natural-language self-critique ("I tried X, the test failed because Y, next time I should Z"), stores it in episodic memory, and conditions the next attempt on it. No weights are updated; the "learning" lives entirely in the context window. Worth reading to make explicit what gradient-based RL actually provides over in-context iteration — Reflexion gets a long way without any training and clarifies the point at which you do need weight updates (transfer across tasks, persistence across context resets, scale).

**Tree of Thoughts (ToT)** — Yao et al., 2023. [arXiv:2305.10601](https://arxiv.org/abs/2305.10601).
Deliberate search over reasoning steps rather than sampling a single chain. The model expands and evaluates partial-solution nodes, with BFS or DFS over the tree. Same first author as ReAct. ToT is the bridge from "one trajectory per problem" to "search the space of trajectories" — useful background for why GRPO-style group sampling (sample K trajectories, baseline against their mean) and MCTS-style methods (Agent Q in section 4) showed up later in the agentic-RL literature. Read it for the search formulation, not the prompting details.

**Voyager** — Wang et al., 2023. [arXiv:2305.16291](https://arxiv.org/abs/2305.16291).
LLM agent in Minecraft that writes its own skills as Python functions and accumulates them into a skill library it queries later. Auto-curriculum proposes the next task; an iterative-prompting loop refines code until execution succeeds; the skill library grows. No RL fine-tuning; the demonstration is that an open-ended setting can be navigated by a frozen model plus the right scaffold. Worth reading because the open-world setting forces decisions about exploration, skill reuse, and curriculum that the constrained SWE-bench world lets you avoid. Most "agentic memory" claims you'll see in product copy trace some intuition back to Voyager.

For the foundations these build on, see [lecture 16](../../../notes/lectures/16-agentic-rl.md).

---

## 2. Browser and web agents

The web-agent line is older than the coding-agent line and the action space is messier. The same model has to plan, parse the DOM, follow links, and sometimes deal with the visual layout. The benchmarks have gotten progressively more realistic over time — from "rank these answers" to "actually fill out this multi-page checkout flow against a live e-commerce site."

**WebGPT** — Nakano et al., 2021. [arXiv:2112.09332](https://arxiv.org/abs/2112.09332).
One of the earliest applications of RLHF to an agentic setting. A fine-tuned GPT-3 issues search queries, clicks links, scrolls, quotes passages, and finally writes an answer with citations. The action space is a fixed, small set; the reward is from human comparisons of full trajectory-and-answer outputs. Bootstrapped with imitation on human demonstrations, refined with RLHF. The structural blueprint for everything that follows — define the action space, collect comparisons on trajectories, RL the policy — even though everyone since has scaled it up and broken pieces of it.

**Mind2Web** — Deng et al., 2023. [arXiv:2306.06070](https://arxiv.org/abs/2306.06070).
A dataset of over 2,000 open-ended tasks across 137 real websites with crowd-collected human demonstrations. Tasks span shopping, travel booking, social media, information lookup. Less an RL paper than a "what does general web use actually look like" paper — the value is in the task distribution and the demonstration that off-the-shelf models do poorly when you stop hand-curating the environment. Often cited as the dataset for SFT/imitation pretraining of web agents before downstream RL.

**WebArena** — Zhou et al., 2023. [arXiv:2307.13854](https://arxiv.org/abs/2307.13854).
Self-hostable end-to-end web environments (e-commerce, forums, software-development tooling, content management, maps), each cloned from a real open-source application. Functional evaluation — the agent's actions have to actually accomplish the goal (item added to cart, post created, issue closed), not just match a reference trajectory. The score is read from the resulting application state. Currently one of the most-used web-agent benchmarks and the one where leaderboard numbers tend to mean something.

**VisualWebArena** — Koh et al., 2024. [arXiv:2401.13649](https://arxiv.org/abs/2401.13649).
WebArena's visual sibling — tasks where the goal description or the page itself requires understanding screenshots, not just the DOM. Same group as WebArena. Useful for thinking about how much of "web agent performance" depends on the DOM serialization (a lot, it turns out) versus genuine visual reasoning.

A practical note: pure-DOM agents and screenshot-based agents are different beasts, and the literature splits accordingly. WebGPT/Mind2Web/WebArena are mostly DOM-based — the agent reads a cleaned text representation of the page and emits structured actions. VisualWebArena and the computer-use papers below operate on pixels and rely on multimodal models. The training methods overlap but the failure modes don't.

---

## 3. Computer-use agents

Operating the desktop, not just the browser. The action space here is roughly "any mouse click, any keystroke" — much wider than a curated set of tools. The reward grounding ("did the file actually save? did the email actually send?") is harder to verify than test pass/fail because there isn't always a clean programmatic check. And the observation is usually a screenshot, which means the model has to do visual grounding before it can plan an action.

**OSWorld** — Xie et al., 2024. [arXiv:2404.07972](https://arxiv.org/abs/2404.07972).
369 real-computer tasks across OS utilities, office suite, browser, code editor, and multi-app workflows. Tasks ship with execution-based evaluators that inspect the resulting filesystem and application state (file contents, settings, sent email, etc.). The current standard benchmark for "can the agent operate a computer" and the one that made it possible to compare computer-use models on something other than vibes. Read the appendix for the evaluator design — that's where most of the methodological work is.

**Set-of-Mark prompting (SoM)** — Yang et al., 2023. [arXiv:2310.11441](https://arxiv.org/abs/2310.11441).
Overlay numbered marks on segmented regions of an image so a multimodal model can reference UI elements by ID ("click 7") instead of by pixel coordinates ("click at 412, 836"). Originally framed as a GPT-4V grounding technique, but the same trick shows up in computer-use scaffolds now — turning pixel coordinates into a discrete-ish action space the model is actually good at predicting. Worth knowing about even if you don't use it directly, because the alternative (predicting raw coordinates) is one of the things that makes computer-use agents fragile.

**Anthropic Computer Use** — model card / announcement: <https://www.anthropic.com/news/3-5-models-and-computer-use> (October 2024).
Claude takes screenshots, moves a virtual cursor, types into UIs, runs into the same brittleness everyone else does (off-by-pixel clicks, hallucinated buttons). Public material covers capability claims, evaluation results on OSWorld and related benchmarks, and safety mitigations. The training procedure — how the model learned to operate a computer, whether it was RL on agent trajectories or supervised on a dataset of screen recordings or both — is not disclosed.

**OpenAI Operator** — announcement: <https://openai.com/index/introducing-operator/> (January 2025).
Browser-operating agent built on a "Computer-Using Agent" (CUA) model derived from GPT-4o. Lives in a separate web product rather than as a tool in the main chat surface. As with Anthropic's, the announcement covers the capability and the eval numbers; the training procedure is not published.

For the lecture-level treatment, see [lecture 24: computer-use agents](../../../notes/lectures/24-computer-use-agents.md).

---

## 4. Multi-turn RL training

This is the section where the gap between "what's known publicly" and "what's actually deployed" is widest. Most of the high-scoring agentic systems in 2024–2025 are described in blog posts and model cards as "trained with RL on agentic trajectories" without specifying the reward function, the scaffold, the number of environment steps, or how credit is assigned across a long trajectory. Treat that as a real limitation of what you can learn from the published literature alone — much of this is in industry reports and not in formal papers yet (as of mid-2025).

What you can read:

**SWE-RL** — Wei et al., 2025. [arXiv:2502.18449](https://arxiv.org/abs/2502.18449).
RL fine-tuning of Llama 3 on a large corpus of open-source software-evolution data — issues, diffs, and the regression tests attached to them. The reward is a similarity score between the generated patch and the reference fix from the original PR. Llama3-SWE-RL-70B reaches 41.0% on SWE-bench Verified, with the paper attributing roughly 10 points of that to the RL stage on top of SFT. Published enough detail to reason about, and currently the closest thing to "the public reference implementation" of agentic-RL for code. See [lecture 16, "SWE-bench"](../../../notes/lectures/16-agentic-rl.md).

**Agent Q** — Putta et al., 2024. [arXiv:2408.07199](https://arxiv.org/abs/2408.07199).
MCTS over agent trajectories + AI self-critique + DPO-style fine-tuning on the trajectories the search identifies as good. Evaluated on WebShop (simulated e-commerce) and a real booking benchmark, with a reported 340% relative improvement on the real benchmark for a Llama-3 70B model. One of the few public end-to-end agentic-RL recipes for a web task. Read it as a recipe rather than as the last word — the search-then-distill pattern (MCTS produces trajectories, supervised/preference-based loss imitates them) is what's worth learning from it.

**GRPO applied to multi-turn settings** — there's no canonical paper called "GRPO for agentic tasks." The reasoning is in [lecture 16, "Credit assignment"](../../../notes/lectures/16-agentic-rl.md): treat each rollout as a trajectory, compute the group-relative advantage from K trajectories per task as `(R_i - mean) / std`, and weight every action token in trajectory `i` by that advantage. The original GRPO paper (Shao et al., 2024, DeepSeekMath, [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)) is single-turn math; the agentic adaptation is folk knowledge dressed up in implementation details that vary by group. If you're implementing this yourself, the DeepSeekMath formulation is what you'll be porting.

**DeepSeek-R1** — DeepSeek-AI, 2025. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948).
Not a pure agentic-RL paper, but the most detailed public account in 2025 of "RL on a base model produces a useful reasoning policy without much SFT in between." The pure-RL R1-Zero variant is especially relevant because it shows reasoning behaviour (longer chains, self-verification, backtracking) emerging from outcome reward alone — the same pattern people are trying to elicit in agentic settings. Read for the training-recipe details and the ablations on reward shaping.

Things that exist but aren't a paper:

- **"Reasoning models with tool use"** (o1/o3, Claude 3.5/3.7/4 extended thinking, DeepSeek-R1 derivatives that call tools). These clearly use multi-turn RL when the tool calls are part of the trained behaviour rather than a wrapper. Training details are partly public for DeepSeek-R1 (above) and largely opaque for the rest.
- **Top-of-leaderboard SWE-bench systems.** Most cite "RL on agentic trajectories" in their model cards; none publish the training pipeline in enough detail to reproduce. The published training literature lags behind what's deployed by roughly a year.
- **"RL on tool-use sequences" inside frontier-lab training pipelines.** Reported in passing in tech reports; not specified.

If you're trying to build something in this area, the honest summary is: the published methods cover the principles, and you'll be filling in a lot of the engineering yourself. Expect this section to age fast — papers that nail down the recipes are likely in the 2025–2026 pipeline.

---

## 5. Coding agents

Where agentic RL is most mature, because the reward is concrete: tests either pass or they don't. This is the sub-field that established "hidden tests as the reward function," and most of the methodological work — action-space design, sandboxing, episode-length budgeting, credit assignment — was either invented here or pressure-tested here first.

**SWE-bench** — Jimenez et al., 2023 (ICLR 2024). [arXiv:2310.06770](https://arxiv.org/abs/2310.06770).
2,294 tasks drawn from real GitHub pull requests across 12 Python repositories. Each task hands the agent an issue description and a repository snapshot; the reward is whether the hidden regression-test suite passes after the agent's edits. The benchmark that anchored the whole sub-field. **SWE-bench Verified** — a ~500-task human-validated subset — is what most papers report now, because the full benchmark has tasks where the original PR is ambiguous or the hidden tests are unreasonable. If you read one paper here, read this one. Covered in [lecture 16, "SWE-bench"](../../../notes/lectures/16-agentic-rl.md).

**SWE-agent** — Yang et al., 2024. [arXiv:2405.15793](https://arxiv.org/abs/2405.15793).
Introduces the Agent-Computer Interface (ACI): a designed set of commands (`open`, `goto`, `edit`, `submit`, `search_file`, etc.) tuned for what language models actually use well, rather than raw shell access. The finding is that giving the model a curated, narrower interface beats giving it bash — because bash is brittle to argument formatting, and a misformatted argument burns a step. Same authors as SWE-bench. Useful for thinking about action-space design as a first-class problem distinct from the policy. The ACI is also a worked example of "the environment is the reward function" — what the agent can do is what you've decided the agent can do.

**OpenHands** (formerly OpenDevin) — Wang et al., 2024. [arXiv:2407.16741](https://arxiv.org/abs/2407.16741).
Open platform for building general-purpose AI software-engineering agents. Pluggable LLMs (OpenAI, Anthropic, local models via providers), sandboxed Docker environments, multi-agent compositions, support for browser actions alongside code actions. The paper is more about the platform than any specific training method, but the system has been used as the harness for several published agent experiments. Active codebase as of writing.

**AutoCodeRover** — Zhang et al., 2024. [arXiv:2404.05427](https://arxiv.org/abs/2404.05427).
Agentic patching for GitHub issues using code-structure-aware navigation — AST search, spectrum-based fault localization, class/method-level lookups — rather than raw file reading. Different point in the design space: less "general LLM with tools," more "LLM with explicit program-analysis tools." Read alongside SWE-agent to see two different answers to "what should the action space for a code agent be." AutoCodeRover bets on the structure of code as a navigation aid; SWE-agent bets on a narrower text-editor abstraction.

For benchmark methodology and reward design in this area, see [lecture 13: RLHF for code generation](../../../notes/lectures/13-rlhf-code-generation.md) and [`../LLM-Code-Generation/`](../LLM-Code-Generation/).

---

## 6. Long-horizon and credit assignment

The single hardest problem in agentic RL: a 30-step episode with one terminal reward gives every action in the trajectory the same gradient signal. The variance of policy-gradient estimates scales with trajectory length. The failure modes — mostly-successful trajectory with one disastrous step that doesn't get punished hard enough, mostly-failing trajectory with one essential step that doesn't get reinforced — are exactly what naive REINFORCE cannot see. Everything in this section is some answer to "how do we get better than 'multiply every action by the terminal reward.'"

**Lecture 25: long-horizon and credit assignment** — [`../../../notes/lectures/25-long-horizon-credit.md`](../../../notes/lectures/25-long-horizon-credit.md).
Being written in parallel with this reading list; may not exist yet when you click the link. The plan is to cover GRPO-style group baselines, process reward models, hierarchical decomposition, return decomposition, and the trade-offs between them. The dedicated lecture goes further than the credit-assignment section of lecture 16.

**Process reward models in math** — Lightman et al., 2023, "Let's Verify Step by Step." [arXiv:2305.20050](https://arxiv.org/abs/2305.20050).
Trains a model (a PRM) to score the correctness of each step in a chain of reasoning, and uses it to rerank or to weight training. The clearest published evidence that step-level supervision beats outcome-only supervision on long reasoning tasks — at the cost of needing per-step labels from somewhere. Most "PRM in agentic RL" claims trace back to this paper's setup. Worth knowing about even if you never train a PRM yourself, because the failure modes (PRM gets fooled by plausible-looking-but-wrong steps; PRM is itself a learned model with its own biases) are the same ones that resurface in agentic settings.

**Group-relative baselines (GRPO)** — Shao et al., 2024 ([arXiv:2402.03300](https://arxiv.org/abs/2402.03300)).
Originated for single-turn math. Sample K completions per prompt, compute the advantage of each as its deviation from the group mean normalized by the group std, weight every action token in that completion by that advantage. The reason this matters for agentic RL: it sidesteps the need for a separately trained value-function critic, which would have to score an entire conversation-plus-environment history. You pay for it in compute (K rollouts per task instead of 1), but the engineering simplification is large enough that most public agentic-RL pipelines use some variant of this.

**Process reward models for code/agents** — there isn't a single canonical paper yet. The math PRM idea has been adapted to code-step scoring and to general agent-step scoring; results in the public literature are mixed and the labelling cost is the main blocker. See lecture 23 for the conceptual treatment.

The lecture treatment in [lecture 16, "Credit assignment"](../../../notes/lectures/16-agentic-rl.md) walks through the math; the dedicated lecture 25 goes further into hierarchical and PRM-based approaches.

---

## 7. Benchmarks worth knowing

The benchmarks are how this sub-field communicates. A method without numbers on at least one of these is hard to compare against anything; numbers on these are the closest thing to a shared language for progress.

**SWE-bench / SWE-bench Verified** — Jimenez et al., 2023. [arXiv:2310.06770](https://arxiv.org/abs/2310.06770).
Real GitHub issues, hidden-test reward. The standard for coding agents. The full benchmark has tasks where the original PR is ambiguous or the hidden tests are unreasonable; **SWE-bench Verified** is a ~500-task human-vetted subset and is what most modern papers report. When someone says "X% on SWE-bench" without qualification, ask which one.

**WebArena** — Zhou et al., 2023. [arXiv:2307.13854](https://arxiv.org/abs/2307.13854).
Self-hosted web environments (e-commerce, forums, content management, software-dev tooling, maps) with functional, state-based evaluation. Standard for browser agents. The fact that you can run it offline is a feature — your training loop doesn't depend on a live website that might change.

**VisualWebArena** — Koh et al., 2024. [arXiv:2401.13649](https://arxiv.org/abs/2401.13649).
Visual variant of WebArena. Tasks require understanding the rendered page, not just the DOM. The split between DOM agents and visual agents tends to be reported here.

**OSWorld** — Xie et al., 2024. [arXiv:2404.07972](https://arxiv.org/abs/2404.07972).
Real-computer tasks across desktop applications (OS utilities, office suite, browser, code editor, multi-app workflows). Execution-based evaluators check application/filesystem state. Standard for computer-use agents. Has become the headline benchmark for any "agent that operates a computer" claim.

**GAIA** — Mialon et al., 2023. [arXiv:2311.12983](https://arxiv.org/abs/2311.12983).
466 general-assistant questions that require multi-step tool use (web search, file reading, multimodal reasoning) and have unambiguous answers. Reportedly around 92% accuracy for humans, historically much lower for systems. Tracks "general agent" capability rather than a specific domain. Useful as a cross-section read on agent quality when SWE-bench/OSWorld feel too narrow.

**HumanEval / MBPP / APPS** — function- and program-level code benchmarks.
Lower-level than SWE-bench (single function vs. multi-file edit) but cheap to score and useful as sanity checks. Generally not agentic — they expect a single completion — but they're how most code models get evaluated before they're put into an agent harness. Covered in [`../LLM-Code-Generation/`](../LLM-Code-Generation/).

**BFCL — Berkeley Function-Calling Leaderboard** — leaderboard, not a paper. Cite as <https://gorilla.cs.berkeley.edu/leaderboard.html>.
Evaluates function/tool-calling correctness across many APIs. Current versions cover single-turn calls, multi-turn (multi-call) scenarios, and various flavours of parameter validation. Useful when "did the agent call the right tool with the right arguments" is the thing you actually care about — distinct from "did the agent solve the task," which is what SWE-bench/WebArena/OSWorld measure.

**Cybench** — cybersecurity capture-the-flag tasks for agents. Cite by URL: <https://cybench.github.io/>.
Forty CTF challenges from real competitions, structured for agent evaluation. Distinct from other cyber-eval projects with overlapping names — there are several, and the naming is unstable. Check what you're citing before quoting numbers. The cyber-eval area is moving fast and most of it predates established benchmark conventions.

What's missing from this list: most multimodal-agent and embodied-agent benchmarks; specialized math benchmarks (covered in [lecture 26: RL for math reasoning](../../../notes/lectures/26-rl-math-reasoning.md) and elsewhere); robotics-RL benchmarks (covered in [lecture 33](../../../notes/lectures/33-robotics-rl.md)). This list is deliberately scoped to "agent operates an information environment" rather than "agent operates a physical robot."

---

## Cross-references in this repo

- [Lecture 16: agentic RL — tool use and multi-turn interaction](../../../notes/lectures/16-agentic-rl.md) — the conceptual entry point. Covers scaffolds, the environment-as-reward-function framing, credit assignment, and a worked code-fixing environment.
- [Lecture 24: computer-use agents](../../../notes/lectures/24-computer-use-agents.md) — pixels and mouse, not just text APIs.
- [Lecture 25: long-horizon and credit assignment](../../../notes/lectures/25-long-horizon-credit.md) — the credit-assignment problem in depth (may be in progress).
- [Lecture 15: RL with verifiable rewards](../../../notes/lectures/15-rl-verifiable-rewards.md) — the single-turn ancestor of everything here. RLVR scales naturally to agentic RL once you accept that the trajectory is now the unit of optimization.
- [Lecture 23: process reward models](../../../notes/lectures/23-process-reward-models.md) — the alternative to terminal-only rewards.
- [`../LLM-Code-Generation/`](../LLM-Code-Generation/) — coding-reward design, sandboxing, execution feedback as a signal.
- [`../RLHF-and-Alignment/`](../RLHF-and-Alignment/) — single-turn alignment background; the methods (PPO, DPO, GRPO) carry over.

---

## A note on what's missing

Things that should arguably be on this list but aren't, with reasons:

- **Specific agent products** (Cursor, Devin, Claude Code, Codex, Replit Agent, etc.). No training papers, and announcement posts and changelogs aren't a citation. Their behaviour is worth tracking informally; their training method isn't part of the published record and won't be cited here until it is.
- **"Survey of agentic RL" papers.** Several exist as of 2024–2025; most were written too early to be useful and copy each other's reference lists. Skim a recent one to find primary sources, then cite the primary sources. A survey isn't a substitute for reading the originals when the originals are this recent.
- **Multi-agent setups.** Covered separately in [lecture 21: multi-agent RL](../../../notes/lectures/21-multi-agent-rl.md). The methods overlap with what's here but the central problems (communication protocols, role assignment, credit assignment across agents) are different enough to deserve their own treatment.
- **Specific tools like LangChain, AutoGen, CrewAI.** These are agent-orchestration frameworks rather than training methods. Useful if you're building; not what this list is for. If you want the engineering side, look at OpenHands above as a reference for how a research-grade harness is put together.
- **Reward hacking and safety in agentic settings.** Covered in [lecture 28: reward hacking](../../../notes/lectures/28-reward-hacking.md). The agentic version of the problem (the agent deletes the tests to "pass" them; the agent navigates to a pre-cached success page) is a recurring theme rather than a single paper.
- **Frontier-lab tech reports** that mention "RL on agentic data" without specifying how. They're worth tracking for capability claims; they're not citable as training methods.

## Reading-order suggestions for specific goals

If you have a specific goal, the path through this list is different:

- **"I want to train a code agent."** Read SWE-bench, then SWE-agent, then SWE-RL. Skim OpenHands for harness ideas. Then look at GRPO and the credit-assignment material.
- **"I want to train a web agent."** Read WebGPT for the historical setup, WebArena for the modern eval, Agent Q for one published recipe. Then look at VisualWebArena if you care about visual grounding.
- **"I want to understand the credit-assignment problem."** Skip the benchmark papers; read lecture 16's credit-assignment section, then Lightman et al. (PRM), then DeepSeekMath (GRPO). Section 6 above is for you.
- **"I just want to know what's going on in the field."** Read the three "where to start" papers, plus the section 4 intro paragraph for the honest accounting of what's published vs. what's deployed.

---

## Vocabulary you'll see across the papers

A few terms get used inconsistently across the literature; here's the rough convention used in this list and in the lectures.

- **Agent.** A model that produces actions in a loop against an environment, as opposed to a model that produces a single output. The boundary is fuzzy — a tool-using model with one tool call is "barely" agentic — but the loop-vs.-one-shot distinction is the load-bearing one.
- **Scaffold.** The prompting/formatting convention that defines what counts as a thought, an action, and an observation. ReAct is a scaffold. The scaffold lives outside the model; the policy lives inside it.
- **Trajectory** or **episode.** The full sequence `(s_1, a_1, o_1, s_2, a_2, o_2, ..., R)` of state-action-observation triples and the terminal reward. The unit of optimization for most agentic-RL methods.
- **Tool call.** A structured action that invokes an external function (calculator, search, code execution, API). Distinct from a generic action in that it has a typed signature you can validate.
- **Action space.** The set of actions the policy can emit. For SWE-agent it's a curated command set; for OSWorld it's "click anywhere, type anything"; for WebArena it's a small set of structured DOM operations. Design choice with large consequences.
- **Hidden tests / held-out evaluation.** Tests the agent cannot see during the episode, used only for terminal scoring. The standard answer to reward gaming in code agents.
- **Process reward model (PRM).** A model that scores intermediate steps for plausibility/correctness, as opposed to an outcome reward model (ORM) that scores only the final answer.
- **GRPO.** Group Relative Policy Optimization — sample K rollouts per task, baseline each by the group mean, normalize by group std. The advantage formula is the load-bearing piece; the rest is policy-gradient bookkeeping.

## Recurring themes across the papers

Reading the above in sequence, a few patterns show up often enough to be worth naming explicitly. None of these is novel to any one paper; they're the connective tissue.

**The environment is the reward function.** Almost everything here defines success by an environment check — pytest exit code, application state after the click, hidden-test pass — rather than by a learned reward model. This is the agentic version of RLVR and it's what makes the training loop tractable. It's also where the failure modes live: anything the environment will accept as success, the trained policy will eventually find a way to produce. See [lecture 16, "The environment is the reward function"](../../../notes/lectures/16-agentic-rl.md) and [lecture 28: reward hacking](../../../notes/lectures/28-reward-hacking.md).

**Action-space design is half the work.** SWE-agent's ACI, AutoCodeRover's program-analysis tools, Set-of-Mark's numbered overlays, and the various "give the model a curated set of commands rather than raw shell" choices are not incidental engineering — they're the part of the system that determines what the policy can express. A good action space makes the credit-assignment problem easier; a bad one makes a good policy untrainable.

**Sample inefficiency is the bottleneck.** Each step in an agentic episode is a model call plus an environment call. The environment call might be running pytest, rendering a webpage, or executing a shell command in a sandboxed container. A 30-step episode at 30 environment calls per episode, K=8 trajectories per task, batch of 16 tasks per update step — that's thousands of external calls per gradient step. This is orders of magnitude more expensive than single-turn RL and is the practical reason group-based methods (which amortize one task across K rollouts) get used over per-step value-network methods (which would need a separate model).

**The published literature trails the deployed systems.** Almost every entry in section 4 above carries a version of "this is partially open" or "this is opaque." Plan around that — read the papers for principles, expect to do your own engineering, and don't assume the leaderboard scores correspond to anything reproducible.

**Benchmarks set the research direction.** SWE-bench made "resolve a real issue" the headline number for code agents; OSWorld did the same for computer use. This is partly good (concrete, checkable) and partly bad (everyone optimizes the benchmark, including in ways the benchmark wasn't meant to measure). Treat top-of-leaderboard claims with the standard skepticism — ask whether the model has seen the benchmark data, whether the scaffold is benchmark-specific, whether the reported pass rate is on Verified or full SWE-bench, and so on. Most of the methodological discipline in this sub-field comes from the benchmark designers; most of the over-claiming comes from people using their benchmarks.

## Regenerating `PAPERS.md`

The `PAPERS.md` file in this directory (when it exists) is generated by the arXiv collector — don't hand-edit it; re-run the collector when it's stale.

To regenerate `PAPERS.md` in this directory: run `tools/arxiv-collector/arxiv_paper_collector.py` — see [`../README.md`](../README.md).

Add or update the topic-keyword queries in the collector config if you want this directory's listing to track a different slice of arXiv. The hand-curated entries in this README are the trusted set; `PAPERS.md` is a wider net for finding things to add here.
