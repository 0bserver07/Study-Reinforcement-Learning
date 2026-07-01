# Changelog

Notable changes to the repo. Not a release log — there are no releases — just a record of what moved and why.

## 2026-05-15 — promote the hand-written notes out of Archive/

The trusted, hand-written 2017 notes had been sitting in a folder called `Archive/` — which connotes "old/dead" — while the unreviewed AI-drafted lecture series occupied `notes/`. Backwards. Fixed:

- `Archive/2017-Course-Notes/CS294-DeepRL-Berkeley/` → `notes/cs294-2017/` (with `imgs/` intact and image links unchanged).
- `Archive/2017-Course-Notes/Elements-Of-RL/` → `notes/sutton-barto-digest/`.
- Both moved files got a `<!-- status: hand-written -->` header.
- `Archive/` directory deleted (`Archive/README.md` was a wrapper; no content lost).
- Root `readme.md` "What's here" section restructured to lead with the trusted, hand-written content (the CS294 notes, the Sutton & Barto digest, the curated talks/books/courses, the tested exercises) and clearly demote the AI-drafted lecture series as scaffold-with-skepticism. "Start here" reordered to lead with safer paths (talks/books → exercises → drafts).
- `notes/README.md` rewritten in the same spirit — hand-written content first, lecture series second with a clear caveat about what `unreviewed` means.
- `AGENTS.md` and `CLAUDE.md` updated: the layout table now points at `notes/cs294-2017/` and `notes/sutton-barto-digest/` as the trusted, frozen, never-reword material instead of `Archive/`.
- GitHub topics refreshed: dropped `guideline` and `study` (generic), added `rlhf`, `llm-alignment`, `dpo`, `grpo`, `ppo`, `rlvr`, `agentic-rl`, `lecture-notes`, `study-notes`, `deepseek-r1`, `constitutional-ai`, `policy-gradient`, `q-learning`, `sutton-barto`. Description sharpened.

## 2026-05-16 — bulk add: 15 new lectures, 4 new exercises, 4 new cheat sheets, 2 new reading lists

Context: the lecture series stopped at 19 (offline RL) and was missing most of the 2024–2025 material — exploration, multi-agent, world models, the agentic / reasoning / RLAIF / reward-hacking deep-dives, the systems and hardware layer, and the meta-RL / robotics adjacencies. Twenty-five parallel agents wrote one chunk each, each to a unique path, each `unreviewed`. The orchestrator wired them into the index files; no existing material was reworded.

Lectures added (notes/lectures/, all `unreviewed`, ~10,200 lines total):

- 20 — Exploration: ε-greedy → UCB → RND → ICM → NGU → Go-Explore, plus how exploration shows up differently in LLM-RL (temperature / best-of-N rather than dedicated bonuses).
- 21 — Multi-agent RL and self-play: stochastic games, CTDE (MADDPG, QMIX, MAPPO), self-play (AlphaZero, AlphaStar, OpenAI Five), PSRO and exploitability.
- 22 — World models: the Dreamer family (PlaNet → V3), MuZero / EfficientZero / Sampled MuZero, transformer world models (IRIS, GAIA-1, Genie), the LLM-as-world-model thread.
- 23 — Process reward models vs outcome reward models: PRM800K, Math-Shepherd, why DeepSeek skipped PRMs, best-of-N re-ranking, the autoPRM line.
- 24 — Computer use and browser agents: WebGPT → Mind2Web → WebArena → VisualWebArena → OSWorld; Set-of-Marks; Anthropic Computer Use and OpenAI Operator; prompt-injection failure modes.
- 25 — Long-horizon credit assignment: GAE/PPO at long horizons, hindsight relabeling, Reflexion, ToT, MCTS-then-train, the open question of value functions at 1000+ steps.
- 26 — RL for mathematical reasoning: GSM8K/MATH/AIME/OlympiadBench/MiniF2F; STaR; PRM800K; DeepSeekMath/GRPO; DeepSeek-R1; the rollout/verifier loop; tool-augmented math RL (PAL, Lean/Coq).
- 27 — RLAIF and synthetic preferences: Lee et al RLAIF, OAIF, UltraFeedback; LLM-judge biases (position, length, self-preference, sycophancy) and mitigations; weak-to-strong generalization.
- 28 — Reward hacking and verifier design: Goodhart, CoastRunners; Stiennon overoptimization, Gao scaling laws (KL as the controller); RLVR-specific verifier hacks (test-overfitting, format exploits, judge-injection); mitigations.
- 29 — Distributed RL systems: A3C → IMPALA → Ape-X → SEED RL → R2D2; Ray RLlib, ACME; the LLM-RL stack (TRL, OpenRLHF, verl/HybridFlow, DeepSpeed-Chat) with vLLM rollouts and FSDP/ZeRO training.
- 30 — RL inference infrastructure: why ~70–90% of LLM-RL wall-clock is decode; PagedAttention, continuous batching, RadixAttention; speculative decoding caveats for RL; weight-sync patterns.
- 31 — Hardware for RL: classical (CPU-env-bottlenecked) vs LLM-RL (HBM-bandwidth-bound); accelerators (H100/H200/B200, MI300X, TPU v5p/Trillium); FlashAttention 1/2/3; Triton; FP8/INT4 quantization for rollouts; interconnect geometry.
- 32 — Meta-RL and in-context RL: MAML and Reptile (gradient-based); RL² and "Learning to RL" (recurrence-based); Decision Transformer, Trajectory Transformer, Algorithm Distillation, DPT; the bridge to LLM in-context learning.
- 33 — Robotics RL: PILCO, domain randomization, OpenAI dexterous hand; the data-driven shift (RT-1, RT-2, Octo, OpenVLA, π₀, Open X-Embodiment); residual policies; Isaac Gym, MuJoCo, Brax, Drake.
- 34 — Self-distillation and self-improvement loops: STaR, ReST^EM, Self-Instruct, CAI's RL phase, self-rewarding LMs; failure modes (mode collapse, hallucination amplification, self-preference); the connection to RLVR.

Cheat sheets added (notes/cheat-sheets/, all `unreviewed`):

- `RLHF-vs-DPO-vs-GRPO.md` — side-by-side of PPO-RLHF / DPO / IPO / KTO / ORPO / SimPO / GRPO / RLVR, with comparison table, per-method blocks, decision tree.
- `RL-LLM-loops-2026.md` — ASCII data-flow diagrams of every major LLM-RL training loop (SFT, RLHF, DPO, iterative DPO, RLVR/GRPO, CAI, R1-Zero, R1-distill, agentic).
- `KL-control.md` — KL penalties across TRPO/PPO/RLHF/DPO/GRPO with formulas, K1/K2/K3 estimators, β tuning rules of thumb.
- `RL-loss-functions.md` — one block per algorithm (23 total) with loss in symbols, gradient, ~5-10 line PyTorch snippet, stability tradeoff, "watch in training" diagnostic.

Exercises added (`exercises/`, each tested against its reference solution and passing):

- `05-ppo/` — PPO with GAE on CartPole-v1. Five filled-in pieces (ActorCriticNet, compute_gae, ppo_clip_loss, collect_rollouts, train). 13 tests, ~13s wall-clock, integration test threshold mean_last10 > 150.
- `09-reward-model/` — Bradley-Terry reward model on synthetic preferences (4-dim features, known true reward, BT-labeled pairs). 5 tests including `test_bradley_terry_loss_tied` (asserting loss == log(2)) and a Spearman-correlation integration test > 0.85. ~3s.
- `11-dpo/` — DPO on a toy per-prompt categorical policy. 6 tests including `test_dpo_loss_starts_at_log2` (when policy == ref) and an integration test that the greedy policy mean true reward goes from ~0.09 (uniform baseline) to ~1.7 after 2000 steps. ~2.5s.
- `20-exploration/` — RND on a sparse-reward 20-state chain MDP. 11 tests: `test_q_learning_alone_fails` (mean reward < 0.1 after 200 episodes, never sees the goal), `test_train_with_intrinsic_succeeds` (mean reward > 0.5 after 200 episodes with intrinsic_coef=0.1). ~4.5s.

Reading lists added (hand-curated `reference/papers/<topic>/README.md` files; the auto-generated `PAPERS.md` companions still need a collector run):

- `GRPO-RLVR/README.md` — 20 verified arXiv IDs across foundational PPO/InstructGPT/STaR, the GRPO / RLVR / reasoning lineage (DeepSeekMath, R1, Kimi k1.5, Qwen2.5-Math, Open-Reasoner-Zero), PRMs (Uesato, Lightman, Math-Shepherd, OmegaPRM), code-RL with verifiable rewards (CodeRL, CodeT, SWE-bench, SWE-RL), and verifier design / reward hacking (Stiennon, Gao, Pan in-context reward hacking).
- `Agentic-RL/README.md` — 21 verified arXiv IDs across tool use (ReAct, Toolformer, Reflexion, ToT, Voyager), browser agents (WebGPT, Mind2Web, WebArena, VisualWebArena), computer-use (OSWorld, Set-of-Marks) + Anthropic Computer Use / OpenAI Operator by URL, coding agents (SWE-agent, OpenHands, AutoCodeRover), and benchmarks (GAIA, BFCL, Cybench).

Index updates:

- `notes/README.md` — table extended from 19 rows to 34; new cheat sheets listed; the "Planned: an exploration lecture" line removed (lecture 20 covers it).
- `CURRICULUM.md` — added Block 6 (modern RL deep-dives: 20, 21, 22, 32, 33), Block 7 (reasoning, agents, LLM modern stack continued: 23–28, 34), Block 8 (systems and infrastructure: 29, 30, 31), each with prereqs / time / checkpoints. Removed the "Optionally: an exploration lecture" from Planned.
- `exercises/README.md` — table extended from 5 rows to 9; the "PPO exercise on a continuous-control env is the next obvious one" line softened (`05-ppo/` now covers the discrete-action case; continuous control still open).

Caveats:

- All 15 lectures, 4 cheat sheets, and 2 reading lists are `unreviewed`. Per AGENTS.md, that means citations / math / claims should be treated as unverified until a person reads them end to end. Each subagent reported its own verification status in its summary; the high-confidence citations are the ones a subagent fetched against `arxiv.org/abs/<id>` and confirmed title+authors+date.
- Specifically flagged by subagents as wanting a careful second look: lecture 27 (RLAIF) — three citations (Tunstall 2310.16944, Meng 2405.14734, Lambert 2403.13787) were added without live arxiv fetches; lecture 34 (self-distillation) — the ReST^EM ID was set to 2312.06585 (Singh, "Beyond Human Data") distinct from Gulcehre's ReST 2308.08998; cheat sheet `RLHF-vs-DPO-vs-GRPO.md` — IPO and KTO loss formulas are simplifications, flagged inline with "verify against §4 of the paper"; cheat sheet `KL-control.md` — the InstructGPT β ≈ 0.02 and the "6 nats" adaptive-KL target were quoted from memory rather than re-verified.
  - **Resolved 2026-05-15** (the "_Verified_" tags in the lecture files were ahead of this caveat; the caveat was stale). All five flagged arXiv IDs fetched live against `arxiv.org/abs/<id>` and confirmed title + first author + month: Zephyr / Tunstall / Oct 2023 (2310.16944); SimPO / Meng / May 2024 (2405.14734); RewardBench / Lambert / Mar 2024 (2403.13787); ReST / Gulcehre / Aug 2023 (2308.08998); "Beyond Human Data" (ReST^EM) / Singh / Dec 2023 (2312.06585). The lecture-34 ID split was correct. Still not fully closed: InstructGPT β ≈ 0.02 — the term "KL reward coefficient, β" is confirmed in the paper (2203.02155) and a secondary RLHF write-up corroborates 0.02, but Appendix C wasn't machine-extractable, so the cheat sheet keeps its "verify against the paper" hedge; the IPO/KTO formulas remain inline-flagged sketches by design.
- All 4 new exercises pass their tests on a laptop CPU. The PPO integration test has only ~20 points of margin above its threshold at seed=0 (mean_last10 ≈ 169.6 vs threshold 150); seeds 1, 2, 7, 42 all clear with bigger margin. The exploration test is single-seed only.
- The new GRPO-RLVR and Agentic-RL reading lists have only the hand-curated README so far; `tools/arxiv-collector/arxiv_paper_collector.py` should be re-run to generate their `PAPERS.md` companions.

## 2026-05-12 — restructure: separate the layers, set up rules

Context: the repo had grown two layers — the original 2017 notes, and a much larger newer layer added in 2025 (a 13-lecture series, scraped paper lists, a content tool). The newer layer was unmarked, wrote in a first person it hadn't earned, and shipped broken links, phantom lectures, and made-up citations. This pass separates the two so nobody has to guess what's trustworthy, and sets up conventions so they can coexist.

Structural:

- Added `AGENTS.md` (and `CLAUDE.md` pointing to it): how the repo is laid out, the `status:` convention, voice rules, citation rules, how lectures and exercises work, what agents should and shouldn't do.
- Added the review-status convention. Every doc under `notes/` and `reference/` carries a `<!-- status: hand-written | reviewed | unreviewed -->` comment plus a one-line visible note. `unreviewed` means nobody has checked it. Only a person promotes a file to `reviewed`.
- Reorganized:
  - `self-study-lectures/` → `notes/` (`notes/lectures/`, `notes/cheat-sheets/`, `notes/diagrams/`).
  - `Modern-RL-Research/` → `reference/papers/` (it's a reading list, not the main content).
  - `scripts/` → `tools/arxiv-collector/`; `content-pipeline/` → `tools/content-pipeline/`.
  - Added `exercises/`, `drafts/`, `CURRICULUM.md`.
- Added `.gitignore`; untracked the committed `.DS_Store` files.

Content:

- Stamped status headers on all 13 lectures, both cheat sheets, and the diagrams file. All `unreviewed`.
- `notes/README.md` rewritten: fixed four broken lecture links; removed the two phantom lectures (14 Constitutional AI, 15 Test-Time Compute — the files never existed; listed as "planned" instead); removed the first-person framing; de-slopped.
- `readme.md` rewritten: kept the original curated talks/books/courses and the Sutton & Barto agent diagram; replaced the marketing-voice sections; removed an invented paper ("RL for Safe LLM Code Generation, Berkeley 2025"); gave the paper list resolvable identifiers (arXiv IDs / venues).
- `tools/content-pipeline/README.md` rewritten to describe only the scripts that exist — the old one described ~10 scripts and several directories that were never built.
- `reference/papers/README.md` rewritten: removed the wall of significance puffery, the unverifiable survey citations (one was an "arXiv:2509.16679" labeled "2024" — that ID is September 2025), and the wrong author/affiliation list; kept a short landmark-papers list with IDs that resolve.
- `Archive/README.md` lightly de-slopped; fixed the dead `../Modern-RL-Research/` link.
- Built three exercises, each with a task, a starter file with TODOs, `pytest` tests, a reference solution, and graduated hints: `01-mdps` (value iteration on a gridworld), `02-policy-gradients` (REINFORCE on CartPole), `03-q-learning` (tabular Q-learning on non-slippery FrozenLake). Plus `exercises/README.md` and `exercises/requirements.txt`. Each test suite was run against its reference solution and passes (`03-q-learning` uses optimistic Q-initialization, `02-policy-gradients` uses `lr=1e-3` + gradient clipping — earlier configs that looked plausible didn't actually train, which is exactly the kind of thing these tests exist to catch).
- Lecture 02: fixed the broken link to lecture 03; fixed a code bug (`import gym` while using the new Gymnasium reset/step API → `import gymnasium as gym`); replaced fabricated "expected output" numbers with ranges; removed the first-person debugging-war-story framing; gave the references real arXiv IDs; linked the exercise. Still `unreviewed`.
- Lecture 01: removed a fabricated value-function output matrix (it didn't match the code's own reward structure — a goal-adjacent cell should be ≈ +10, not negative); removed the first-person framing; added a missing import to a code snippet; tightened the references with venues / arXiv IDs; de-slopped headings. Still `unreviewed`.
- Lecture 03: fixed a dead reference (`/Modern-RL-Research/RLHF-and-Alignment/PAPERS.md` — that path no longer exists, and those DQN papers were never in the LLM-focused reading lists anyway); added a missing `import torch.nn.functional as F` to the "complete DQN" code block (it used `F.mse_loss`); added a note that `FrozenLake-v1` is slippery by default; added arXiv IDs to the references; de-slopped headings; linked the exercise. Still `unreviewed`.
- Lectures 04–13 reviewed (de-slop + fixes), each still `unreviewed`. Notable: dead `Modern-RL-Research/` reference paths corrected (04, 07, 08, 13); old-API Gymnasium calls fixed (06: `import gym` → `import gymnasium as gym`; 07, 08: 4-tuple `env.step()` → 5-tuple); missing `import torch.nn.functional as F` added to snippets using `F.…` (07, 11, 12); a wrong next-lecture link fixed (10 pointed at lecture 13, now 11); a code bug fixed (04: a list-of-tensors used where a tensor was needed); fabricated outputs removed — invented Atari training times (05), a made-up GSM8K/MATH cross-method comparison table (12), invented AlphaCode pass@k figures (13); fabricated or misattributed citations corrected/removed — CodeRL was credited to Meta throughout, it's Salesforce / Le et al. (13); a nonexistent "Beyond DPO: A Comprehensive Study" (12) and a nonexistent "Anthropic 2024 — Constitutional AI with DPO" (11) removed; KTO mis-credited to Anthropic → Contextual AI, and "Scaling Laws for Reward Model Overoptimization" mis-credited to Anthropic → OpenAI (12); unverified compute/cost claims removed (10: "16 hours on 256 GPUs", "~$1M"); first-person diaries and breathless epigraphs stripped throughout; arXiv IDs / venues added to references in every lecture. (Done by parallel subagents on disjoint files.)
- `reference/papers/`'s three sub-READMEs and `tools/arxiv-collector/README.md` rewritten short and honest: removed the tutorial bloat and significance-puffery; fixed dead `Modern-RL-Research/` and `scripts/` paths; removed unverifiable or made-up citations — an "arXiv:2509.16679" survey labeled "2024" (that ID is September 2025), a Berkeley master's thesis presented as a peer paper with GoEx as "its" contribution, a "cRLHF / Wong et al." with no resolvable identifier, a "Process-Supervised RL for Code Generation (2025)" with no author/ID, and assorted product/system marketing claims.
- Fixed two stale `/self-study-lectures/lectures/` paths in the cheat sheets → `../lectures/`.
- Lectures 14–17 added (drafted by parallel agents, all `unreviewed`, ~2,200 lines total): 14 — Constitutional AI / RLAIF / self-improvement (LLM-as-judge, RLAIF, the CAI two-phase recipe, self-rewarding LMs, SPIN); 15 — RL with verifiable rewards & reasoning models (GRPO in depth, the DeepSeek-R1 / R1-Zero recipe, process vs. outcome reward models, STaR/ReST); 16 — agentic RL (multi-turn rollouts, tool use, ReAct, SWE-bench-style training, long-horizon credit assignment); 17 — online & iterative preference optimization + generative reward models (why offline DPO underperforms PPO, iterative/online DPO, reward over-optimization, the 2024–25 stack). Citations verified against arXiv where a paper exists; o1 and the closed agentic systems are flagged as having no public technical paper. The earlier phantom Lectures 14–15 are now real; lecture 13 points forward to 14; the index (`notes/README.md`) and `CURRICULUM.md` (new "Block 4") updated.
- Added `tools/lit-builder/` — a copy of the local `iclr-lit-builder` tool, retuned: `configs/keywords.yaml` replaced with an RL / RLHF / reasoning / agentic keyword set; installed in an isolated venv; ran `fetch → ingest → filter` on ICLR 2026 (19,813 papers ingested → 4,329 matched the RL keyword set). The `score` step (LLM triage 0–3 with a reason) needs a credential — `ANTHROPIC_API_KEY` (Claude Haiku, default) or `OLLAMA_API_KEY` (cloud models) — and is queued; `tools/lit-builder/README.md` has the exact commands. Once scored, the top papers get `deepen`-ed into digests and folded into `reference/papers/<topic>/README.md`; the auto-scraped `PAPERS.md` files stay as the unfiltered appendix. The user's original `iclr-lit-builder` was not touched.
- Lectures 18 and 19 added (parallel agents, all `unreviewed`): 18 — Distillation of reasoning models (the R1-distill recipe; why imitation works for reasoning when the teacher is checkable; STaR/ReST as the self-distillation cousins; limits — can't exceed the teacher); 19 — Offline RL (BCQ, BEAR, CQL, IQL, Decision Transformer; the bridge to DPO as offline preference learning). ~830 lines combined, all citations verified against arXiv. Lecture 17's ending updated to point forward to 18; lecture 19 is now the last in the series. `CURRICULUM.md` got a new "Block 5 — foundational topic that didn't fit earlier" for 19.
- Cheat sheets and diagrams audited (parallel agents). The substantive findings: a **wrong KL-divergence direction** in `RL-Math-Formulas.md` (it described `KL(p||q)` as "from q to p" — corrected to "from p to q"); a **wrong DPO loss** in `RL-Algorithm-Diagrams.md` (was missing the chosen-rejected log-ratio difference — corrected to `-log σ(β(log π(y_w)/π_ref(y_w) − log π(y_l)/π_ref(y_l)))`); a **wrong GRPO advantage** in the same diagrams file (was showing rank-order values like `[+1, -1, +1, -1]` — corrected to the actual `A_i = (r_i − μ)/σ` group-relative form); a too-low DQN replay-buffer recommendation (10K → 100K min); a typo'd CS285 URL; ~14 arXiv IDs added to the quick-reference paper list. All three files de-slopped (emoji, hype banners, fake-first-person debugging-diary footers stripped). Still `unreviewed`.
- Two new tested exercises (parallel agents, both verified against their reference solutions): `exercises/04-actor-critic/` (REINFORCE → A2C with a learned value baseline on CartPole; 14 tests pass in ~70s; integration test asserts `max(returns) ≥ 195` and `mean(last 100) > 100`) and `exercises/15-grpo-rlvr/` (a tiny GRPO loop on a verifiable arithmetic toy task — 9 prompts, K=8 samples per prompt, group-relative advantage, PPO clipped surrogate; 16 tests pass in ~2s; toy policy converges to ~perfect accuracy). Both registered in `exercises/README.md`.
- Slop sweep across 15 root docs / READMEs (`readme.md`, `AGENTS.md`, `CLAUDE.md`, `CURRICULUM.md`, `CHANGELOG.md`, the various `README.md`s under `notes/`, `reference/`, `tools/`, `exercises/`, `drafts/`, `Archive/`): zero matches against the slop blacklist (`comprehensive`, `powerful`, `cutting-edge`, `groundbreaking`, `seamless`, `leverage`, `delve`, `tapestry`, figurative `landscape`/`navigate`, "let's dive", "this is huge", "it's not just X — it's Y", throat-clearing openers, etc.).

Still TODO:

- Run `lit score` on the lit-builder data (needs a credential), `deepen` the top ~20–30 per area, and write the curated digests into `reference/papers/<topic>/README.md`. Once a credential is set, this is a one-shot run plus the curation pass.
- A person reviews the lectures end to end and promotes the ones that hold up to `reviewed` (an agent can't do that). The 14–19 drafts and especially the 20–34 drafts (added 2026-05-16) — newer, faster-moving, machine-written.
- Decide whether `tools/arxiv-collector/papers_database.json` (large, regenerable) should stay tracked.
- A PPO exercise on a continuous-control env (`Pendulum-v1` or `LunarLanderContinuous-v2`) — `05-ppo/` covers the discrete-action case.
- Run `tools/arxiv-collector/arxiv_paper_collector.py` against the two new topics (`GRPO-RLVR/`, `Agentic-RL/`) to populate their `PAPERS.md` companions.
