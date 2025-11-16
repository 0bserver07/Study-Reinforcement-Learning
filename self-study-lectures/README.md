# Self-Study Lecture Series: Deep RL to LLM Alignment

> "These are my notes to future me. If past-me had this, I would've saved months."

This series documents my journey learning reinforcement learning from scratch to implementing RLHF for LLMs. Each lecture includes:
- **Theory** with actual intuition (not just equations)
- **Python implementations** that actually work
- **Paper references** with what they really contribute
- **Gotchas** I discovered the hard way
- **Applications** explained like a human, not a research paper

---

## 📚 Lecture Series Structure

### Foundation Block (Don't Skip This!)
1. **[Lecture 01: MDPs and Bellman Equations](./lectures/01-mdps-bellman.md)** - The actual foundation
2. **[Lecture 02: Policy Gradients from Scratch](./lectures/02-policy-gradients.md)** - REINFORCE to PPO
3. **[Lecture 03: Value Functions and Q-Learning](./lectures/03-value-functions.md)** - When to use what
4. **[Lecture 04: Actor-Critic Methods](./lectures/04-actor-critic.md)** - Why they dominate
5. **[Lecture 05: Trust Regions and TRPO](./lectures/05-trpo.md)** - The math people skip

### Modern RL Block (The Good Stuff)
6. **[Lecture 06: PPO - Practical Policy Optimization](./lectures/06-ppo.md)** - Industry standard
7. **[Lecture 07: Off-Policy Learning](./lectures/07-off-policy.md)** - SAC, TD3, and replay buffers
8. **[Lecture 08: Model-Based RL](./lectures/08-model-based.md)** - When you can't afford millions of samples

### LLM Alignment Block (Why You're Really Here)
9. **[Lecture 09: Reward Modeling Fundamentals](./lectures/09-reward-modeling.md)** - RLHF Part 1
10. **[Lecture 10: PPO for Language Models](./lectures/10-ppo-for-llms.md)** - RLHF Part 2
11. **[Lecture 11: Direct Preference Optimization](./lectures/11-dpo.md)** - The PPO killer?
12. **[Lecture 12: Beyond DPO - GRPO, IPO, RRHF](./lectures/12-beyond-dpo.md)** - 2024-2025 methods

### Applications Block (Make It Real)
13. **[Lecture 13: RLHF for Code Generation](./lectures/13-rlhf-code.md)** - AlphaCode, CodeRL
14. **[Lecture 14: Constitutional AI](./lectures/14-constitutional-ai.md)** - Safety alignment
15. **[Lecture 15: Test-Time Compute Scaling](./lectures/15-test-time-compute.md)** - o1, DeepSeek R1

---

## 🎯 How to Use This Series

### If You're Starting From Scratch
1. Do lectures 1-5 in order. **Do not skip**.
2. Implement every code example yourself
3. Break things and fix them
4. Move to lecture 6 only when you can explain PPO to a rubber duck

### If You Know RL But Not LLMs
- Skim lectures 1-5 for notation
- Deep dive lectures 9-12
- Study lectures 13-15 for applications

### If You're Here for Code Generation Specifically
- Read lecture 2 (policy gradients intuition)
- Read lecture 10 (PPO for LLMs)
- Deep dive lectures 11-14

---

## 💡 Philosophy of These Notes

### What Makes These Different

**Not This**: "The policy gradient theorem states that ∇J(θ) = E[...]"
**But This**: "Okay so we want to make good actions more likely. The gradient tells us which direction to push the probabilities. Here's why..."

**Not This**: "Implement the algorithm"
**But This**: "Here's the code. It will break at line 47 because of numerical instability. Here's why and how to fix it."

**Not This**: "See [Smith et al. 2023]"
**But This**: "Smith's paper solves the reward hacking problem I mentioned in lecture 9. Their key insight is X. I'll use their technique in lecture 13."

### What You'll Actually Learn

- **The Why**: Why does this algorithm work? What's the intuition?
- **The When**: When should you use this vs that?
- **The How**: Working code you can run today
- **The Gotchas**: Where will this break? What are the edge cases?
- **The History**: How did we get here? What papers matter?

---

## 🔧 Prerequisites

### Math You Need
- Calculus: derivatives, chain rule, gradients
- Probability: expectations, distributions, KL divergence
- Linear algebra: vectors, matrices, basic operations

**Don't panic**: I explain the math as we go. If you can code, you can learn this.

### Programming
- Python intermediate level
- PyTorch or JAX basics (I'll use PyTorch)
- Numpy, basic ML concepts

### Time Investment
- Each lecture: 2-4 hours (reading + coding + debugging)
- Full series: ~60-80 hours
- Worth it: Absolutely

---

## 📖 How Each Lecture Is Structured

### 1. The Setup (What Problem Are We Solving?)
Real scenario, not abstract math

### 2. Intuition First (Before The Math)
Mental models, analogies, diagrams

### 3. The Math (With Running Commentary)
Equations with line-by-line explanation

### 4. The Code (That Actually Runs)
Complete implementation with comments

### 5. The Gotchas (Where You'll Get Stuck)
Common bugs, numerical issues, hyperparameter traps

### 6. Paper Trail (Who Figured This Out)
Key papers with what they actually contributed

### 7. When To Use This (Practical Decision Making)
Problem characteristics, trade-offs, alternatives

### 8. Exercises (Build Understanding)
Small projects to solidify concepts

---

## 🎓 Study Tips From My Experience

### What Worked
1. **Code everything yourself** - Don't copy-paste. Type it out.
2. **Break it on purpose** - Change hyperparameters. Make it fail. Understand why.
3. **Explain out loud** - If you can't explain it simply, you don't get it yet.
4. **Connect to papers** - After coding, read the original paper. Makes way more sense.
5. **Build projects** - After every 3-4 lectures, build something small.

### What Didn't Work
1. Just reading papers (nothing sticks)
2. Watching videos without coding (feels like learning, isn't)
3. Using high-level libraries without understanding underneath
4. Skipping the "boring" math lectures (you'll get stuck later)

### When You Get Stuck
1. Code the simplest possible version
2. Add one feature at a time
3. Print everything (seriously, print all the shapes)
4. Compare to reference implementation
5. Ask: "What would happen if...?"

---

## 🗺️ The Journey Ahead

Here's what you'll be able to do after this series:

**After Foundations (Lectures 1-5)**
- Understand any RL paper's core contribution
- Implement basic RL algorithms from scratch
- Debug reward functions and policy networks
- Know when RL is the right tool (and when it isn't)

**After Modern RL (Lectures 6-8)**
- Implement PPO, SAC, model-based methods
- Train policies on real tasks
- Handle high-dimensional continuous control
- Optimize sample efficiency

**After LLM Alignment (Lectures 9-12)**
- Understand RLHF pipeline deeply
- Implement reward models
- Train LLMs with PPO and DPO
- Evaluate alignment quality

**After Applications (Lectures 13-15)**
- Build code generation systems
- Implement safety constraints
- Use test-time compute effectively
- Design your own alignment methods

---

## 📊 Progress Tracking

As you go through each lecture, mark your progress:

- [ ] Lecture 01: MDPs and Bellman Equations
- [ ] Lecture 02: Policy Gradients from Scratch
- [ ] Lecture 03: Value Functions and Q-Learning
- [ ] Lecture 04: Actor-Critic Methods
- [ ] Lecture 05: Trust Regions and TRPO
- [ ] Lecture 06: PPO - Practical Policy Optimization
- [ ] Lecture 07: Off-Policy Learning
- [ ] Lecture 08: Model-Based RL
- [ ] Lecture 09: Reward Modeling Fundamentals
- [ ] Lecture 10: PPO for Language Models
- [ ] Lecture 11: Direct Preference Optimization
- [ ] Lecture 12: Beyond DPO
- [ ] Lecture 13: RLHF for Code Generation
- [ ] Lecture 14: Constitutional AI
- [ ] Lecture 15: Test-Time Compute Scaling

---

## 🤝 Using These Notes

### They're For You
- Add your own observations
- Write in the margins
- Disagree with me (and note why)
- Expand sections that were unclear

### They're Not Perfect
- I'm learning too
- There are probably bugs in the code
- Some explanations could be better
- That's okay - iterate on them

### Share What You Learn
- If you figure out a better explanation, write it down
- Found a bug? Fix it and document why it happened
- Discovered a gotcha I missed? Add it
- Built something cool? Link it

---

## 🚀 Let's Begin

Start with **[Lecture 01: MDPs and Bellman Equations](./lectures/01-mdps-bellman.md)**.

No shortcuts. No hand-waving. Let's actually understand this stuff.

---

*"The best time to start was yesterday. The second best time is now."*

---

## 📚 Supplementary Resources

I'll reference these throughout:
- Sutton & Barto (2017) - The RL Bible
- Spinning Up in Deep RL (OpenAI) - Practical guide
- David Silver's Lectures - Foundation
- Recent papers in `../Modern-RL-Research/`

But these lectures are self-contained. You can succeed with just these notes.

---

*Last Updated: 2025*
