# Modern RL Research (2022-2025)

## Overview

This directory contains curated resources on cutting-edge Reinforcement Learning research, focusing on applications to Large Language Models, Code Generation, and Program Synthesis.

---

## 📂 Directory Structure

### 1. [LLM + RL for Program Synthesis](./LLM-RL-Program-Synthesis/)
**Focus**: Competition-level code generation and automated program synthesis

**Key Topics**:
- AlphaCode (DeepMind, 2022) - Near-human performance on competitive programming
- CodeRL (Salesforce, 2022) - RL framework for code generation
- Process-supervised RL for code
- Test-time compute scaling (o1, DeepSeek R1)

**Why It Matters**: Demonstrates that RL can enable LLMs to solve complex reasoning tasks that require search, verification, and iterative refinement.

---

### 2. [LLM Code Generation with RL](./LLM-Code-Generation/)
**Focus**: Practical code generation for real-world software engineering

**Key Topics**:
- RLHF for code quality and style
- Execution feedback as reward signals
- Safety and security in code generation
- Sandboxed training environments
- Benchmark evaluation (HumanEval, MBPP, APPS)

**Why It Matters**: Shows how to safely deploy RL-trained code models in production, addressing security, reliability, and alignment challenges.

---

### 3. [RLHF and Alignment](./RLHF-and-Alignment/)
**Focus**: Aligning code generation models with human preferences and safety requirements

**Key Topics**:
- PPO, DPO, GRPO comparison
- Reward model training
- Constitutional AI for code
- Multi-objective optimization
- Reward hacking prevention

**Why It Matters**: RLHF is now the dominant paradigm for training production LLMs. Understanding these techniques is essential for modern AI development.

---

## 🎯 Why This Matters

### The RL + LLM Revolution (2022-2025)

The past three years have seen a fundamental shift in how we train and deploy language models:

1. **RLHF Became Standard** (2022-2023)
   - ChatGPT demonstrated the power of RLHF
   - All major LLMs now use some form of RL alignment
   - Code generation particularly benefits from execution feedback

2. **Reasoning Models Emerged** (2024-2025)
   - OpenAI o1: Test-time RL for chain-of-thought reasoning
   - DeepSeek R1: Open approach to reasoning with RL
   - Claude Sonnet 4+: Long-horizon reasoning capabilities
   - These models excel at math, coding, and complex problem-solving

3. **Beyond Token Prediction**
   - Traditional supervised learning hits limits on complex tasks
   - RL enables exploration, self-correction, and goal-directed behavior
   - Combination of pre-training, SFT, and RL is now standard

---

## 🚀 Quick Start Guide

### For Researchers
1. Start with [RLHF and Alignment](./RLHF-and-Alignment/) to understand modern training methods
2. Study [Program Synthesis](./LLM-RL-Program-Synthesis/) papers (AlphaCode, CodeRL)
3. Explore safety considerations in [Code Generation](./LLM-Code-Generation/)

### For Practitioners
1. Review [Code Generation](./LLM-Code-Generation/) for production best practices
2. Understand [RLHF pipelines](./RLHF-and-Alignment/) for deployment
3. Study safety frameworks (GoEx, sandboxing, Constitutional AI)

### For Students
1. Ensure you understand classic RL first (see [Archive](../Archive/))
2. Read survey papers in each subdirectory
3. Implement simple RLHF pipelines with existing frameworks (TRL, etc.)

---

## 🔑 Key Insights

### What We've Learned (2022-2025)

1. **Execution Feedback is Critical**
   - For code, test outcomes are far better than token prediction
   - Unit tests provide objective, dense reward signals
   - Enables models to learn from mistakes

2. **Safety Cannot Be an Afterthought**
   - Code models can generate vulnerabilities
   - Sandboxing is now standard practice
   - Constitutional AI and safety post-training are essential

3. **RLHF Scales**
   - Works for models from 1B to 100B+ parameters
   - Compute-intensive but dramatically improves quality
   - DPO and GRPO offer more efficient alternatives to PPO

4. **Test-Time Compute Matters**
   - Reasoning models use additional compute at inference
   - Search, verification, and refinement improve outputs
   - Quality vs. latency tradeoff

5. **Self-Improvement is Possible**
   - Models can generate training data for themselves
   - Iterative refinement with RL improves over time
   - Still requires initial human feedback for alignment

---

## 📊 Impact on the Field

### Research Impact
- **1000+ papers** on RLHF and RL for LLMs (2022-2025)
- Major conferences (NeurIPS, ICLR, ICML) have dedicated workshops
- New benchmarks emerge regularly (LiveCodeBench, SWE-bench)

### Industry Impact
- All major AI labs use RLHF (OpenAI, Anthropic, Google, Meta)
- Code assistants are now commonplace (Copilot, Cursor, Replit)
- Estimated 40%+ of code written with AI assistance by 2025

### Open Source Momentum
- Open weights models competitive with proprietary (DeepSeek, Qwen)
- Tools like TRL make RLHF accessible
- Datasets and benchmarks freely available

---

## 🔬 Research Frontiers

### Current Hot Topics (2025)

1. **Scaling Test-Time Compute**
   - How much better can models get with more inference compute?
   - Optimal search strategies
   - Cost-benefit analysis

2. **Multi-Modal Code Generation**
   - From diagrams, sketches, natural language
   - Vision + language for UI generation
   - Audio inputs for accessibility

3. **Formal Verification + RL**
   - Combining theorem provers with neural models
   - Provably correct code generation
   - Certified robustness

4. **Personalization and Adaptation**
   - Learning individual/team coding styles
   - Few-shot adaptation to new domains
   - Continual learning without catastrophic forgetting

5. **Beyond Code: Scientific Reasoning**
   - Mathematical theorem proving
   - Scientific hypothesis generation
   - Experiment design and execution

---

## 📚 Essential Reading

### Must-Read Surveys
1. **"Enhancing Code LLMs with RL"** (Dec 2024) - arXiv:2412.20367
2. **"RL Meets LLMs: A Survey"** (2024) - arXiv:2509.16679
3. **"LLMs for Code: A Comprehensive Survey"** (2025) - arXiv:2503.01245

### Landmark Papers
1. **AlphaCode** (Science 2022) - Proved RL can reach human-level competitive programming
2. **CodeRL** (NeurIPS 2022) - Framework for RL-based program synthesis
3. **DPO** (2023) - Simpler alternative to PPO for alignment
4. **"RL for Safe Code Generation"** (Berkeley 2025) - GoEx runtime for safety

### Practical Guides
- Hugging Face TRL documentation
- OpenAI RLHF best practices
- Anthropic Constitutional AI paper

---

## 🛠️ Tools and Frameworks

### Training Frameworks
- **TRL (Transformer Reinforcement Learning)**: Hugging Face's RLHF library
- **DeepSpeed**: Efficient training for large models
- **Composer**: Mosaic ML's training framework

### Evaluation Tools
- **HumanEval**: OpenAI's code benchmark
- **MBPP**: Google's Python problems
- **LiveCodeBench**: Continuously updated benchmark
- **SWE-bench**: Real GitHub issues

### Safety Tools
- **GoEx**: Berkeley's safe execution runtime
- **Bandit/Semgrep**: Static analysis for security
- **Docker/gVisor**: Sandboxing technologies

---

## 🤝 Community Resources

### GitHub Organizations
- [opendilab](https://github.com/opendilab) - RL algorithms and RLHF
- [huggingface](https://github.com/huggingface) - TRL and model hub
- [salesforce](https://github.com/salesforce) - CodeRL and related work

### Active Researchers to Follow
- Sergey Levine (Berkeley) - Model-based RL, robotics
- Chelsea Finn (Stanford) - Meta-learning, few-shot adaptation
- Yann Dubois (Stanford/Anthropic) - RLHF and alignment
- Sida Wang (OpenAI) - o1 and reasoning models

### Conferences
- **NeurIPS**: Largest ML conference, strong RL track
- **ICLR**: Focus on representation learning and RL
- **ICML**: Broad ML conference with RL papers
- **EMNLP/ACL**: NLP conferences increasingly covering code

---

## 📈 Keeping Up to Date

This is a rapidly evolving field. To stay current:

1. **arXiv alerts**: Set up notifications for "reinforcement learning" + "code generation"
2. **Twitter/X**: Follow researchers and labs
3. **Papers with Code**: Track SOTA on benchmarks
4. **Conference workshops**: Attend talks on RL for LLMs
5. **This repo**: Watch for updates!

---

## 💡 Contributing

Found a great paper or resource? Please contribute!

- Open an issue with suggestions
- Submit a pull request with new content
- Share your own experiments and findings

---

## 🎓 Learning Path Recommendation

### Week 1-2: Foundations
- Review classic RL if needed ([Archive](../Archive/))
- Read RLHF survey
- Understand PPO basics

### Week 3-4: Code Generation
- Study AlphaCode and CodeRL papers
- Experiment with HumanEval benchmark
- Try TRL library for RLHF

### Week 5-6: Advanced Topics
- Explore DPO and GRPO
- Study safety frameworks
- Investigate reasoning models (o1, DeepSeek R1)

### Week 7-8: Hands-On Project
- Implement RLHF pipeline for code task
- Compare PPO vs DPO
- Evaluate on standard benchmarks

---

*Last Updated: 2025*

*This is a living document. Star and watch this repo for updates!*
