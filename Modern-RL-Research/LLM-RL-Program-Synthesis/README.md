# Reinforcement Learning for Program Synthesis with LLMs

## Overview

This directory contains resources on the intersection of Reinforcement Learning, Large Language Models, and Program Synthesis. Recent advances (2022-2025) have shown that RL-enhanced LLMs can achieve near-human performance on competitive programming tasks.

---

## Key Papers

### 1. **CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning** (NeurIPS 2022)
- **Authors**: Le et al., Salesforce Research
- **arXiv**: [2207.01780](https://arxiv.org/abs/2207.01780)
- **GitHub**: [salesforce/CodeRL](https://github.com/salesforce/CodeRL)

**Key Contributions**:
- Treats code-generating LM as an actor network
- Introduces a critic network trained to predict functional correctness
- Provides dense feedback signals during training
- Uses unit test execution as reward signals

**Approach**: During training, CodeRL uses the LM as an actor and trains a critic to predict if generated programs are functionally correct, enabling RL training with execution feedback.

---

### 2. **Competition-Level Code Generation with AlphaCode** (Science 2022)
- **Authors**: Li et al., DeepMind
- **Paper**: Science, Feb 2022
- **arXiv**: [2203.07814](https://arxiv.org/abs/2203.07814)

**Key Achievements**:
- Achieved approximately human-level performance on Codeforces
- Ranked in top 54.3% in simulated programming competitions
- Generates millions of diverse programs using transformer networks
- Filters and clusters programs to top 10 submissions

**Technical Approach**:
- Uses GOLD (Generation by Off-policy Learning from Demonstrations)
- Offline RL algorithm focusing on most likely solutions
- Massive-scale sampling with intelligent filtering

---

### 3. **Process-Supervised Reinforcement Learning for Code Generation** (2025)
- **Focus**: Improving upon AlphaCode's approach
- **Innovation**: Process-level supervision rather than outcome-only feedback

---

## Core Techniques

### Reinforcement Learning Methods
1. **PPO (Proximal Policy Optimization)**: Standard RL training for code models
2. **DPO (Direct Preference Optimization)**: Eliminates separate reward model (2023)
3. **GRPO (Group Relative Policy Optimization)**: Samples multiple answers for relative quality assessment (DeepSeekMath 2024)

### Reward Signals
- **Unit Tests**: Primary signal for functional correctness
- **Execution Feedback**: Runtime behavior and output validation
- **Human Feedback**: RLHF for alignment with coding preferences

---

## Key Challenges

### 1. Functional Correctness vs Token Similarity
- Traditional metrics (BLEU, ROUGE) don't correlate with code correctness
- Need execution-based validation through unit tests
- Token similarity ≠ functional equivalence

### 2. Safety and Security
- LLMs can generate destructive code (database overwrites, harmful API calls)
- Need sandboxed execution environments
- Berkeley's GoEx runtime: deterministic undo for all operations

### 3. Reward Design
- Sparse rewards from test passing
- Need for dense intermediate feedback
- Balancing correctness with efficiency/style

---

## Recent Developments (2023-2025)

### Test-Time and Post-Training RL
Starting in 2024, advanced LLMs showed substantial improvements using:
- Test-time search and refinement
- Self-trained self-correction
- Chain-of-thought reasoning with RL

**Notable Systems**:
- OpenAI o1 (2024)
- Anthropic Claude 3.7/4 (2024-2025)
- DeepSeek R1 (2024)
- Kimi K1.5 (2024)
- Qwen 3 (2024)

---

## State-of-the-Art Approaches

### Sandbox Isolation
- **Standard Practice**: Resource-bounded containers for code execution
- **Security**: Isolated environments prevent harmful operations
- **Validation**: Success judged solely by unit-test suites

### Model Architectures
- Transformer-based LLMs (GPT, Claude, LLaMA family)
- Specialized code models (CodeGen, StarCoder, Code LLaMA)
- Actor-Critic architectures for RL training

---

## Resources

### Surveys and Comprehensive Reviews
- **"Enhancing Code LLMs with Reinforcement Learning in Code Generation: A Survey"** (2024)
  - arXiv: [2412.20367](https://arxiv.org/abs/2412.20367)

- **"A Survey on Large Language Models for Code Generation"** (TOSEM 2025)
  - arXiv: [2503.01245](https://arxiv.org/abs/2503.01245)
  - GitHub: [juyongjiang/CodeLLMSurvey](https://github.com/juyongjiang/CodeLLMSurvey)

- **"Reinforcement Learning Meets Large Language Models"** (2024)
  - arXiv: [2509.16679](https://arxiv.org/abs/2509.16679)
  - Covers advancements across LLM lifecycle

### GitHub Repositories
- [TsinghuaC3I/Awesome-RL-for-LRMs](https://github.com/TsinghuaC3I/Awesome-RL-for-LRMs) - RL for Large Reasoning Models
- [opendilab/awesome-RLHF](https://github.com/opendilab/awesome-RLHF) - Curated RLHF resources

---

## Key Takeaways

1. **RL is Essential**: Self-correction and reasoning in program synthesis require RL
2. **Execution-Based Rewards**: Unit tests provide concrete functionality measures
3. **Safety First**: Sandboxed execution is now standard practice
4. **Beyond Supervised Learning**: Pure supervised fine-tuning insufficient for competitive programming
5. **Active Research Area**: Significant progress in 2024-2025 with new methods emerging

---

## Future Directions

- Multi-modal program synthesis (combining code, documentation, tests)
- Better exploration strategies for large program spaces
- Incorporating formal verification alongside RL
- Transfer learning across programming languages
- Real-world software engineering tasks beyond competitive programming

---

*Last Updated: 2025*
