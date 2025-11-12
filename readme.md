## Study Reinforcement Learning & Deep RL Guide

A comprehensive collection of resources for studying Reinforcement Learning, from foundational concepts to cutting-edge applications in Large Language Models and Program Synthesis.

**🚀 New here? [Start with GETTING_STARTED.md](./GETTING_STARTED.md) to find your learning path!**

---

## 📚 Repository Structure

### 🆕 Modern RL Research (2022-2025)
Cutting-edge research on RL applied to LLMs, code generation, and program synthesis:
- **[LLM + RL for Program Synthesis](./Modern-RL-Research/LLM-RL-Program-Synthesis/)** - AlphaCode, CodeRL, and competition-level code generation
  - 📄 50 recent papers collected
- **[LLM Code Generation with RL](./Modern-RL-Research/LLM-Code-Generation/)** - Practical applications, safety, and real-world deployment
  - 📄 271 recent papers collected
- **[RLHF and Alignment](./Modern-RL-Research/RLHF-and-Alignment/)** - PPO, DPO, GRPO, and aligning code models with human preferences
  - 📄 111 recent papers collected

**📊 Total: 432 papers automatically collected from arXiv!**

### 📦 Archive - Classic RL Resources
Foundational materials and course notes from 2017:
- **[CS294 Deep RL (Berkeley 2017)](./Archive/2017-Course-Notes/CS294-DeepRL-Berkeley/)** - Notes from Levine, Schulman, and Abbeel
- **[Elements of RL](./Archive/2017-Course-Notes/Elements-Of-RL/)** - Core concepts from Sutton & Barto

### 🤖 Research Automation
- **[ArXiv Paper Collector](./scripts/)** - Automatically fetch and organize latest RL+LLM papers
  - Run `python3 scripts/arxiv_paper_collector.py` to update
  - Keep your research collection current with monthly runs!

---

## 🚀 Quick Start Paths

### Path 1: New to RL? Start with Fundamentals
1. Watch the introductory talks (below)
2. Read Sutton & Barto's book (below)
3. Take David Silver's course (below)
4. Check out the archived CS294 notes

### Path 2: Interested in LLMs + RL?
1. Review basic RL concepts (talks and books below)
2. Dive into [Modern RL Research](./Modern-RL-Research/)
3. Start with [RLHF and Alignment](./Modern-RL-Research/RLHF-and-Alignment/)
4. Explore [Program Synthesis](./Modern-RL-Research/LLM-RL-Program-Synthesis/)

---

## 🎯 What's New in RL (2024-2025)

The field has seen explosive growth in applying RL to language models:
- **RLHF** (Reinforcement Learning from Human Feedback) is now standard for LLM training
- **Code Generation**: Models like AlphaCode achieve near-human performance on competitive programming
- **Reasoning Models**: OpenAI o1, DeepSeek R1, Claude Sonnet use RL for chain-of-thought reasoning
- **New Methods**: DPO and GRPO offer alternatives to traditional PPO-based RLHF
- **Safety Focus**: Secure sandboxing and constitutional AI for safe code generation

---

### Talks to check out first:
----

* [Introduction to Reinforcement Learning](http://videolectures.net/deeplearning2016_pineau_reinforcement_learning/) by Joelle Pineau, McGill University:
	* Applications of RL.
	* When to use RL?
	* RL vs supervised learning
	* What is MDP? Markov Decision Process
	* Components of an RL agent:
		- states
		- actions (Probabilistic effects)
		- Reward function
		- Initial state distribution
			```
			                             +-----------------+
			      +--------------------- |                 |
			      |                      |      Agent      |
			      |                      |                 | +---------------------+
			      |         +----------> |                 |                       |
			      |         |            +-----------------+                       |
			      |         |                                                      |
			state |         | reward                                               | action
			S(t)  |         | r(t)                                                 | a(t)
			      |         |                                                      |
			      |         | +                                                    |
			      |         | |  r(t+1) +----------------------------+             |
			      |         +-----------+                            |             |
			      |           |         |                            | <-----------+
			      |           |         |      Environment           |
			      |           |  S(t+1) |                            |
			      +---------------------+                            |
			                  |         +----------------------------+
			                  +

			* Sutton and Barto (1998)

			```

	* Explanation of the Markov Property:
	* Why Maximizing utility in:
		- Episodic tasks
		- Continuing tasks
			+ The discount factor, gamma γ
	* What is the policy & what to do with it?
		- A policy defines the action-selection strategy at every state:
	* Value functions:
		- The value of a policy equations are (two forms of) Bellman’s equation.
		- (This is a dynamic programming algorithm).
		- Iterative Policy Evaluation:
			+ Main idea: turn Bellman equations into update rules.
	* Optimal policies and optimal value functions.
		* Finding a good policy: Policy Iteration (Check the talk Below By Peter Abeel)
		* Finding a good policy: Value iteration
			- Asynchronous value iteration:
			- Instead of updating all states on every iteration, focus on important states.
	* Key challenges in RL:
		- Designing the problem domain
			- State representation
			– Action choice
			– Cost/reward signal
		- Acquiring data for training
			– Exploration / exploitation
			– High cost actions
			– Time-delayed cost/reward signal
		- Function approximation
		- Validation / confidence measures
	* The RL lingo.
	* In large state spaces: Need approximation:
		- Fitted Q-iteration:
			+ Use supervised learning to estimate the Q-function from a batch of training data:
			+ Input, Output and Loss.
				* i.e: The Arcade Learning Environment
	* Deep Q-network (DQN) and tips.

* [Deep Reinforcement Learning](http://videolectures.net/deeplearning2016_abbeel_deep_reinforcement/) by Pieter Abbeel, EE & CS, UC Berkeley
	- Why Policy Optimization?
	- Cross Entropy Method (CEM) / Finite Differences / Fixing Random Seed
	- Likelihood Ratio (LR) Policy Gradient
	- Natural Gradient / Trust Regions (-> TRPO)
	- Actor-Critic (-> GAE, A3C)
	- Path Derivatives (PD) (-> DPG, DDPG, SVG)
	- Stochastic Computation Graphs (generalizes LR / PD)
	- Guided Policy Search (GPS)
	- Inverse Reinforcement Learning
		+ Inverse RL vs. behavioral cloning

	- Explanation with Implementation for some of the topics mentioned in the Deep Reinforcement Learning talk, written by [Arthur Juliani](https://github.com/awjuliani)
		* The TF / Python implementations [can be found here](https://github.com/awjuliani/DeepRL-Agents).
		* [Part 0 — Q-Learning Agents](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0#.kghmcex46)
		* [Part 1 — Two-Armed Bandit](https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-1-fd544fab149#.bqvzsrvh7)
		* [Part 1.5 — Contextual Bandits](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-1-5-contextual-bandits-bff01d1aad9c#.h2c63t3om)
		* [Part 2 — Policy-Based Agents](https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-2-ded33892c724#.v0hnvh4tw)
		* [Part 3 — Model-Based RL](https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-3-model-based-rl-9a6fe0cce99#.i8pgqg8xa)
		* [Part 4 — Deep Q-Networks and Beyond](https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df#.qecef59on)
		* [Part 5 — Visualizing an Agent’s Thoughts and Actions](https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-5-visualizing-an-agents-thoughts-and-actions-4f27b134bb2a#.60nyejzep)
		* [Part 6 — Partial Observability and Deep Recurrent Q-Networks](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-6-partial-observability-and-deep-recurrent-q-68463e9aeefc#.w22xh551q)
		* [Part 7 — Action-Selection Strategies for Exploration](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-7-action-selection-strategies-for-exploration-d3a97b7cceaf#.vxsnvalt7)
		* [Part 8 — Asynchronous Actor-Critic Agents (A3C)](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2#.9nns6digz)




### Books:
---

- Before starting out the books, here is a neat overview by Yuxi Li about Deep RL:
	- [Deep Reinforcement Learning: An Overview](https://arxiv.org/abs/1701.07274v2)
- [Reinforcement Learning: An Introduction by Richard S. Sutton and Andrew G. Barto](http://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html)
- [Algorithms for Reinforcement Learning.](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf)
- [Reinforcement Learning and Dynamic Programming using Function Approximators.](https://orbi.ulg.ac.be/bitstream/2268/27963/1/book-FA-RL-DP.pdf)



### Courses:
---

* [Reinforcement Learning](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html) by David Silver.
	* Lecture 1: Introduction to Reinforcement Learning
	* Lecture 2: Markov Decision Processes
	* Lecture 3: Planning by Dynamic Programming
	* Lecture 4: Model-Free Prediction
	* Lecture 5: Model-Free Control
	* Lecture 6: Value Function Approximation
	* Lecture 7: Policy Gradient Methods
	* Lecture 8: Integrating Learning and Planning
	* Lecture 9: Exploration and Exploitation
	* Lecture 10: Case Study: RL in Classic Games

* [CS 294: Deep Reinforcement Learning, Spring 2017](http://rll.berkeley.edu/deeprlcoursesp17/#lecture-videos) by John Schulman and Pieter Abbeel.
	* Instructors: Sergey Levine, John Schulman, Chelsea Finn:
	* [My Notes from 2017](./Archive/2017-Course-Notes/CS294-DeepRL-Berkeley/) (archived)










---

## 🔬 Modern RL Resources (2024-2025)

### Recent Courses
* **[Deep RL Course by Hugging Face](https://huggingface.co/learn/deep-rl-course/unit0/introduction)** (2024) - Free, hands-on course with modern tools
* **[CS 285: Deep Reinforcement Learning (Berkeley)](https://rail.eecs.berkeley.edu/deeprlcourse/)** - Updated version of CS294 with recent advances
* **[Spinning Up in Deep RL (OpenAI)](https://spinningup.openai.com/)** - Comprehensive educational resource

### Key Papers for Modern RL + LLMs
* **AlphaCode** (Science 2022) - Competition-level code generation
* **CodeRL** (NeurIPS 2022) - RL for program synthesis
* **Direct Preference Optimization** (2023) - Alternative to PPO for RLHF
* **"RL for Safe LLM Code Generation"** (Berkeley 2025) - Safety in code generation

### Communities and Resources
* [r/reinforcementlearning](https://reddit.com/r/reinforcementlearning) - Active community
* [Hugging Face RL](https://huggingface.co/learn/deep-rl-course) - Practical tutorials
* [Papers with Code - RL](https://paperswithcode.com/area/reinforcement-learning) - Latest benchmarks

---

## 🤝 Contributing

This repository is continually updated with new resources. Feel free to suggest additions or corrections via issues or pull requests.

---

## 📄 License

![cc](https://licensebuttons.net/l/by-nc-nd/3.0/88x31.png)

This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License.

---

*Last Updated: 2025*
