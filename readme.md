## Study Reinforcement Learning & (Deep RL) Guide:

* Simple guide and collective to study RL/DeepRL in one to 2.5 months of time.

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
	* [My Bad Notes](/CS294-Spring17-RL)










-----

![cc](https://licensebuttons.net/l/by-nc-nd/3.0/88x31.png)
