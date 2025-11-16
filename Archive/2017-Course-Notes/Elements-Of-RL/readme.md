#### Elements Of Reinforcement Learning: (Derived from Barto and Sutton '17 and Li '17)

* A policy defines the learning agent’s way of behaving at a given time. Roughly speaking, a policy is a mapping from perceived states of the environment to actions to be taken when in those states.
	* Actor-Critic
	* Policy Gradient
	* Comb Policy Grad + Off-Policy RL

* A reward signal defines the goal in a reinforcement learning problem. On each time step, the environment sends to the reinforcement learning agent a single number called the reward. The agent’s sole objective is to maximize the total reward it receives over the long run.
	* rewards can be sparse signals
	* reward functions exist often times to handle the scarcity.

* A value function specifies what is good in the long run. Roughly speaking, the value of a state is the total amount of reward an agent can expect to accumulate over the future, starting from that state.
	* Deep DQN
	* Double DQN
	* Dueling Archs

* The fourth and final element of some reinforcement learning systems is a model of the environment. This is something that mimics the behavior of the environment, or more generally, that allows inferences to be made about how the environment will behave.

---

Upgraded:

* Planning
* Exploration
