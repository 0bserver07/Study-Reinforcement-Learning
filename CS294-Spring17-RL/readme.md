## Notes taken from CS 294: Deep Reinforcement Learning, Spring 2017 (Berkeley)

### 1 Introduction and course overview (Levine (L), Finn (F), Schulman (S)):

* L: Course information and todos:
	* Overall the course will cover:
		* supervised learning to decision making
		* Basic RL: Q-Learning and Policy gradients
		* Advanced model learning, prediction, distillation, reward learning.
		* DeepRL: Trust Region Policy, Actor-Critic Methods, Exploration
		* Open Problems
	+ HW:
		+ HW1: Imitation Learning (Control via supervised learning)
		+ HW2: Basic (Sh) RL
		+ HW3: Deep Q-Learning:
		+ HW4: Deep Policy grads
		* HW5: Final Project (Research)
	* Using more sensory information form environment can allow the agent to navigate better towards reward and explore.
		* Hence, the need for vision, audio and other sensory inputs.
		* Problem: How do we process all these information coming from the sensory inputs?
			* Answer: Deep in the Deep Learning, using neural networks and function approximation to process complex inputs into the RL algorithm which can choose an action.
		+ Let's be clear here:
			+ Deep Learning for processing the complex information
			+ RL to select the next action using processed inputs by NNs
	* Interesting work: [Reinforcement learning in the brain](https://www.princeton.edu/~yael/Publications/Niv2009.pdf) - Yael Niv
		* *Sutton and Barto (1990) suggested the temporal difference learning rule as a model of prediction learning in Pavlovian conditioning.*
		* *Basal ganglia appears to be related to reward system*
	+ What can deep learning and RL do well now?
		+ high degree of proficiency in domains with known rules.
		+ learn skills with low sensory info, but a lot of repetition.
		+ learning from human imitation.
	- The Hard things:
		- Humans (we) can learn real quick! VS RL is usually slow.
		- Humans (we) can reuse previous information! VS **Transfer learning in deep RL is an open problem**.
		- Not clear what the reward function is.
		- Not clear what the role of prediction should be.
	* Turing said: Why try rebuilding an adult brain? Try simulating a child's one. (Hence the shitty robo-cops falling into water fountains)

* F: Beyond learning rewards
	* child's play vs rl agent in an environment. (In the real world, humans don’t get a score.)
	* reward funcs are essential to the structure the problem as RL.
		* real-world domains: reward/cost often difficult to specify:
			* robotic manipulation
			* autonomous driving
			* dialog systems
			* virtual assistants
			* ...
	- List all the forms of supervision?
		-  imitation: autonomous-cars
			-  inferring intentions:
				-  Inverse RL:  demonstrations
				-  Inverse RL: learned behavior
				-  Inverse RL: demonstrations
		-  model-based, self-supervision: (Behavior via Prediction)
		-  auxiliary tasks:
			-  Sources of Auxiliary Supervision:
			-  touch, audio, depth)
			-  learning multiple, related tasks
			-  task-relevant properties of the world
* S: RL is a branch of ML that concerns about taking sequences of actions.
	* Formalized as partially observable Markov decision
process (POMDP)
	* Observations, Actions, Rewards (this can be applied to real-world stuff, as in Business Operations)
		* O: current levels / score
		* A: number of steps to make / purchase
		* R: profit / reheal or something to continue
	+ Other ML problems falling into this realm:
		+ Classification with Hard Attention
		+ Sequential/structured prediction, e.g., machine translation
	- Deep Reinforcement Learning = RL learning using NNs to ~~~ functions.
		- Policies (next actions)
		- Value Funcs (measure how good state actions-pairs are)
		- Dynamic Models (predict next states and rewards)
	* Supervised learning:
	* Contextual bandits:
	* In RL vs other ML areas:
		* we don't have access to the full information.
		* we also have to be careful, because next step is dependent on the current step taken, or current one depends on previous Xt-1. X for actions, t for Time.
	+ TD-backgammon yester-era
	+ Atari DeepQ, policy gradients, DAgger
		+ policy search
		+ AlphaGo:  supervised learning + policy gradients + value functions + Monte-Carlo tree search