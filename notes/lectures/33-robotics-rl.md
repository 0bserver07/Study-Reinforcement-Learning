<!-- status: unreviewed | last-reviewed: never -->

# Lecture 33: Robotics RL

_Unreviewed — no one has checked this end to end. Treat the math, code, and citations as unverified._

**Time**: ~2-3 h · **Prerequisites**: Lectures 04, 07, 08

---

## Where this fits

Everything so far has assumed the agent lives in a simulator or in a text stream. Atari resets in microseconds. A GSM8K problem hands you a clean prompt and a clean answer. PPO runs across thousands of parallel environments because spinning up an environment is free.

A robot is none of that. The agent has a body. The body has motors that wear, sensors that drift, and a price tag that goes up when you break it. A "reset" is a person walking over and putting the block back on the table. A "step" takes 10–50 ms in the real world, regardless of how fast your GPU is. The action space is continuous and high-dimensional, the observation space is partly raw pixels and partly noisy proprioception, and the reward function is whatever you can compute from sensors that may or may not be telling the truth.

This lecture is about what changes when RL has to deal with that. The first half covers the classical robotics RL story — sample efficiency, sim-to-real, the algorithms that actually get deployed (SAC, TD3, PPO at scale in simulation). The second half covers the 2022–2025 shift toward foundation models for robotics — RT-1, RT-2, Octo, OpenVLA, π₀ — and where RL specifically sits in that pipeline.

The short version: in 2025, the dominant recipe is "imitation learning on lots of demonstrations, then RL to fine-tune." Pure RL on a real robot from scratch is rare. RL in simulation, transferred to the real world, is common. RL on top of a vision-language-action backbone is the active research frontier.

---

## What's different about RL on robots

A short list of properties that make the algorithms from Lectures 02–08 break or strain when you try to apply them on hardware.

### Real-world samples are expensive

In Atari you can collect millions of frames per hour on a single machine. On a real robot arm, a thousand episodes is a long campaign — maybe several days of operator time if anything goes wrong (and it will). The Andrychowicz et al. dexterous hand work (arXiv:1808.00177) reports needing ~100 years of equivalent simulation experience for the in-hand cube manipulation task. They got it by running massively parallel simulation, not by collecting it on hardware.

This shifts the algorithm choice. PPO, which throws data away after every update (Lecture 06), is fine if you have unlimited simulation but ruinous on hardware. SAC and TD3 (Lecture 07) reuse every transition many times via the replay buffer. That replay reuse is why they show up almost everywhere in real-robot work.

### Safety

A policy in CartPole can take an arbitrarily bad action. A policy on a real robot can drive the arm into the table, hit a person, or break a $30k motor. Exploration is constrained — usually by torque limits, joint-angle bounds, and a "stop button" run-time monitor that takes over if the policy proposes anything dangerous.

This rules out a lot of the exploration toys (Lecture 20). Random exploration in joint space is not safe. Adding Gaussian noise to actions, the default for DDPG/TD3, is bounded by what magnitude of noise you can tolerate without breaking things — usually much smaller than what works best in simulation.

### Reset is non-trivial

Classical RL assumes you can call `env.reset()` and start fresh. On a robot, reset means:

- For manipulation: someone walks over, picks up the cube, puts it back at the start position. Or you write a reset policy that does this autonomously (which is itself an RL problem).
- For locomotion: the robot has fallen. Pick it up. Reposition it on the ground. Re-zero the joints.

Reset bottlenecks throughput as much as the episode itself. Two practical responses:

1. Run multiple robots in parallel — the Andrychowicz et al. work and Google's RT-1 data collection both used fleets of robots in parallel cells.
2. Design tasks with implicit resets — "keep walking" doesn't require a reset; "place the cube on the shelf" does.

### Continuous, high-dimensional everything

A typical manipulator has 6–7 joints; a humanoid has 20–30. The observation is usually a mix:

- Proprioception (joint angles, joint velocities, motor currents) — low-dim, accurate, low latency.
- Force/torque sensors at the wrist or fingertips — useful for contact, often noisy.
- One or more RGB or RGB-D cameras — high-dim, slow, hard to interpret.

The action is joint torques, joint velocity commands, or end-effector pose deltas, depending on the abstraction level. Lower-level (torques) gives the policy more authority but requires the policy to learn motor control from scratch. Higher-level (end-effector pose) outsources some control to an inverse-kinematics solver and shortens the horizon.

### Non-stationarity in subtle ways

Motors wear in. The same torque command produces slightly different motion after a week of training. The lighting in the lab changes between morning and afternoon. The gripper picks up a smudge of grease and the friction coefficient shifts. None of these break the policy outright, but they all chip at any policy that has overfit to the exact conditions it was trained in.

This is one reason domain randomization in simulation (covered below) and frequent on-robot fine-tuning are both standard.

### What a real-robot RL campaign actually looks like

To make the above concrete: a typical pure on-robot RL training session for a manipulation task circa 2025 might look like this.

- A single robot cell with a manipulator arm, a table, an overhead camera, and a "stop" button on the wall.
- An operator who sits with the robot for the duration. Resets the task by hand every episode. Hits stop if the policy proposes anything dangerous. Logs notes when the behavior changes.
- Episode length: 10–60 seconds. Reset overhead: 5–30 seconds.
- Throughput: maybe 100–500 episodes per day. A "long campaign" is a week — call it 2000–5000 episodes total. Compare this to a Brax run that hits 5000 episodes in under a minute.
- The replay buffer at the end of the campaign holds a few tens of thousands of transitions. SAC trained on this data is fine. PPO would be impossible — it would have thrown most of it away.

This is why the field has largely abandoned pure on-robot RL in favor of sim-to-real and imitation-then-RL pipelines. The economics of one operator + one robot for a week don't beat "train in sim overnight, deploy in the morning" unless the simulator is genuinely unable to capture the task.

---

## Classical era: model-based, sample-efficient methods

The early-2010s answer to "RL is too sample-inefficient for robots" was model-based RL with a strong Bayesian prior.

### PILCO

PILCO ("Probabilistic Inference for Learning Control"; Deisenroth & Rasmussen, ICML 2011) fit a Gaussian process model to the system dynamics, then did analytic policy improvement under that model. The GP captured uncertainty over dynamics, so the policy didn't overcommit to its model's predictions in regions where the model was unsure. PILCO learned to swing up a cart-pole from scratch in around a dozen episodes of real interaction.

The reason PILCO is mostly historical now is that GPs don't scale to high-dimensional state spaces. A 30-dim humanoid breaks the cubic-in-data-size complexity of GP inference. The conceptual idea — quantify model uncertainty, plan under it — persisted in deep model-based RL (PETS, MBPO — Lecture 08), but the GP machinery was replaced by neural ensembles.

### SAC and TD3 as workhorses

Once neural off-policy methods stabilized (Lecture 07), SAC and TD3 became the default for continuous control. They show up in:

- MuJoCo locomotion benchmarks (HalfCheetah, Walker, Humanoid) — the standard tuning ground for new continuous-control ideas.
- Real-robot manipulation when training from scratch is feasible — usually in simulation, sometimes on hardware with careful safety scaffolding.
- The fine-tuning step on top of imitation-learning policies (residual RL — covered below).

SAC's automatic entropy tuning matters more on robots than in simulation. The right amount of exploration noise depends on torque limits, gear ratios, and how much the robot can physically tolerate; tuning a fixed temperature by hand is annoying. SAC adjusting it during training is one less thing to grid-search.

### Sim-to-real and domain randomization

If you can't train on the robot, train in simulation and transfer. The problem: simulators are wrong. Friction models are approximations. Contact dynamics are even worse approximations. Motor torque-vs-current curves don't match the data sheet. A policy that exploits some artifact of the simulator (a slightly-too-slippery floor, a contact model that lets the gripper pass through objects under load) will fail on hardware.

Tobin et al. 2017 ("Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World," arXiv:1703.06907) is the canonical paper on the visual side. The idea: during training in simulation, randomize the textures, lighting, camera positions, and object colors so aggressively that the real world looks like just another random variant to the policy. The policy learns features that are robust across the randomized distribution, which (with luck) includes the real world.

Dynamics randomization is the same idea applied to physics: randomize friction, mass, motor torque limits, sensor noise. The OpenAI dexterous-hand work (Andrychowicz et al. 2018, arXiv:1808.00177) used both visual and dynamics randomization to train an in-hand cube-rotation policy entirely in simulation, then deployed it on a real Shadow Hand. It worked, with some quirks — the policy displayed some of the same "behaviors" humans use (finger gaiting), and it generalized to novel object shapes the simulation never showed it.

The catch with domain randomization: more randomization means the policy has a harder learning problem (it has to solve all variants, not one). The policy ends up more conservative — more robust to perturbation, but lower asymptotic performance than a policy trained on the exact target environment. You're trading peak performance for robustness, and the tradeoff is task-specific.

---

## Sim and sim-to-real, 2020s

The simulator situation in 2025 is much better than it was in 2017. The dominant tools:

- **MuJoCo** — the classic. Originally Roboti LLC, now open-source under DeepMind. Fast on CPU, accurate enough for most rigid-body problems. The standard benchmark suite (Gymnasium-MuJoCo) runs here.
- **Isaac Gym / Isaac Lab** (NVIDIA) — GPU-parallel physics. Makoviychuk et al. 2021 (arXiv:2108.10470) describes the original Isaac Gym; Isaac Lab is the current iteration. The selling point is throughput: thousands of robot instances simulated in parallel on a single GPU, with the policy network also on the GPU, so the whole loop avoids CPU-GPU transfer.
- **Brax** (Google) — Freeman et al. 2021 (arXiv:2106.13281). Differentiable, JAX-based, GPU-parallel. Cleaner than Isaac for research, less faithful for contact-rich manipulation. Brax demonstrated millions of environment steps per second on a single TPU/GPU for tasks like Ant and Humanoid.
- **Drake** (originally MIT/TRI) — focused on contact-rich manipulation and model-based control. Slower than the GPU-parallel ones, but more accurate for tasks where contact really matters.

Massively parallel simulation changed what's tractable. Training PPO on a humanoid in 2018 took days on a cluster. With Isaac Gym or Brax in 2021, the same training run finishes in minutes on a single workstation, because you're collecting 100x more data per wall-clock second. This makes domain randomization much cheaper — you can sweep the randomization range broadly without worrying about wall-clock cost.

### A note on action space choices in sim

When you set up an RL problem in simulation, you pick an action space. The common options for a manipulator:

- **Raw joint torques** — the policy outputs torques directly. Most expressive, least structured. Hardest to learn from scratch but gives the policy access to dynamic behaviors (whipping motions, controlled compliance) that higher-level abstractions don't.
- **Joint position or velocity commands** — the policy outputs targets that a lower-level controller (PD, impedance) tracks. Smoother learning, but the policy can't do anything the lower controller can't track.
- **End-effector pose deltas** — the policy outputs a small Cartesian offset for the gripper, and inverse kinematics computes the joint angles. Most structured, easiest to learn for pick-and-place style tasks. Loses contact awareness — the IK solver doesn't know about forces.

For sim-to-real, the higher-level abstractions transfer better because the lower-level controller absorbs much of the sim-real gap. The downside is that some behaviors (catching a falling object, controlled compliance against a surface) aren't expressible in the high-level space at all.

### Where the sim-to-real gap still bites

The gap is smallest for locomotion on flat ground in known conditions, and largest for contact-rich manipulation. "Sim looks like real but doesn't move like real" is the standard summary.

Specific failure modes:

- **Stiffness and damping mismatches**: motors in simulation respond as instantly-rigid actuators; real motors have backlash, friction, and finite bandwidth. A policy that depends on bang-bang torque switching will fail because the real motors can't keep up.
- **Contact**: simulators use approximate contact models (penalty-based, impulse-based, complementarity-based). Each has artifacts. A policy that exploits a particular contact artifact — say, getting "stuck" to a surface in a way the simulator allows but physics doesn't — won't transfer.
- **Cable, deformable, or fluid dynamics**: rarely modeled at all. If your task involves a rope, a cloth, or liquid, sim-to-real for the manipulation strategy is much harder.
- **Latency**: simulators step instantaneously; real robots have observation-to-action latency of 10–50 ms. Policies that don't account for this can oscillate.

The standard mitigations:

- **System identification** — measure the real-robot parameters (motor bandwidth, friction) and set the simulator to match. Useful but tedious.
- **Real-world fine-tuning** — train in sim, then collect a small amount of real-robot data and fine-tune. Common pattern.
- **Domain randomization with conservative ranges** — wide enough to span reality, narrow enough that the task remains learnable.
- **Online adaptation** — RMA (Kumar et al. 2021, arXiv:2107.04034) and related work train a context encoder that infers the environment parameters from a short history of observations, then conditions the policy on the inferred context. The policy adapts on the fly without explicit fine-tuning.

---

## The data-driven shift: foundation models for robotics

The most visible change in robotics RL between 2022 and 2025 is the move toward large-scale pretrained policies. The premise: instead of training a policy from scratch on a single task on a single robot, train one big model on demonstrations from many tasks across many robots, then specialize.

This is more imitation learning than RL — most of these systems learn from behavior cloning on demonstrations rather than from a reward signal. But RL keeps showing up at the edges (fine-tuning, residual policies, reward modeling), and the foundation models are increasingly the right starting point for any RL on a new task.

### RT-1

Brohan et al. 2022 ("RT-1: Robotics Transformer for Real-World Control at Scale," arXiv:2212.06817) trained a transformer policy on ~130k demonstrations collected over 17 months across 13 robots in Google's Everyday Robots fleet. The architecture is straightforward: image patches and text instructions go in, discretized actions come out. The training is behavior cloning.

The interesting result was scaling. RT-1 generalized to new tasks, new objects, and new instructions in a way that single-task policies didn't. The paper's framing — that scale of demonstration data, not algorithmic cleverness, was the bottleneck — set the tone for the next two years of work.

### RT-2

Brohan et al. 2023 ("RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control," arXiv:2307.15818) took the next step: co-train a vision-language model (PaLI-X / PaLM-E backbone) with robot action tokens. Actions become just another vocabulary that the VLM emits, alongside text. The VLM is pretrained on web-scale image-text data, fine-tuned on robot demonstrations.

The payoff is semantic generalization. RT-2 can follow instructions like "pick up the extinct animal" (and select the dinosaur toy) because the VLM understands what "extinct animal" means; the policy maps that understanding to actions. RT-1 couldn't do this — it could only handle instructions that looked like the ones in its training set.

### Open X-Embodiment

The Open X-Embodiment collaboration (RT-X collaborators 2023, arXiv:2310.08864) pooled robot data across 21 institutions and 22 different robot platforms. The dataset covered ~527 skills across ~160k tasks. The companion RT-X models showed that training on this multi-robot mixture and then fine-tuning to a specific target robot outperformed training from scratch on the target robot alone.

This is the dataset that the next generation of foundation models trained on.

### Octo

The Octo Model Team 2024 ("Octo: An Open-Source Generalist Robot Policy," arXiv:2405.12213) released an open-source transformer policy trained on 800k trajectories from Open X-Embodiment. The architecture supports both language-conditioned and goal-image-conditioned tasks. Fine-tuning to a new robot or task takes hours, not days, because the pretrained model already knows how to manipulate.

Octo is the first widely-used open-source robotics foundation model. Most academic work on foundation-model robotics in 2024–2025 starts from Octo (or OpenVLA) rather than training a new policy from scratch.

### OpenVLA

Kim et al. 2024 ("OpenVLA: An Open-Source Vision-Language-Action Model," arXiv:2406.09246) is a 7B-parameter VLA model trained on 970k Open X-Embodiment demonstrations. It's positioned as the open-source counterpart to RT-2 (which is closed). The paper reports that OpenVLA outperforms RT-2-X (a 55B closed model) by ~16.5% on a 29-task evaluation despite being much smaller, attributed to better data curation and the use of a Llama-2 backbone.

The practical importance: OpenVLA is the first VLA you can download and fine-tune yourself. Combined with parameter-efficient fine-tuning (LoRA), this brought VLA fine-tuning into a single-GPU budget.

### π₀

Black, Brown, Driess et al. 2024 ("π₀: A Vision-Language-Action Flow Model for General Robot Control," arXiv:2410.24164) from Physical Intelligence is the current state of the art at the time of writing. The differences from prior VLAs:

- A flow-matching action head rather than discretized token output. This produces continuous actions directly, which matters for fine-grained manipulation.
- Trained on a larger and more diverse robot data mixture, including dexterous bimanual tasks.
- Demonstrations include genuinely complex skills — laundry folding, table cleaning, box assembly — that previous VLAs struggled with.

π₀ is closed; the architecture is described but the weights aren't released.

### A note on action tokenization

Several of these models (RT-2, OpenVLA) discretize continuous actions into tokens that the language-model decoder can emit alongside text. The standard approach: pick 256 bins per action dimension, map each bin to a token in the model's vocabulary. The policy outputs a sequence of action tokens which get detokenized back to a continuous action vector.

This is convenient — actions become "just another language" — but it has costs. The discretization adds quantization noise. A 7-DOF arm with 256 bins per dim gives 256^7 ≈ 4.7×10^16 possible actions, so the joint action space isn't enumerable, but per-dimension precision is limited. For coarse pick-and-place this is fine. For fine assembly it's not.

π₀'s flow-matching head was partly a response to this: keep the LM backbone but generate continuous actions directly through a separate diffusion-style head. The reported result is smoother action profiles and better performance on dexterous tasks. Expect this design to spread.

### Where this is going

The 2024–2025 picture, in one line: robot policies are converging on the "pretrain a VLM, fine-tune for actions" pattern that LLMs converged on around 2020–2022. The pretraining data is still the bottleneck — there's no internet-scale corpus of robot demonstrations, so it has to be collected.

The next big unknown is whether self-supervised pretraining on video (without action labels) can substitute for some of the demonstration data. Several lines of work in 2024 (V-JEPA, world-model pretraining) are betting yes. If it works, robotics data becomes much cheaper and the foundation models get much better. If it doesn't, demonstration collection at scale remains the rate-limiting step.

---

## Where RL specifically shows up

The above is mostly imitation learning. So where does RL fit in?

### Residual policies

A common pattern: behavior cloning (or a VLA fine-tune) produces a base policy that gets you most of the way to a working system but isn't quite good enough. Train a small residual policy with RL on top that outputs a correction to the base policy's action.

The idea (Johannink et al. 2018, arXiv:1812.03201, and follow-ons): the base policy provides a strong prior, so the RL agent only has to learn a small local correction. The RL problem is much easier than training the full policy from scratch — narrower action range, shorter effective horizon, more sample-efficient.

This is one of the few places where on-robot RL fine-tuning is genuinely practical. The residual is small enough that exploration is bounded (you can clip it to a small range), the base policy keeps the robot in safe regions of state space, and the RL signal only has to refine, not learn from scratch.

### Reward modeling for manipulation

Reward design on hardware is hard. A force/torque sensor reading "the cube is grasped" is gameable — the policy can wedge the cube against the table to trigger the sensor without actually grasping. A vision-based reward ("the cube is in the goal zone") needs a separate vision model that itself can fail.

Learned reward models from preferences (Lecture 09 territory, but applied to manipulation) are one answer. Collect pairs of robot trajectories, have a human pick which is better at the task, train a reward model on the comparisons, then optimize with RL. This is mostly a research direction rather than a deployed technique in 2025, but it's active — see, e.g., the line of work on preference-based reward learning for robotics that traces back to Christiano et al. 2017 (arXiv:1706.03741).

### Sim-to-real with RL in sim

The most common deployment of RL in robotics is: train PPO or SAC entirely in simulation with heavy domain randomization, then deploy zero-shot or with light fine-tuning. The locomotion line of work — ANYmal (Hwangbo et al. 2019), MIT mini-cheetah, Unitree's quadrupeds — almost all uses this recipe. The RL algorithm is unremarkable (PPO with a standard MLP policy); the wins come from the simulation, the reward shaping, and the randomization.

For manipulation, this is harder because contact dynamics don't transfer as cleanly. Locomotion gets away with it because the relevant contact (foot on ground) is intermittent and well-modeled; in-hand manipulation has continuous, multi-point contact that simulators still struggle with.

This locomotion-vs-manipulation split is worth flagging because it runs through everything. The papers and demos that look most impressive — quadrupeds traversing rough terrain, humanoid backflips — are almost all locomotion. The papers that struggle to scale beyond a single lab — bimanual cloth folding, peg-in-hole assembly, in-hand reorientation — are almost all manipulation. The algorithm is rarely the difference; it's the simulator fidelity, the reward designability, and the safety of exploration.

### RL fine-tuning on VLA policies

This is the frontier. Take a pretrained VLA (Octo, OpenVLA), fine-tune it with RL on a specific task to push past the imitation-learning ceiling. The challenges:

- The action space is high-dim continuous (or discretized to high-cardinality tokens). PPO/SAC at this scale on a model with billions of parameters is expensive.
- The reward is whatever you can compute, often sparse (task succeeded / task failed). RLVR-style verifiable rewards (Lecture 15) work for some manipulation tasks (did the robot place the object in the right zone?) but not for tasks where success is subjective.
- The KL penalty back to the SFT policy matters a lot, same as in LLM RLHF — you want to refine the VLA without destroying its general capabilities.

Practical patterns that have shown up in 2024–2025 work:

- **LoRA-only fine-tuning.** Freeze the VLA backbone entirely; train only LoRA adapters on the attention layers. The number of trainable parameters drops by 100–1000x, which makes RL on a single GPU feasible and reduces the catastrophic-forgetting risk. The cost: you can't shift the policy as far from the base.
- **Separate critic networks.** PPO/SAC on a billion-parameter actor with a matched billion-parameter critic doubles memory. Common workaround: a much smaller critic (an MLP head on the VLA's hidden states), or removing the critic entirely with a GRPO-style group baseline (Lecture 15). The latter is appealing because it mirrors what works for LLM reasoning, but the group-rollout cost on a robot is much higher than for LLM generation.
- **Action-token-only RL.** If the VLA uses discretized action tokens, you can run RL on just the action-token logits and leave the rest of the language-modeling head alone. This treats the RL update as analogous to RLHF on a specific tail of the vocabulary.

This is an active research area, not a settled recipe. Expect the picture to change.

---

## A short code sketch: SAC on a Brax environment

The following shows the structure of a SAC training loop on a Brax env. Brax is GPU-parallel, so you collect from thousands of envs simultaneously. The actor and critic definitions are the same as the SAC code in Lecture 07; what changes is the env interaction and the throughput.

```python
import jax
import jax.numpy as jnp
import torch
import brax
from brax import envs
from brax.io import torch as brax_torch

# Note: This is a sketch. A production Brax+SAC setup typically either keeps
# everything in JAX (using Brax's reference algorithm implementations) or
# uses brax.envs.create with a torch wrapper. Mixing JAX envs and PyTorch
# policies has overhead at the boundary; for serious training keep both in
# the same framework. The sketch below is for clarity, not throughput.

# Assume SAC, StochasticActor, Critic, and ReplayBuffer come from Lecture 07.
from sac_lecture_07 import SAC, ReplayBuffer

NUM_ENVS = 2048              # Parallel envs on the GPU
TOTAL_STEPS = 10_000_000     # Total env-steps across all parallel envs
WARMUP_STEPS = 10_000        # Random-policy steps to fill buffer
UPDATES_PER_STEP = 1         # Gradient steps per env step
BATCH_SIZE = 256


def main():
    # Create a vectorized Brax env. "ant" is a 27-dim observation, 8-dim
    # continuous action standard benchmark.
    env = envs.create(
        env_name="ant",
        batch_size=NUM_ENVS,
        episode_length=1000,
        action_repeat=1,
    )

    state_dim = env.observation_size
    action_dim = env.action_size
    max_action = 1.0  # Brax actions are usually in [-1, 1]

    agent = SAC(state_dim, action_dim, max_action)
    buffer = ReplayBuffer(capacity=1_000_000)

    # JIT-compile the env step. This is where most of Brax's speedup comes from.
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    key = jax.random.PRNGKey(0)
    state = jit_reset(rng=key)

    steps_done = 0
    while steps_done < TOTAL_STEPS:
        # Sample actions. During warmup use random actions to seed the buffer.
        obs_torch = brax_torch.jax_to_torch(state.obs)
        if steps_done < WARMUP_STEPS:
            actions_torch = torch.empty(NUM_ENVS, action_dim).uniform_(-1, 1)
        else:
            with torch.no_grad():
                actions_torch, _ = agent.actor.sample(obs_torch)

        # Step the env. brax_torch handles the jax<->torch conversion.
        actions_jax = brax_torch.torch_to_jax(actions_torch)
        next_state = jit_step(state, actions_jax)

        # Store transitions in the replay buffer.
        # Each parallel env contributes one transition per step.
        rewards = brax_torch.jax_to_torch(next_state.reward)
        dones = brax_torch.jax_to_torch(next_state.done)
        next_obs_torch = brax_torch.jax_to_torch(next_state.obs)
        for i in range(NUM_ENVS):
            buffer.add(
                obs_torch[i].cpu().numpy(),
                actions_torch[i].cpu().numpy(),
                rewards[i].item(),
                next_obs_torch[i].cpu().numpy(),
                bool(dones[i].item()),
            )

        steps_done += NUM_ENVS
        state = next_state

        # Train the agent. With 2048 parallel envs and one gradient step per
        # env step, you do 2048 updates per "outer loop" — too many. Common
        # practice is to subsample, e.g. one update per ~64 env steps.
        if steps_done > WARMUP_STEPS and (steps_done // NUM_ENVS) % 4 == 0:
            for _ in range(UPDATES_PER_STEP):
                agent.train(buffer, batch_size=BATCH_SIZE)

        if (steps_done // NUM_ENVS) % 100 == 0:
            mean_reward = rewards.mean().item()
            print(f"step={steps_done:>10} | mean_reward={mean_reward:+.3f}")


if __name__ == "__main__":
    main()
```

Things to notice:

- Most of the throughput comes from `jax.jit` on the env step. The Python loop overhead is amortized across `NUM_ENVS=2048` parallel envs.
- The replay buffer write loop in Python is slow. In production you'd batch this — keep the buffer on the GPU and append batches directly. The naive loop above is for clarity.
- The update-per-step ratio is the main hyperparameter to tune. Too many updates per step and the policy overfits to recent data; too few and you waste samples.

For real work, look at the Brax repo's reference SAC and PPO implementations — they stay in JAX end to end and avoid the cross-framework cost.

---

## A short code sketch: residual policy

The structure of a residual policy is small. The base policy is frozen; the residual is a small network trained with RL.

```python
import torch
import torch.nn as nn

# Assume SAC, Actor, Critic, ReplayBuffer from Lecture 07.
from sac_lecture_07 import SAC


class ResidualActor(nn.Module):
    """
    Wraps a frozen base policy with a learnable residual.

    Final action = clip(base(s) + alpha * residual(s), -1, 1)

    The residual is small (a few hundred params) and outputs a correction
    bounded by `alpha`. This bounds exploration around the base policy and
    keeps the robot in safe states.
    """

    def __init__(self, base_policy, state_dim, action_dim, alpha=0.1):
        super().__init__()
        # Freeze the base. Its weights will not be updated.
        self.base = base_policy
        for p in self.base.parameters():
            p.requires_grad = False

        # Small residual network. Output is bounded by tanh and scaled by alpha.
        self.residual_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )
        self.alpha = alpha

    def forward(self, state):
        with torch.no_grad():
            base_action = self.base(state)
        residual = self.alpha * self.residual_net(state)
        return torch.clamp(base_action + residual, -1.0, 1.0)

    def sample(self, state):
        """SAC-style stochastic action with reparameterization."""
        with torch.no_grad():
            base_action = self.base(state)
        # The residual head outputs a Gaussian, same as a normal SAC actor,
        # but its mean and std are scaled by alpha to bound the correction.
        # Implementation left as an exercise — mirror the StochasticActor
        # from Lecture 07, then multiply mean and std by self.alpha and add
        # base_action before clipping.
        raise NotImplementedError(
            "Mirror StochasticActor from Lecture 07; scale by alpha; add base_action."
        )


def train_residual_on_real_robot(env, base_policy_path, total_steps=20_000):
    """
    Sketch of residual RL fine-tuning on a real robot.

    Key safety properties:
    - alpha is small (0.05-0.2). The residual can only nudge the base action.
    - We start training only after a warmup of pure-base rollouts to fill the
      buffer with safe data.
    - A run-time monitor (not shown) clips actions outside safe joint limits.
    """
    base_policy = torch.load(base_policy_path)  # Frozen BC or VLA backbone.
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Construct a SAC agent whose actor is a ResidualActor instead of the
    # default StochasticActor. The critic is unchanged.
    agent = SAC(state_dim, action_dim, max_action=1.0)
    agent.actor = ResidualActor(base_policy, state_dim, action_dim, alpha=0.1)
    # Re-create the actor optimizer to point at only the residual params,
    # because the base is frozen.
    agent.actor_optimizer = torch.optim.Adam(
        agent.actor.residual_net.parameters(), lr=3e-4
    )

    obs, _ = env.reset()
    for step in range(total_steps):
        action = agent.select_action(obs, evaluate=False)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.replay_buffer.add(obs, action, reward, next_obs, done)
        obs = next_obs if not done else env.reset()[0]

        # Only update after the buffer has some real data.
        if step > 1000:
            agent.train(agent.replay_buffer, batch_size=256)

        if step % 200 == 0:
            print(f"step={step} | reward={reward:+.3f}")
```

Things to notice:

- The base policy is frozen. Only the residual trains. This is what makes the gradient signal manageable on a small dataset.
- `alpha` bounds the correction magnitude. Setting it small (0.05–0.2 of the action range) keeps the robot in a neighborhood of the base policy's behavior, which keeps exploration safe.
- The critic learns the full Q-function for `base + residual`. It has to be reset or re-trained from scratch when you swap base policies, because the action distribution it's evaluating has changed.

In practice you might also clip the residual in observation-space units (e.g., max 5 cm of end-effector deviation per step) rather than in normalized action units, depending on how the action is parameterized.

---

## Where it breaks

A short field guide to robotics RL failure modes, drawn from the patterns above.

**Sim-to-real for contact-rich tasks.** Locomotion transfers reasonably well. Manipulation that involves sustained contact (in-hand manipulation, peg insertion, deformable objects) doesn't. The simulator's contact model is wrong in ways the policy will exploit. Mitigations: simpler reward functions that don't reward exploiting contact artifacts; on-robot fine-tuning; better contact simulators (Drake helps here, at a throughput cost).

**Reward sensors that aren't ground truth.** A force/torque sensor reading "grasped" can be wedged. A vision-based "object in zone" classifier can be fooled by occlusion. The policy will find whatever path through state space maximizes the sensor reading, even if it doesn't correspond to task success. Mitigations: use multiple independent sensors and AND them; use human-in-the-loop reward at low rate; design rewards that are harder to spoof (e.g., reward sustained-success over multiple seconds rather than single-frame triggers).

**Reset-induced distribution shift.** If a human places the cube in slightly different positions each reset, the policy learns to handle that distribution. If an autonomous reset places the cube identically every time, the policy overfits to that one configuration and fails when conditions differ at deployment. Mitigations: randomize resets explicitly; collect data with multiple operators; include "perturbation" phases where the cube is moved during the episode.

**Hardware drift over a training run.** Motors heat up, friction changes, calibration drifts. A 12-hour training run can see meaningfully different dynamics at the start and end. The policy can either over-adapt to whichever conditions dominated the buffer, or learn a robust policy that's mediocre on both. Mitigations: shorter training runs with restarts; explicit context inference (RMA-style); periodic recalibration.

**Foundation-model fine-tuning that destroys general capability.** Fine-tuning a VLA with RL on a narrow task can wreck its broader behavior, same as catastrophic forgetting in LLMs. Mitigations: strong KL penalty to the SFT base; mix in continued imitation-learning loss during RL; freeze most of the backbone and only train an action head.

**Reward shaping that creates the wrong policy.** Dense rewards that approximate the true goal often have side effects. "Reward proportional to distance to goal" makes the robot rush, ignoring obstacles. "Reward for being near the cube" makes the robot hover instead of grasping. Mitigations: prefer sparse outcome rewards when feasible; verify shaped rewards by looking at the resulting behavior, not just the reward curve; use process-reward-style intermediate verifications (Lecture 23 territory) when the task decomposes naturally.

---

## When to reach for what

A rough decision tree for robotics RL in 2025.

```
Do you have a real robot or just simulation?
├── Just simulation
│     Use PPO or SAC. PPO if you have massive parallelism (Brax, Isaac Gym).
│     SAC if env interaction is the bottleneck. Domain randomization if you
│     plan to transfer to hardware later. (Lecture 06, 07.)
│
└── Real robot
    Do you have a working base policy (BC, VLA, classical controller)?
    ├── Yes → Residual RL fine-tune. Small alpha, safety monitor, SAC on the
    │         residual. (This section.)
    │
    └── No  → Is there a good simulator for your task?
        ├── Yes → Train in sim with domain randomization, deploy with light
        │         on-robot fine-tune. (PILCO-era thinking still applies: be
        │         data-efficient on the real-robot data, treat it as precious.)
        │
        └── No  → Collect demonstrations first. Behavior cloning gets you a
                  base policy. THEN consider RL fine-tuning. Pure on-robot RL
                  from scratch is rarely the right tool in 2025.
```

For foundation-model-era work: start from Octo or OpenVLA, fine-tune on your task with imitation learning, then add RL only if you need to push past the imitation ceiling. Pure-RL from a VLA backbone is research-frontier and not a settled recipe.

---

## Recap

Robotics RL inherits all the algorithms from Lectures 02–08, but the constraints — sample cost, safety, reset cost, continuous high-dim action and observation spaces — change which algorithms work in practice. SAC and TD3 dominate continuous control because they reuse data via replay buffers. PPO dominates when massive parallel simulation is available. Pure on-robot RL from scratch is rare; the more common pattern is RL in simulation with domain randomization, transferred to hardware.

The 2022–2025 shift is the emergence of vision-language-action foundation models — RT-1, RT-2, Octo, OpenVLA, π₀ — trained on aggregate datasets like Open X-Embodiment. These are mostly trained with imitation learning, not RL. RL specifically shows up as a fine-tuning step on top (residual policies, RL fine-tuning of VLAs), and in the sim-to-real pipeline that produces the lower-level controllers underneath the foundation models.

The frontier in 2025 is RL fine-tuning on VLA backbones — how to do it without destroying the backbone's general capability, what reward signals to use, how to make exploration safe on hardware. None of this is settled.

---

## References

All arXiv IDs and venues verified against arxiv.org and primary sources.

**Classical robotics RL**

- Deisenroth & Rasmussen. 2011. "PILCO: A Model-Based and Data-Efficient Approach to Policy Search." ICML 2011. — The Gaussian-process model-based RL paper that established the "few episodes of real interaction" benchmark.

**Off-policy continuous control (prerequisites — see Lecture 07)**

- Fujimoto, van Hoof, Meger. 2018. "Addressing Function Approximation Error in Actor-Critic Methods" (TD3). ICML 2018. arXiv:1802.09477.
- Haarnoja, Zhou, Abbeel, Levine. 2018. "Soft Actor-Critic" (SAC). ICML 2018. arXiv:1801.01290.

**Sim-to-real**

- Tobin, Fong, Ray, Schneider, Zaremba, Abbeel. 2017. "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World." arXiv:1703.06907. — Canonical domain randomization paper.
- OpenAI / Andrychowicz et al. 2018. "Learning Dexterous In-Hand Manipulation." arXiv:1808.00177. — Shadow Hand cube manipulation, dynamics + visual randomization.
- Kumar, Fu, Pathak, Malik. 2021. "RMA: Rapid Motor Adaptation for Legged Robots." arXiv:2107.04034. — Online context inference for sim-to-real adaptation.

**Residual policies**

- Johannink, Bahl, Nair, Luo, Kumar, Loskyll, Ojea, Solowjow, Levine. 2018. "Residual Reinforcement Learning for Robot Control." arXiv:1812.03201. — Base controller + small RL residual on top.

**Simulators**

- Freeman, Frey, Raichuk, Girgin, Mordatch, Bachem. 2021. "Brax — A Differentiable Physics Engine for Large Scale Rigid Body Simulation." arXiv:2106.13281. — GPU-parallel, JAX-based.
- Makoviychuk, Wawrzyniak, Guo, Lu, Storey, Macklin, Hoeller, Rudin, Allshire, Handa, State. 2021. "Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning." arXiv:2108.10470.

**Robotics foundation models**

- Brohan et al. 2022. "RT-1: Robotics Transformer for Real-World Control at Scale." arXiv:2212.06817. — Transformer policy on 130k demos.
- Brohan et al. 2023. "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control." arXiv:2307.15818. — Co-trained VLM + robot actions.
- Open X-Embodiment Collaboration. 2023. "Open X-Embodiment: Robotic Learning Datasets and RT-X Models." arXiv:2310.08864. — Pooled multi-robot dataset.
- Octo Model Team. 2024. "Octo: An Open-Source Generalist Robot Policy." arXiv:2405.12213. — Open transformer policy trained on Open X-Embodiment.
- Kim et al. 2024. "OpenVLA: An Open-Source Vision-Language-Action Model." arXiv:2406.09246. — 7B VLA, Llama-2 backbone, outperforms RT-2-X.
- Black, Brown, Driess et al. 2024. "π₀: A Vision-Language-Action Flow Model for General Robot Control." arXiv:2410.24164. — Physical Intelligence's flow-matching VLA.

**Preference-based reward learning (for the reward-modeling subsection)**

- Christiano, Leike, Brown, Martic, Legg, Amodei. 2017. "Deep Reinforcement Learning from Human Preferences." arXiv:1706.03741. — The line of work that became RLHF; originally demonstrated on robotics tasks.

**Locomotion (for the locomotion/manipulation split)**

- Hwangbo, Lee, Dosovitskiy, Bellicoso, Tsounis, Koltun, Hutter. 2019. "Learning agile and dynamic motor skills for legged robots." Science Robotics 4(26). arXiv:1901.08652. — ANYmal quadruped trained in simulation, deployed on hardware. The canonical sim-to-real locomotion result.

---

## Next lecture

You're past the main sequence. Related material in the curriculum: Lecture 22 (world models) covers Dreamer-style latent-dynamics learning that's relevant to model-based robotics; Lecture 24 (computer-use agents) covers the digital-action counterpart to physical robot policies; Lecture 28 (reward hacking) goes deeper on the reward-design pitfalls that show up especially badly on hardware.

---

## Exercises

No formal exercise set for this lecture (yet). Things worth implementing if you want hands-on practice:

- **SAC on a Brax env**: take the sketch above, finish it, train on `ant` until you hit ~5000 episode reward. Measure throughput in env-steps per second. The benchmark is somewhere in the range of 100k-1M env-steps/sec on a single modern GPU; if you're much below that, the JAX-PyTorch boundary is the bottleneck.
- **Domain randomization sweep**: train PPO on a MuJoCo locomotion task (Walker or Humanoid). Add randomization to friction, mass, and motor torque limits. Measure asymptotic performance with vs without randomization. The expected result: randomized policy has lower peak reward in the nominal environment but degrades less when you change the parameters at test time.
- **Residual policy on a sim manipulation task**: take a behavior-cloned policy on a simple pick-and-place task in MuJoCo (or use a heuristic controller as the "base"). Fine-tune a residual with SAC. Measure how much the residual improves over the base, and how sensitive performance is to `alpha`.
- **Read and reproduce a result from one of the foundation-model papers**: pick OpenVLA (the most accessible — code and weights are public). Load the pretrained model, run inference on a robot demonstration dataset (e.g., RoboMimic), and measure action prediction accuracy on held-out demos. Then fine-tune on a new task and see how much fine-tuning data you need to recover full performance.

The robotics RL literature changes fast. Anything in this lecture about specific foundation models, datasets, or benchmarks may be obsolete within a year. Treat the algorithmic structure (SAC/TD3, residual policies, sim-to-real recipes) as the long-lived part and the specific systems as a snapshot of mid-2025.

One last note: there's a tendency in robotics writeups to make policies sound smarter than they are. "The robot learned to fold laundry" can mean anything from "achieved 80% success on a fixed setup with one type of shirt" to "generalizes to arbitrary garments." When reading papers, look for the evaluation protocol (number of trials, range of conditions, definition of success) before deciding what a result means. The same skepticism applies to anything in this lecture.
