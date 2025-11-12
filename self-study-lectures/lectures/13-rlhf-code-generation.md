# Lecture 13: RLHF for Code Generation

**Duration:** ~90 minutes
**Prerequisites:** Lecture 10 (PPO for LLMs), Lecture 06 (PPO), basic Python/PyTorch
**Goal:** Understand how to apply RLHF to code generation, implement CodeRL, and handle execution-based rewards

---

## Why This Matters

Remember in Lecture 10 when we trained LLMs with human preferences? Well, **code has something even better than human preferences: unit tests**. Code either works or it doesn't. This lecture is about using that crisp signal.

When DeepMind created AlphaCode, they weren't just doing supervised learning on code. When Meta built CodeLlama, they used RL. When OpenAI made ChatGPT good at coding, they used RLHF. **This is how state-of-the-art code generation actually works**.

### What You'll Learn

1. Why code generation is the perfect RL domain
2. How to use execution feedback as rewards
3. AlphaCode's approach (filtering + RL fine-tuning)
4. CodeRL framework implementation
5. Pass@k evaluation (the industry standard)
6. Real implementation with all the gotchas

---

## Part 1: Why Code is Perfect for RL

### The Supervised Learning Problem

Standard approach: train on GitHub code
```python
# Training data
problem = "Write a function to sort a list"
code = "def sort(lst): return sorted(lst)"
```

**Problem:** This only teaches the model what code *looks like*, not what code *does*.

Real-world issue I hit:
```python
# Model generates syntactically correct but wrong code
def fibonacci(n):
    # Looks plausible, wrong algorithm
    return [i*i for i in range(n)]  # This is squares, not Fibonacci!
```

This passes all syntax checks, looks like Python, but is completely wrong.

### The RL Advantage

With RL, we can actually **run the code**:

```python
def reward_function(generated_code, test_cases):
    """
    Execute code and check if it passes tests.
    Binary reward: works or doesn't work.
    """
    try:
        # Execute generated code
        exec(generated_code, globals())

        # Run test cases
        passed = 0
        for input_data, expected_output in test_cases:
            actual_output = eval(f"solution({input_data})")
            if actual_output == expected_output:
                passed += 1

        # Reward is proportion of tests passed
        return passed / len(test_cases)
    except Exception as e:
        # Code doesn't even run
        return 0.0
```

**This is way better than human feedback** because:
1. **Objective:** No ambiguity, code either passes tests or doesn't
2. **Scalable:** Can generate unlimited test cases
3. **Fast:** Execution is instant (no waiting for humans)
4. **Precise:** Know exactly which test cases fail

---

## Part 2: AlphaCode's Approach

AlphaCode (DeepMind, 2022) was the first to reach competitive programming level. Here's their RL pipeline:

### Stage 1: Supervised Pre-training

```python
# Standard next-token prediction on code
for problem, solution in github_dataset:
    loss = cross_entropy(model(problem), solution)
    loss.backward()
```

Nothing special. Train on millions of GitHub repos.

### Stage 2: Fine-tune on Competitive Programming

```python
# CodeContests dataset: problems with test cases
for problem, solution, tests in codecontests:
    loss = cross_entropy(model(problem), solution)
    loss.backward()
```

Still supervised, but now on high-quality competitive programming problems.

### Stage 3: Massive Sampling + Filtering

Here's where it gets interesting:

```python
def alphacode_generate(problem, model, k=1000000):
    """
    Generate 1 million candidate solutions.
    Filter to top 10 by clustering + test execution.
    """
    # Sample 1M solutions (yes, one million)
    candidates = []
    for i in range(k):
        code = model.generate(problem, temperature=0.8)
        candidates.append(code)

    # Execute on public test cases
    valid_candidates = []
    for code in candidates:
        if passes_public_tests(code, problem.public_tests):
            valid_candidates.append(code)

    # Cluster by behavior (which tests they pass)
    # Keep top 10 from different clusters
    diverse_solutions = cluster_and_select(valid_candidates, n=10)

    return diverse_solutions
```

**Key insight:** Generate massive diversity, filter by execution.

### Stage 4: RL Fine-tuning (The Secret Sauce)

Now apply RLHF with execution-based rewards:

```python
class AlphaCodeRLTrainer:
    def __init__(self, model, ref_model):
        self.policy = model
        self.ref_model = ref_model  # Frozen copy from Stage 2
        self.beta = 0.01  # KL penalty

    def compute_reward(self, problem, generated_code):
        """
        Reward = test_pass_rate - β * KL(π||π_ref)
        """
        # Execute on hidden test cases
        test_reward = self.execute_tests(problem, generated_code)

        # Compute KL penalty (keep close to supervised model)
        kl_penalty = self.compute_kl_divergence(
            self.policy, self.ref_model, problem, generated_code
        )

        return test_reward - self.beta * kl_penalty

    def execute_tests(self, problem, code):
        """Run code against test suite."""
        try:
            # Set up isolated execution environment
            namespace = {}
            exec(code, namespace)

            # Run all test cases
            passed = 0
            for test_input, expected_output in problem.tests:
                try:
                    actual = namespace['solution'](test_input)
                    if actual == expected_output:
                        passed += 1
                except:
                    pass

            return passed / len(problem.tests)
        except:
            return 0.0

    def train_step(self, batch_problems):
        """One PPO training step."""
        rewards = []
        log_probs = []

        for problem in batch_problems:
            # Generate solution
            code, log_prob = self.policy.generate_with_logprobs(problem)

            # Get reward from execution
            reward = self.compute_reward(problem, code)

            rewards.append(reward)
            log_probs.append(log_prob)

        # PPO update (see Lecture 10)
        advantages = self.compute_advantages(rewards)
        policy_loss = self.ppo_loss(log_probs, advantages)
        policy_loss.backward()

        return rewards
```

**Why KL penalty matters:** Without it, model might generate code that passes tests but looks nothing like human code (unmaintainable).

---

## Part 3: CodeRL Framework (Complete Implementation)

CodeRL (Meta, 2022) is a simpler, more practical framework. Let me show you the full implementation:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess
import tempfile
import os

class CodeRLTrainer:
    """
    CodeRL: Reinforcement Learning for Code Generation

    Key differences from AlphaCode:
    1. Uses critic network (actor-critic, not just policy gradient)
    2. Unit test execution feedback
    3. Simpler, more efficient
    """

    def __init__(
        self,
        model_name="Salesforce/codegen-350M-mono",
        learning_rate=1e-5,
        beta=0.01,  # KL penalty
        gamma=1.0,  # No discounting (sparse reward at end)
        value_coef=0.1,
        entropy_coef=0.01
    ):
        # Load pre-trained code model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.policy = AutoModelForCausalLM.from_pretrained(model_name)
        self.ref_model = AutoModelForCausalLM.from_pretrained(model_name)

        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(self.policy.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Optimizer
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.parameters()},
            {'params': self.critic.parameters()}
        ], lr=learning_rate)

        self.beta = beta
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def generate_code(self, problem_text, max_length=512):
        """Generate code solution for problem."""
        prompt = f"# Problem: {problem_text}\n# Solution:\n"
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Generate with sampling
        outputs = self.policy.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )

        # Extract generated code
        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        code = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Compute log probabilities
        log_probs = []
        for i, token_id in enumerate(generated_ids):
            if i < len(outputs.scores):
                logits = outputs.scores[i][0]
                log_prob = F.log_softmax(logits, dim=-1)[token_id]
                log_probs.append(log_prob)

        return code, torch.stack(log_probs)

    def execute_code(self, code, test_cases, timeout=5):
        """
        Execute code against test cases in isolated environment.
        Returns (pass_rate, execution_info).
        """
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Write code to file
            f.write(code)
            f.write("\n\n")

            # Write test execution code
            f.write("import sys\n")
            f.write("import json\n")
            f.write("results = []\n")
            for i, (input_data, expected) in enumerate(test_cases):
                f.write(f"try:\n")
                f.write(f"    result = solution({repr(input_data)})\n")
                f.write(f"    results.append(result == {repr(expected)})\n")
                f.write(f"except Exception as e:\n")
                f.write(f"    results.append(False)\n")
            f.write("print(json.dumps(results))\n")

            temp_path = f.name

        try:
            # Execute in subprocess with timeout
            result = subprocess.run(
                ['python', temp_path],
                capture_output=True,
                timeout=timeout,
                text=True
            )

            # Parse results
            if result.returncode == 0:
                import json
                test_results = json.loads(result.stdout.strip())
                pass_rate = sum(test_results) / len(test_results)
                return pass_rate, {"passed": test_results, "error": None}
            else:
                # Runtime error
                return 0.0, {"passed": [], "error": result.stderr}

        except subprocess.TimeoutExpired:
            return 0.0, {"passed": [], "error": "Timeout"}
        except Exception as e:
            return 0.0, {"passed": [], "error": str(e)}
        finally:
            # Clean up
            os.unlink(temp_path)

    def compute_reward(self, problem, code, log_probs):
        """
        Compute reward for generated code.
        Reward = execution_score - β * KL(π||π_ref)
        """
        # Get execution reward
        exec_reward, info = self.execute_code(code, problem['test_cases'])

        # Compute KL divergence with reference model
        with torch.no_grad():
            prompt = f"# Problem: {problem['text']}\n# Solution:\n"
            inputs = self.tokenizer(prompt, return_tensors="pt")
            code_ids = self.tokenizer.encode(code, return_tensors="pt")[0]

            # Get reference model log probs
            ref_outputs = self.ref_model(
                input_ids=torch.cat([inputs.input_ids[0], code_ids]).unsqueeze(0)
            )
            ref_logits = ref_outputs.logits[0, inputs.input_ids.shape[1]-1:-1]
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            ref_log_probs = ref_log_probs[range(len(code_ids)), code_ids]

        # KL divergence: KL(π||π_ref) = Σ π(a) * (log π(a) - log π_ref(a))
        kl_div = (log_probs - ref_log_probs).mean()

        # Total reward
        total_reward = exec_reward - self.beta * kl_div.item()

        return total_reward, exec_reward, kl_div.item(), info

    def train_step(self, batch_problems):
        """
        One training step of CodeRL.

        For each problem:
        1. Generate code solution
        2. Execute and get reward
        3. Update policy with PPO
        """
        all_rewards = []
        all_exec_rewards = []
        all_kl_divs = []

        for problem in batch_problems:
            # Generate code
            code, log_probs = self.generate_code(problem['text'])

            # Compute reward
            reward, exec_reward, kl_div, info = self.compute_reward(
                problem, code, log_probs
            )

            # Sparse reward: only at end of generation
            rewards = torch.zeros(len(log_probs))
            rewards[-1] = reward  # All reward at final token

            # Compute value estimates
            # (In practice, you'd compute hidden states during generation)
            # Here, simplified version:
            values = torch.zeros(len(log_probs))  # Placeholder

            # Compute advantages (GAE)
            advantages = rewards - values

            # Policy loss (PPO)
            ratio = torch.exp(log_probs - log_probs.detach())
            policy_loss = -torch.min(
                ratio * advantages,
                torch.clamp(ratio, 0.8, 1.2) * advantages
            ).mean()

            # Value loss
            value_loss = F.mse_loss(values, rewards)

            # Entropy bonus (encourage exploration)
            entropy = -(log_probs * torch.exp(log_probs)).mean()

            # Total loss
            loss = (
                policy_loss +
                self.value_coef * value_loss -
                self.entropy_coef * entropy
            )

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

            # Logging
            all_rewards.append(reward)
            all_exec_rewards.append(exec_reward)
            all_kl_divs.append(kl_div)

        return {
            'mean_reward': sum(all_rewards) / len(all_rewards),
            'mean_exec_reward': sum(all_exec_rewards) / len(all_exec_rewards),
            'mean_kl_div': sum(all_kl_divs) / len(all_kl_divs)
        }
```

---

## Part 4: Complete Training Example

Here's how to actually use CodeRL:

```python
# Training script
def train_coderl():
    # Initialize trainer
    trainer = CodeRLTrainer(
        model_name="Salesforce/codegen-350M-mono",
        beta=0.01,
        learning_rate=1e-5
    )

    # Load training problems (e.g., from APPS dataset)
    problems = [
        {
            'text': 'Write a function that returns the sum of two numbers',
            'test_cases': [
                ((2, 3), 5),
                ((0, 0), 0),
                ((-1, 1), 0),
                ((100, 200), 300)
            ]
        },
        {
            'text': 'Write a function that reverses a string',
            'test_cases': [
                (('hello',), 'olleh'),
                (('',), ''),
                (('a',), 'a'),
                (('racecar',), 'racecar')
            ]
        },
        # ... more problems
    ]

    # Training loop
    for epoch in range(100):
        # Sample batch of problems
        import random
        batch = random.sample(problems, k=4)

        # Train
        metrics = trainer.train_step(batch)

        print(f"Epoch {epoch}:")
        print(f"  Mean Reward: {metrics['mean_reward']:.3f}")
        print(f"  Exec Reward: {metrics['mean_exec_reward']:.3f}")
        print(f"  KL Div: {metrics['mean_kl_div']:.3f}")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save(trainer.policy.state_dict(), f'coderl_epoch_{epoch}.pt')

if __name__ == '__main__':
    train_coderl()
```

---

## Part 5: Pass@k Evaluation (Industry Standard)

In code generation, we don't just generate one solution. We generate **k solutions** and check if **any** pass all tests.

### Why Pass@k?

```python
# Problem: Sort a list
# Model might generate multiple valid solutions:

# Solution 1 (bubble sort)
def solution(lst):
    for i in range(len(lst)):
        for j in range(len(lst)-1):
            if lst[j] > lst[j+1]:
                lst[j], lst[j+1] = lst[j+1], lst[j]
    return lst

# Solution 2 (built-in)
def solution(lst):
    return sorted(lst)

# Solution 3 (quicksort)
def solution(lst):
    if len(lst) <= 1:
        return lst
    pivot = lst[len(lst)//2]
    left = [x for x in lst if x < pivot]
    middle = [x for x in lst if x == pivot]
    right = [x for x in lst if x > pivot]
    return solution(left) + middle + solution(right)
```

All three are correct! Pass@k captures this.

### Implementation

```python
def evaluate_pass_at_k(model, problems, k=100, num_samples=200):
    """
    Evaluate pass@k metric.

    Args:
        model: Code generation model
        problems: List of problems with test cases
        k: Number of samples to consider
        num_samples: Total samples to generate (>= k)

    Returns:
        pass@k: Probability that at least one of k samples passes
    """
    import numpy as np

    total_correct = 0
    total_problems = len(problems)

    for problem in problems:
        # Generate k solutions
        solutions = []
        for _ in range(k):
            code, _ = model.generate_code(problem['text'])
            solutions.append(code)

        # Execute each solution
        any_passed = False
        for code in solutions:
            pass_rate, _ = model.execute_code(code, problem['test_cases'])
            if pass_rate == 1.0:  # All tests pass
                any_passed = True
                break

        if any_passed:
            total_correct += 1

    pass_at_k = total_correct / total_problems
    return pass_at_k

# Usage
pass_at_1 = evaluate_pass_at_k(trainer, test_problems, k=1)
pass_at_10 = evaluate_pass_at_k(trainer, test_problems, k=10)
pass_at_100 = evaluate_pass_at_k(trainer, test_problems, k=100)

print(f"Pass@1: {pass_at_1*100:.1f}%")
print(f"Pass@10: {pass_at_10*100:.1f}%")
print(f"Pass@100: {pass_at_100*100:.1f}%")
```

### Real Numbers

AlphaCode on CodeContests:
- Pass@1: ~5%
- Pass@10: ~15%
- Pass@100: ~30%

After RL fine-tuning:
- Pass@1: ~8%
- Pass@10: ~25%
- Pass@100: ~45%

**That's a huge improvement!**

---

## Part 6: Gotchas from Real Implementation

### Gotcha 1: Sandboxing is Critical

I once ran student code without sandboxing:

```python
# Student's "solution" to reverse a string
def solution(s):
    import os
    os.system("rm -rf /")  # NEVER run this!
    return s[::-1]
```

**Solution:** Use Docker containers or Python's `RestrictedPython`:

```python
import docker

def execute_safely(code, test_cases):
    client = docker.from_env()

    # Create isolated container
    container = client.containers.run(
        'python:3.9-slim',
        f'python -c "{code}"',
        detach=True,
        mem_limit='256m',
        cpu_period=100000,
        cpu_quota=50000,  # 50% of one CPU
        network_disabled=True
    )

    # Wait for completion
    result = container.wait(timeout=5)
    logs = container.logs()
    container.remove()

    return logs
```

### Gotcha 2: Infinite Loops

```python
# This will hang forever
def solution(n):
    while True:
        n += 1
    return n
```

**Solution:** Always use timeouts (shown in `execute_code` above).

### Gotcha 3: Reward Hacking

Early in training, models discover exploits:

```python
# Problem: Sort a list
# Model's "solution":
def solution(lst):
    # Just return the expected output for common test cases!
    if lst == [3, 1, 2]:
        return [1, 2, 3]
    if lst == [5, 4]:
        return [4, 5]
    return lst  # Hope for the best
```

**Solution:** Large, diverse test suites that can't be memorized.

### Gotcha 4: Import Handling

```python
# Model tries to import unavailable libraries
def solution(lst):
    import super_rare_library  # Not installed
    return super_rare_library.sort(lst)
```

**Solution:** Provide standard library, catch import errors:

```python
try:
    exec(code)
except ImportError:
    return 0.0  # Penalize unavailable imports
```

### Gotcha 5: Test Case Leakage

Don't let the model see test cases during training!

```python
# BAD: Model can overfit to test cases
prompt = f"""
Problem: {problem}
Test cases: {test_cases}  # DON'T DO THIS
Write solution:
"""
```

**Solution:** Only show problem description, hide tests.

---

## Part 7: Real-World Results

### AlphaCode (DeepMind)

- CodeContests dataset (competitive programming)
- Ranked in top 54% of human competitors
- 1M samples → filter to 10 → submit
- RL improved pass@100 by 15 percentage points

### CodeRL (Meta)

- APPS dataset (introductory to competition level)
- Pass@1 improved from 19% → 27%
- More efficient than AlphaCode (fewer samples)
- Used actor-critic instead of pure policy gradient

### CodeT (Microsoft, 2023)

- Combined CodeRL with self-debugging
- Model generates code → executes → reads error → fixes
- Pass@1 on HumanEval: 65.8%

---

## Part 8: When to Use RLHF for Code

**Use RLHF when:**
1. You have a reliable execution environment
2. Problems have clear test suites
3. You need correctness over style
4. You have compute budget (RL is expensive)

**Don't use RLHF when:**
1. No test cases available (use supervised learning)
2. Code quality matters more than correctness (use human feedback)
3. Very simple problems (SL is enough)
4. Limited compute (fine-tuning on high-quality data is cheaper)

---

## Summary

1. **Code generation is perfect for RL** because execution provides objective feedback
2. **AlphaCode's approach:** Massive sampling + filtering + RL fine-tuning
3. **CodeRL framework:** Actor-critic with execution-based rewards
4. **Key components:**
   - Execution sandbox (Docker/subprocess)
   - Test case suite
   - KL penalty to reference model
   - Sparse reward (at end of generation)
5. **Pass@k evaluation:** Industry standard metric
6. **Gotchas:** Sandboxing, timeouts, reward hacking, import handling

---

## What's Next?

**Next lecture (14):** Constitutional AI - How to make models helpful, harmless, and honest

**The connection:** Instead of execution feedback, we'll use AI feedback to improve AI. Same RL principles, different reward signal.

---

## Paper Trail

1. **AlphaCode (2022):** "Competition-Level Code Generation with AlphaCode" - DeepMind
2. **CodeRL (2022):** "CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning" - Meta
3. **CodeT (2023):** "CodeT: Code Generation with Generated Tests" - Microsoft
4. **APPS Dataset (2021):** "Measuring Coding Challenge Competence With APPS" - Dan Hendrycks et al.
5. **HumanEval (2021):** "Evaluating Large Language Models Trained on Code" - OpenAI

All in `/Modern-RL-Research/LLM-Code-Generation/PAPERS.md`

---

## Exercise for Yourself

Implement a simple version:

1. Take a small code generation model (CodeGen-350M)
2. Create 10 simple problems with test cases
3. Implement execution feedback
4. Train for 50 epochs
5. Measure pass@1 before and after

**You'll see improvement even with this toy setup!**

That's the power of RL for code generation. You're literally teaching the model to write code that *works*, not just code that *looks good*.
