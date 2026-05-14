"""GRPO on a verifiable toy task — exercise starter.

Fill in the TODOs, then from the repo root:

    pip install -r exercises/requirements.txt
    pytest exercises/15-grpo-rlvr/

See README.md for the task and HINTS.md if you're stuck. The reference
implementation is in solution/grpo_rlvr.py — look after you've tried.

The toy task: integer addition.
  - Prompts are (a, b) pairs with a, b in {1, 2, 3}.
  - Possible answers: integers 0..9 (10 options; correct answers are 2..6).
  - Reward: 1.0 if answer == a + b, else 0.0.
  - Policy: nn.Embedding(n_prompts, n_answers) — one row of logits per prompt.

GRPO update per step:
  1. Sample K answers from the current policy for one prompt.
  2. Compute reward for each answer via the verifier.
  3. Group-relative advantage: A_i = (r_i - mean(r)) / (std(r) + eps).
  4. PPO-clipped surrogate loss (mean over K), with log-prob ratio.
  5. Backward, step.
"""

import torch
import torch.nn as nn

# ── task constants ────────────────────────────────────────────────────────────

# Prompts: all (a, b) with a, b in {1, 2, 3}. Fixed order so we can use
# integer indices into an embedding.
PROMPTS = [(a, b) for a in range(1, 4) for b in range(1, 4)]  # 9 prompts
N_PROMPTS = len(PROMPTS)   # 9
N_ANSWERS = 10             # possible answers: 0 .. 9


# ── verifier ─────────────────────────────────────────────────────────────────

def verifier(prompt: tuple, answer: int) -> float:
    """Return 1.0 if answer == a + b, else 0.0.

    This is given to you — the student doesn't implement this. It's the
    'checker' that replaces a learned reward model.

    Args:
        prompt: (a, b) tuple of ints.
        answer: the integer the policy chose.
    """
    a, b = prompt
    return 1.0 if answer == a + b else 0.0


# ── policy ───────────────────────────────────────────────────────────────────

class GRPOPolicy(nn.Module):
    """Per-prompt categorical policy, parameterized as an embedding.

    `forward(prompt_idx)` returns logits of shape (N_ANSWERS,) for that prompt.
    `prompt_idx` is an integer index into PROMPTS.

    Args:
        n_prompts: number of distinct prompts (rows in the embedding).
        n_answers: number of possible answers (columns — the vocab size).
    """

    def __init__(self, n_prompts: int = N_PROMPTS, n_answers: int = N_ANSWERS):
        super().__init__()
        # TODO: create self.embed = nn.Embedding(n_prompts, n_answers).
        # That's the whole thing — each row is a logit vector for one prompt.
        raise NotImplementedError("GRPOPolicy.__init__")

    def forward(self, prompt_idx: int) -> torch.Tensor:
        # TODO: look up prompt_idx in self.embed and return the logit row.
        # Hint: nn.Embedding expects a LongTensor. torch.tensor(prompt_idx)
        # works, but make sure the dtype is torch.long.
        raise NotImplementedError("GRPOPolicy.forward")


# ── group sampling ────────────────────────────────────────────────────────────

def sample_group(
    policy: GRPOPolicy,
    prompt_idx: int,
    K: int,
) -> tuple:
    """Sample K answers from the policy for one prompt.

    Returns:
        answers:   LongTensor of shape (K,) — the sampled answer indices.
        log_probs: Tensor of shape (K,) — log π(answer_i | prompt), WITH
                   gradient (do not .detach() here; the loss needs to
                   backprop through these).

    Steps:
      - logits = policy(prompt_idx)                      shape: (N_ANSWERS,)
      - dist = torch.distributions.Categorical(logits=logits)
      - answers = dist.sample((K,))                      shape: (K,)
      - log_probs = dist.log_prob(answers)               shape: (K,)
    """
    # TODO
    raise NotImplementedError("sample_group")


# ── group-relative advantage ──────────────────────────────────────────────────

def group_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute group-relative advantages from a reward tensor.

    Formula: A_i = (r_i - mean(r)) / (std(r) + eps)

    Edge cases — return zeros(K) when:
      - K == 1 (can't compute a meaningful group baseline from one sample).
      - All rewards are equal (std == 0; advantages would all be 0 anyway,
        and dividing by eps gives artificially large values).

    Args:
        rewards: float Tensor of shape (K,).
        eps:     small constant added to std to avoid division by zero.

    Returns:
        Tensor of shape (K,), same dtype as rewards.
    """
    # TODO:
    #   K = rewards.shape[0]
    #   if K == 1 or rewards.std() < eps:
    #       return torch.zeros_like(rewards)
    #   return (rewards - rewards.mean()) / (rewards.std() + eps)
    raise NotImplementedError("group_advantages")


# ── GRPO loss ─────────────────────────────────────────────────────────────────

def grpo_loss(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """PPO-style clipped surrogate loss for one group, averaged over K.

    This is the loss to MINIMIZE (gradient ascent via descent).

    The probability ratio:
        rho_i = exp(log_probs_new_i - log_probs_old_i)

    The clipped surrogate for each element:
        L_i = -min(rho_i * A_i, clip(rho_i, 1 - clip_eps, 1 + clip_eps) * A_i)

    The loss is the mean of L_i over K.

    Args:
        log_probs_new: Tensor (K,) — log-probs from the CURRENT policy
                       (has gradient).
        log_probs_old: Tensor (K,) — log-probs from the policy used to
                       SAMPLE (should be .detach()-ed; treated as a constant).
        advantages:    Tensor (K,) — output of group_advantages.
        clip_eps:      clipping radius (default 0.2, same as PPO).

    Returns:
        Scalar tensor with gradient.
    """
    # TODO:
    #   ratio = (log_probs_new - log_probs_old).exp()
    #   clipped = ratio.clamp(1 - clip_eps, 1 + clip_eps)
    #   per_sample = -torch.min(ratio * advantages, clipped * advantages)
    #   return per_sample.mean()
    raise NotImplementedError("grpo_loss")


# ── training loop ─────────────────────────────────────────────────────────────

def train(
    num_steps: int = 1500,
    K: int = 8,
    lr: float = 0.5,
    clip_eps: float = 0.2,
    seed: int = 0,
) -> list:
    """Train the GRPO policy on the toy addition task.

    Each step:
      1. Pick a prompt uniformly at random.
      2. Sample K answers from the current policy → (answers, log_probs_old).
         Detach log_probs_old immediately — it's the 'sampling policy', treated
         as a constant in the loss.
      3. Compute rewards via verifier.
      4. Compute advantages via group_advantages.
      5. Re-evaluate log_probs_new from the *current* policy on the same answers
         (in this single-update on-policy setting, the policy hasn't moved yet,
         so log_probs_new ≈ log_probs_old — but they're connected to the graph).
      6. Compute grpo_loss(log_probs_new, log_probs_old, advantages).
      7. optimizer.zero_grad() → loss.backward() → optimizer.step().

    Returns:
        List of length num_steps: mean reward over the K-group at each step.
        Use this to verify learning (should rise from ~0.1 toward ~1.0).
    """
    torch.manual_seed(seed)
    rng = torch.Generator()
    rng.manual_seed(seed)

    policy = GRPOPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    step_rewards = []
    for step in range(num_steps):
        # TODO:
        #   prompt_idx = torch.randint(N_PROMPTS, (1,), generator=rng).item()
        #   prompt = PROMPTS[prompt_idx]
        #
        #   answers, log_probs_old = sample_group(policy, prompt_idx, K)
        #   log_probs_old = log_probs_old.detach()
        #
        #   rewards = torch.tensor(
        #       [verifier(prompt, int(a)) for a in answers], dtype=torch.float32
        #   )
        #
        #   adv = group_advantages(rewards)
        #
        #   # Re-evaluate log-probs from the current policy on the same answers.
        #   logits = policy(prompt_idx)
        #   dist = torch.distributions.Categorical(logits=logits)
        #   log_probs_new = dist.log_prob(answers)
        #
        #   loss = grpo_loss(log_probs_new, log_probs_old, adv, clip_eps)
        #
        #   optimizer.zero_grad()
        #   loss.backward()
        #   optimizer.step()
        #
        #   step_rewards.append(rewards.mean().item())
        raise NotImplementedError("train: the update step")

    return step_rewards


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rewards = train()
    last200 = rewards[-200:]
    print(
        f"steps: {len(rewards)}  "
        f"mean reward (last 200): {sum(last200) / len(last200):.3f}  "
        f"(random guessing ≈ 0.10)"
    )
