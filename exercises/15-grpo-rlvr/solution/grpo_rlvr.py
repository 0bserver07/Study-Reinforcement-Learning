"""GRPO on a verifiable toy task — reference solution.

Look at this after you've made a real attempt at ../starter.py.

Run it directly:  python3 exercises/15-grpo-rlvr/solution/grpo_rlvr.py
"""

import torch
import torch.nn as nn

# ── task constants ────────────────────────────────────────────────────────────

PROMPTS = [(a, b) for a in range(1, 4) for b in range(1, 4)]
N_PROMPTS = len(PROMPTS)   # 9
N_ANSWERS = 10             # possible answers: 0 .. 9


# ── verifier ─────────────────────────────────────────────────────────────────

def verifier(prompt: tuple, answer: int) -> float:
    a, b = prompt
    return 1.0 if answer == a + b else 0.0


# ── policy ───────────────────────────────────────────────────────────────────

class GRPOPolicy(nn.Module):
    def __init__(self, n_prompts: int = N_PROMPTS, n_answers: int = N_ANSWERS):
        super().__init__()
        self.embed = nn.Embedding(n_prompts, n_answers)

    def forward(self, prompt_idx: int) -> torch.Tensor:
        idx = torch.tensor(prompt_idx, dtype=torch.long)
        return self.embed(idx)  # shape: (N_ANSWERS,)


# ── group sampling ────────────────────────────────────────────────────────────

def sample_group(policy: GRPOPolicy, prompt_idx: int, K: int) -> tuple:
    logits = policy(prompt_idx)                          # (N_ANSWERS,)
    dist = torch.distributions.Categorical(logits=logits)
    answers = dist.sample((K,))                          # (K,)
    log_probs = dist.log_prob(answers)                   # (K,), with grad
    return answers, log_probs


# ── group-relative advantage ──────────────────────────────────────────────────

def group_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    K = rewards.shape[0]
    if K == 1 or rewards.std() < eps:
        return torch.zeros_like(rewards)
    return (rewards - rewards.mean()) / (rewards.std() + eps)


# ── GRPO loss ─────────────────────────────────────────────────────────────────

def grpo_loss(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    ratio = (log_probs_new - log_probs_old).exp()
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
    per_sample = -torch.min(ratio * advantages, clipped * advantages)
    return per_sample.mean()


# ── training loop ─────────────────────────────────────────────────────────────

def train(
    num_steps: int = 1500,
    K: int = 8,
    lr: float = 0.5,
    clip_eps: float = 0.2,
    seed: int = 0,
) -> list:
    torch.manual_seed(seed)
    rng = torch.Generator()
    rng.manual_seed(seed)

    policy = GRPOPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    step_rewards = []
    for step in range(num_steps):
        prompt_idx = torch.randint(N_PROMPTS, (1,), generator=rng).item()
        prompt = PROMPTS[prompt_idx]

        answers, log_probs_old = sample_group(policy, prompt_idx, K)
        log_probs_old = log_probs_old.detach()

        rewards = torch.tensor(
            [verifier(prompt, int(a)) for a in answers], dtype=torch.float32
        )

        adv = group_advantages(rewards)

        # Re-evaluate log-probs from current policy on the same answers.
        logits = policy(prompt_idx)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs_new = dist.log_prob(answers)

        loss = grpo_loss(log_probs_new, log_probs_old, adv, clip_eps)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step_rewards.append(rewards.mean().item())

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
