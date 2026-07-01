"""DPO on a toy preference dataset — reference solution.

Look at this after you've made a real attempt at ../starter.py.

Run it directly:  python3 exercises/11-dpo/solution/dpo.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── task constants ────────────────────────────────────────────────────────────

N_PROMPTS = 8
N_ANSWERS = 10
TRUE_REWARD_SEED = 0   # fixed seed for the "true" reward; same across runs
DATASET_SEED = 1       # fixed seed for the preference dataset
BETA = 0.1             # DPO KL coefficient
BT_TEMPERATURE = 5.0   # Bradley-Terry temperature for label noise


# ── true reward (fixed, known) ────────────────────────────────────────────────

def make_true_reward(n_prompts: int = N_PROMPTS, n_answers: int = N_ANSWERS) -> torch.Tensor:
    """Generate a fixed (prompt, answer) -> reward table.

    Deterministic for a given seed. Returns a tensor of shape
    (n_prompts, n_answers) with values in roughly [-2, 2].
    """
    g = torch.Generator()
    g.manual_seed(TRUE_REWARD_SEED)
    return torch.randn(n_prompts, n_answers, generator=g)


# ── preference dataset ────────────────────────────────────────────────────────

def make_preference_dataset(
    n_pairs: int,
    true_reward: torch.Tensor,
    seed: int = DATASET_SEED,
    bt_temperature: float = BT_TEMPERATURE,
) -> list:
    """Build a list of (prompt_idx, chosen_answer, rejected_answer) triples.

    For each pair:
      - Pick a prompt uniformly.
      - Sample two answers uniformly (without replacement).
      - Label using Bradley-Terry: P(a > b) = sigmoid(temp * (r_a - r_b)).
    """
    n_prompts, n_answers = true_reward.shape
    g = torch.Generator()
    g.manual_seed(seed)

    data = []
    for _ in range(n_pairs):
        p = int(torch.randint(n_prompts, (1,), generator=g).item())
        # sample two distinct answers
        perm = torch.randperm(n_answers, generator=g)
        a, b = int(perm[0]), int(perm[1])
        r_a = float(true_reward[p, a])
        r_b = float(true_reward[p, b])
        p_a_wins = torch.sigmoid(torch.tensor(bt_temperature * (r_a - r_b))).item()
        u = torch.rand((), generator=g).item()
        if u < p_a_wins:
            chosen, rejected = a, b
        else:
            chosen, rejected = b, a
        data.append((p, chosen, rejected))
    return data


# ── policy ───────────────────────────────────────────────────────────────────

class Policy(nn.Module):
    """Per-prompt categorical policy, parameterized as an embedding.

    Identical in shape to the GRPO exercise: nn.Embedding(n_prompts, n_answers)
    where row p is the logit vector over answers for prompt p.
    """

    def __init__(self, n_prompts: int = N_PROMPTS, n_answers: int = N_ANSWERS):
        super().__init__()
        self.embed = nn.Embedding(n_prompts, n_answers)

    def forward(self, prompt_idx) -> torch.Tensor:
        """Return logits of shape (n_answers,) for a single prompt index,
        or (B, n_answers) for a LongTensor of B prompt indices."""
        if isinstance(prompt_idx, int):
            idx = torch.tensor(prompt_idx, dtype=torch.long)
        else:
            idx = prompt_idx.long()
        return self.embed(idx)


# ── DPO loss ─────────────────────────────────────────────────────────────────

def dpo_loss(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    chosen_answer: torch.Tensor,
    rejected_answer: torch.Tensor,
    beta: float = BETA,
) -> torch.Tensor:
    """Compute the DPO loss for a batch of preference pairs.

    Args:
        policy_logits:  (B, n_answers) logits from the policy being trained.
        ref_logits:     (B, n_answers) logits from the frozen reference policy.
        chosen_answer:   LongTensor (B,) — index of the preferred answer.
        rejected_answer: LongTensor (B,) — index of the dispreferred answer.
        beta: KL coefficient.

    Returns:
        Scalar tensor: -log sigmoid(beta * [(log pi(y_w) - log pi_ref(y_w))
                                           - (log pi(y_l) - log pi_ref(y_l))])
        averaged over the batch.
    """
    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    # Gather the log-probs for chosen and rejected answers
    pi_w = policy_log_probs.gather(-1, chosen_answer.unsqueeze(-1)).squeeze(-1)
    pi_l = policy_log_probs.gather(-1, rejected_answer.unsqueeze(-1)).squeeze(-1)
    ref_w = ref_log_probs.gather(-1, chosen_answer.unsqueeze(-1)).squeeze(-1)
    ref_l = ref_log_probs.gather(-1, rejected_answer.unsqueeze(-1)).squeeze(-1)

    # The DPO logit: bracket inside sigmoid in the loss
    logits = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    return -F.logsigmoid(logits).mean()


# ── evaluation helper ─────────────────────────────────────────────────────────

def greedy_mean_true_reward(policy: Policy, true_reward: torch.Tensor) -> float:
    """Mean true reward of the greedy (argmax) policy across all prompts."""
    n_prompts = true_reward.shape[0]
    with torch.no_grad():
        idx = torch.arange(n_prompts)
        logits = policy(idx)                       # (n_prompts, n_answers)
        greedy = logits.argmax(dim=-1)             # (n_prompts,)
        rewards = true_reward.gather(-1, greedy.unsqueeze(-1)).squeeze(-1)
    return rewards.mean().item()


# ── training loop ─────────────────────────────────────────────────────────────

def train_dpo(
    preference_dataset: list,
    ref_policy: Policy,
    true_reward: torch.Tensor,
    beta: float = BETA,
    n_steps: int = 2000,
    lr: float = 0.05,
    batch_size: int = 32,
    seed: int = 0,
) -> tuple:
    """Train a Policy with DPO against a frozen reference.

    Each step:
      - Sample `batch_size` preference triples uniformly from the dataset.
      - Forward both policy and ref_policy on the batch prompts.
      - Compute dpo_loss, backprop, step.

    Returns:
        losses:        list of per-step loss values.
        mean_rewards:  list of per-step greedy mean true reward.
    """
    torch.manual_seed(seed)
    rng = torch.Generator()
    rng.manual_seed(seed)

    # Fresh policy initialized from the reference so they start equal.
    n_prompts, n_answers = true_reward.shape
    policy = Policy(n_prompts, n_answers)
    policy.load_state_dict(ref_policy.state_dict())

    # Freeze the reference (no gradients through it).
    for p in ref_policy.parameters():
        p.requires_grad_(False)
    ref_policy.eval()

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    # Pre-tensorize the dataset for fast batching.
    prompts = torch.tensor([t[0] for t in preference_dataset], dtype=torch.long)
    chosens = torch.tensor([t[1] for t in preference_dataset], dtype=torch.long)
    rejecteds = torch.tensor([t[2] for t in preference_dataset], dtype=torch.long)
    n = prompts.shape[0]

    losses, mean_rewards = [], []
    for _ in range(n_steps):
        idx = torch.randint(n, (batch_size,), generator=rng)
        p_batch = prompts[idx]
        w_batch = chosens[idx]
        l_batch = rejecteds[idx]

        policy_logits = policy(p_batch)
        with torch.no_grad():
            ref_logits = ref_policy(p_batch)

        loss = dpo_loss(policy_logits, ref_logits, w_batch, l_batch, beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        mean_rewards.append(greedy_mean_true_reward(policy, true_reward))

    return losses, mean_rewards, policy


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    true_reward = make_true_reward()
    dataset = make_preference_dataset(n_pairs=512, true_reward=true_reward)

    ref = Policy()                                  # uniform-ish at init
    losses, rewards, _ = train_dpo(
        dataset, ref, true_reward, beta=BETA, n_steps=2000, lr=0.05, seed=0
    )

    # Baseline = mean reward of a uniform-random policy across all (prompt, answer).
    uniform_baseline = true_reward.mean().item()
    print(
        f"steps: {len(losses)}  "
        f"start loss: {losses[0]:.4f}  "
        f"end loss: {losses[-1]:.4f}  "
        f"end greedy reward: {rewards[-1]:.3f}  "
        f"uniform baseline: {uniform_baseline:.3f}"
    )
