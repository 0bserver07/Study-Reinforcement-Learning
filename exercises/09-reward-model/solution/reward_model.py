"""Bradley-Terry reward model on synthetic preferences — reference solution.

Look at this after you've made a real attempt at ../starter.py.

Run it directly:  python3 exercises/09-reward-model/solution/reward_model.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── synthetic-data constants ──────────────────────────────────────────────────

FEATURE_DIM = 4
# The "true" reward is a fixed linear function of the 4-dim feature vector.
# The student's reward model never sees these weights — it has to recover the
# ranking from preference labels alone.
TRUE_W = torch.tensor([1.0, -0.5, 0.3, 2.0])
# Temperature on the Bradley-Terry preference distribution. Higher β = labels
# are closer to the deterministic argmax; lower β = noisier labels.
BETA = 4.0


# ── synthetic data ────────────────────────────────────────────────────────────

def true_reward(features: torch.Tensor) -> torch.Tensor:
    """Hidden reward function. Linear in the features.

    Args:
        features: (..., FEATURE_DIM) tensor of feature vectors.

    Returns:
        Tensor of scalar rewards with the leading shape of `features`.
    """
    return features @ TRUE_W


def make_preference_dataset(
    n_pairs: int,
    seed: int = 0,
) -> tuple:
    """Generate n_pairs of (chosen_features, rejected_features).

    Each pair is two random feature vectors. The "chosen" one is the one with
    the higher true reward with probability σ(β · (r*_a - r*_b)) — that IS the
    Bradley-Terry model.

    Args:
        n_pairs: number of preference pairs to sample.
        seed:    RNG seed for reproducibility.

    Returns:
        chosen:   (n_pairs, FEATURE_DIM) tensor.
        rejected: (n_pairs, FEATURE_DIM) tensor.
    """
    g = torch.Generator().manual_seed(seed)
    feats_a = torch.randn(n_pairs, FEATURE_DIM, generator=g)
    feats_b = torch.randn(n_pairs, FEATURE_DIM, generator=g)

    r_a = true_reward(feats_a)
    r_b = true_reward(feats_b)

    # P(a is chosen) = σ(β · (r_a - r_b))
    p_a = torch.sigmoid(BETA * (r_a - r_b))
    a_is_chosen = torch.rand(n_pairs, generator=g) < p_a

    chosen = torch.where(a_is_chosen.unsqueeze(-1), feats_a, feats_b)
    rejected = torch.where(a_is_chosen.unsqueeze(-1), feats_b, feats_a)
    return chosen, rejected


def make_eval_set(n: int, seed: int = 1) -> tuple:
    """Held-out features + their true rewards, for Spearman evaluation."""
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(n, FEATURE_DIM, generator=g)
    return feats, true_reward(feats)


# ── reward model ──────────────────────────────────────────────────────────────

class RewardModel(nn.Module):
    """4-dim feature vector → scalar reward, via a small MLP.

    Forward:
        single input,  shape (FEATURE_DIM,)         → scalar tensor, shape ()
        batched input, shape (B, FEATURE_DIM)       → tensor of shape (B, 1)

    The (B, 1) batched shape matches what HuggingFace-style reward heads
    return; the scalar-on-single-input case is a convenience for the
    Bradley-Terry loss tests.
    """

    def __init__(self, feature_dim: int = FEATURE_DIM, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 1:
            # Single input: return a scalar (shape ()).
            return self.net(features).squeeze(-1)
        return self.net(features)  # (B, 1)


# ── Bradley-Terry loss ────────────────────────────────────────────────────────

def bradley_terry_loss(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
) -> torch.Tensor:
    """-log σ(r_chosen - r_rejected), averaged over the batch.

    This is the negative log-likelihood under the Bradley-Terry model:

        P(chosen > rejected) = σ(r_chosen - r_rejected)

    Properties:
      - chosen >> rejected → loss → 0
      - chosen == rejected → loss = log(2) ≈ 0.693 (chance level)
      - chosen << rejected → loss grows linearly in the gap

    Inputs may be either scalars (shape ()), 1-D (shape (B,)), or 2-D
    (shape (B, 1)). The implementation flattens both so the math is the same.

    Args:
        chosen_rewards:   model rewards for the chosen response in each pair.
        rejected_rewards: model rewards for the rejected response in each pair.

    Returns:
        Scalar tensor, mean loss over the batch.
    """
    chosen = chosen_rewards.flatten()
    rejected = rejected_rewards.flatten()
    # logsigmoid is more numerically stable than log(sigmoid(...)).
    return -F.logsigmoid(chosen - rejected).mean()


# ── Spearman correlation (no scipy dep) ───────────────────────────────────────

def spearman_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation between two 1-D tensors.

    Equivalent to scipy.stats.spearmanr(a, b).correlation in the no-ties case
    we get from continuous-valued rewards. Implemented locally to keep scipy
    out of the dependency list.
    """
    a_np = a.detach().cpu().numpy()
    b_np = b.detach().cpu().numpy()
    ra = np.argsort(np.argsort(a_np)).astype(np.float64)
    rb = np.argsort(np.argsort(b_np)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = (np.linalg.norm(ra) * np.linalg.norm(rb))
    if denom == 0.0:
        return 0.0
    return float((ra @ rb) / denom)


# ── training ──────────────────────────────────────────────────────────────────

def train_reward_model(
    n_pairs: int = 4096,
    n_steps: int = 2000,
    batch_size: int = 64,
    lr: float = 3e-3,
    seed: int = 0,
) -> dict:
    """Train the reward model on synthetic Bradley-Terry preferences.

    Per step: sample a minibatch of preference pairs, run both through the
    model, take the BT loss, backprop, step. After each step, evaluate Spearman
    correlation against the true reward on a fixed held-out set.

    Args:
        n_pairs:    total training pairs to generate.
        n_steps:    number of optimizer steps.
        batch_size: minibatch size for each step.
        lr:         Adam learning rate.
        seed:       RNG seed.

    Returns:
        {
            "model":    the trained RewardModel,
            "losses":   list of n_steps floats — per-step training loss,
            "spearman": list of n_steps floats — per-step held-out Spearman.
        }
    """
    torch.manual_seed(seed)

    chosen, rejected = make_preference_dataset(n_pairs, seed=seed)
    eval_feats, eval_true_r = make_eval_set(n=512, seed=seed + 1)

    model = RewardModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    g = torch.Generator().manual_seed(seed + 2)

    losses: list = []
    spearmans: list = []

    for _step in range(n_steps):
        idx = torch.randint(n_pairs, (batch_size,), generator=g)
        r_chosen = model(chosen[idx])      # (B, 1)
        r_rejected = model(rejected[idx])  # (B, 1)

        loss = bradley_terry_loss(r_chosen, r_rejected)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        with torch.no_grad():
            pred_r = model(eval_feats).squeeze(-1)  # (n_eval,)
        spearmans.append(spearman_corr(pred_r, eval_true_r))

    return {"model": model, "losses": losses, "spearman": spearmans}


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out = train_reward_model(seed=0, n_steps=2000)
    final_loss = out["losses"][-1]
    final_spearman = out["spearman"][-1]
    print(
        f"steps: {len(out['losses'])}  "
        f"final loss: {final_loss:.4f}  "
        f"final Spearman (held-out): {final_spearman:.4f}  "
        f"(random baseline ≈ 0.00, perfect ranking = 1.00)"
    )
