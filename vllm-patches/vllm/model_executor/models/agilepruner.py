"""AgilePruner: visual-token pre-pruning for Qwen3-VL.

Reference: Baek et al., ICLR 2026 (https://arxiv.org/abs/2603.01236).
Adapted to Qwen3-VL (no [CLS] token, window-attention vision tower).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_erank(embeds: torch.Tensor, eps: float = 1e-9) -> float:
    """Effective rank via fast N×N covariance form (N ≪ D for LVLMs).

    erank(X) = exp(H(p)), where p = singular_values(X) normalised to sum to 1.
    Equivalent to the SVD definition but cheaper when N ≪ D.
    """
    X = embeds.float()
    C = X @ X.T  # (N, N)
    eig = torch.linalg.eigvalsh(C).clamp(min=0.0)
    S = eig.sqrt()
    total = S.sum()
    if total <= 0:
        return 1.0
    p = S / total
    entropy = -(p * (p + eps).log()).sum()
    return float(torch.exp(entropy).item())


def compute_surrogate_cls_score(attn: torch.Tensor) -> torch.Tensor:
    """Surrogate for [CLS] attention rank in Qwen3-VL.

    Score for token j = mean over heads and queries i (i ≠ j) of attn[h, i, j].
    Captures "how much other tokens attend to j" — the global-importance signal.

    Args:
        attn: (H, N, N) attention weights from a single full-attention layer.

    Returns:
        (N,) per-token score, float32 on the same device.
    """
    H, N, _ = attn.shape
    a = attn.float()
    # Zero out the diagonal so a token can't promote itself.
    diag_mask = torch.eye(N, dtype=torch.bool, device=a.device)
    a = a.masked_fill(diag_mask.unsqueeze(0), 0.0)
    # Mean over heads and queries.
    return a.mean(dim=(0, 1))  # (N,)


def agilepruner_select(
    embeds: torch.Tensor,
    attn: torch.Tensor,
    ratio: float,
    tau_max: float,
    erank_avg: float,
) -> torch.Tensor:
    """Pick K visual tokens per AgilePruner's adaptive iterative threshold.

    Args:
        embeds: (N, D) post-projector visual embeddings (any dtype).
        attn:   (H, N, N) full-attention layer weights for the same N tokens.
        ratio:  K = round(ratio × N), floored to 4 (or N if N ≤ 4).
        tau_max: cap on the per-step similarity threshold.
        erank_avg: dataset-mean erank used to normalise τ.

    Returns:
        1D LongTensor of `kept` indices, length = K (or N when N ≤ 4 or ratio = 1).
        Order = attention-rank order, NOT spatial.
    """
    N, _ = embeds.shape
    device = embeds.device

    if N <= 4 or ratio >= 1.0:
        return torch.arange(N, device=device, dtype=torch.long)

    K = max(round(ratio * N), 4)
    K = min(K, N)

    score = compute_surrogate_cls_score(attn)  # (N,) float32 on attn.device
    score = score.to(device)

    erank_input = compute_erank(embeds)
    tau_base = (erank_input / erank_avg) * 0.01

    ranked = score.argsort(descending=True).tolist()
    X = F.normalize(embeds.float(), dim=1)  # cosine via dot product

    alive = torch.ones(N, dtype=torch.bool, device=device)
    kept: list[int] = []

    for order_i, j in enumerate(ranked, start=1):
        if not alive[j]:
            continue
        tau_i = min(order_i * tau_base, tau_max)
        kept.append(j)
        alive[j] = False
        cos_to_j = (X @ X[j])  # (N,)
        d = 1.0 - cos_to_j
        alive = alive & (d >= tau_i)
        if len(kept) >= K:
            break

    if len(kept) < K:
        kept_set = set(kept)
        for j in ranked:
            if j not in kept_set:
                kept.append(j)
                kept_set.add(j)
                if len(kept) >= K:
                    break

    return torch.tensor(kept, device=device, dtype=torch.long)
