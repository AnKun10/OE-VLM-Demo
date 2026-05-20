"""CPU unit tests for vllm.model_executor.models.agilepruner."""
import math

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.models.agilepruner import (
    agilepruner_select,
    compute_erank,
    compute_l2_norm_score,
    compute_surrogate_cls_score,
)


def test_compute_erank_identity_matrix():
    # X = I_N has all singular values equal → erank should equal N.
    N = 8
    X = torch.eye(N, dtype=torch.float32)
    result = compute_erank(X)
    assert math.isclose(result, float(N), rel_tol=1e-4)


def test_compute_erank_rank_one():
    # X = u·vᵀ has only one non-zero singular value → erank should be 1.
    N, D = 8, 64
    u = torch.randn(N, 1)
    v = torch.randn(1, D)
    X = u @ v
    result = compute_erank(X)
    assert math.isclose(result, 1.0, rel_tol=1e-2, abs_tol=5e-3)


def test_compute_erank_bf16_input():
    # Caller may pass bf16; result must be finite, float32 internally.
    X = torch.randn(16, 64, dtype=torch.bfloat16)
    result = compute_erank(X)
    assert math.isfinite(result)
    assert result > 0


def test_surrogate_cls_uniform_attention():
    # Uniform attention: every token attended equally → all scores equal.
    H, N = 4, 10
    attn = torch.full((H, N, N), 1.0 / N)
    scores = compute_surrogate_cls_score(attn)
    assert scores.shape == (N,)
    assert torch.allclose(scores, scores[0].expand_as(scores), atol=1e-6)


def test_surrogate_cls_excludes_diagonal():
    # Construct attention where only the diagonal is non-zero.
    # After excluding diagonal, all scores should be zero.
    H, N = 2, 5
    attn = torch.zeros(H, N, N)
    for i in range(N):
        attn[:, i, i] = 1.0
    scores = compute_surrogate_cls_score(attn)
    assert torch.allclose(scores, torch.zeros(N), atol=1e-6)


def test_surrogate_cls_one_hot_target():
    # All queries attend only to token 3 → token 3 should have the highest score,
    # and others should be zero.
    H, N = 3, 7
    attn = torch.zeros(H, N, N)
    attn[:, :, 3] = 1.0
    # Then zero out the diagonal contribution at (3, 3).
    attn[:, 3, 3] = 0.0
    scores = compute_surrogate_cls_score(attn)
    top = int(scores.argmax().item())
    assert top == 3
    other_idxs = [i for i in range(N) if i != 3]
    assert torch.allclose(scores[other_idxs], torch.zeros(N - 1), atol=1e-6)


def test_select_returns_exact_K():
    N, D = 20, 64
    embeds = torch.randn(N, D)
    attn = torch.softmax(torch.randn(2, N, N), dim=-1)
    score = compute_surrogate_cls_score(attn)
    indices = agilepruner_select(
        embeds, score, ratio=0.5, tau_max=0.25, erank_avg=95.0
    )
    assert len(indices) == 10
    assert len(set(indices.tolist())) == 10  # no duplicates
    assert all(0 <= i < N for i in indices.tolist())


def test_select_ratio_one_returns_all():
    N, D = 12, 32
    embeds = torch.randn(N, D)
    attn = torch.softmax(torch.randn(1, N, N), dim=-1)
    score = compute_surrogate_cls_score(attn)
    indices = agilepruner_select(
        embeds, score, ratio=1.0, tau_max=0.25, erank_avg=95.0
    )
    assert len(indices) == N
    assert sorted(indices.tolist()) == list(range(N))


def test_select_floor_K_is_4():
    # ratio=0.1 with N=10 → round(1)=1, floor lifts to 4.
    N, D = 10, 32
    embeds = torch.randn(N, D)
    attn = torch.softmax(torch.randn(1, N, N), dim=-1)
    score = compute_surrogate_cls_score(attn)
    indices = agilepruner_select(
        embeds, score, ratio=0.1, tau_max=0.25, erank_avg=95.0
    )
    assert len(indices) == 4


def test_select_small_N_skips_pruning():
    # N ≤ 4 → no pruning; return all.
    N, D = 3, 32
    embeds = torch.randn(N, D)
    attn = torch.softmax(torch.randn(1, N, N), dim=-1)
    score = compute_surrogate_cls_score(attn)
    indices = agilepruner_select(
        embeds, score, ratio=0.5, tau_max=0.25, erank_avg=95.0
    )
    assert sorted(indices.tolist()) == [0, 1, 2]


def test_select_top_attention_token_always_kept():
    # Token with overwhelmingly highest attention score must always survive.
    N, D = 30, 64
    embeds = torch.randn(N, D)
    attn = torch.zeros(1, N, N)
    attn[:, :, 7] = 1.0  # Token 7 is the global-importance peak.
    attn[:, 7, 7] = 0.0
    score = compute_surrogate_cls_score(attn)
    indices = agilepruner_select(
        embeds, score, ratio=0.3, tau_max=0.25, erank_avg=95.0
    )
    assert 7 in indices.tolist()


def test_select_fallback_when_threshold_drains_pool():
    # All embeddings nearly identical → every iter would prune everyone.
    # Fallback should still return exactly K.
    N, D = 20, 64
    base = torch.randn(1, D)
    embeds = base.expand(N, D).clone() + 1e-6 * torch.randn(N, D)
    attn = torch.softmax(torch.randn(1, N, N), dim=-1)
    score = compute_surrogate_cls_score(attn)
    indices = agilepruner_select(
        embeds, score, ratio=0.5, tau_max=0.25, erank_avg=95.0
    )
    assert len(indices) == 10


def test_l2_norm_score_unit_vectors():
    # All rows unit norm → all scores ≈ 1.0.
    N, D = 8, 16
    X = F.normalize(torch.randn(N, D), dim=1)
    scores = compute_l2_norm_score(X)
    assert scores.shape == (N,)
    assert torch.allclose(scores, torch.ones(N), atol=1e-5)


def test_l2_norm_score_known_magnitudes():
    # Construct rows with prescribed norms; verify scores match.
    N, D = 5, 16
    base = F.normalize(torch.randn(N, D), dim=1)
    magnitudes = torch.tensor([1.0, 2.0, 0.5, 3.0, 1.5])
    X = base * magnitudes.unsqueeze(1)
    scores = compute_l2_norm_score(X)
    assert torch.allclose(scores, magnitudes, atol=1e-5)
