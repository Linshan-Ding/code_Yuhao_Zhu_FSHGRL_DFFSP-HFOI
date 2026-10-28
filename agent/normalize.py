"""Per-observation feature normalisation (v2; fixes docs/known-issues.md item 9).

The original networks normalised the processing-rate matrix over the whole forward batch and the due dates
over all valid entries of the batch, so the logits of one observation depended on the other observations in
the batch (the number of environments, the lockstep evaluation batch). Both are now normalised within each
observation: a batch of one and a batch of many give the same numbers (tests/test_known_issue_fixes.py).
Node features were already normalised per observation (``dim=-2`` with ``keepdim``).
"""
from __future__ import annotations

import torch

EPS = 1e-5


def normalize_per_observation(tensor: torch.Tensor) -> torch.Tensor:
    """Zero mean / unit (population) std over every non-batch dimension of each batch element."""
    dims = tuple(range(1, tensor.dim()))
    mean = tensor.mean(dim=dims, keepdim=True)
    std = tensor.std(dim=dims, keepdim=True, unbiased=False)
    return (tensor - mean) / (std + EPS)


def normalize_due_dates(due_dates: torch.Tensor, order_mask: torch.Tensor) -> torch.Tensor:
    """``[B, T, K]`` due dates normalised with the mean / population std of the valid entries (``order_mask``)
    of the same observation; observations without valid entries are returned unchanged."""
    mask = order_mask.to(due_dates.dtype)
    counts = mask.sum(dim=(1, 2), keepdim=True)
    safe = counts.clamp(min=1.0)
    mean = (due_dates * mask).sum(dim=(1, 2), keepdim=True) / safe
    var = (((due_dates - mean) * mask) ** 2).sum(dim=(1, 2), keepdim=True) / safe
    normalized = (due_dates - mean) / (var.sqrt() + EPS)
    return torch.where(counts > 0, normalized, due_dates)
