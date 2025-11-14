# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch


def safe_nanmin(x: torch.Tensor, dim=1, keepdim=True):
    """
    Compute nan-min safely.
    Replace NaNs with +inf before min so they don't affect result.
    """
    mask = torch.isnan(x)
    if mask.any():
        x = x.clone()
        x[mask] = torch.inf
    values, _ = torch.min(x, dim=dim, keepdim=keepdim)
    return values


def safe_nanmax(x: torch.Tensor, dim=1, keepdim=True):
    """
    Compute nan-max safely.
    Replace NaNs with -inf before max so they don't affect result.
    """
    mask = torch.isnan(x)
    if mask.any():
        x = x.clone()
        x[mask] = -torch.inf
    values, _ = torch.max(x, dim=dim, keepdim=keepdim)
    return values


def normalize_time_for_ctrope(
    time_values: torch.Tensor,
    attention_mask: torch.Tensor = None,
    seq_length: int = None,
    alpha: float = 1.0,
):
    """
    Normalize raw time_values to the CT-RoPE scale:
        t_scaled ∈ [0, alpha * (seq_len - 1)]

    Args:
        time_values: [B, L] real timestamps
        attention_mask: [B, L] (1=valid, 0=pad)
        seq_length: L (required)
        alpha: scaling factor (default=1.0)

    Returns:
        normalized_time: [B, L] scaled timestamps
        (t_min, t_max) for caching (for inference consistency)
    """

    time_values = time_values.to(torch.float32)

    if attention_mask is not None:
        # Mask padding positions → NaN
        masked_time = time_values.masked_fill(attention_mask == 0, float("nan"))
    else:
        masked_time = time_values

    # Compute per-sample min/max (ignoring NaNs)
    t_min = safe_nanmin(masked_time, dim=1, keepdim=True)
    t_max = safe_nanmax(masked_time, dim=1, keepdim=True)
    denom = (t_max - t_min).clamp(min=1e-8)

    # Normalize to [0, 1]
    t_norm = (time_values - t_min) / denom

    # Scale to [0, alpha * (seq_len - 1)]
    max_range = alpha * float(seq_length - 1)
    t_scaled = t_norm * max_range

    # Replace NaNs → 0
    t_scaled = torch.nan_to_num(t_scaled, nan=0.0)

    return t_scaled, t_min, t_max