"""
Input preprocessing for PatchTST-FM pretraining.
Available utilities include:

- make_cpm_prediction_mask: create prediction masks for the CPM-style pretraining objective.
- mask_aware_normalize: compute mean/std from visible real values and apply asinh normalization
    while zeroing out all masked positions.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset



def _sample_log_uniform_int(
    low: int,
    high: int,
    device: torch.device,
) -> int:
    """Sample one integer approximately from LogUniform([low, high]).

    The returned value is clamped to the inclusive integer interval [low, high].
    We use torch RNG on the same device as the training tensors, so the sample is
    controlled by torch.manual_seed / torch.cuda.manual_seed_all.
    """
    low = int(low)
    high = int(high)

    if low <= 0:
        raise ValueError(f"Log-uniform lower bound must be positive, got low={low}.")
    if high < low:
        raise ValueError(f"Log-uniform upper bound must be >= lower bound, got low={low}, high={high}.")
    if high == low:
        return low

    u = torch.empty((), device=device).uniform_(math.log(float(low)), math.log(float(high)))
    value = int(round(float(torch.exp(u).item())))
    return int(max(low, min(high, value)))
def make_cpm_prediction_mask(
    observed_mask: torch.Tensor,
    padding_mask: torch.Tensor,
    cfg: any,
    terminal_mask_min_patches: int = 0,
    terminal_mask_max_patches: int = 32,
) -> torch.Tensor:
    """
    Vectorized CPM-style prediction mask.

    This version keeps the same high-level objective:
        1. valid observed range only;
        2. optional forecasting-style terminal mask (Now sampled in TIMESTEPS);
        3. random contiguous patch blocks;
        4. approximately cfg.mask_ratio masked positions.
    """
    device = observed_mask.device

    B, T = observed_mask.shape
    L = int(cfg.patch_size)

    if T % L != 0:
        raise ValueError("T must be divisible by patch_size.")

    N = T // L

    block_patches = int(cfg.cpm_blocks)
    if block_patches <= 0:
        raise ValueError("cfg.cpm_blocks must be positive.")

    terminal_mask_min_patches = int(terminal_mask_min_patches)
    terminal_mask_max_patches = int(terminal_mask_max_patches)

    if terminal_mask_min_patches < 0:
        raise ValueError("terminal_mask_min_patches must be non-negative.")

    if terminal_mask_max_patches < terminal_mask_min_patches:
        raise ValueError("terminal_mask_max_patches must be >= terminal_mask_min_patches.")

    # ------------------------------------------------------------
    # 1. Build valid masks.
    # ------------------------------------------------------------
    valid_time = observed_mask & (~padding_mask)
    valid_patch = valid_time.reshape(B, N, L).any(dim=-1)

    valid_time_count = valid_time.sum(dim=1)
    valid_patch_count = valid_patch.sum(dim=1)

    too_short = valid_time_count <= max(2 * L, 4)
    has_valid_patch = valid_patch_count > 0

    # ------------------------------------------------------------
    # 2. Compute first and last valid patch for each sample.
    # ------------------------------------------------------------
    valid_patch_int = valid_patch.to(torch.int64)

    first_patch = torch.argmax(valid_patch_int, dim=1)
    last_patch = N - 1 - torch.argmax(
        torch.flip(valid_patch_int, dims=[1]),
        dim=1,
    )

    first_patch = torch.where(
        has_valid_patch,
        first_patch,
        torch.zeros_like(first_patch),
    )

    last_patch = torch.where(
        has_valid_patch,
        last_patch,
        torch.zeros_like(last_patch),
    )

    # ------------------------------------------------------------
    # 3. Target number of masked patches.
    # ------------------------------------------------------------
    target_patch_count = torch.ceil(
        float(cfg.mask_ratio) * valid_patch_count.to(torch.float32)
    ).to(torch.long)

    target_patch_count = target_patch_count.clamp_min(1)
    target_patch_count = torch.minimum(target_patch_count, valid_patch_count)

    # ------------------------------------------------------------
    # 4. Forecasting-style terminal mask (NOW IN TIMESTEPS).
    # ------------------------------------------------------------
    max_terminal_by_length = ((valid_time_count - 32).clamp_min(0) // L).to(torch.long)

    max_terminal = torch.full((B,), terminal_mask_max_patches, dtype=torch.long, device=device)
    max_terminal = torch.minimum(max_terminal, max_terminal_by_length)
    max_terminal = torch.minimum(max_terminal, target_patch_count)

    min_terminal = torch.full((B,), terminal_mask_min_patches, dtype=torch.long, device=device)
    min_terminal = torch.minimum(min_terminal, max_terminal)

    # Convert patch limits to exact timestep limits
    min_terminal_t = min_terminal * L
    max_terminal_t = max_terminal * L
    terminal_span_t = (max_terminal_t - min_terminal_t + 1).clamp_min(1)

    # Sample exactly how many TIMESTEPS to mask at the end uniformly
    terminal_timesteps = min_terminal_t + torch.floor(
        torch.rand(B, device=device) * terminal_span_t.to(torch.float32)
    ).to(torch.long)

    terminal_timesteps = torch.where(
        has_valid_patch & (~too_short),
        terminal_timesteps,
        torch.zeros_like(terminal_timesteps),
    )

    # Find the exact timestamp index of the last valid observation
    last_time = T - 1 - torch.argmax(
        torch.flip(valid_time.to(torch.int64), dims=[1]),
        dim=1,
    )
    last_time = torch.where(has_valid_patch, last_time, torch.zeros_like(last_time))

    # Build the precise timestep-level terminal mask
    time_idx = torch.arange(T, device=device).view(1, T)
    terminal_start_time = last_time - terminal_timesteps + 1

    terminal_mask_time = (
        (terminal_timesteps[:, None] > 0)
        & (time_idx >= terminal_start_time[:, None])
        & (time_idx <= last_time[:, None])
        & valid_time
    )

    # Project the time-level terminal mask back up to the patch level. 
    # This initializes `pred_patch` so Step 5 knows which patches are currently occupied.
    terminal_mask_patch = terminal_mask_time.reshape(B, N, L).any(dim=-1)
    pred_patch = terminal_mask_patch.clone()

    # ------------------------------------------------------------
    # 5. Random contiguous patch blocks.
    # ------------------------------------------------------------
    max_blocks = max(
        8,
        int(math.ceil(2.0 * N / float(block_patches))),
    )

    offsets = torch.arange(block_patches, device=device).view(1, block_patches)
    batch_idx = torch.arange(B, device=device).view(B, 1)

    valid_span = (last_patch - first_patch + 1).clamp_min(1)

    for _ in range(max_blocks):
        # 当前已经被 terminal mask 或前面 CPM blocks 占用的有效 patch 数。
        current_patch_count = (pred_patch & valid_patch).sum(dim=1)

        # 每个样本还允许新增多少个 patch。
        remaining_patch_count = (
            target_patch_count - current_patch_count
        ).clamp_min(0)

        # 只有还没有达到目标的有效样本才继续生成 block。
        active = (
            (remaining_patch_count > 0)
            & has_valid_patch
            & (~too_short)
        )

        # 为每条序列随机选择一个 block 起点。
        starts = first_patch + torch.floor(
            torch.rand(B, device=device)
            * valid_span.to(torch.float32)
        ).to(torch.long)

        # 构造长度为 block_patches 的候选 block。
        positions = starts[:, None] + offsets
        positions_clamped = positions.clamp(0, N - 1)

        # 删除超过该序列最后一个有效 patch 的位置。
        in_range = positions <= last_patch[:, None]

        block_mask = torch.zeros(
            (B, N),
            dtype=torch.bool,
            device=device,
        )

        block_mask[
            batch_idx.expand_as(positions_clamped)[in_range],
            positions_clamped[in_range],
        ] = True

        # 只统计：
        #   1. 位于本次候选 block 中；
        #   2. 本身是有效 patch；
        #   3. 尚未被之前的 mask 占用；
        #   4. 当前样本仍处于 active 状态
        # 的新增候选位置。
        candidate_patch = (
            block_mask
            & valid_patch
            & (~pred_patch)
            & active[:, None]
        )

        # 对候选 patch 从左到右编号：
        #
        # candidate_patch = [0, 1, 1, 0, 1]
        # candidate_rank  = [0, 1, 2, 2, 3]
        #
        # 如果 remaining_patch_count=2，只接受 rank 1 和 2。
        candidate_rank = candidate_patch.to(torch.long).cumsum(dim=1)

        accepted_patch = candidate_patch & (
            candidate_rank <= remaining_patch_count[:, None]
        )

        # 只加入不超过剩余额度的部分。
        pred_patch = pred_patch | accepted_patch
    # for _ in range(max_blocks):
    #     current_patch_count = (pred_patch & valid_patch).sum(dim=1)

    #     active = (
    #         (current_patch_count < target_patch_count)
    #         & has_valid_patch
    #         & (~too_short)
    #     )

    #     starts = first_patch + torch.floor(
    #         torch.rand(B, device=device) * valid_span.to(torch.float32)
    #     ).to(torch.long)

    #     positions = starts[:, None] + offsets
    #     positions_clamped = positions.clamp(0, N - 1)

    #     in_range = positions <= last_patch[:, None]

    #     block_mask = torch.zeros(
    #         (B, N),
    #         dtype=torch.bool,
    #         device=device,
    #     )

    #     block_mask[
    #         batch_idx.expand_as(positions_clamped)[in_range],
    #         positions_clamped[in_range],
    #     ] = True

    #     pred_patch = pred_patch | (block_mask & active[:, None] & valid_patch)

    # ------------------------------------------------------------
    # 6. Expand patch-level mask back to timestamp-level mask.
    # ------------------------------------------------------------
    # Isolate strictly the random patches added in Step 5 so we don't accidentally
    # full-mask a partial terminal patch upon expansion.
    random_blocks_patch = pred_patch & (~terminal_mask_patch)
    
    # Expand the random blocks into timesteps
    pred_mask = random_blocks_patch[:, :, None].expand(B, N, L).reshape(B, T)

    # Merge the expanded random patches with our highly-precise terminal mask
    pred_mask = pred_mask | terminal_mask_time

    # Never mask padding or missing values.
    pred_mask = pred_mask & valid_time

    # Keep old behavior: very short series receive no prediction mask.
    pred_mask = pred_mask & (~too_short[:, None])

    # ------------------------------------------------------------
    # 7. Safety rule:
    # keep at least two visible points for stable mean/std.
    # ------------------------------------------------------------
    visible_count = (valid_time & (~pred_mask)).sum(dim=1)

    need_fix = (visible_count < 2) & (valid_time_count >= 2)

    valid_rank = valid_time.to(torch.long).cumsum(dim=1)
    first_two_valid = valid_time & (valid_rank <= 2)

    pred_mask = pred_mask & (~(need_fix[:, None] & first_two_valid))

    return pred_mask

def mask_aware_normalize_for_training(
    x: torch.Tensor,
    observed_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    padding_mask: torch.Tensor,
    cfg: any,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mask-aware normalization.

    Statistics are computed only from visible real, non-padding values:
        visible = observed and not prediction-masked and not padding.

    The model input is
        asinh((x - mean) / std)
    but all prediction/missing/padding positions are zeroed.
    """
    missing_mask = (~observed_mask) & (~padding_mask)
    union_mask = pred_mask | missing_mask | padding_mask

    visible_mask = observed_mask & (~pred_mask) & (~padding_mask)
    denom = visible_mask.sum(dim=1, keepdim=True).clamp_min(1).to(x.dtype)

    x_visible = torch.where(visible_mask, x, torch.zeros_like(x))
    mean = x_visible.sum(dim=1, keepdim=True) / denom

    var = torch.where(visible_mask, (x - mean).pow(2), torch.zeros_like(x)).sum(dim=1, keepdim=True) / denom
    std = torch.sqrt(var + cfg.eps)

    target_norm = torch.asinh((x - mean) / std)
    x_norm_input = torch.where(union_mask, torch.zeros_like(target_norm), target_norm)

    B, T = x.shape
    L, N = cfg.patch_size, cfg.num_patches
    patch_padding = padding_mask.reshape(B, N, L).all(dim=-1)

    return x_norm_input, target_norm, union_mask, patch_padding


def mask_aware_normalize_for_inference(
    x: torch.Tensor,
    observed_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    padding_mask: torch.Tensor,
    cfg: PatchTSTFMConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize using only visible historical values.

    Sparse-history handling
    -----------------------
    Rolling windows, especially after multivariate-to-univariate expansion,
    can contain channels with zero or one finite historical value.
    We should still return a valid forecast instead of aborting evaluation.

    Let

        n_i = number of visible finite historical values in series i.

    We use:
      - n_i >= 2: usual mask-aware mean/std;
      - n_i = 1 : mean = the single observed value, std = 1;
      - n_i = 0 : mean = 0, std = 1.

    This keeps forecasts finite and lets GluonTS mask invalid labels during
    metric computation.
    """
    missing_mask = (~observed_mask) & (~padding_mask) & (~pred_mask)
    union_mask = pred_mask | missing_mask | padding_mask

    visible_mask = observed_mask & (~pred_mask) & (~padding_mask)
    count = visible_mask.sum(dim=1, keepdim=True).to(x.dtype)

    x_visible = torch.where(visible_mask, x, torch.zeros_like(x))
    safe_count = count.clamp_min(1.0)

    raw_mean = x_visible.sum(dim=1, keepdim=True) / safe_count
    mean = torch.where(count > 0, raw_mean, torch.zeros_like(raw_mean))

    raw_var = torch.where(visible_mask, (x - mean).pow(2), torch.zeros_like(x)).sum(
        dim=1, keepdim=True
    ) / safe_count

    # If there are fewer than two visible points, variance is not identifiable.
    # Use unit scale rather than sqrt(eps), because sqrt(eps) would make the
    # reverse normalization almost constant and numerically brittle.
    std = torch.where(count >= 2, torch.sqrt(raw_var + cfg.eps), torch.ones_like(raw_var))

    x_norm_all = torch.asinh((x - mean) / std)
    x_norm_input = torch.where(union_mask, torch.zeros_like(x_norm_all), x_norm_all)

    B, T = x.shape
    L, N = cfg.patch_size, cfg.num_patches
    patch_padding = padding_mask.reshape(B, N, L).all(dim=-1)

    return x_norm_input, mean, std, union_mask, patch_padding


def inverse_asinh_normalize(
    x_norm: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Map normalized model outputs back to the original value scale.

    Inverts the preprocessing transform:

        x_norm = asinh((x - mean) / std)
    """
    return torch.sinh(x_norm) * std + mean
