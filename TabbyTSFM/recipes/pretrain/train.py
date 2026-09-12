#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tabby-Pretrain mixed-data training entry point.

The released 165K recipe uses BLAST, pre-generated KernelSynth Arrow shards,
and online CauKer V2 samples. The optional GIFT loader is retained only for
backward compatibility and is disabled by default; it was not used to train
Tabby-Pretrain.
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
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset

from tabby.models.PatchTSTFM import PatchTSTFM, PatchTSTFMConfig
from tabby.data.real_data import (
    BlastRealIterableDataset,
    concat_mixed_batches,
    discover_blast_shards,
)
from tabby.data.gift_eval_pretrain import (
    GiftEvalPretrainRealIterableDataset,
    discover_hf_dataset_dirs as discover_gift_dataset_dirs,
)
from tabby.data.synthetic_data import (
    ArrowSyntheticIterableDataset,
    ArrowCauKerSCMIterableDataset,
    OnlineCauKerIterableDataset,
    discover_arrow_files,
)
from tabby.utils.input_preprocessing import (
    make_cpm_prediction_mask,
    mask_aware_normalize_for_training as mask_aware_normalize,
)

# 1. Distributed utilities
# =============================================================================

def setup_distributed() -> Tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    else:
        rank, world_size, local_rank = 0, 1, 0
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def set_seed(seed: int, rank: int = 0) -> None:
    seed = seed + rank * 100_003
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def strip_state_dict_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip common wrapper prefixes (DDP 'module.', torch.compile '_orig_mod.') from checkpoint keys."""
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "_orig_mod."):
            while new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def resolve_checkpoint_bin_path(path: Path) -> Path:
    """Accept either a checkpoint directory (containing pytorch_model.bin) or a direct .bin path."""
    if path.is_dir():
        bin_path = path / "pytorch_model.bin"
        if not bin_path.exists():
            raise FileNotFoundError(f"No pytorch_model.bin found inside checkpoint directory: {path}")
        return bin_path
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    return path


def load_init_checkpoint(
    init_checkpoint: str,
    model: nn.Module,
    device: torch.device,
    strict: bool = True,
) -> None:
    """Warm-start model weights only (no optimizer state, no step counter) from a checkpoint.

    This is distinct from --resume: it does not restore training progress, LR schedule
    position, or optimizer moments. Use it to initialize training from a pretrained
    checkpoint (e.g. a different run, an earlier stage, or a released model) while still
    starting the optimizer and step counter fresh.
    """
    ckpt_path = resolve_checkpoint_bin_path(Path(init_checkpoint))
    if is_main_process():
        print(f"Initializing model weights from checkpoint: {ckpt_path.resolve()}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state_dict = strip_state_dict_prefixes(state_dict)

    target_model = unwrap_model(model)
    missing, unexpected = target_model.load_state_dict(state_dict, strict=strict)

    if is_main_process():
        if missing:
            print(f"  init_checkpoint: {len(missing)} missing keys (not found in checkpoint): {missing[:10]}"
                  + (" ..." if len(missing) > 10 else ""))
        if unexpected:
            print(f"  init_checkpoint: {len(unexpected)} unexpected keys (ignored from checkpoint): {unexpected[:10]}"
                  + (" ..." if len(unexpected) > 10 else ""))
        if not missing and not unexpected:
            print("  init_checkpoint: all keys matched exactly.")

    del ckpt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# 2. Parameter-free multi-exit wrapper for intermediate-layer supervision
# =============================================================================

def resolve_flow_exit_layers(
    num_layers: int,
    explicit_layers: Optional[List[int]],
    num_exits: int,
) -> List[int]:
    """Resolve 0-indexed transport endpoints and 1-indexed block exits.

    Exit 0 decodes the patch embeddings before any Transformer block. Exit K
    decodes the output after all K blocks. When explicit exits are omitted,
    `num_exits` points (including 0 and K) are placed approximately uniformly.
    """
    if num_layers <= 0:
        raise ValueError("num_layers must be positive.")

    if explicit_layers is not None:
        exits = sorted(set(int(k) for k in explicit_layers))
    else:
        if num_exits < 2:
            raise ValueError("--num_flow_exits must be at least 2.")
        exits = sorted(
            set(
                int(round(i * num_layers / (num_exits - 1)))
                for i in range(num_exits)
            )
        )

    exits = sorted(set([0, num_layers, *exits]))
    bad = [k for k in exits if k < 0 or k > num_layers]
    if bad:
        raise ValueError(
            f"Every --flow_exit_layers value must be in [0, {num_layers}], "
            f"got invalid values {bad}."
        )
    return exits


class PatchTSTFMWithFlowExits(PatchTSTFM):
    """PatchTSTFM with shared-head decoding at selected intermediate layers.

    This subclass adds no trainable modules. Its state_dict is therefore
    identical to the original PatchTSTFM state_dict, so existing checkpoints
    and the original evaluation model remain strictly compatible.
    """

    def __init__(self, cfg: PatchTSTFMConfig, exit_layers: List[int]):
        super().__init__(cfg)
        self.exit_layers = tuple(sorted(set(exit_layers)))
        if not self.exit_layers or self.exit_layers[-1] != cfg.num_layers:
            raise ValueError("The terminal layer must be included in exit_layers.")

    def _decode_quantiles(
        self,
        h: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Apply the original shared monotone quantile head to one hidden state."""
        cfg = self.config
        N = cfg.num_patches
        L = cfg.patch_size
        K = cfg.num_quantiles

        h = self.backbone.out_layer(h)
        q_raw = h.reshape(batch_size, N, K + 1, L).permute(0, 2, 1, 3)
        q = q_raw[:, 0, :, :].unsqueeze(1) + torch.cumsum(
            nn.functional.softplus(q_raw[:, 1:, :, :]) / K,
            dim=1,
        )
        return q.permute(0, 2, 3, 1).reshape(
            batch_size,
            cfg.context_length,
            K,
        )

    def forward(
        self,
        x_norm: torch.Tensor,
        union_mask: torch.Tensor,
        patch_key_padding_mask: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        """Return `{layer_index: quantile_prediction}` for selected exits."""
        cfg = self.config
        B, T = x_norm.shape
        if T != cfg.context_length:
            raise ValueError(f"Expected T={cfg.context_length}, got {T}.")

        L = cfg.patch_size
        N = cfg.num_patches
        bb = self.backbone

        value_patch = x_norm.reshape(B, N, L)
        mask_patch = union_mask.reshape(B, N, L).to(x_norm.dtype)
        h = bb.in_layer(torch.cat([value_patch, 1.0 - mask_patch], dim=-1))
        h = bb.pos_embed(h)

        # Same key-only attention mask as the original PatchTSTFM wrapper.
        attn_mask = ~patch_key_padding_mask.bool()[:, None, None, :]

        exit_preds: Dict[int, torch.Tensor] = {}
        if 0 in self.exit_layers:
            exit_preds[0] = self._decode_quantiles(h, B)

        for layer_idx, block in enumerate(bb.blocks, start=1):
            h = block(h, attn_mask)
            if layer_idx in self.exit_layers:
                exit_preds[layer_idx] = self._decode_quantiles(h, B)

        return exit_preds


# =============================================================================
# 5. Masking, normalization, and loss
# =============================================================================


# def pinball_loss(
#     q_hat: torch.Tensor,
#     target_norm: torch.Tensor,
#     loss_mask: torch.Tensor,
#     quantiles: torch.Tensor,
# ) -> torch.Tensor:
#     """Pinball loss averaged over masked timestamps and quantile levels."""
#     y = target_norm.unsqueeze(-1)
#     err = y - q_hat
#     q = quantiles.view(1, 1, -1)

#     loss = torch.maximum(q * err, (q - 1.0) * err)
#     mask = loss_mask.unsqueeze(-1).to(loss.dtype)
#     denom = mask.sum().clamp_min(1.0) * q_hat.shape[-1]
#     return (loss * mask).sum() / denom
def pinball_loss(
    q_hat: torch.Tensor,
    target_norm: torch.Tensor,
    loss_mask: torch.Tensor,
    quantiles: torch.Tensor,
    length_weight_alpha: float = 0.5,
) -> torch.Tensor:
    """Length-balanced pinball loss.

    First compute the mean pinball loss independently for every
    sequence, then aggregate sequences using:

        weight_i = masked_count_i ** length_weight_alpha

    alpha = 0.0:
        every sequence has equal weight.

    alpha = 0.5:
        square-root length weighting.

    alpha = 1.0:
        equivalent to averaging over all masked timestamps.
    """
    if not 0.0 <= length_weight_alpha <= 1.0:
        raise ValueError(
            "length_weight_alpha must be in [0, 1], "
            f"got {length_weight_alpha}."
        )

    active = loss_mask.bool()              # [B, T]
    active_3d = active.unsqueeze(-1)       # [B, T, 1]

    # 防止 missing/padding 位置上的 NaN 进入 loss。
    safe_target = torch.where(
        active,
        target_norm,
        torch.zeros_like(target_norm),
    )

    safe_q_hat = torch.where(
        active_3d,
        q_hat,
        torch.zeros_like(q_hat),
    )

    y = safe_target.unsqueeze(-1)          # [B, T, 1]
    q = quantiles.view(1, 1, -1)           # [1, 1, Q]

    err = y - safe_q_hat

    raw_loss = torch.maximum(
        q * err,
        (q - 1.0) * err,
    )                                      # [B, T, Q]

    raw_loss = torch.where(
        active_3d,
        raw_loss,
        torch.zeros_like(raw_loss),
    )

    # 每条序列实际参加 loss 的时间点数。
    masked_count = active.sum(dim=1)        # [B]
    valid_series = masked_count > 0

    if not valid_series.any():
        # 保留计算图，但梯度为零。
        return q_hat.sum() * 0.0

    masked_count_float = masked_count.to(raw_loss.dtype)

    # 每条序列先独立平均。
    per_series_sum = raw_loss.sum(dim=(1, 2))

    per_series_denom = (
        masked_count_float
        * q_hat.shape[-1]
    ).clamp_min(1.0)

    per_series_loss = (
        per_series_sum
        / per_series_denom
    )

    per_series_loss = per_series_loss[valid_series]
    valid_count = masked_count_float[valid_series]

    # 折中长度权重。
    series_weights = valid_count.pow(
        length_weight_alpha
    )

    return (
        per_series_loss * series_weights
    ).sum() / series_weights.sum().clamp_min(1e-12)


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    length_weight_alpha: float = 0.5,
) -> torch.Tensor:
    """Length-balanced masked MSE used by the flow-path constraint."""
    active = loss_mask.bool()
    active_3d = active.unsqueeze(-1)

    safe_prediction = torch.where(
        active_3d,
        prediction,
        torch.zeros_like(prediction),
    )
    safe_target = torch.where(
        active_3d,
        target,
        torch.zeros_like(target),
    )
    squared_error = torch.where(
        active_3d,
        (safe_prediction - safe_target).square(),
        torch.zeros_like(safe_prediction),
    )

    masked_count = active.sum(dim=1)
    valid_series = masked_count > 0
    if not valid_series.any():
        return prediction.sum() * 0.0

    masked_count_float = masked_count.to(squared_error.dtype)
    per_series_denom = (
        masked_count_float * prediction.shape[-1]
    ).clamp_min(1.0)
    per_series_mse = squared_error.sum(dim=(1, 2)) / per_series_denom

    per_series_mse = per_series_mse[valid_series]
    series_weights = masked_count_float[valid_series].pow(length_weight_alpha)
    return (
        per_series_mse * series_weights
    ).sum() / series_weights.sum().clamp_min(1e-12)


def quantile_flow_loss(
    exit_preds: Dict[int, torch.Tensor],
    target_norm: torch.Tensor,
    loss_mask: torch.Tensor,
    quantiles: torch.Tensor,
    num_layers: int,
    lambda_ds: float,
    lambda_fm: float,
    ds_gamma: float,
    length_weight_alpha: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Final pinball + intermediate supervision + flow-path consistency.

    Let K be the number of Transformer layers and E the selected exits:

        L = L_final
          + lambda_ds * sum_{k in E, 0<k<K} (k/K)^gamma L_pinball(q_k, y)
          + lambda_fm * sum_{k in E, 0<k<K}
                MSE(q_k, (1-k/K) sg(q_0) + (k/K) sg(q_K)).

    Both endpoints are stop-gradient targets. Thus the flow term shapes only
    intermediate representations; the final endpoint is still trained by the
    original pinball objective.
    """
    K = num_layers
    if K not in exit_preds:
        raise ValueError(f"Terminal exit {K} is missing.")

    q_final = exit_preds[K]
    terminal = pinball_loss(
        q_hat=q_final,
        target_norm=target_norm,
        loss_mask=loss_mask,
        quantiles=quantiles,
        length_weight_alpha=length_weight_alpha,
    )

    deep_sup = q_final.new_zeros(())
    if lambda_ds > 0.0:
        for k in sorted(exit_preds):
            if k in (0, K):
                continue
            deep_sup = deep_sup + (k / K) ** ds_gamma * pinball_loss(
                q_hat=exit_preds[k],
                target_norm=target_norm,
                loss_mask=loss_mask,
                quantiles=quantiles,
                length_weight_alpha=length_weight_alpha,
            )

    flow_match = q_final.new_zeros(())
    if lambda_fm > 0.0:
        if 0 not in exit_preds:
            raise ValueError("Flow matching requires exit 0 as the path start.")
        q_start = exit_preds[0].detach()
        q_end = q_final.detach()
        for k in sorted(exit_preds):
            if k in (0, K):
                continue
            t_k = k / K
            path_target = (1.0 - t_k) * q_start + t_k * q_end
            flow_match = flow_match + masked_mse(
                prediction=exit_preds[k],
                target=path_target,
                loss_mask=loss_mask,
                length_weight_alpha=length_weight_alpha,
            )

    total = terminal + lambda_ds * deep_sup + lambda_fm * flow_match
    parts = {
        "terminal": terminal.detach(),
        "deep_sup": deep_sup.detach(),
        "flow_match": flow_match.detach(),
    }
    return total, parts

# =============================================================================
# 6. Optimizer and checkpointing
# =============================================================================

def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    betas: Tuple[float, float],
) -> torch.optim.Optimizer:
    """AdamW with no decay on bias, norms, and positional embedding."""
    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower() or "pos_embed" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=betas,
        fused=torch.cuda.is_available(),
    )



# def lr_at_step(
#     step: int,
#     total_steps: int,
#     warmup_steps: int,
#     peak_lr: float,
#     min_lr: float,
#     scheduler: str = "cosine",
#     min_lr_steps: int = 5_000,
#     num_cycles: int = 2,
#     T: float = 10.0,
# ) -> float:
#     """Linear warmup plus cosine or cyclic WSD (Warmup-Stable-Decay) with exponential fall."""

#     # Global linear warmup (happens only once at the beginning of training)
#     if step < warmup_steps:
#         return peak_lr * float(step + 1) / float(max(1, warmup_steps))

#     if scheduler == "cosine":
#         progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
#         progress = min(1.0, max(0.0, progress))
#         cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
#         return min_lr + (peak_lr - min_lr) * cosine

#     elif scheduler == "wsd":
#         # 1. Calculate length of a single cycle and current position within it
#         cycle_length = total_steps // max(1, num_cycles)
#         cycle_step = step % cycle_length

#         # 2. Determine where the stable phase ends and the rapid decay begins
#         decay_start = cycle_length - min_lr_steps

#         if cycle_step < decay_start:
#             # Stable phase: Hold the peak learning rate
#             return peak_lr
#         else:
#             # Decay phase: Exponentially fall to min_lr
#             relative_step = float(cycle_step - decay_start)

#             # Apply the exponential decay formula
#             return min_lr + (peak_lr - min_lr) * math.exp(-relative_step / T)

#     else:
#         raise ValueError(f"Unknown scheduler: {scheduler}")

def lr_at_step(
    step: int,
    total_steps: int,
    warmup_steps: int,
    peak_lr: float,
    min_lr: float,
    scheduler: str = "cosine",
    decay_steps: int = 5_000,
    num_cycles: int = 1,
) -> float:
    """
    Compute the learning rate for a given global optimization step.

    Supported schedules:
      1. cosine:
         linear warmup followed by cosine decay over the entire run.

      2. wsd:
         linear warmup, stable learning rate, and a cosine decay
         during the final ``decay_steps`` steps of every cycle.

    Notes:
      - ``step`` is the global optimizer step, not the number of steps
        since resuming.
      - ``total_steps`` is the absolute final training step.
      - For a continuation ending at the Tabby release step, set:
            total_steps=165000
            num_cycles=1
            decay_steps=10000
    """
    if total_steps <= 0:
        raise ValueError(
            f"total_steps must be positive, got {total_steps}."
        )

    if warmup_steps < 0:
        raise ValueError(
            f"warmup_steps must be non-negative, got {warmup_steps}."
        )

    if decay_steps <= 0:
        raise ValueError(
            f"decay_steps must be positive, got {decay_steps}."
        )

    if num_cycles <= 0:
        raise ValueError(
            f"num_cycles must be positive, got {num_cycles}."
        )

    if peak_lr < min_lr:
        raise ValueError(
            f"peak_lr ({peak_lr}) must be >= min_lr ({min_lr})."
        )

    # -------------------------------------------------------------
    # 1. Linear warmup
    # -------------------------------------------------------------
    # When warmup_steps=0, this branch is skipped.
    if step < warmup_steps:
        warmup_progress = float(step + 1) / float(
            max(1, warmup_steps)
        )
        return peak_lr * warmup_progress

    # -------------------------------------------------------------
    # 2. Standard global cosine schedule
    # -------------------------------------------------------------
    if scheduler == "cosine":
        decay_duration = max(
            1,
            total_steps - warmup_steps,
        )

        progress = float(
            step - warmup_steps
        ) / float(decay_duration)

        progress = min(
            1.0,
            max(0.0, progress),
        )

        cosine_factor = 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

        return min_lr + (
            peak_lr - min_lr
        ) * cosine_factor

    # -------------------------------------------------------------
    # 3. Warmup-Stable-Decay schedule
    # -------------------------------------------------------------
    # if scheduler == "wsd":
    #     cycle_length = total_steps // num_cycles

    #     if cycle_length <= 0:
    #         raise ValueError(
    #             "total_steps must be >= num_cycles."
    #         )

    #     if decay_steps > cycle_length:
    #         raise ValueError(
    #             f"decay_steps ({decay_steps}) cannot exceed "
    #             f"cycle_length ({cycle_length})."
    #         )

    #     # Position inside the current cycle.
    #     cycle_step = step % cycle_length

    #     # The last decay_steps updates form the decay stage.
    #     decay_start = cycle_length - decay_steps

    #     # Stable stage.
    #     if cycle_step < decay_start:
    #         return peak_lr

    #     # Decay stage:
    #     # decay_index ranges from 0 to decay_steps - 1.
    #     decay_index = cycle_step - decay_start

    #     if decay_steps == 1:
    #         return min_lr

    #     progress = float(decay_index) / float(
    #         decay_steps - 1
    #     )

    #     progress = min(
    #         1.0,
    #         max(0.0, progress),
    #     )

    #     # Smooth cosine decay from peak_lr to min_lr.
    #     cosine_factor = 0.5 * (
    #         1.0 + math.cos(math.pi * progress)
    #     )

    #     return min_lr + (
    #         peak_lr - min_lr
    #     ) * cosine_factor
    if scheduler == "wsd":
        cycle_length = total_steps // num_cycles

        if cycle_length <= 0:
            raise ValueError(
                "total_steps must be >= num_cycles."
            )

        if decay_steps > cycle_length:
            raise ValueError(
                f"decay_steps ({decay_steps}) cannot exceed "
                f"cycle_length ({cycle_length})."
            )

        # ---------------------------------------------------------
        # 1. Determine the current cycle.
        #
        # cycle_index is zero-based:
        #   first cycle  -> cycle_index = 0
        #   second cycle -> cycle_index = 1
        #   third cycle  -> cycle_index = 2
        #
        # min(..., num_cycles - 1) ensures that any remainder
        # caused by integer division belongs to the final cycle.
        # ---------------------------------------------------------
        cycle_index = min(
            step // cycle_length,
            num_cycles - 1,
        )

        cycle_start = cycle_index * cycle_length

        # Put all remainder steps into the final cycle.
        if cycle_index == num_cycles - 1:
            cycle_end = total_steps
        else:
            cycle_end = cycle_start + cycle_length

        current_cycle_length = cycle_end - cycle_start
        cycle_step = step - cycle_start

        # ---------------------------------------------------------
        # 2. Reduce the peak learning rate in every new cycle.
        #
        # cycle 1: peak_lr / 2**0 = peak_lr
        # cycle 2: peak_lr / 2**1 = peak_lr / 2
        # cycle 3: peak_lr / 2**2 = peak_lr / 4
        # ---------------------------------------------------------
        cycle_peak_lr = peak_lr / (2.0 ** cycle_index)

        # The last decay_steps updates form the decay stage.
        decay_start = current_cycle_length - decay_steps

        # Stable stage.
        if cycle_step < decay_start:
            return cycle_peak_lr

        # Decay stage.
        decay_index = cycle_step - decay_start

        if decay_steps == 1:
            return min_lr

        progress = float(decay_index) / float(
            decay_steps - 1
        )

        progress = min(
            1.0,
            max(0.0, progress),
        )

        cosine_factor = 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

        # Cosine decay from the current cycle's peak to min_lr.
        return min_lr + (
            cycle_peak_lr - min_lr
        ) * cosine_factor

    raise ValueError(
        f"Unknown scheduler: {scheduler}"
    )

def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: PatchTSTFMConfig,
    args: argparse.Namespace,
) -> None:
    if not is_main_process():
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / f"step_{step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": asdict(cfg),
        "args": vars(args),
    }
    torch.save(payload, ckpt_dir / "pytorch_model.bin")

    with open(ckpt_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    latest = output_dir / "latest"
    tmp = output_dir / "latest.tmp"
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    tmp.symlink_to(ckpt_dir.name, target_is_directory=True)
    tmp.replace(latest)


# =============================================================================
# 7. Training
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # BLAST data.  The legacy --real-* option names remain accepted aliases.
    parser.add_argument(
        "--blast_data_root",
        "--real_data_root",
        dest="real_data_root",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--blast_ratio",
        "--real_ratio",
        dest="real_ratio",
        type=float,
        required=True,
        help="Fraction of each per-device batch drawn from BLAST.",
    )
    parser.add_argument(
        "--num_blast_workers",
        "--num_real_workers",
        dest="num_real_workers",
        type=int,
        default=4,
    )
    parser.add_argument("--num_synth_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument(
        "--min_real_length",
        type=int,
        default=2,
        help="Minimum finite observations required in BLAST, Kernel, or mixup crops.",
    )
    parser.add_argument(
        "--min_blast_sample_length",
        "--min_real_sample_length",
        dest="min_real_sample_length",
        type=int,
        default=96,
    )
    parser.add_argument(
        "--max_blast_sample_length",
        "--max_real_sample_length",
        dest="max_real_sample_length",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--real_domain_balance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deprecated compatibility flag; BLAST is already pattern-balanced.",
    )
    parser.add_argument(
        "--blast_max_sample_attempts",
        "--real_max_sample_attempts",
        dest="real_max_sample_attempts",
        type=int,
        default=32,
        help="Maximum retries when a sampled BLAST row is invalid.",
    )

    # Optional compatibility stream. This stream is disabled in the released
    # Tabby-Pretrain recipe and was not used for the 165K checkpoint.
    parser.add_argument(
        "--gift_data_root",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--gift_ratio",
        type=float,
        default=0.0,
        help=(
            "Optional GIFT-Eval-Pretrain batch fraction. The released "
            "Tabby-Pretrain recipe keeps this at 0."
        ),
    )
    parser.add_argument("--num_gift_workers", type=int, default=4)
    parser.add_argument("--min_gift_sample_length", type=int, default=96)
    parser.add_argument("--max_gift_sample_length", type=int, default=8192)
    parser.add_argument(
        "--gift_domain_balance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Uniformly sample GIFT-Eval-Pretrain sub-datasets so large domains do not dominate.",
    )
    parser.add_argument(
        "--gift_max_sample_attempts",
        type=int,
        default=32,
        help="Maximum retries when a sampled GIFT-Eval-Pretrain row is invalid.",
    )

    # Pre-generated Kernel/Arrow crop lengths.
    parser.add_argument("--min_synth_sample_length", type=int, default=96)
    parser.add_argument("--max_synth_sample_length", type=int, default=8192)

    # Online multi-base CauKer V2 data.
    parser.add_argument(
        "--synthetic_mode",
        type=str,
        default="online_cauker",
        choices=["online_cauker", "arrow", "arrow_scm", "kernel_mixed"],
        help=(
            "Synthetic data source: "
            "online_cauker generates online CauKer data; "
            "arrow reads pre-generated Arrow files directly; "
            "arrow_scm reads Arrow files as SCM root nodes and applies a CauKer-style DAG; "
            "kernel_mixed uses Arrow files as kernel synthetic SCM root nodes (requires --use_cauker_V2 for the CauKer V2 online stream)."
             ),
    )
    parser.add_argument(
        "--synthetic_arrow_root",
        type=str,
        default=None,
        help=(
            "Folder containing pre-generated KernelSynth .arrow files. This "
            "repository provides the reader, not the offline shard generator."
        ),
    )
    parser.add_argument(
        "--synthetic_arrow_balance_files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Uniformly cycle synthetic Arrow files before sampling examples inside each file.",
    )
    parser.add_argument(
        "--mixup_arrow_root",
        type=str,
        default=None,
        help="Folder containing real-kernel mixup .arrow shards generated by generate_real_kernel_mixup_arrow.py.",
    )
    parser.add_argument(
        "--mixup_ratio",
        type=float,
        default=0.0,
        help="Fraction of each per-device batch drawn from real-kernel mixup Arrow shards.",
    )
    parser.add_argument(
        "--mixup_arrow_balance_files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Uniformly cycle mixup Arrow shards before sampling examples inside each file.",
    )
    parser.add_argument(
        "--num_mixup_workers",
        type=int,
        default=None,
        help="DataLoader workers for the mixup stream. Defaults to --num_synth_workers when omitted.",
    )
    parser.add_argument("--min_series_length", type=int, default=96)
    parser.add_argument("--max_series_length", type=int, default=2048)
    parser.add_argument("--length_sampling", type=str, default="uniform", choices=["uniform", "log_uniform"])
    parser.add_argument(
        "--cauker_features",
        type=int,
        default=4,
        help="V2: number of observed SCM channels; each channel becomes one univariate training sample.",
    )
    parser.add_argument(
        "--cauker_max_parents",
        type=int,
        default=6,
        help="Maximum number of DAG parents for each synthetic SCM channel.",
    )
    parser.add_argument(
        "--cauker_num_nodes",
        type=int,
        default=18,
        help="Backward-compatible name. In CauKer V2 this is the number of primitive base sources, i.e. latent_dim.",
    )
    parser.add_argument("--cauker_v2_weights_json", type=str, default=None)
    parser.add_argument("--use_cauker_V2", action="store_true", default=False,
                        help="Enable CauKer V2 online data generation as an additional synthetic data stream.")
    parser.add_argument("--cauker_V2_ratio", type=float, default=0.2,
                        help="Fraction of per-device batch drawn from the CauKer V2 synthetic stream (requires --use_cauker_V2).")

    # Model.
    parser.add_argument("--context_length", type=int, default=8192)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=20)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_quantiles", type=int, default=99)
    parser.add_argument("--mask_ratio", type=float, default=0.40)
    parser.add_argument(
        "--cpm_blocks",
        type=int,
        default=8,
        help="CPM contiguous mask block length in patches. Random block starts may overlap.",
    )
    parser.add_argument(
        "--terminal_mask_min",
        type=int,
        default=0,
        help="Minimum terminal forecasting-mask length in patches.",
    )
    parser.add_argument(
        "--terminal_mask_max",
        type=int,
        default=32,
        help="Maximum terminal forecasting-mask length in patches.",
    )
    parser.add_argument(
        "--flow_supervision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable parameter-free intermediate exits with deep supervision "
            "and displacement-interpolation flow matching."
        ),
    )
    parser.add_argument(
        "--flow_exit_layers",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Transformer exits to decode, using 0 for the pre-block embedding "
            "and num_layers for the terminal output. Endpoints are added "
            "automatically. Default: approximately uniform exits."
        ),
    )
    parser.add_argument(
        "--num_flow_exits",
        type=int,
        default=5,
        help=(
            "Number of approximately uniform exits, including endpoints, when "
            "--flow_exit_layers is omitted. With 20 layers and 5 exits this "
            "gives [0, 5, 10, 15, 20]."
        ),
    )
    parser.add_argument(
        "--lambda_ds",
        type=float,
        default=0.5,
        help="Weight of intermediate-layer pinball supervision.",
    )
    parser.add_argument(
        "--lambda_fm",
        type=float,
        default=0.1,
        help="Weight of intermediate flow-path consistency.",
    )
    parser.add_argument(
        "--ds_gamma",
        type=float,
        default=1.0,
        help="Depth weighting exponent (k / num_layers) ** ds_gamma.",
    )

    # Optimization.
    parser.add_argument("--total_steps", type=int, default=165_000)
    parser.add_argument("--warmup_steps", type=int, default=5_000)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "wsd"])
    parser.add_argument("--num_cycles" , type=int, default=2, help="Number of cycles for WSD scheduler.")
    parser.add_argument(
        "--decay_steps",
        type=int,
        default=5_000,
        help=(
            "Number of optimizer steps used by the decay stage "
            "at the end of each WSD cycle."
        ),
    )
    parser.add_argument("--per_device_batch_size", type=int, default=16)
    parser.add_argument("--grad_accum_steps", type=int, default=32)
    parser.add_argument("--peak_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])

    # System.
    parser.add_argument("--output_dir", type=str, default="./outputs/tabby-pretrain-165k")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=100, help="Print averaged training loss every N steps.")
    parser.add_argument(
        "--log_file",
        type=str,
        default="train_log.jsonl",
        help="Training metric log path. Relative paths are written under output_dir.",
    )
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from the latest checkpoint found in --output_dir.",
    )
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint used to initialize model weights before training starts, "
            "instead of random initialization. Accepts either a checkpoint directory "
            "(containing pytorch_model.bin) or a direct path to a .bin file. Only the model "
            "weights are loaded — optimizer state, step counter, and LR schedule position all "
            "start fresh. Mutually exclusive with --resume (which restores full training state "
            "from --output_dir instead)."
        ),
    )
    parser.add_argument(
        "--init_checkpoint_strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require an exact key match between --init_checkpoint and the model. Set "
            "--no-init_checkpoint_strict to allow partial loading (e.g. warm-starting a "
            "backbone while leaving new/changed heads randomly initialized)."
        ),
    )
    parser.add_argument(
        "--skip_bad_grad_update",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip optimizer.step() when grad norm is non-finite or extremely large.",
    )

    parser.add_argument(
        "--skip_grad_norm_threshold",
        type=float,
        default=50,
        help="Skip optimizer.step() if the pre-clipping grad norm is larger than this value.",
    )
    parser.add_argument(
        "--length_weight_alpha",
        type=float,
        default=0.5,
        help=(
            "Sequence-length weighting exponent for pinball loss. "
            "0.0 means equal weight per sequence; "
            "0.5 means square-root weighting; "
            "1.0 means equal weight per masked timestamp."
        ),
    )

    return parser.parse_args()


def make_loader(
    dataset: IterableDataset,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
) -> Optional[DataLoader]:
    if batch_size <= 0:
        return None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=True,
    )


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    set_seed(args.seed, rank)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    if args.context_length % args.patch_size != 0:
        raise ValueError("--context_length must be divisible by --patch_size.")
    if args.min_series_length < 4:
        raise ValueError("--min_series_length must be at least 4.")
    if args.max_series_length < args.min_series_length:
        raise ValueError("--max_series_length must be >= --min_series_length.")
    if args.max_series_length > args.context_length:
        raise ValueError("--max_series_length must be <= --context_length.")
    if args.log_every <= 0:
        raise ValueError("--log_every must be positive.")
    if args.synthetic_mode in {"arrow", "arrow_scm", "kernel_mixed"} and not args.synthetic_arrow_root:
        raise ValueError(
            "--synthetic_arrow_root is required when --synthetic_mode is 'arrow', 'arrow_scm' or 'kernel_mixed'."
        )
    for option_name, ratio in (
        ("--blast_ratio", args.real_ratio),
        ("--gift_ratio", args.gift_ratio),
        ("--mixup_ratio", args.mixup_ratio),
        ("--cauker_V2_ratio", args.cauker_V2_ratio),
    ):
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"{option_name} must be in [0, 1].")
    if args.mixup_ratio > 0.0 and not args.mixup_arrow_root:
        raise ValueError("--mixup_arrow_root is required when --mixup_ratio > 0.")
    if args.real_ratio > 0.0 and not args.real_data_root:
        raise ValueError("--blast_data_root is required when --blast_ratio > 0.")
    if args.gift_ratio > 0.0 and not args.gift_data_root:
        raise ValueError("--gift_data_root is required when --gift_ratio > 0.")
    if args.num_mixup_workers is None:
        args.num_mixup_workers = args.num_synth_workers
    for option_name, num_workers in (
        ("--num_blast_workers", args.num_real_workers),
        ("--num_synth_workers", args.num_synth_workers),
        ("--num_mixup_workers", args.num_mixup_workers),
        ("--num_gift_workers", args.num_gift_workers),
    ):
        if num_workers < 0:
            raise ValueError(f"{option_name} must be non-negative.")
    active_cauker_V2_ratio = args.cauker_V2_ratio if args.use_cauker_V2 else 0.0

    explicit_ratio_sum = (
        args.real_ratio
        + args.gift_ratio
        + args.mixup_ratio
        + active_cauker_V2_ratio
    )
    if explicit_ratio_sum > 1.0 + 1e-8:
        raise ValueError(
            "blast_ratio + gift_ratio + mixup_ratio + cauker_V2_ratio must be <= 1.0 "
            "(where cauker_V2_ratio counts only when --use_cauker_V2 is set)."
        )
    synth_ratio = max(0.0, 1.0 - explicit_ratio_sum)

    length_ranges = (
        ("BLAST", args.min_real_sample_length, args.max_real_sample_length),
        ("Kernel/Arrow", args.min_synth_sample_length, args.max_synth_sample_length),
        ("GIFT-Eval-Pretrain", args.min_gift_sample_length, args.max_gift_sample_length),
    )
    for source_name, min_length, max_length in length_ranges:
        if min_length <= 0:
            raise ValueError(f"{source_name} minimum sample length must be positive.")
        if max_length < min_length:
            raise ValueError(f"{source_name} maximum sample length must be >= its minimum.")
        if max_length > args.context_length:
            raise ValueError(f"{source_name} maximum sample length must be <= --context_length.")
    if args.max_real_sample_length > 4096:
        raise ValueError("BLAST maximum sample length must be <= 4096.")

    if args.terminal_mask_min < 0:
        raise ValueError("--terminal_mask_min must be non-negative.")
    if args.terminal_mask_max < args.terminal_mask_min:
        raise ValueError("--terminal_mask_max must be >= --terminal_mask_min.")
    if args.terminal_mask_max > args.context_length // args.patch_size:
        raise ValueError("--terminal_mask_max must be <= --context_length / --patch_size.")
    if args.lambda_ds < 0.0 or args.lambda_fm < 0.0:
        raise ValueError("--lambda_ds and --lambda_fm must be non-negative.")
    if args.ds_gamma < 0.0:
        raise ValueError("--ds_gamma must be non-negative.")
    if args.init_checkpoint and args.resume:
        raise ValueError(
            "--init_checkpoint and --resume are mutually exclusive: --resume already restores "
            "model weights (plus optimizer state and step count) from --output_dir/latest. "
            "Use --init_checkpoint only when starting a fresh run warm-started from another "
            "checkpoint's weights."
        )

    # Convert user-level mixture ratios into integer per-device batch sizes.
    # Largest-remainder rounding keeps the five stream sizes summing exactly to
    # --per_device_batch_size.
    stream_ratios = {
        "real": args.real_ratio,       # BLAST (legacy internal name)
        "synth": synth_ratio,          # kernel/arrow/online stream selected by --synthetic_mode
        "mixup": args.mixup_ratio,     # optional pre-generated real-kernel mixup stream
        "cauker_V2": active_cauker_V2_ratio,
        "gift": args.gift_ratio,
    }
    raw_stream_sizes = {name: args.per_device_batch_size * ratio for name, ratio in stream_ratios.items()}
    stream_batch_sizes = {name: int(math.floor(size)) for name, size in raw_stream_sizes.items()}
    remaining = args.per_device_batch_size - sum(stream_batch_sizes.values())
    for name in sorted(
        stream_ratios,
        key=lambda k: (raw_stream_sizes[k] - math.floor(raw_stream_sizes[k]), stream_ratios[k]),
        reverse=True,
    ):
        if remaining <= 0:
            break
        stream_batch_sizes[name] += 1
        remaining -= 1

    real_batch_size = stream_batch_sizes["real"]
    synth_batch_size = stream_batch_sizes["synth"]
    mixup_batch_size = stream_batch_sizes["mixup"]
    cauker_V2_batch_size = stream_batch_sizes["cauker_V2"]
    gift_batch_size = stream_batch_sizes["gift"]

    for source_name, ratio, batch_size in (
        ("BLAST", args.real_ratio, real_batch_size),
        ("Kernel/synthetic", synth_ratio, synth_batch_size),
        ("mixup", args.mixup_ratio, mixup_batch_size),
        ("CauKer V2", active_cauker_V2_ratio, cauker_V2_batch_size),
        ("GIFT-Eval-Pretrain", args.gift_ratio, gift_batch_size),
    ):
        if ratio > 0.0 and batch_size == 0:
            raise ValueError(
                f"{source_name} ratio is positive but its batch size rounded to 0. "
                "Increase --per_device_batch_size."
            )

    cfg = PatchTSTFMConfig(
        context_length=args.context_length,
        patch_size=args.patch_size,
        num_layers=args.num_layers,
        d_model=args.d_model,
        head_dim=args.head_dim,
        dropout=args.dropout,
        num_quantiles=args.num_quantiles,
        mask_ratio=args.mask_ratio,
        cpm_blocks=args.cpm_blocks,
    )
    flow_exit_layers = (
        resolve_flow_exit_layers(
            num_layers=args.num_layers,
            explicit_layers=args.flow_exit_layers,
            num_exits=args.num_flow_exits,
        )
        if args.flow_supervision
        else [args.num_layers]
    )

    effective_batch = args.per_device_batch_size * world_size * args.grad_accum_steps

    if is_main_process():
        blast_shards = discover_blast_shards(args.real_data_root) if real_batch_size > 0 else []
        gift_dirs = discover_gift_dataset_dirs(args.gift_data_root) if gift_batch_size > 0 else []
        print("=" * 110)
        title_parts = ["PatchTST-FM mixed pretraining"]
        if args.real_ratio > 0:
            title_parts.append(f"{args.real_ratio:.0%} BLAST")
        if args.gift_ratio > 0:
            title_parts.append(f"{args.gift_ratio:.0%} GIFT-Eval-Pretrain")
        if args.synthetic_mode == "kernel_mixed" and synth_ratio > 0:
            title_parts.append(f"{synth_ratio:.0%} kernel synthetic ({args.synthetic_mode})")
        elif synth_batch_size > 0:
            title_parts.append(f"{synth_ratio:.0%} {args.synthetic_mode} synthetic data")
        if args.mixup_ratio > 0:
            title_parts.append(f"{args.mixup_ratio:.0%} real-kernel mixup data")
        if args.use_cauker_V2:
            title_parts.append(f"{active_cauker_V2_ratio:.0%} CauKer V2 online synthetic data")
        print(" + ".join(title_parts))
        print(f"world_size              : {world_size}")
        print(f"context_length          : {args.context_length}")
        print(f"blast_data_root         : {args.real_data_root}")
        print(f"blast_shards            : {len(blast_shards)}")
        print(f"blast_ratio             : {args.real_ratio}")
        print(f"blast sample length     : Uniform({args.min_real_sample_length}, {args.max_real_sample_length})")
        print(f"kernel/synth ratio      : {synth_ratio}")
        print(f"kernel sample length    : Uniform({args.min_synth_sample_length}, {args.max_synth_sample_length})")
        print(f"cauker sample length    : {args.length_sampling}({args.min_series_length}, {args.max_series_length})")
        print(f"gift_data_root          : {args.gift_data_root}")
        print(f"gift_dataset_dirs       : {len(gift_dirs)}")
        print(f"gift_ratio              : {args.gift_ratio}")
        print(f"gift_domain_balance     : {args.gift_domain_balance}")
        print(f"gift sample length      : Uniform({args.min_gift_sample_length}, {args.max_gift_sample_length})")
        print(f"mixup_ratio             : {args.mixup_ratio}")
        print(
            f"batch split per device  : "
            f"blast={real_batch_size}, synth/kernel={synth_batch_size}, "
            f"mixup={mixup_batch_size}, cauker_V2={cauker_V2_batch_size}, "
            f"gift={gift_batch_size}"
        )
        if args.use_cauker_V2:
            print(f"cauker_V2 batch size    : {cauker_V2_batch_size} per device")
            print(f"cauker_V2_ratio         : {active_cauker_V2_ratio}")
        print(f"blast workers per rank  : {args.num_real_workers}")
        print(f"synth workers per rank  : {args.num_synth_workers}")
        print(f"mixup workers per rank  : {args.num_mixup_workers}")
        print(f"gift workers per rank   : {args.num_gift_workers}")
        print(f"total blast workers     : {args.num_real_workers * world_size}")
        print(f"total synth workers     : {args.num_synth_workers * world_size}")
        print(f"total mixup workers     : {args.num_mixup_workers * world_size}")
        print(f"total gift workers      : {args.num_gift_workers * world_size}")
        print(f"prefetch_factor         : {args.prefetch_factor}")
        print(f"effective batch size    : {effective_batch}")
        print(f"model config            : {cfg}")
        print(f"flow supervision        : {args.flow_supervision}")
        print(f"flow exits              : {flow_exit_layers}")
        if args.flow_supervision:
            print(
                f"flow loss weights        : "
                f"lambda_ds={args.lambda_ds}, lambda_fm={args.lambda_fm}, "
                f"ds_gamma={args.ds_gamma}"
            )
        print(f"cpm block length        : {args.cpm_blocks} patches")
        print(f"terminal mask length    : Uniform({args.terminal_mask_min}, {args.terminal_mask_max}) patches")
        print(f"synthetic_mode          : {args.synthetic_mode}")
        if args.init_checkpoint:
            print(f"init_checkpoint         : {args.init_checkpoint}")
            print(f"init_checkpoint_strict  : {args.init_checkpoint_strict}")
        if args.synthetic_mode in {"arrow", "arrow_scm", "kernel_mixed"}:
            arrow_files = discover_arrow_files(args.synthetic_arrow_root) if synth_batch_size > 0 else []
            print(f"synthetic_arrow_root    : {args.synthetic_arrow_root}")
            print(f"synthetic_arrow_files   : {len(arrow_files)}")
            print(f"synthetic arrow balance : {args.synthetic_arrow_balance_files}")

            if args.synthetic_mode == "arrow_scm":
                print(f"arrow_scm source        : Arrow-rooted CauKer-style SCM")
                print(f"arrow_scm DAG nodes     : {args.cauker_num_nodes}")
                print(f"arrow_scm output chans  : {args.cauker_features}")
                print(f"arrow_scm max_parents   : {args.cauker_max_parents}")
        else:
            print(f"synthetic source        : multi-base CauKer V2")
            print(f"cauker_v2 latent_dim    : {args.cauker_num_nodes}")
            print(f"cauker_v2 n_channels    : {args.cauker_features}")
            print(f"cauker_v2 weights_json  : {args.cauker_v2_weights_json}")

        if args.use_cauker_V2:
            print(f"cauker_v2 latent roots  : {args.cauker_num_nodes}")
            print(f"cauker_v2 observed nodes: {args.cauker_features}")
            print(f"cauker_v2 total DAG nodes: {args.cauker_num_nodes + args.cauker_features}")
            print(f"cauker_v2 max length    : {args.max_series_length}")

        if mixup_batch_size > 0:
            mixup_arrow_files = discover_arrow_files(args.mixup_arrow_root)
            print(f"mixup_arrow_root        : {args.mixup_arrow_root}")
            print(f"mixup_arrow_files       : {len(mixup_arrow_files)}")
            print(f"mixup arrow balance     : {args.mixup_arrow_balance_files}")
        print("=" * 110, flush=True)

    real_loader = None
    if real_batch_size > 0:
        real_dataset = BlastRealIterableDataset(
            root=args.real_data_root,
            context_length=args.context_length,
            seed=args.seed,
            min_real_length=args.min_real_length,
            min_sample_length=args.min_real_sample_length,
            max_sample_length=args.max_real_sample_length,
            shuffle_datasets=True,
            shuffle_examples=True,
            domain_balance=args.real_domain_balance,
            max_sample_attempts=args.real_max_sample_attempts,
        )
        real_loader = make_loader(
            dataset=real_dataset,
            batch_size=real_batch_size,
            num_workers=args.num_real_workers,
            prefetch_factor=args.prefetch_factor,
        )

    synth_loader = None
    if synth_batch_size > 0:
        if args.synthetic_mode == "arrow":
            synth_dataset = ArrowSyntheticIterableDataset(
                root=args.synthetic_arrow_root,
                context_length=args.context_length,
                seed=args.seed,
                min_sample_length=args.min_synth_sample_length,
                max_sample_length=args.max_synth_sample_length,
                min_real_length=args.min_real_length,
                max_sample_attempts=args.real_max_sample_attempts,
                balance_files=args.synthetic_arrow_balance_files,
            )

        elif args.synthetic_mode == "arrow_scm":
            synth_dataset = ArrowCauKerSCMIterableDataset(
                root=args.synthetic_arrow_root,
                context_length=args.context_length,
                seed=args.seed,
                min_sample_length=args.min_synth_sample_length,
                max_sample_length=args.max_synth_sample_length,
                min_real_length=args.min_real_length,
                max_sample_attempts=args.real_max_sample_attempts,
                balance_files=args.synthetic_arrow_balance_files,
                num_features=args.cauker_features,
                max_parents=args.cauker_max_parents,
                num_nodes=args.cauker_num_nodes,
                length_sampling=args.length_sampling,
            )

        elif args.synthetic_mode == "kernel_mixed":
            synth_dataset = ArrowSyntheticIterableDataset(
                root=args.synthetic_arrow_root,
                context_length=args.context_length,
                seed=args.seed,
                min_sample_length=args.min_synth_sample_length,
                max_sample_length=args.max_synth_sample_length,
                min_real_length=args.min_real_length,
                max_sample_attempts=args.real_max_sample_attempts,
                balance_files=args.synthetic_arrow_balance_files,
            )

        else:
            synth_dataset = OnlineCauKerIterableDataset(
                context_length=args.context_length,
                min_length=args.min_series_length,
                max_length=args.max_series_length,
                num_features=args.cauker_features,
                max_parents=args.cauker_max_parents,
                num_nodes=args.cauker_num_nodes,
                seed=args.seed,
                length_sampling=args.length_sampling,
                weights_json=args.cauker_v2_weights_json,
            )
        synth_loader = make_loader(
            dataset=synth_dataset,
            batch_size=synth_batch_size,
            num_workers=args.num_synth_workers,
            prefetch_factor=args.prefetch_factor,
        )

    mixup_loader = None
    if mixup_batch_size > 0:
        mixup_dataset = ArrowSyntheticIterableDataset(
            root=args.mixup_arrow_root,
            context_length=args.context_length,
            seed=args.seed + 3_333_333,
            min_sample_length=args.min_real_sample_length,
            max_sample_length=args.max_real_sample_length,
            min_real_length=args.min_real_length,
            max_sample_attempts=args.real_max_sample_attempts,
            balance_files=args.mixup_arrow_balance_files,
        )
        mixup_loader = make_loader(
            dataset=mixup_dataset,
            batch_size=mixup_batch_size,
            num_workers=args.num_mixup_workers,
            prefetch_factor=args.prefetch_factor,
        )

    cauker_V2_loader = None
    if cauker_V2_batch_size > 0:
        cauker_V2_dataset = OnlineCauKerIterableDataset(
            context_length=args.context_length,
            min_length=args.min_series_length,
            max_length=args.max_series_length,
            num_features=args.cauker_features,
            max_parents=args.cauker_max_parents,
            num_nodes=args.cauker_num_nodes,
            seed=args.seed + 7777777,
            length_sampling=args.length_sampling,
            weights_json=args.cauker_v2_weights_json,
        )
        cauker_V2_loader = make_loader(
            dataset=cauker_V2_dataset,
            batch_size=cauker_V2_batch_size,
            num_workers=args.num_synth_workers,
            prefetch_factor=args.prefetch_factor,
        )

    gift_loader = None
    if gift_batch_size > 0:
        gift_dataset = GiftEvalPretrainRealIterableDataset(
            root=args.gift_data_root,
            context_length=args.context_length,
            seed=args.seed + 9_999_991,
            min_real_length=args.min_real_length,
            min_sample_length=args.min_gift_sample_length,
            max_sample_length=args.max_gift_sample_length,
            shuffle_datasets=True,
            shuffle_examples=True,
            domain_balance=args.gift_domain_balance,
            max_sample_attempts=args.gift_max_sample_attempts,
        )
        gift_loader = make_loader(
            dataset=gift_dataset,
            batch_size=gift_batch_size,
            num_workers=args.num_gift_workers,
            prefetch_factor=args.prefetch_factor,
        )

    real_iter = iter(real_loader) if real_loader is not None else None
    synth_iter = iter(synth_loader) if synth_loader is not None else None
    mixup_iter = iter(mixup_loader) if mixup_loader is not None else None
    cauker_V2_iter = iter(cauker_V2_loader) if cauker_V2_loader is not None else None
    gift_iter = iter(gift_loader) if gift_loader is not None else None

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = PatchTSTFMWithFlowExits(cfg, flow_exit_layers).to(device)

    # ---- Warm-start weights from a checkpoint (model only; optimizer/step stay fresh) ----
    if args.init_checkpoint:
        load_init_checkpoint(
            init_checkpoint=args.init_checkpoint,
            model=model,
            device=device,
            strict=args.init_checkpoint_strict,
        )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    if args.compile:
        model = torch.compile(model)

    if dist.is_available() and dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = build_optimizer(
        unwrap_model(model),
        lr=args.peak_lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )

    if args.num_quantiles == 99:
        q_levels = torch.arange(1, 100, dtype=torch.float32, device=device) / 100.0
    else:
        q_levels = torch.linspace(
            1.0 / (args.num_quantiles + 1),
            args.num_quantiles / (args.num_quantiles + 1),
            args.num_quantiles,
            dtype=torch.float32,
            device=device,
        )

    if args.precision == "bf16":
        amp_dtype = torch.bfloat16
        use_amp = True
        scaler = None
    elif args.precision == "fp16":
        amp_dtype = torch.float16
        use_amp = True
        scaler = torch.cuda.amp.GradScaler()
    else:
        amp_dtype = torch.float32
        use_amp = False
        scaler = None

    output_dir = Path(args.output_dir)
    log_path = Path(args.log_file)
    if not log_path.is_absolute():
        log_path = output_dir / log_path
    if is_main_process():
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Resume from checkpoint ----
    start_step = 0
    if args.resume:
        latest = output_dir / "latest"
        if latest.exists():
            ckpt_path = latest / "pytorch_model.bin"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"--resume: found {latest} but missing {ckpt_path}")
            if is_main_process():
                print(f"Resuming from checkpoint: {latest.resolve()}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            unwrap_model(model).load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_step = ckpt["step"]
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            del ckpt
            torch.cuda.empty_cache()
            if is_main_process():
                print(f"Resumed at step {start_step}, will train to step {args.total_steps}")
            if start_step >= args.total_steps:
                if is_main_process():
                    print(f"Already completed {start_step}/{args.total_steps} steps. Nothing to do.")
                cleanup_distributed()
                return
        else:
            raise FileNotFoundError(
                f"--resume was set but no checkpoint found at {latest}. "
                f"Run without --resume to start fresh."
            )

    optimizer.zero_grad(set_to_none=True)

    start_time = time.time()
    # Accumulate loss on-device — no .item() sync until log steps
    running_loss_device = torch.zeros((), device=device)
    # Raw loss components, accumulated with the same grad-accumulation/logging
    # convention as running_loss_device: [terminal, deep_sup, flow_match].
    running_flow_parts_device = torch.zeros(3, device=device)
    running_loss_count = 0

    running_real_loss = 0.0
    running_synth_loss = 0.0
    running_mixup_loss = 0.0
    running_cauker_V2_loss = 0.0
    running_gift_loss = 0.0
    running_real_loss_count = 0
    running_synth_loss_count = 0
    running_mixup_loss_count = 0
    running_cauker_V2_loss_count = 0
    running_gift_loss_count = 0

    running_mask_ratio = 0.0
    running_real_mask_ratio = 0.0
    running_synth_mask_ratio = 0.0
    running_mixup_mask_ratio = 0.0
    running_cauker_V2_mask_ratio = 0.0
    running_gift_mask_ratio = 0.0
    running_mask_ratio_count = 0
    running_real_mask_ratio_count = 0
    running_synth_mask_ratio_count = 0
    running_mixup_mask_ratio_count = 0
    running_cauker_V2_mask_ratio_count = 0
    running_gift_mask_ratio_count = 0

    # Per-phase timing accumulators (reset every log interval)
    t_data = 0.0         # dataloader fetch + concat
    t_h2d = 0.0          # .to(device) host-to-device transfers
    t_preprocess = 0.0    # mask generation + normalization
    t_forward = 0.0       # model forward + loss
    t_backward = 0.0      # loss.backward()
    t_optim = 0.0         # grad clip + optimizer.step + zero_grad
    t_diag = 0.0          # diagnostic-only computation (log steps)
    t_post_step = 0.0     # allreduce + .item() sync + logging + checkpoint
    log_interval_steps = 0
    log_interval_samples = 0
    running_skipped_updates = 0
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    print(f"Starting training from step {start_step} to {args.total_steps} with effective batch size {effective_batch}...")

    for step in range(start_step, args.total_steps):
        lr = lr_at_step(
            step=step,
            total_steps=args.total_steps,
            warmup_steps=args.warmup_steps,
            peak_lr=args.peak_lr,
            min_lr=args.min_lr,
            scheduler=args.scheduler,
            decay_steps=args.decay_steps,
            num_cycles=args.num_cycles,
        )
        set_optimizer_lr(optimizer, lr)

        step_loss_accum = torch.zeros((), device=device)
        step_flow_parts_accum = torch.zeros(3, device=device)

        should_log = (step + 1) % args.log_every == 0 or step == 0

        step_real_loss = 0.0
        step_synth_loss = 0.0
        step_mixup_loss = 0.0
        step_cauker_V2_loss = 0.0
        step_gift_loss = 0.0
        step_real_loss_count = 0
        step_synth_loss_count = 0
        step_mixup_loss_count = 0
        step_cauker_V2_loss_count = 0
        step_gift_loss_count = 0

        step_mask_ratio = 0.0
        step_real_mask_ratio = 0.0
        step_synth_mask_ratio = 0.0
        step_mixup_mask_ratio = 0.0
        step_cauker_V2_mask_ratio = 0.0
        step_gift_mask_ratio = 0.0
        step_mask_ratio_count = 0
        step_real_mask_ratio_count = 0
        step_synth_mask_ratio_count = 0
        step_mixup_mask_ratio_count = 0
        step_cauker_V2_mask_ratio_count = 0
        step_gift_mask_ratio_count = 0

        for accum_idx in range(args.grad_accum_steps):
            ctx = model.no_sync() if (accum_idx < args.grad_accum_steps - 1) else nullcontext()
            with ctx:
                # ---- DATA FETCH ----
                t0 = time.time()

                parts: List[Dict[str, torch.Tensor]] = []

                current_real_bs = 0
                current_synth_bs = 0
                current_mixup_bs = 0
                current_cauker_V2_bs = 0
                current_gift_bs = 0

                if real_iter is not None:
                    real_part = next(real_iter)
                    current_real_bs = real_part["x"].shape[0]
                    real_part["source_id"] = torch.full(
                        (current_real_bs,), 0, dtype=torch.long
                    )
                    parts.append(real_part)

                if synth_iter is not None:
                    synth_part = next(synth_iter)
                    current_synth_bs = synth_part["x"].shape[0]
                    synth_part["source_id"] = torch.full(
                        (current_synth_bs,), 1, dtype=torch.long
                    )
                    parts.append(synth_part)

                if mixup_iter is not None:
                    mixup_part = next(mixup_iter)
                    current_mixup_bs = mixup_part["x"].shape[0]
                    mixup_part["source_id"] = torch.full(
                        (current_mixup_bs,), 2, dtype=torch.long
                    )
                    parts.append(mixup_part)

                if cauker_V2_iter is not None:
                    cauker_V2_part = next(cauker_V2_iter)
                    current_cauker_V2_bs = cauker_V2_part["x"].shape[0]
                    cauker_V2_part["source_id"] = torch.full(
                        (current_cauker_V2_bs,), 3, dtype=torch.long
                    )
                    parts.append(cauker_V2_part)

                if gift_iter is not None:
                    gift_part = next(gift_iter)
                    current_gift_bs = gift_part["x"].shape[0]
                    gift_part["source_id"] = torch.full(
                        (current_gift_bs,), 4, dtype=torch.long
                    )
                    parts.append(gift_part)

                batch = concat_mixed_batches(parts)

                t_data += time.time() - t0

                # ---- HOST-TO-DEVICE (fused transfer + cast) ----
                t1 = time.time()
                x = batch["x"].to(device=device, dtype=torch.float32, non_blocking=True)
                observed_mask = batch["observed_mask"].to(device=device, dtype=torch.bool, non_blocking=True)
                padding_mask = batch["padding_mask"].to(device=device, dtype=torch.bool, non_blocking=True)
                source_id = batch["source_id"].to(device=device, dtype=torch.long, non_blocking=True)
                batch_size = x.shape[0]
                t_h2d += time.time() - t1

                # ---- PREPROCESSING (mask + normalize) ----
                t2 = time.time()

                pred_mask = make_cpm_prediction_mask(
                    observed_mask=observed_mask,
                    padding_mask=padding_mask,
                    cfg=cfg,
                    terminal_mask_min_patches=args.terminal_mask_min,
                    terminal_mask_max_patches=args.terminal_mask_max,
                )

                x_norm_input, target_norm, union_mask, patch_padding = mask_aware_normalize(
                    x=x,
                    observed_mask=observed_mask,
                    pred_mask=pred_mask,
                    padding_mask=padding_mask,
                    cfg=cfg,
                )

                t_preprocess += time.time() - t2

                # ---- FORWARD ----
                t3 = time.time()

                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    exit_preds = model(
                        x_norm=x_norm_input,
                        union_mask=union_mask,
                        patch_key_padding_mask=patch_padding,
                    )
                    loss, flow_parts = quantile_flow_loss(
                        exit_preds=exit_preds,
                        target_norm=target_norm,
                        loss_mask=pred_mask,
                        quantiles=q_levels,
                        num_layers=args.num_layers,
                        lambda_ds=args.lambda_ds if args.flow_supervision else 0.0,
                        lambda_fm=args.lambda_fm if args.flow_supervision else 0.0,
                        ds_gamma=args.ds_gamma,
                        length_weight_alpha=args.length_weight_alpha,
                    )
                    q_hat = exit_preds[args.num_layers]

                    loss = loss / args.grad_accum_steps

                t_forward += time.time() - t3

                # ---- BACKWARD ----
                t4 = time.time()

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                t_backward += time.time() - t4

                step_loss_accum += loss.detach()
                step_flow_parts_accum += torch.stack(
                    [
                        flow_parts["terminal"],
                        flow_parts["deep_sup"],
                        flow_parts["flow_match"],
                    ]
                )
                

                log_interval_samples += batch_size

                # Diagnostics only on last accum sub-step of log steps
                if should_log and accum_idx == args.grad_accum_steps - 1:
                    t5 = time.time()
                    with torch.no_grad():
                        # source_id is shuffled together with the tensors in
                        # concat_mixed_batches, so these masks remain correct after
                        # within-batch shuffling.
                        is_real = source_id == 0
                        is_synth = source_id == 1
                        is_mixup = source_id == 2
                        is_cauker_V2 = source_id == 3
                        is_gift = source_id == 4

                        if is_real.any():
                            real_loss = pinball_loss(
                                q_hat=q_hat[is_real],
                                target_norm=target_norm[is_real],
                                loss_mask=pred_mask[is_real],
                                quantiles=q_levels,
                                length_weight_alpha=args.length_weight_alpha,
                            )
                            step_real_loss = float(real_loss.cpu())
                            step_real_loss_count = 1

                        if is_synth.any():
                            synth_loss = pinball_loss(
                                q_hat=q_hat[is_synth],
                                target_norm=target_norm[is_synth],
                                loss_mask=pred_mask[is_synth],
                                quantiles=q_levels,
                                length_weight_alpha=args.length_weight_alpha,
                            )
                            step_synth_loss = float(synth_loss.cpu())
                            step_synth_loss_count = 1

                        if is_mixup.any():
                            mixup_loss = pinball_loss(
                                q_hat=q_hat[is_mixup],
                                target_norm=target_norm[is_mixup],
                                loss_mask=pred_mask[is_mixup],
                                quantiles=q_levels,
                                length_weight_alpha=args.length_weight_alpha,
                            )
                            step_mixup_loss = float(mixup_loss.cpu())
                            step_mixup_loss_count = 1

                        if is_cauker_V2.any():
                            cauker_V2_loss = pinball_loss(
                                q_hat=q_hat[is_cauker_V2],
                                target_norm=target_norm[is_cauker_V2],
                                loss_mask=pred_mask[is_cauker_V2],
                                quantiles=q_levels,
                                length_weight_alpha=args.length_weight_alpha,
                            )
                            step_cauker_V2_loss = float(cauker_V2_loss.cpu())
                            step_cauker_V2_loss_count = 1

                        if is_gift.any():
                            gift_loss = pinball_loss(
                                q_hat=q_hat[is_gift],
                                target_norm=target_norm[is_gift],
                                loss_mask=pred_mask[is_gift],
                                quantiles=q_levels,
                                length_weight_alpha=args.length_weight_alpha,
                            )
                            step_gift_loss = float(gift_loss.cpu())
                            step_gift_loss_count = 1

                        valid_mask = observed_mask & (~padding_mask)

                        mask_ratio = pred_mask.float().sum() / valid_mask.float().sum().clamp_min(1.0)
                        step_mask_ratio = float(mask_ratio.cpu())
                        step_mask_ratio_count = 1

                        if is_real.any():
                            real_valid_mask = valid_mask[is_real]
                            real_pred_mask = pred_mask[is_real]
                            real_mask_ratio = real_pred_mask.float().sum() / real_valid_mask.float().sum().clamp_min(1.0)
                            step_real_mask_ratio = float(real_mask_ratio.cpu())
                            step_real_mask_ratio_count = 1

                        if is_synth.any():
                            synth_valid_mask = valid_mask[is_synth]
                            synth_pred_mask = pred_mask[is_synth]
                            synth_mask_ratio = synth_pred_mask.float().sum() / synth_valid_mask.float().sum().clamp_min(1.0)
                            step_synth_mask_ratio = float(synth_mask_ratio.cpu())
                            step_synth_mask_ratio_count = 1

                        if is_mixup.any():
                            mixup_valid_mask = valid_mask[is_mixup]
                            mixup_pred_mask = pred_mask[is_mixup]
                            mixup_mask_ratio = mixup_pred_mask.float().sum() / mixup_valid_mask.float().sum().clamp_min(1.0)
                            step_mixup_mask_ratio = float(mixup_mask_ratio.cpu())
                            step_mixup_mask_ratio_count = 1

                        if is_cauker_V2.any():
                            cauker_V2_valid_mask = valid_mask[is_cauker_V2]
                            cauker_V2_pred_mask = pred_mask[is_cauker_V2]
                            cauker_V2_mask_ratio = cauker_V2_pred_mask.float().sum() / cauker_V2_valid_mask.float().sum().clamp_min(1.0)
                            step_cauker_V2_mask_ratio = float(cauker_V2_mask_ratio.cpu())
                            step_cauker_V2_mask_ratio_count = 1

                        if is_gift.any():
                            gift_valid_mask = valid_mask[is_gift]
                            gift_pred_mask = pred_mask[is_gift]
                            gift_mask_ratio = gift_pred_mask.float().sum() / gift_valid_mask.float().sum().clamp_min(1.0)
                            step_gift_mask_ratio = float(gift_mask_ratio.cpu())
                            step_gift_mask_ratio_count = 1

                    t_diag += time.time() - t5

        # ---- OPTIMIZER STEP ----
        t_opt_start = time.time()

        skipped_update = False

        if scaler is not None:
            # For fp16 AMP, gradients are scaled before backward.
            # We must unscale before computing/clipping grad norm.
            scaler.unscale_(optimizer)

            # clip_grad_norm_ returns the pre-clipping total grad norm.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.max_grad_norm,
                error_if_nonfinite=False,
            )

            if args.skip_bad_grad_update:
                # Device-side boolean: True if grad norm is NaN/Inf or extremely large.
                skip_tensor = (
                    (~torch.isfinite(grad_norm))
                    | (grad_norm > args.skip_grad_norm_threshold)
                ).to(device=device, dtype=torch.int32)

                # In DDP, all ranks must make the same skip/step decision.
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(skip_tensor, op=dist.ReduceOp.MAX)

                skipped_update = bool(skip_tensor.item())

            if skipped_update:
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
            else:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        else:
            # bf16/fp32 branch.
            # clip_grad_norm_ returns the pre-clipping total grad norm.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.max_grad_norm,
                error_if_nonfinite=False,
            )

            if args.skip_bad_grad_update:
                # Device-side boolean: True if grad norm is NaN/Inf or extremely large.
                skip_tensor = (
                    (~torch.isfinite(grad_norm))
                    | (grad_norm > args.skip_grad_norm_threshold)
                ).to(device=device, dtype=torch.int32)

                # In DDP, all ranks must make the same skip/step decision.
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(skip_tensor, op=dist.ReduceOp.MAX)

                skipped_update = bool(skip_tensor.item())

            if skipped_update:
                optimizer.zero_grad(set_to_none=True)
            else:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        if skipped_update:
            running_skipped_updates += 1

        t_optim += time.time() - t_opt_start

        # ---- POST-STEP: device-side accumulation (no sync on non-log steps) ----
        t_post_start = time.time()

        log_interval_steps += 1

        # Accumulate loss on device — NO allreduce, NO .item(), NO CUDA sync
        running_loss_device += step_loss_accum * args.grad_accum_steps
        running_flow_parts_device += step_flow_parts_accum
        running_loss_count += 1

        if should_log:
            # Single sync point per log interval: allreduce the accumulated loss
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(running_loss_device, op=dist.ReduceOp.AVG)
                dist.all_reduce(running_flow_parts_device, op=dist.ReduceOp.AVG)

            extra_stats = torch.tensor(
                [
                    step_real_loss,
                    step_synth_loss,
                    step_mixup_loss,
                    step_cauker_V2_loss,
                    float(step_real_loss_count),
                    float(step_synth_loss_count),
                    float(step_mixup_loss_count),
                    float(step_cauker_V2_loss_count),
                    step_mask_ratio,
                    step_real_mask_ratio,
                    step_synth_mask_ratio,
                    step_mixup_mask_ratio,
                    step_cauker_V2_mask_ratio,
                    float(step_mask_ratio_count),
                    float(step_real_mask_ratio_count),
                    float(step_synth_mask_ratio_count),
                    float(step_mixup_mask_ratio_count),
                    float(step_cauker_V2_mask_ratio_count),
                    step_gift_loss,
                    float(step_gift_loss_count),
                    step_gift_mask_ratio,
                    float(step_gift_mask_ratio_count),
                ],
                device=device,
            )

            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(extra_stats, op=dist.ReduceOp.SUM)

            global_real_loss = float(extra_stats[0].item())
            global_synth_loss = float(extra_stats[1].item())
            global_mixup_loss = float(extra_stats[2].item())
            global_cauker_V2_loss = float(extra_stats[3].item())
            global_real_loss_count = int(extra_stats[4].item())
            global_synth_loss_count = int(extra_stats[5].item())
            global_mixup_loss_count = int(extra_stats[6].item())
            global_cauker_V2_loss_count = int(extra_stats[7].item())

            global_mask_ratio = float(extra_stats[8].item())
            global_real_mask_ratio = float(extra_stats[9].item())
            global_synth_mask_ratio = float(extra_stats[10].item())
            global_mixup_mask_ratio = float(extra_stats[11].item())
            global_cauker_V2_mask_ratio = float(extra_stats[12].item())
            global_mask_ratio_count = int(extra_stats[13].item())
            global_real_mask_ratio_count = int(extra_stats[14].item())
            global_synth_mask_ratio_count = int(extra_stats[15].item())
            global_mixup_mask_ratio_count = int(extra_stats[16].item())
            global_cauker_V2_mask_ratio_count = int(extra_stats[17].item())
            global_gift_loss = float(extra_stats[18].item())
            global_gift_loss_count = int(extra_stats[19].item())
            global_gift_mask_ratio = float(extra_stats[20].item())
            global_gift_mask_ratio_count = int(extra_stats[21].item())

            running_real_loss += global_real_loss
            running_synth_loss += global_synth_loss
            running_mixup_loss += global_mixup_loss
            running_cauker_V2_loss += global_cauker_V2_loss
            running_gift_loss += global_gift_loss
            running_real_loss_count += global_real_loss_count
            running_synth_loss_count += global_synth_loss_count
            running_mixup_loss_count += global_mixup_loss_count
            running_cauker_V2_loss_count += global_cauker_V2_loss_count
            running_gift_loss_count += global_gift_loss_count

            running_mask_ratio += global_mask_ratio
            running_real_mask_ratio += global_real_mask_ratio
            running_synth_mask_ratio += global_synth_mask_ratio
            running_mixup_mask_ratio += global_mixup_mask_ratio
            running_cauker_V2_mask_ratio += global_cauker_V2_mask_ratio
            running_gift_mask_ratio += global_gift_mask_ratio
            running_mask_ratio_count += global_mask_ratio_count
            running_real_mask_ratio_count += global_real_mask_ratio_count
            running_synth_mask_ratio_count += global_synth_mask_ratio_count
            running_mixup_mask_ratio_count += global_mixup_mask_ratio_count
            running_cauker_V2_mask_ratio_count += global_cauker_V2_mask_ratio_count
            running_gift_mask_ratio_count += global_gift_mask_ratio_count

            # Snapshot the averaged loss before resetting (rank 0 reads it below)
            mean_loss_snapshot = float(running_loss_device.item()) / max(1, running_loss_count)
            flow_parts_snapshot = (
                running_flow_parts_device / max(1, running_loss_count)
            ).tolist()
            # Reset on ALL ranks so the next interval starts clean
            running_loss_device.zero_()
            running_flow_parts_device.zero_()
            running_loss_count = 0

        if (step + 1) % args.save_every == 0:
            save_checkpoint(output_dir, model, optimizer, step + 1, cfg, args)

        t_post_step += time.time() - t_post_start

        if is_main_process() and should_log:
            mean_loss = mean_loss_snapshot

            mean_real_loss = running_real_loss / max(1, running_real_loss_count)
            mean_synth_loss = running_synth_loss / max(1, running_synth_loss_count)
            mean_mixup_loss = running_mixup_loss / max(1, running_mixup_loss_count)
            mean_cauker_V2_loss = running_cauker_V2_loss / max(1, running_cauker_V2_loss_count)
            mean_gift_loss = running_gift_loss / max(1, running_gift_loss_count)

            mean_mask_ratio = running_mask_ratio / max(1, running_mask_ratio_count)
            mean_real_mask_ratio = running_real_mask_ratio / max(1, running_real_mask_ratio_count)
            mean_synth_mask_ratio = running_synth_mask_ratio / max(1, running_synth_mask_ratio_count)
            mean_mixup_mask_ratio = running_mixup_mask_ratio / max(1, running_mixup_mask_ratio_count)
            mean_cauker_V2_mask_ratio = running_cauker_V2_mask_ratio / max(1, running_cauker_V2_mask_ratio_count)
            mean_gift_mask_ratio = running_gift_mask_ratio / max(1, running_gift_mask_ratio_count)

            running_real_loss = 0.0
            running_synth_loss = 0.0
            running_mixup_loss = 0.0
            running_cauker_V2_loss = 0.0
            running_gift_loss = 0.0
            running_real_loss_count = 0
            running_synth_loss_count = 0
            running_mixup_loss_count = 0
            running_cauker_V2_loss_count = 0
            running_gift_loss_count = 0

            running_mask_ratio = 0.0
            running_real_mask_ratio = 0.0
            running_synth_mask_ratio = 0.0
            running_mixup_mask_ratio = 0.0
            running_cauker_V2_mask_ratio = 0.0
            running_gift_mask_ratio = 0.0
            running_mask_ratio_count = 0
            running_real_mask_ratio_count = 0
            running_synth_mask_ratio_count = 0
            running_mixup_mask_ratio_count = 0
            running_cauker_V2_mask_ratio_count = 0
            running_gift_mask_ratio_count = 0

            elapsed = time.time() - start_time
            grad_norm_value = float(grad_norm)
            skipped_updates_value = running_skipped_updates

            # Per-phase breakdown — averaged over the log interval
            t_phases = t_data + t_h2d + t_preprocess + t_forward + t_backward + t_optim + t_diag + t_post_step
            t_total = max(t_phases, 1e-12)
            pct = lambda t: 100.0 * t / t_total
            n_steps = max(1, log_interval_steps)
            n_microbatches = n_steps * args.grad_accum_steps

            # Per-step averages (ms)
            avg_data      = 1000.0 * t_data / n_steps
            avg_h2d       = 1000.0 * t_h2d / n_steps
            avg_preproc   = 1000.0 * t_preprocess / n_steps
            avg_fwd       = 1000.0 * t_forward / n_steps
            avg_bwd       = 1000.0 * t_backward / n_steps
            avg_optim     = 1000.0 * t_optim / n_steps
            avg_diag      = 1000.0 * t_diag / n_steps
            avg_post_step = 1000.0 * t_post_step / n_steps
            avg_total     = 1000.0 * t_total / n_steps

            # Throughput
            samples_per_sec = log_interval_samples / max(t_total, 1e-12)
            steps_per_sec = n_steps / max(t_total, 1e-12)
            sec_per_step = t_total / n_steps
            ms_per_microbatch = 1000.0 * t_total / max(n_microbatches, 1)


            print(
                f"  timing %: "
                f"data={pct(t_data):.1f}% "
                f"h2d={pct(t_h2d):.1f}% "
                f"preproc={pct(t_preprocess):.1f}% "
                f"fwd={pct(t_forward):.1f}% "
                f"bwd={pct(t_backward):.1f}% "
                f"optim={pct(t_optim):.1f}% "
                f"diag={pct(t_diag):.1f}% "
                f"post_step={pct(t_post_step):.1f}%",
                flush=True,
            )
            print(
                f"  avg/step: "
                f"data={avg_data:.0f}ms "
                f"h2d={avg_h2d:.0f}ms "
                f"preproc={avg_preproc:.0f}ms "
                f"fwd={avg_fwd:.0f}ms "
                f"bwd={avg_bwd:.0f}ms "
                f"optim={avg_optim:.0f}ms "
                f"diag={avg_diag:.0f}ms "
                f"post_step={avg_post_step:.0f}ms "
                f"total={avg_total:.0f}ms "
                f"| {samples_per_sec:.0f}samp/s "
                f"{steps_per_sec:.2f}step/s "
                f"(over {n_steps} steps)",
                flush=True,
            )
            log_record = {
                "step": step + 1,
                "total_steps": args.total_steps,
                "loss": mean_loss,
                "loss_terminal": flow_parts_snapshot[0],
                "loss_deep_sup": flow_parts_snapshot[1],
                "loss_flow_match": flow_parts_snapshot[2],
                "flow_supervision": args.flow_supervision,
                "flow_exit_layers": flow_exit_layers,
                "lambda_ds": args.lambda_ds,
                "lambda_fm": args.lambda_fm,
                "ds_gamma": args.ds_gamma,
                "lr": lr,
                "grad_norm": grad_norm_value,
                "blast_batch_size": real_batch_size,
                "real_batch_size": real_batch_size,
                "synth_batch_size": synth_batch_size,
                "mixup_batch_size": mixup_batch_size,
                "cauker_V2_batch_size": cauker_V2_batch_size,
                "gift_batch_size": gift_batch_size,
                "elapsed_hours": elapsed / 3600.0,
                "world_size": world_size,
                "effective_batch_size": effective_batch,
                "synthetic_mode": args.synthetic_mode,
                "blast_ratio": args.real_ratio,
                "blast_data_root": args.real_data_root,
                "synth_ratio": synth_ratio,
                "mixup_ratio": args.mixup_ratio,
                "mixup_arrow_root": args.mixup_arrow_root,
                "use_cauker_V2": args.use_cauker_V2,
                "cauker_V2_ratio": active_cauker_V2_ratio,
                "gift_ratio": args.gift_ratio,
                "gift_data_root": args.gift_data_root,
                "blast_loss": mean_real_loss,
                "real_loss": mean_real_loss,
                "synth_loss": mean_synth_loss,
                "mixup_loss": mean_mixup_loss,
                "cauker_V2_loss": mean_cauker_V2_loss,
                "gift_loss": mean_gift_loss,
                "mask_ratio": mean_mask_ratio,
                "blast_mask_ratio": mean_real_mask_ratio,
                "real_mask_ratio": mean_real_mask_ratio,
                "synth_mask_ratio": mean_synth_mask_ratio,
                "mixup_mask_ratio": mean_mixup_mask_ratio,
                "cauker_V2_mask_ratio": mean_cauker_V2_mask_ratio,
                "gift_mask_ratio": mean_gift_mask_ratio,
                # Per-phase average per step (ms)
                "avg_data_ms": avg_data,
                "avg_h2d_ms": avg_h2d,
                "avg_preprocess_ms": avg_preproc,
                "avg_forward_ms": avg_fwd,
                "avg_backward_ms": avg_bwd,
                "avg_optim_ms": avg_optim,
                "avg_diag_ms": avg_diag,
                "avg_post_step_ms": avg_post_step,
                "avg_step_ms": avg_total,
                # Per-phase percentage
                "t_data_pct": pct(t_data),
                "t_h2d_pct": pct(t_h2d),
                "t_preprocess_pct": pct(t_preprocess),
                "t_forward_pct": pct(t_forward),
                "t_backward_pct": pct(t_backward),
                "t_optim_pct": pct(t_optim),
                "t_diag_pct": pct(t_diag),
                "t_post_step_pct": pct(t_post_step),
                # Throughput
                "samples_per_sec": samples_per_sec,
                "steps_per_sec": steps_per_sec,
                "sec_per_step": sec_per_step,
                "ms_per_microbatch": ms_per_microbatch,
                "log_interval_steps": n_steps,
                "log_interval_samples": log_interval_samples,
                "skipped_updates": skipped_updates_value,
                "skip_grad_norm_threshold": args.skip_grad_norm_threshold,
            }
            print(
                f"step={step+1:07d}/{args.total_steps} "
                f"loss={mean_loss:.6f} "
                f"term={flow_parts_snapshot[0]:.6f} "
                f"ds={flow_parts_snapshot[1]:.6f} "
                f"fm={flow_parts_snapshot[2]:.6f} "
                f"blast_loss={mean_real_loss:.6f} "
                f"synth_loss={mean_synth_loss:.6f} "
                f"mixup_loss={mean_mixup_loss:.6f} "
                f"cvr2_loss={mean_cauker_V2_loss:.6f} "
                f"gift_loss={mean_gift_loss:.6f} "
                f"lr={lr:.3e} "
                f"grad_norm={grad_norm_value:.3f} "
                f"skipped_updates={skipped_updates_value} "
                f"elapsed={elapsed/3600:.2f}h",
                flush=True,
            )
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_record, ensure_ascii=True) + "\n")
            t_data = 0.0
            t_h2d = 0.0
            t_preprocess = 0.0
            t_forward = 0.0
            t_backward = 0.0
            t_optim = 0.0
            t_diag = 0.0
            t_post_step = 0.0
            log_interval_steps = 0
            log_interval_samples = 0
            running_skipped_updates = 0

    save_checkpoint(output_dir, model, optimizer, args.total_steps, cfg, args)
    cleanup_distributed()


if __name__ == "__main__":
    main()
