#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Checkpoint loader and latent-embedding wrapper for Tabby-Pretrain.

Usage:
    from benchmarks.anomaly.predictor import PatchTSTFMPredictor

    predictor = PatchTSTFMPredictor(
        checkpoint="/path/to/step_0165000", device="cuda:0"
    )
    latents = predictor.embed(windows)   # per-patch encoder embeddings

This module reuses the canonical implementation under :mod:`tabby.models`; it
does not carry a second copy of the pretrained model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from tabby.checkpoint import load_backbone
from tabby.models.PatchTSTFM import PatchTSTFM, PatchTSTFMConfig
from tabby.utils.input_preprocessing import mask_aware_normalize_for_inference

logger = logging.getLogger(__name__)


# =============================================================================
# Checkpoint loading
# =============================================================================

def load_model(
    checkpoint: str,
    device: torch.device,
) -> Tuple[PatchTSTFM, PatchTSTFMConfig, int]:
    """Load a PatchTSTFM model from a checkpoint.

    Accepts every layout :mod:`tabby.checkpoint` supports: a Hugging Face
    snapshot directory (``config.json`` + ``model.safetensors``), an in-house
    ``pytorch_model.bin``, the step directory holding one, a parent directory
    with a ``latest`` symlink, or a Hub repository id.

    Returns (model, config, step).  The model is on ``device`` in eval mode.
    """
    model, cfg, step = load_backbone(checkpoint, device)
    logger.info("Loaded model at step=%d  config=%s", step, cfg)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return model, cfg, step


# =============================================================================
# Predictor
# =============================================================================

class PatchTSTFMPredictor:
    """Loads a PatchTSTFM checkpoint once and exposes latent embeddings.

    Parameters
    ----------
    checkpoint : str
        Path to a checkpoint directory, a ``pytorch_model.bin`` file,
        or a parent directory containing a ``latest`` symlink.
    batch_size : int
        Inference batch size (number of univariate series per forward pass).
    device : str
        PyTorch device string, e.g. ``"cuda:0"`` or ``"cpu"``.
    precision : str
        One of ``"bf16"``, ``"fp16"``, ``"fp32"``.
    """

    def __init__(
        self,
        checkpoint: str,
        batch_size: int = 256,
        device: str = "cuda",
        precision: str = "bf16",
    ) -> None:
        self.device = torch.device(
            device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        )
        self.model, self.cfg, self.step = load_model(checkpoint, self.device)
        self.batch_size = int(batch_size)

        if precision == "bf16":
            self.amp_dtype = torch.bfloat16
            self.use_amp = self.device.type == "cuda"
        elif precision == "fp16":
            self.amp_dtype = torch.float16
            self.use_amp = self.device.type == "cuda"
        else:
            self.amp_dtype = torch.float32
            self.use_amp = False

        logger.info(
            "PatchTSTFMPredictor ready: step=%d, device=%s, precision=%s",
            self.step, self.device, precision,
        )

    @torch.no_grad()
    def embed(
        self,
        windows: Sequence[np.ndarray],
        layer: int = -1,
    ) -> np.ndarray:
        """Per-patch latent embeddings for each window (no forecasting).

        Each window's real content is right-aligned inside the model's fixed
        context buffer (left-padded, with the pad marked as padding so it is
        masked in attention); nothing is masked for prediction. The frozen
        encoder is run and the pre-head hidden state returned.

        Returns ``np.ndarray`` of shape ``[len(windows), n_patch, d_model]``.
        ``layer`` is passed to :meth:`PatchTSTFM.encode`: a negative value runs
        all blocks, while a non-negative value is clamped to at least one block.
        """
        W = self.cfg.context_length
        out_chunks: List[np.ndarray] = []

        for start in range(0, len(windows), self.batch_size):
            wb = windows[start: start + self.batch_size]
            B = len(wb)

            x = np.zeros((B, W), dtype=np.float32)
            observed = np.zeros((B, W), dtype=np.bool_)
            padding = np.ones((B, W), dtype=np.bool_)
            pred = np.zeros((B, W), dtype=np.bool_)   # nothing predicted -> pure encode

            for i, win in enumerate(wb):
                h = np.asarray(win, dtype=np.float32).ravel()
                if h.shape[0] > W:
                    h = h[-W:]
                finite = np.isfinite(h)
                h_clean = np.where(finite, h, 0.0)
                off = W - h_clean.shape[0]              # right-align real content
                x[i, off:] = h_clean
                observed[i, off:] = finite
                padding[i, off:] = False

            xt = torch.from_numpy(x).to(device=self.device, dtype=torch.float32)
            obst = torch.from_numpy(observed).to(device=self.device, dtype=torch.bool)
            padt = torch.from_numpy(padding).to(device=self.device, dtype=torch.bool)
            predt = torch.from_numpy(pred).to(device=self.device, dtype=torch.bool)

            x_norm, mean, std, union_mask, patch_pad = mask_aware_normalize_for_inference(
                x=xt, observed_mask=obst, pred_mask=predt, padding_mask=padt, cfg=self.cfg,
            )
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                h = self.model.encode(
                    x_norm=x_norm,
                    union_mask=union_mask,
                    patch_key_padding_mask=patch_pad,
                    layer=layer,
                )
            out_chunks.append(h.float().cpu().numpy())

        return np.concatenate(out_chunks, axis=0)  # [n_windows, n_patch, d_model]
