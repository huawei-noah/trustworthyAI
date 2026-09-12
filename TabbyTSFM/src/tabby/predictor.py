#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GluonTS-compatible predictor for Tabby-Pretrain.
 
Usage:
    from tabby.predictor import PatchTSTFMPredictor
 
    predictor = PatchTSTFMPredictor(
        checkpoint="/path/to/step_0165000",
        prediction_length=96,
        device="cuda:0",
    )
    forecasts = predictor.predict(test_data.input)
"""
 
from __future__ import annotations
 
import logging
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
 
import numpy as np
import torch
 
from gluonts.model.forecast import Forecast, QuantileForecast
 
from .checkpoint import load_backbone
from .models.PatchTSTFM import PatchTSTFM, PatchTSTFMConfig
from .utils.input_preprocessing import (
    inverse_asinh_normalize,
    mask_aware_normalize_for_inference,
)
 
logger = logging.getLogger(__name__)
 
 
# =============================================================================
# Checkpoint loading
# =============================================================================
 
def load_model(
    checkpoint: str,
    device: torch.device,
) -> Tuple[PatchTSTFM, PatchTSTFMConfig, int]:
    """Load a PatchTSTFM model from a checkpoint.
 
    Accepts a Hugging Face snapshot directory (``config.json`` +
    ``model.safetensors``, such as the published Tabby-Pretrain weights or a
    ``huggingface-cli download`` target), an in-house ``pytorch_model.bin``, the
    step directory holding one, a parent directory with a ``latest`` symlink, or
    a Hub repository id.
 
    Returns (model, config, step).  The model is on ``device`` in eval mode.
    """
    model, cfg, step = load_backbone(checkpoint, device)
    logger.info("Loaded model at step=%d  config=%s", step, cfg)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return model, cfg, step
 
 
# =============================================================================
# Quantile-level helpers
# =============================================================================
 
def model_quantile_levels(num_quantiles: int) -> np.ndarray:
    """Reproduce the quantile grid used during training."""
    if num_quantiles == 99:
        return np.arange(1, 100, dtype=np.float64) / 100.0
    return np.linspace(
        1.0 / (num_quantiles + 1),
        num_quantiles / (num_quantiles + 1),
        num_quantiles,
        dtype=np.float64,
    )
 
 
def _resolve_quantile_indices(
    model_q: np.ndarray,
    requested_q: Sequence[float],
) -> List[int]:
    """Find model quantile indices closest to each requested level."""
    indices = []
    for q in requested_q:
        idx = int(np.argmin(np.abs(model_q - float(q))))
        if abs(float(model_q[idx]) - float(q)) > 1e-6:
            raise ValueError(
                f"Requested quantile {q} not available in model "
                f"(nearest is {model_q[idx]:.6f} at index {idx})."
            )
        indices.append(idx)
    return indices
 
 
# =============================================================================
# Input construction
# =============================================================================
 
def _target_to_univariate_list(target: object) -> List[np.ndarray]:
    """Split a GluonTS target into one or more univariate float32 arrays."""
    arr = np.asarray(target, dtype=np.float32)
    if arr.ndim == 1:
        return [arr]
    if arr.ndim != 2:
        raise ValueError(f"Unsupported target ndim={arr.ndim}, shape={arr.shape}")
    # Convention: [channels, time] if shape[0] <= shape[1], else [time, channels].
    if arr.shape[0] <= arr.shape[1]:
        return [arr[i] for i in range(arr.shape[0])]
    return [arr[:, i] for i in range(arr.shape[1])]
 
 
def _target_time_length(target: object) -> int:
    """Return the time-axis length regardless of 1D/2D layout."""
    arr = np.asarray(target)
    if arr.ndim == 1:
        return arr.shape[0]
    if arr.ndim == 2:
        return arr.shape[1] if arr.shape[0] <= arr.shape[1] else arr.shape[0]
    raise ValueError(f"Unsupported target ndim={arr.ndim}")
 
 
def _build_forecast_inputs(
    histories: Sequence[np.ndarray],
    horizon: int,
    context_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build [left-pad | visible history | masked future] tensors.
 
    Layout (T = context_length):
        [padding ... | history values | horizon zeros]
         padding=True  padding=False    padding=False
         observed=F    observed=finite  observed=False
         pred=False    pred=False       pred=True
 
    Returns (x, observed_mask, padding_mask, pred_mask), all shape [B, T].
    """
    B = len(histories)
    T = context_length
 
    if not (1 <= horizon < T):
        raise ValueError(
            f"horizon={horizon} must satisfy 1 <= horizon < context_length={T}."
        )
 
    max_hist = T - horizon
    future_start = T - horizon
 
    x = np.zeros((B, T), dtype=np.float32)
    observed = np.zeros((B, T), dtype=np.bool_)
    padding = np.ones((B, T), dtype=np.bool_)
    pred = np.zeros((B, T), dtype=np.bool_)
 
    # Future horizon: not padding, is prediction target.
    padding[:, future_start:] = False
    pred[:, future_start:] = True
 
    for i, hist in enumerate(histories):
        h = np.asarray(hist, dtype=np.float32).ravel()
        finite = np.isfinite(h)
        h_clean = np.where(finite, h, 0.0)
 
        # Truncate to max_hist if longer.
        if h_clean.shape[0] > max_hist:
            h_clean = h_clean[-max_hist:]
            finite = finite[-max_hist:]
 
        n = h_clean.shape[0]
        start = future_start - n
 
        if n > 0:
            x[i, start:future_start] = h_clean
            observed[i, start:future_start] = finite
            padding[i, start:future_start] = False
 
    return (
        torch.from_numpy(x),
        torch.from_numpy(observed),
        torch.from_numpy(padding),
        torch.from_numpy(pred),
    )
 
 
# =============================================================================
# Core inference
# =============================================================================
 
@torch.no_grad()
def _forecast_single_horizon(
    model: PatchTSTFM,
    cfg: PatchTSTFMConfig,
    histories: Sequence[np.ndarray],
    horizon: int,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> np.ndarray:
    """Forecast ``horizon`` steps in a single model call (batched).
 
    Returns np.ndarray of shape [len(histories), horizon, num_quantiles]
    in the **original value scale** (denormalized).
    """
    all_chunks: List[np.ndarray] = []
 
    for start in range(0, len(histories), batch_size):
        batch_hist = histories[start : start + batch_size]
 
        x, obs, pad, pred = _build_forecast_inputs(batch_hist, horizon, cfg.context_length)
        x = x.to(device=device, dtype=torch.float32)
        obs = obs.to(device=device, dtype=torch.bool)
        pad = pad.to(device=device, dtype=torch.bool)
        pred = pred.to(device=device, dtype=torch.bool)
 
        # ---- Normalize (inference path returns mean, std for denorm) ----
        x_norm, mean, std, union_mask, patch_pad = mask_aware_normalize_for_inference(
            x=x,
            observed_mask=obs,
            pred_mask=pred,
            padding_mask=pad,
            cfg=cfg,
        )
 
        # ---- Forward ----
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            q_norm = model(
                x_norm=x_norm,
                union_mask=union_mask,
                patch_key_padding_mask=patch_pad,
            )
 
        # ---- Denormalize only the future slice ----
        # q_norm: [B, T, K]  →  slice the last `horizon` timesteps
        q_norm_future = q_norm[:, cfg.context_length - horizon :, :].float()
        # mean, std: [B, 1]  →  broadcast to [B, 1, 1] against [B, H, K]
        q_future = inverse_asinh_normalize(q_norm_future, mean.unsqueeze(-1), std.unsqueeze(-1))
 
        all_chunks.append(q_future.cpu().numpy())
 
    return np.concatenate(all_chunks, axis=0)  # [total_series, H, K]
 
 
def _forecast_recursive(
    model: PatchTSTFM,
    cfg: PatchTSTFMConfig,
    histories: Sequence[np.ndarray],
    horizon: int,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> np.ndarray:
    """Recursive median-extension for horizons longer than context_length - 1."""
    max_single = cfg.context_length - 1
 
    logger.info(
        "Recursive forecast: horizon=%d, max_single=%d, chunks needed=%d",
        horizon, max_single, (horizon + max_single - 1) // max_single,
    )
 
    model_q = model_quantile_levels(cfg.num_quantiles)
    median_idx = int(np.argmin(np.abs(model_q - 0.5)))
 
    current = [np.asarray(h, dtype=np.float32).ravel().copy() for h in histories]
    remaining = horizon
    chunks: List[np.ndarray] = []
 
    while remaining > 0:
        chunk_h = min(remaining, max_single)
        chunk_q = _forecast_single_horizon(
            model, cfg, current, chunk_h, batch_size, device, amp_dtype, use_amp,
        )
        chunks.append(chunk_q)
 
        # Extend histories with the median forecast for the next chunk.
        median = chunk_q[:, :, median_idx]  # [B, chunk_h]
        for i in range(len(current)):
            current[i] = np.concatenate([current[i], median[i]], axis=0)
 
        remaining -= chunk_h
 
    return np.concatenate(chunks, axis=1)  # [B, horizon, K]
 
 
# =============================================================================
# GluonTS Predictor
# =============================================================================
 
class PatchTSTFMPredictor:
    """GluonTS-compatible predictor for PatchTSTFM.
 
    Loads the model once at init. Call ``predict(test_data_input)`` to get
    a list of ``QuantileForecast`` objects.
 
    Parameters
    ----------
    checkpoint : str
        A Hugging Face snapshot directory (``config.json`` +
        ``model.safetensors``), an in-house ``pytorch_model.bin`` file, the
        step directory holding one, a parent directory containing a
        ``latest`` symlink, or a Hugging Face Hub repository id.
    prediction_length : int
        Number of future timesteps to forecast.
    batch_size : int
        Inference batch size (number of univariate series per forward pass).
    device : str
        PyTorch device string, e.g. ``"cuda:0"`` or ``"cpu"``.
    precision : str
        One of ``"bf16"``, ``"fp16"``, ``"fp32"``.
    quantile_levels : sequence of float
        Which quantiles to include in the output forecasts.
        Must be a subset of the model's trained quantile grid.
    """
 
    def __init__(
        self,
        checkpoint: str,
        prediction_length: int,
        batch_size: int = 256,
        device: str = "cuda",
        precision: str = "bf16",
        quantile_levels: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    ) -> None:
        self.device = torch.device(
            device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        )
        self.model, self.cfg, self.step = load_model(checkpoint, self.device)
        self.prediction_length = int(prediction_length)
        self.batch_size = int(batch_size)
 
        # AMP setup
        if precision == "bf16":
            self.amp_dtype = torch.bfloat16
            self.use_amp = self.device.type == "cuda"
        elif precision == "fp16":
            self.amp_dtype = torch.float16
            self.use_amp = self.device.type == "cuda"
        else:
            self.amp_dtype = torch.float32
            self.use_amp = False
 
        # Resolve which model quantile indices correspond to the requested levels.
        model_q = model_quantile_levels(self.cfg.num_quantiles)
        self.quantile_indices = _resolve_quantile_indices(model_q, quantile_levels)
        self.quantile_levels = [float(q) for q in quantile_levels]
 
        logger.info(
            "PatchTSTFMPredictor ready: step=%d, prediction_length=%d, "
            "device=%s, precision=%s, quantiles=%s",
            self.step, self.prediction_length, self.device, precision,
            self.quantile_levels,
        )
 
    def _forecast_flat(self, histories: Sequence[np.ndarray]) -> np.ndarray:
        """Run inference on a flat list of univariate histories.
 
        Returns [len(histories), prediction_length, len(quantile_levels)]
        in original scale.
        """
        max_single = self.cfg.context_length - 1
 
        if self.prediction_length <= max_single:
            raw = _forecast_single_horizon(
                self.model, self.cfg, histories, self.prediction_length,
                self.batch_size, self.device, self.amp_dtype, self.use_amp,
            )
        else:
            raw = _forecast_recursive(
                self.model, self.cfg, histories, self.prediction_length,
                self.batch_size, self.device, self.amp_dtype, self.use_amp,
            )
 
        # Select only the requested quantile levels.
        return raw[:, :, self.quantile_indices]  # [B, H, Q]
 
    def predict(
        self,
        test_data_input: Iterable[Dict],
    ) -> List[Forecast]:
        """Generate quantile forecasts for each entry in ``test_data_input``.
 
        Each entry is a dict with at least ``"target"`` and ``"start"`` keys,
        following the GluonTS convention.
 
        Multivariate targets are expanded to univariate channels, forecast
        independently, and reassembled into a single ``QuantileForecast``.
        """
        items = list(test_data_input)
 
        # Flatten all items into univariate histories.
        flat_histories: List[np.ndarray] = []
        channel_counts: List[int] = []
 
        for item in items:
            channels = _target_to_univariate_list(item["target"])
            flat_histories.extend(channels)
            channel_counts.append(len(channels))
 
        # Batched inference over all univariate series at once.
        flat_q = self._forecast_flat(flat_histories)  # [total_series, H, Q]
 
        # Reassemble into per-item QuantileForecast objects.
        forecasts: List[Forecast] = []
        offset = 0
 
        for item, n_ch in zip(items, channel_counts):
            arr = flat_q[offset : offset + n_ch]  # [C, H, Q]
            offset += n_ch
 
            forecast_start = item["start"] + _target_time_length(item["target"])
 
            if n_ch == 1:
                # Univariate: QuantileForecast expects [Q, H]
                forecast_arrays = arr[0].T  # [H, Q] → [Q, H]
            else:
                # Multivariate: QuantileForecast expects [Q, H, C]
                forecast_arrays = np.transpose(arr, (2, 1, 0))  # [C, H, Q] → [Q, H, C]
 
            forecasts.append(
                QuantileForecast(
                    forecast_arrays=forecast_arrays.astype(np.float32),
                    forecast_keys=[str(q) for q in self.quantile_levels],
                    start_date=forecast_start,
                )
            )
 
        return forecasts
