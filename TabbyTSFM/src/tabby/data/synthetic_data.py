#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CauKer V2 generation and pre-generated KernelSynth Arrow readers.

The offline KernelSynth shard generator is intentionally not part of this
repository. ArrowSyntheticIterableDataset consumes shards produced ahead of
training and does not generate those shards itself.
"""

from __future__ import annotations

import functools
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info

from .real_data import extract_target_from_entry, format_real_training_sample


# 2. Online multi-base CauKer V2 generator
# =============================================================================

# =============================================================================
# 0. Global generator weights
# =============================================================================

DEFAULT_GENERATOR_WEIGHTS: Dict[str, float] = {
    "kernel_gp": 0.1,
    "trend_seasonality": 0.1,
    "sde_ou": 0.1,
    "arima_state_space": 0.10,
    "step_changepoint": 0.10,
    "spike_event": 0.3,
    "waveform": 0.05,
    "fbm_fgn": 0.05,
    "garch_volatility": 0.05,
    "chaotic_timesynth_audio": 0.05,
}


# =============================================================================
# 1. Utility functions
# =============================================================================


def _safe_std(x: np.ndarray, eps: float = 1e-6) -> float:
    """Return a numerically safe standard deviation."""
    std = float(np.nanstd(x))
    if not np.isfinite(std) or std < eps:
        return 1.0
    return std


def robust_normalize(x: np.ndarray, clip: float = 8.0, eps: float = 1e-6) -> np.ndarray:
    """
    Normalize a 1D series to approximately zero mean and unit variance.

    Steps:
        1. Replace non-finite values by zero.
        2. Subtract the median, which is robust to spikes.
        3. Divide by a safe standard deviation.
        4. Clip extreme values.
        5. Re-standardize after clipping.
    """
    y = np.asarray(x, dtype=np.float64).copy()
    y[~np.isfinite(y)] = 0.0

    med = float(np.median(y))
    y = y - med

    y = y / _safe_std(y, eps=eps)
    y = np.clip(y, -clip, clip)

    y = y - float(np.mean(y))
    y = y / _safe_std(y, eps=eps)
    return y.astype(np.float64)


def normalize_matrix_columns(x: np.ndarray, clip: float = 8.0) -> np.ndarray:
    """Apply robust_normalize independently to each column of a [T, P] matrix."""
    out = np.empty_like(x, dtype=np.float64)
    for j in range(x.shape[1]):
        out[:, j] = robust_normalize(x[:, j], clip=clip)
    return out


def log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    """Sample from LogUniform(low, high)."""
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def colored_noise(length: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    """
    Generate approximate 1/f^beta colored noise by shaping Fourier amplitudes.

    beta = 0 gives white noise.
    beta = 1 gives pink-like noise.
    beta = 2 gives brown-like noise.
    """
    white = rng.normal(size=length)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(length)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    scale = 1.0 / np.power(freqs, beta / 2.0)
    spectrum = spectrum * scale
    y = np.fft.irfft(spectrum, n=length)
    return robust_normalize(y)


def triangle_wave(phase: np.ndarray) -> np.ndarray:
    """Triangle wave with phase measured in cycles."""
    frac = phase - np.floor(phase)
    return 4.0 * np.abs(frac - 0.5) - 1.0


def sawtooth_wave(phase: np.ndarray, width: float = 1.0) -> np.ndarray:
    """
    Sawtooth wave with phase measured in cycles.

    width=1.0 gives an increasing ramp followed by a jump.
    width=0.0 gives a decreasing ramp.
    Intermediate width gives asymmetric sawtooth.
    """
    frac = phase - np.floor(phase)
    width = float(np.clip(width, 1e-3, 1.0 - 1e-3))
    y = np.where(
        frac < width,
        2.0 * frac / width - 1.0,
        1.0 - 2.0 * (frac - width) / (1.0 - width),
    )
    return y


def square_wave(phase: np.ndarray, duty: float = 0.5) -> np.ndarray:
    """Square wave with phase measured in cycles."""
    frac = phase - np.floor(phase)
    return np.where(frac < duty, 1.0, -1.0)


def build_weighted_schedule(
    total: int,
    weights: Dict[str, float],
    rng: np.random.Generator,
    ensure_min_one_when_possible: bool = True,
) -> List[str]:
    """
    Build an approximately exact weighted schedule of generator names.

    This makes the empirical generator proportions match the requested weights
    more closely than independent categorical sampling, especially for small data.
    """
    if total <= 0:
        return []

    names = list(weights.keys())
    w = np.array([weights[name] for name in names], dtype=np.float64)
    w = w / w.sum()

    raw = w * total
    counts = np.floor(raw).astype(int)

    if ensure_min_one_when_possible and total >= len(names):
        for i in range(len(names)):
            if counts[i] == 0 and w[i] > 0:
                counts[i] = 1

    # Fix over-allocation caused by the min-one rule.
    while counts.sum() > total:
        idx = int(np.argmax(counts - raw))
        counts[idx] -= 1

    # Distribute the remaining slots by largest fractional remainders.
    remainder = total - int(counts.sum())
    if remainder > 0:
        frac_order = np.argsort(-(raw - np.floor(raw)))
        for idx in frac_order[:remainder]:
            counts[int(idx)] += 1

    schedule: List[str] = []
    for name, count in zip(names, counts):
        schedule.extend([name] * int(count))

    rng.shuffle(schedule)
    return schedule


# =============================================================================
# 2. Primitive/base generators
# =============================================================================

class CauKerKernel:
    """Small dependency-free kernel algebra matching the CauKer kernel bank."""

    def __init__(
        self,
        kind: str,
        *,
        periodicity: float = 1.0,
        sigma_0: float = 0.0,
        length_scale: float = 1.0,
        alpha: float = 1.0,
        noise_level: float = 1.0,
        left: Optional["CauKerKernel"] = None,
        right: Optional["CauKerKernel"] = None,
    ) -> None:
        self.kind = kind
        self.periodicity = float(periodicity)
        self.sigma_0 = float(sigma_0)
        self.length_scale = float(length_scale)
        self.alpha = float(alpha)
        self.noise_level = float(noise_level)
        self.left = left
        self.right = right

    def __add__(self, other: "CauKerKernel") -> "CauKerKernel":
        return CauKerKernel("sum", left=self, right=other)

    def __mul__(self, other: "CauKerKernel") -> "CauKerKernel":
        return CauKerKernel("product", left=self, right=other)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x1 = np.asarray(x, dtype=np.float64).reshape(-1, 1)
        if self.kind == "sum":
            if self.left is None or self.right is None:
                raise ValueError("Composite sum kernel is missing operands.")
            return self.left(x1) + self.right(x1)
        if self.kind == "product":
            if self.left is None or self.right is None:
                raise ValueError("Composite product kernel is missing operands.")
            return self.left(x1) * self.right(x1)

        diff = x1 - x1.T
        sqdist = diff * diff
        if self.kind == "exp_sine_squared":
            period = max(abs(self.periodicity), 1e-6)
            length_scale = max(abs(self.length_scale), 1e-6)
            sine = np.sin(np.pi * diff / period)
            return np.exp(-2.0 * sine * sine / (length_scale * length_scale))
        if self.kind == "dot_product":
            return x1 @ x1.T + self.sigma_0 * self.sigma_0
        if self.kind == "rbf":
            length_scale = max(abs(self.length_scale), 1e-6)
            return np.exp(-0.5 * sqdist / (length_scale * length_scale))
        if self.kind == "rational_quadratic":
            alpha = max(abs(self.alpha), 1e-6)
            length_scale = max(abs(self.length_scale), 1e-6)
            return np.power(1.0 + sqdist / (2.0 * alpha * length_scale * length_scale), -alpha)
        if self.kind == "white":
            return self.noise_level * np.eye(x1.shape[0], dtype=np.float64)
        if self.kind == "constant":
            return np.ones((x1.shape[0], x1.shape[0]), dtype=np.float64)
        raise ValueError(f"Unknown CauKer kernel kind: {self.kind}")


def build_cauker_kernel_bank(time_length: int) -> List[CauKerKernel]:
    """Return CauKer's time-length-parameterized base kernel bank."""
    length = max(1, int(time_length))
    return [
        CauKerKernel("exp_sine_squared", periodicity=24 / length),
        CauKerKernel("exp_sine_squared", periodicity=48 / length),
        CauKerKernel("exp_sine_squared", periodicity=96 / length),
        CauKerKernel("exp_sine_squared", periodicity=24 * 7 / length),
        CauKerKernel("exp_sine_squared", periodicity=48 * 7 / length),
        CauKerKernel("exp_sine_squared", periodicity=96 * 7 / length),
        CauKerKernel("exp_sine_squared", periodicity=7 / length),
        CauKerKernel("exp_sine_squared", periodicity=14 / length),
        CauKerKernel("exp_sine_squared", periodicity=30 / length),
        CauKerKernel("exp_sine_squared", periodicity=60 / length),
        CauKerKernel("exp_sine_squared", periodicity=365 / length),
        CauKerKernel("exp_sine_squared", periodicity=365 * 2 / length),
        CauKerKernel("exp_sine_squared", periodicity=4 / length),
        CauKerKernel("exp_sine_squared", periodicity=26 / length),
        CauKerKernel("exp_sine_squared", periodicity=52 / length),
        CauKerKernel("exp_sine_squared", periodicity=4 / length),
        CauKerKernel("exp_sine_squared", periodicity=6 / length),
        CauKerKernel("exp_sine_squared", periodicity=12 / length),
        CauKerKernel("exp_sine_squared", periodicity=4 / length),
        CauKerKernel("exp_sine_squared", periodicity=(4 * 10) / length),
        CauKerKernel("exp_sine_squared", periodicity=10 / length),
        CauKerKernel("dot_product", sigma_0=0.0),
        CauKerKernel("dot_product", sigma_0=1.0),
        CauKerKernel("dot_product", sigma_0=10.0),
        CauKerKernel("rbf", length_scale=0.1),
        CauKerKernel("rbf", length_scale=1.0),
        CauKerKernel("rbf", length_scale=10.0),
        CauKerKernel("rational_quadratic", alpha=0.1),
        CauKerKernel("rational_quadratic", alpha=1.0),
        CauKerKernel("rational_quadratic", alpha=10.0),
        CauKerKernel("white", noise_level=0.1),
        CauKerKernel("white", noise_level=1.0),
        CauKerKernel("constant"),
    ]


def random_cauker_kernel_combination(
    a: CauKerKernel,
    b: CauKerKernel,
    rng: np.random.Generator,
) -> CauKerKernel:
    """Randomly combine two CauKer kernels by addition or product."""
    return a + b if rng.random() < 0.5 else a * b


def cauker_zero_mean(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.zeros_like(x, dtype=np.float64)


def cauker_linear_mean(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1.0, 1.0) * x + rng.uniform(-1.0, 1.0)


def cauker_exponential_mean(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.5, 1.5) * np.exp(rng.uniform(0.5, 1.5) * x)


def cauker_anomaly_mean(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    m = np.zeros_like(x, dtype=np.float64)
    for _ in range(int(rng.integers(1, 6))):
        m[int(rng.integers(0, len(x)))] += rng.uniform(-5.0, 5.0)
    return m


def random_cauker_mean_combination(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Pick two CauKer mean functions and combine them by addition or product."""
    mean_functions = [
        cauker_zero_mean,
        cauker_linear_mean,
        cauker_exponential_mean,
        cauker_anomaly_mean,
    ]
    i1, i2 = rng.integers(0, len(mean_functions), size=2)
    m1, m2 = mean_functions[int(i1)], mean_functions[int(i2)]
    return m1(x, rng) + m2(x, rng) if rng.random() < 0.5 else m1(x, rng) * m2(x, rng)


def sample_from_cauker_gp_prior(
    kernel: CauKerKernel,
    x: np.ndarray,
    rng: np.random.Generator,
    mean_vec: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Draw one CauKer GP-prior realization with a NumPy backend."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = x.shape[0]
    mean = np.zeros(n, dtype=np.float64) if mean_vec is None else np.asarray(mean_vec, dtype=np.float64)
    cov = kernel(x)
    cov = 0.5 * (cov + cov.T)
    cov = cov + 1e-6 * np.eye(n, dtype=np.float64)
    try:
        chol = np.linalg.cholesky(cov)
        return mean + chol @ rng.normal(size=n)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 0.0, None)
        return mean + eigvecs @ (np.sqrt(eigvals) * rng.normal(size=n))


def generate_kernel_gp(length: int, rng: np.random.Generator) -> np.ndarray:
    """
    CauKer kernel GP generator.

    The kernel bank and random kernel algebra are taken from the provided CauKer
    implementation. Other primitive generators in this file are intentionally
    left unchanged.
    """
    t = np.linspace(0.0, 1.0, length, dtype=np.float64)
    kernel_bank = build_cauker_kernel_bank(length)
    selected_indices = rng.integers(0, len(kernel_bank), size=int(rng.integers(1, 8)))
    selected = [kernel_bank[int(i)] for i in selected_indices]
    kernel = functools.reduce(lambda a, b: random_cauker_kernel_combination(a, b, rng), selected)
    mean_vec = random_cauker_mean_combination(t, rng)
    y = sample_from_cauker_gp_prior(kernel, t, rng, mean_vec=mean_vec)
    return robust_normalize(y)


def generate_trend_seasonality(length: int, rng: np.random.Generator) -> np.ndarray:
    """
    ETS/TSI/ForecastPFN-like generator.

    It combines trend, seasonality, and irregular components in additive or
    multiplicative forms.
    """
    t = np.linspace(0.0, 1.0, length, dtype=np.float64)

    # Trend component.
    trend_type = str(rng.choice(["none", "linear", "quadratic", "exponential", "damped", "piecewise"],
                                p=[0.08, 0.27, 0.20, 0.15, 0.15, 0.15]))
    trend = np.zeros(length, dtype=np.float64)
    if trend_type == "linear":
        trend = rng.uniform(-3.0, 3.0) * (t - 0.5)
    elif trend_type == "quadratic":
        trend = rng.uniform(-4.0, 4.0) * (t - 0.5) ** 2 + rng.uniform(-2.0, 2.0) * (t - 0.5)
    elif trend_type == "exponential":
        rate = rng.uniform(-3.0, 3.0)
        trend = rng.uniform(0.3, 2.0) * (np.exp(rate * t) - np.exp(rate * 0.5))
    elif trend_type == "damped":
        rate = rng.uniform(1.0, 8.0)
        trend = rng.uniform(-3.0, 3.0) * (1.0 - np.exp(-rate * t))
    elif trend_type == "piecewise":
        cp = int(rng.integers(length // 5, max(length // 5 + 1, 4 * length // 5)))
        slope1 = rng.uniform(-3.0, 3.0)
        slope2 = rng.uniform(-3.0, 3.0)
        trend[:cp] = slope1 * (t[:cp] - t[0])
        trend[cp:] = trend[cp - 1] + slope2 * (t[cp:] - t[cp])

    # Seasonality component.
    season = np.zeros(length, dtype=np.float64)
    n_seasons = int(rng.integers(0, 4))
    for _ in range(n_seasons):
        period = float(rng.choice([4, 6, 7, 12, 24, 48, 52, 96, 168, 336]))
        if period > length / 2:
            period = rng.uniform(8.0, max(9.0, length / 3.0))
        amp = rng.uniform(0.2, 2.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        seasonal_type = str(rng.choice(["sin", "triangle", "step", "impulse"], p=[0.55, 0.20, 0.15, 0.10]))
        cycles = np.arange(length, dtype=np.float64) / period + phase / (2.0 * np.pi)
        if seasonal_type == "sin":
            season += amp * np.sin(2.0 * np.pi * cycles)
        elif seasonal_type == "triangle":
            season += amp * triangle_wave(cycles)
        elif seasonal_type == "step":
            season += amp * square_wave(cycles, duty=float(rng.uniform(0.2, 0.8)))
        elif seasonal_type == "impulse":
            frac = cycles - np.floor(cycles)
            width = rng.uniform(0.02, 0.12)
            season += amp * np.exp(-0.5 * (np.minimum(frac, 1.0 - frac) / width) ** 2)

    # Irregular component.
    irregular_type = str(rng.choice(["white", "colored", "rw", "fbm_like"], p=[0.35, 0.35, 0.15, 0.15]))
    if irregular_type == "white":
        irregular = rng.normal(scale=rng.uniform(0.05, 0.5), size=length)
    elif irregular_type == "colored":
        irregular = rng.uniform(0.05, 0.5) * colored_noise(length, beta=rng.uniform(0.3, 1.5), rng=rng)
    elif irregular_type == "rw":
        irregular = np.cumsum(rng.normal(scale=rng.uniform(0.005, 0.05), size=length))
    else:
        irregular = rng.uniform(0.05, 0.5) * generate_fbm_fgn(length, rng)

    combination = str(rng.choice(["additive", "multiplicative", "mixed"], p=[0.55, 0.20, 0.25]))
    if combination == "additive":
        y = trend + season + irregular
    elif combination == "multiplicative":
        base = 1.0 + 0.25 * robust_normalize(trend)
        seasonal = 1.0 + 0.25 * robust_normalize(season) if np.std(season) > 1e-8 else 1.0
        noise = 1.0 + 0.15 * robust_normalize(irregular)
        y = base * seasonal * noise
    else:
        y = trend + season * (1.0 + 0.15 * robust_normalize(irregular)) + irregular

    return robust_normalize(y)


def generate_arima_state_space(length: int, rng: np.random.Generator) -> np.ndarray:
    """Generate an ARIMA/SARIMA-like linear stochastic sequence."""
    burn = 300
    total = length + burn

    p = int(rng.choice([0, 1, 2, 3], p=[0.1, 0.3, 0.4, 0.2]))
    q = int(rng.choice([0, 1, 2, 3], p=[0.1, 0.3, 0.4, 0.2]))
    if p == 0 and q == 0:
        p = 1
    d = int(rng.choice([0, 1, 2], p=[0.50, 0.40, 0.10]))

    ar = rng.uniform(-0.8, 0.8, size=p)
    if p > 0:
        ar_sum = float(np.sum(np.abs(ar)))
        if ar_sum > 0.92:
            ar = ar * (0.92 / ar_sum)

    ma = rng.uniform(-0.8, 0.8, size=q)
    if q > 0:
        ma_sum = float(np.sum(np.abs(ma)))
        if ma_sum > 0.92:
            ma = ma * (0.92 / ma_sum)

    sigma = log_uniform(rng, 0.02, 1.0)
    eps = rng.normal(scale=sigma, size=total + max(p, q) + 10)
    y = np.zeros_like(eps)

    for t in range(max(p, q) + 1, len(eps)):
        ar_part = 0.0
        for j in range(1, p + 1):
            ar_part += ar[j - 1] * y[t - j]
        ma_part = 0.0
        for j in range(1, q + 1):
            ma_part += ma[j - 1] * eps[t - j]
        y[t] = ar_part + eps[t] + ma_part

    y = y[-total:]

    # Optional seasonal AR component.
    if rng.random() < 0.35:
        m = int(rng.choice([4, 7, 12, 24, 52, 96]))
        phi_s = rng.uniform(-0.5, 0.5)
        for t in range(m, total):
            y[t] += phi_s * y[t - m]

    # Integration.
    for _ in range(d):
        y = np.cumsum(y)

    return robust_normalize(y[burn:])


def generate_sde_ou(length: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a regime-switching Ornstein-Uhlenbeck process."""
    k = int(rng.choice([2, 3, 4], p=[0.50, 0.30, 0.20]))
    base_mu = rng.uniform(-2.0, 2.0)
    offsets = rng.choice([-1.0, 1.0], size=k) * rng.uniform(0.5, 4.0, size=k)
    mus = base_mu + offsets
    thetas = rng.uniform(0.03, 3.0, size=k)
    sigmas = rng.uniform(0.03, 1.5, size=k)

    # Transition matrix with strong diagonal concentration.
    transition = np.zeros((k, k), dtype=np.float64)
    concentration = rng.uniform(10.0, 50.0)
    for i in range(k):
        alpha = np.ones(k)
        alpha[i] = concentration
        transition[i] = rng.dirichlet(alpha)

    x = np.zeros(length, dtype=np.float64)
    state = int(rng.integers(0, k))
    x[0] = mus[state] + rng.normal(scale=sigmas[state])

    # Optional slow seasonal mean perturbation.
    seasonal_amp = rng.uniform(0.0, 0.8) if rng.random() < 0.4 else 0.0
    seasonal_period = rng.uniform(24.0, max(25.0, length / 2.0))
    phase = rng.uniform(0.0, 2.0 * np.pi)

    for t in range(1, length):
        state = int(rng.choice(k, p=transition[state]))
        theta = thetas[state]
        sigma = sigmas[state]
        mu_t = mus[state] + seasonal_amp * np.sin(2.0 * np.pi * t / seasonal_period + phase)
        decay = math.exp(-theta)
        innovation_std = sigma * math.sqrt(max(1e-8, (1.0 - math.exp(-2.0 * theta)) / (2.0 * theta)))
        x[t] = mu_t + (x[t - 1] - mu_t) * decay + innovation_std * rng.normal()

    return robust_normalize(x)


def generate_step_changepoint(length: int, rng: np.random.Generator) -> np.ndarray:
    """Generate piecewise-constant or piecewise-smooth level-shift sequences."""
    nseg = int(min(max(2, rng.geometric(0.15) + 1), max(3, length // 8)))
    proportions = rng.dirichlet(np.ones(nseg) * rng.uniform(0.5, 2.0))
    lengths = np.maximum(1, np.round(proportions * length).astype(int))
    diff = length - int(lengths.sum())
    lengths[-1] += diff
    lengths = np.maximum(1, lengths)
    while lengths.sum() > length:
        idx = int(np.argmax(lengths))
        lengths[idx] -= 1
    while lengths.sum() < length:
        lengths[int(rng.integers(0, nseg))] += 1

    mode = str(rng.choice(["uniform", "random_walk", "clustered"], p=[0.35, 0.40, 0.25]))
    if mode == "uniform":
        levels = rng.uniform(-4.0, 4.0, size=nseg)
    elif mode == "random_walk":
        levels = np.zeros(nseg, dtype=np.float64)
        levels[0] = rng.uniform(-3.0, 3.0)
        for i in range(1, nseg):
            levels[i] = levels[i - 1] + rng.normal(scale=rng.uniform(0.3, 1.8))
    else:
        nclusters = int(rng.integers(2, min(5, nseg) + 1))
        centers = rng.uniform(-4.0, 4.0, size=nclusters)
        assignments = rng.integers(0, nclusters, size=nseg)
        levels = centers[assignments] + rng.normal(scale=0.15, size=nseg)

    y = np.repeat(levels, lengths)[:length].astype(np.float64)

    transition_type = str(rng.choice(["hard", "ramp", "sigmoid"], p=[0.5, 0.3, 0.2]))
    if transition_type != "hard":
        cps = np.cumsum(lengths)[:-1]
        for cp in cps:
            width = int(max(3, rng.uniform(0.01, 0.05) * length))
            left = max(0, int(cp) - width // 2)
            right = min(length, int(cp) + width // 2)
            if right <= left + 1:
                continue
            start_val = y[left]
            end_val = y[right - 1]
            grid = np.linspace(0.0, 1.0, right - left)
            if transition_type == "ramp":
                blend = grid
            else:
                steep = rng.uniform(5.0, 15.0)
                blend = 1.0 / (1.0 + np.exp(-steep * (grid - 0.5)))
            y[left:right] = (1.0 - blend) * start_val + blend * end_val

    if rng.random() < 0.70:
        y += rng.normal(scale=rng.uniform(0.02, 0.30), size=length)
    if rng.random() < 0.35:
        y += 0.3 * generate_trend_seasonality(length, rng)

    return robust_normalize(y)


def generate_spike_event(length: int, rng: np.random.Generator) -> np.ndarray:
    """Generate baseline signals with spikes, bursts, holiday-like events, or shocks."""
    baseline_type = str(rng.choice(["kernel", "trend", "flat", "ou"], p=[0.25, 0.45, 0.15, 0.15]))
    if baseline_type == "kernel":
        y = 0.4 * generate_kernel_gp(length, rng)
    elif baseline_type == "trend":
        y = 0.5 * generate_trend_seasonality(length, rng)
    elif baseline_type == "ou":
        y = 0.4 * generate_sde_ou(length, rng)
    else:
        y = rng.normal(scale=0.05, size=length)

    n_events = int(rng.poisson(lam=max(1.0, length / 160.0)))
    n_events = int(np.clip(n_events, 1, max(2, length // 16)))

    event_mode = str(rng.choice(["point", "gaussian", "plateau", "shock_recovery", "periodic"],
                                p=[0.25, 0.30, 0.20, 0.15, 0.10]))

    if event_mode == "periodic":
        period = int(rng.integers(16, max(17, length // 4)))
        positions = list(range(int(rng.integers(0, period)), length, period))
    else:
        positions = list(rng.choice(np.arange(length), size=n_events, replace=False))

    for pos in positions:
        amp = rng.choice([-1.0, 1.0]) * log_uniform(rng, 0.8, 6.0)
        if event_mode == "point":
            y[int(pos)] += amp
        elif event_mode == "gaussian" or event_mode == "periodic":
            width = rng.uniform(1.0, max(2.0, length / 80.0))
            grid = np.arange(length, dtype=np.float64)
            y += amp * np.exp(-0.5 * ((grid - float(pos)) / width) ** 2)
        elif event_mode == "plateau":
            width = int(rng.integers(2, max(3, length // 20)))
            end = min(length, int(pos) + width)
            y[int(pos):end] += amp
        elif event_mode == "shock_recovery":
            width = int(rng.integers(4, max(5, length // 12)))
            end = min(length, int(pos) + width)
            decay = np.exp(-np.linspace(0.0, 4.0, end - int(pos)))
            y[int(pos):end] += amp * decay

    y += rng.normal(scale=rng.uniform(0.01, 0.25), size=length)
    return robust_normalize(y)


def generate_waveform(length: int, rng: np.random.Generator) -> np.ndarray:
    """Generate mixtures of sawtooth, square, and triangle waves."""
    t = np.linspace(0.0, 1.0, length, dtype=np.float64)
    y = np.zeros(length, dtype=np.float64)

    nwaves = int(rng.integers(1, 4))
    for _ in range(nwaves):
        wave_type = str(rng.choice(["sawtooth", "square", "triangle"]))
        amp = rng.uniform(0.3, 3.0)
        freq = rng.uniform(1.0, 50.0)
        phase = rng.uniform(0.0, 1.0)
        cycles = freq * t + phase

        if wave_type == "sawtooth":
            width = float(rng.choice([0.0, 1.0]))
            width = 0.001 if width == 0.0 else 0.999
            comp = sawtooth_wave(cycles, width=width)
        elif wave_type == "square":
            comp = square_wave(cycles, duty=float(rng.uniform(0.2, 0.8)))
        else:
            comp = triangle_wave(cycles)

        if rng.random() < 0.30:
            am_freq = rng.uniform(0.5, 5.0)
            envelope = 1.0 + rng.uniform(0.1, 0.8) * np.sin(2.0 * np.pi * am_freq * t + rng.uniform(0, 2 * np.pi))
            comp = comp * envelope

        y += amp * comp

    if rng.random() < 0.30:
        y += rng.uniform(-2.0, 2.0) * (t - 0.5)
    if rng.random() < 0.70:
        y += rng.normal(scale=rng.uniform(0.01, 0.30), size=length)

    return robust_normalize(y)


def generate_fbm_fgn(length: int, rng: np.random.Generator) -> np.ndarray:
    """Generate approximate fBm or fGn using colored-noise spectral synthesis."""
    hurst = rng.uniform(0.1, 0.9)
    output_type = str(rng.choice(["fgn", "fbm"]))

    if output_type == "fgn":
        # Approximate fGn spectral slope: beta = 2H - 1.
        beta = float(np.clip(2.0 * hurst - 1.0, -0.8, 0.8))
        y = colored_noise(length, beta=beta, rng=rng)
    else:
        # Approximate fBm by integrating fGn-like increments.
        beta = float(np.clip(2.0 * hurst - 1.0, -0.8, 0.8))
        increments = colored_noise(length, beta=beta, rng=rng)
        y = np.cumsum(increments)

    scale = log_uniform(rng, 0.1, 5.0)
    return robust_normalize(scale * y)


def _standardized_student_t(rng: np.random.Generator, df: float, size: int) -> np.ndarray:
    """Student-t noise standardized to unit variance when df > 2."""
    z = rng.standard_t(df=df, size=size)
    if df > 2.0:
        z = z / math.sqrt(df / (df - 2.0))
    return z


def generate_garch_volatility(length: int, rng: np.random.Generator) -> np.ndarray:
    """Generate GARCH/GJR-GARCH/EGARCH-like volatility-clustered series."""
    burn = 500
    total = length + burn
    model = str(rng.choice(["garch", "gjr", "egarch"]))
    innovations = str(rng.choice(["normal", "student_t"]))
    mean_type = str(rng.choice(["zero", "const", "ar1"], p=[0.55, 0.20, 0.25]))

    if innovations == "student_t":
        df = rng.uniform(3.0, 10.0)
        z = _standardized_student_t(rng, df=df, size=total)
    else:
        z = rng.normal(size=total)

    mu = rng.normal(scale=0.05) if mean_type == "const" else 0.0
    phi = rng.uniform(-0.3, 0.3) if mean_type == "ar1" else 0.0
    unconditional_sigma = log_uniform(rng, 0.005, 0.5)

    eps = np.zeros(total, dtype=np.float64)
    returns = np.zeros(total, dtype=np.float64)
    sigma2 = np.ones(total, dtype=np.float64) * unconditional_sigma ** 2

    if model == "garch":
        persistence = rng.uniform(0.70, 0.97)
        alpha = persistence * rng.uniform(0.02, 0.25)
        beta = persistence - alpha
        omega = max(1e-8, unconditional_sigma ** 2 * (1.0 - alpha - beta))
        for t in range(1, total):
            sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
            eps[t] = math.sqrt(max(sigma2[t], 1e-12)) * z[t]
            returns[t] = mu + phi * (returns[t - 1] - mu) + eps[t]

    elif model == "gjr":
        persistence = rng.uniform(0.70, 0.97)
        alpha = persistence * rng.uniform(0.02, 0.20)
        gamma = rng.uniform(0.01, 0.15)
        beta = max(0.0, persistence - alpha - 0.5 * gamma)
        omega = max(1e-8, unconditional_sigma ** 2 * (1.0 - alpha - beta - 0.5 * gamma))
        for t in range(1, total):
            leverage = gamma if eps[t - 1] < 0.0 else 0.0
            sigma2[t] = omega + (alpha + leverage) * eps[t - 1] ** 2 + beta * sigma2[t - 1]
            eps[t] = math.sqrt(max(sigma2[t], 1e-12)) * z[t]
            returns[t] = mu + phi * (returns[t - 1] - mu) + eps[t]

    else:
        beta = rng.uniform(0.70, 0.97)
        alpha = rng.uniform(0.05, 0.25)
        gamma = rng.uniform(-0.20, 0.20)
        omega = math.log(unconditional_sigma ** 2 + 1e-8) * (1.0 - beta)
        log_sigma2 = np.ones(total, dtype=np.float64) * math.log(unconditional_sigma ** 2 + 1e-8)
        expected_abs_z = math.sqrt(2.0 / math.pi)
        for t in range(1, total):
            prev_std = math.sqrt(max(math.exp(log_sigma2[t - 1]), 1e-12))
            z_prev = eps[t - 1] / prev_std if prev_std > 0 else 0.0
            log_sigma2[t] = omega + alpha * (abs(z_prev) - expected_abs_z) + gamma * z_prev + beta * log_sigma2[t - 1]
            log_sigma2[t] = float(np.clip(log_sigma2[t], -20.0, 8.0))
            sigma2[t] = math.exp(log_sigma2[t])
            eps[t] = math.sqrt(max(sigma2[t], 1e-12)) * z[t]
            returns[t] = mu + phi * (returns[t - 1] - mu) + eps[t]

    y = returns[burn:]
    if rng.random() < 0.5:
        y = np.cumsum(y)

    return robust_normalize(y)


def _generate_lorenz(length: int, rng: np.random.Generator) -> np.ndarray:
    """Generate one coordinate of a Lorenz attractor."""
    burn = 800
    total = length + burn
    dt = rng.uniform(0.005, 0.02)
    sigma = rng.uniform(8.0, 12.0)
    rho = rng.uniform(24.0, 30.0)
    beta = rng.uniform(2.0, 3.5)
    state = rng.uniform(-1.0, 1.0, size=3)

    def f(s: np.ndarray) -> np.ndarray:
        x, y, z = s
        return np.array([
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z,
        ], dtype=np.float64)

    traj = np.zeros((total, 3), dtype=np.float64)
    for i in range(total):
        k1 = f(state)
        k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2)
        k4 = f(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        traj[i] = state

    coord = int(rng.integers(0, 3))
    return robust_normalize(traj[burn:, coord])


def _generate_mackey_glass(length: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a Mackey-Glass-like delayed nonlinear sequence."""
    burn = 1000
    total = length + burn
    tau = int(rng.integers(15, 31))
    n = int(rng.integers(8, 13))
    beta = rng.uniform(0.15, 0.25)
    gamma = rng.uniform(0.05, 0.15)
    dt = rng.uniform(0.5, 2.0)
    x = np.ones(total + tau + 1, dtype=np.float64) * rng.uniform(0.8, 1.0)
    x[:tau + 1] += rng.normal(scale=0.01, size=tau + 1)
    for t in range(tau, total + tau):
        delayed = x[t - tau]
        dx = beta * delayed / (1.0 + delayed ** n) - gamma * x[t]
        x[t + 1] = x[t] + dt * dx
    return robust_normalize(x[tau + burn:tau + burn + length])


def _generate_narma(length: int, rng: np.random.Generator) -> np.ndarray:
    """Generate NARMA-like nonlinear autoregressive moving-average sequence."""
    order = int(rng.integers(5, 15))
    total = length + order + 50
    u = rng.uniform(0.0, 0.5, size=total)
    y = np.zeros(total, dtype=np.float64)
    c1 = rng.uniform(0.6, 0.9)
    c2 = rng.uniform(0.02, 0.08)
    c3 = rng.uniform(1.0, 2.0)
    c4 = rng.uniform(0.05, 0.15)
    for t in range(order, total - 1):
        y[t + 1] = (
            c1 * y[t]
            + c2 * y[t] * np.sum(y[t - order + 1:t + 1])
            + c3 * u[t - order + 1] * u[t]
            + c4
        )
        if not np.isfinite(y[t + 1]) or abs(y[t + 1]) > 1e6:
            y[t + 1] = 0.0
    return robust_normalize(y[-length:])


def _generate_audio_like(length: int, rng: np.random.Generator) -> np.ndarray:
    """Generate procedural audio-inspired multi-scale rhythm/fractal signal."""
    t = np.linspace(0.0, 1.0, length, dtype=np.float64)
    y = np.zeros(length, dtype=np.float64)

    # Multi-scale oscillatory layers.
    n_layers = int(rng.integers(2, 6))
    for _ in range(n_layers):
        freq = log_uniform(rng, 1.0, 80.0)
        amp = log_uniform(rng, 0.1, 1.5)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        fm = rng.uniform(0.0, 0.3) * np.sin(2.0 * np.pi * rng.uniform(0.2, 5.0) * t + rng.uniform(0, 2 * np.pi))
        y += amp * np.sin(2.0 * np.pi * freq * t + fm + phase)

    # Event rhythm layer.
    if rng.random() < 0.7:
        period = int(rng.integers(8, max(9, length // 8)))
        start = int(rng.integers(0, period))
        for pos in range(start, length, period):
            width = rng.uniform(1.0, max(2.0, period / 5.0))
            amp = rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 3.0)
            grid = np.arange(length, dtype=np.float64)
            y += amp * np.exp(-0.5 * ((grid - pos) / width) ** 2)

    # Fractal colored noise.
    y += rng.uniform(0.1, 0.8) * colored_noise(length, beta=rng.uniform(0.5, 2.0), rng=rng)
    return robust_normalize(y)


def generate_chaotic_timesynth_audio(length: int, rng: np.random.Generator) -> np.ndarray:
    """Generate low-weight complex nonlinear/diversity sources."""
    subtype = str(rng.choice(["lorenz", "mackey_glass", "narma", "audio_like"],
                             p=[0.25, 0.25, 0.25, 0.25]))
    if subtype == "lorenz":
        return _generate_lorenz(length, rng)
    if subtype == "mackey_glass":
        return _generate_mackey_glass(length, rng)
    if subtype == "narma":
        return _generate_narma(length, rng)
    return _generate_audio_like(length, rng)


GENERATOR_REGISTRY: Dict[str, Callable[[int, np.random.Generator], np.ndarray]] = {
    "kernel_gp": generate_kernel_gp,
    "trend_seasonality": generate_trend_seasonality,
    "sde_ou": generate_sde_ou,
    "arima_state_space": generate_arima_state_space,
    "step_changepoint": generate_step_changepoint,
    "spike_event": generate_spike_event,
    "waveform": generate_waveform,
    "fbm_fgn": generate_fbm_fgn,
    "garch_volatility": generate_garch_volatility,
    "chaotic_timesynth_audio": generate_chaotic_timesynth_audio,
}


# =============================================================================
# 3. CauKer-style SCM configuration and transformation
# =============================================================================


@dataclass
class SCMConfig:
    """Configuration for the CauKer DAG SCM transformation."""

    n_channels: int = 8
    latent_dim: int = 8
    max_parents: int = 3
    source_scale_low: float = 0.6
    source_scale_high: float = 1.8
    noise_scale_low: float = 0.01
    noise_scale_high: float = 0.15
    clip: float = 8.0


def sample_scm_structure(
    cfg: SCMConfig,
    rng: np.random.Generator,
) -> Dict[str, object]:
    """
    Sample a two-layer CauKer-style random DAG SCM.

    The graph contains ``latent_dim`` root nodes and ``n_channels`` observed
    non-root nodes, for a total of ``latent_dim + n_channels`` nodes.  Every
    primitive base series is assigned one-to-one to a root, and every root is
    forced to be a parent of at least one observed node.  Additional edges may
    connect any earlier node, including earlier observed nodes, which preserves
    random DAG depth while guaranteeing that no generated base series is unused.
    """
    n_channels = int(cfg.n_channels)
    latent_dim = int(cfg.latent_dim)
    max_parents = int(cfg.max_parents)

    if n_channels <= 0 or latent_dim <= 0:
        raise ValueError("n_channels and latent_dim must both be positive.")
    if max_parents <= 0:
        raise ValueError("max_parents must be positive for the rooted SCM.")
    if latent_dim > n_channels * max_parents:
        raise ValueError(
            "Cannot connect every latent root to an observed node: "
            f"latent_dim={latent_dim} exceeds n_channels*max_parents="
            f"{n_channels * max_parents}."
        )

    num_nodes = latent_dim + n_channels
    root_nodes = [int(v) for v in rng.permutation(latent_dim)]
    observed_nodes = [
        int(v) for v in rng.permutation(np.arange(latent_dim, num_nodes))
    ]
    order = root_nodes + observed_nodes

    parents: List[List[int]] = [[] for _ in range(num_nodes)]
    edge_weights: List[List[float]] = [[] for _ in range(num_nodes)]
    activations: List[Dict[str, float]] = [
        {"name": "linear", "a": 1.0, "b": 0.0}
        for _ in range(num_nodes)
    ]

    # Split the roots across observed nodes.  With the current 16-root,
    # 16-channel configuration this assigns exactly one distinct anchor root to
    # every observed node.  The general form also supports multiple roots per
    # observed node when latent_dim > n_channels.
    shuffled_roots = [int(v) for v in rng.permutation(root_nodes)]
    required_root_groups = [
        [int(v) for v in group]
        for group in np.array_split(np.asarray(shuffled_roots, dtype=np.int64), n_channels)
    ]

    for output_pos, node in enumerate(observed_nodes):
        required = required_root_groups[output_pos]
        possible_parents = order[: latent_dim + output_pos]
        optional = [p for p in possible_parents if p not in required]

        min_count = max(1, len(required))
        max_count = min(max_parents, len(possible_parents))
        n_parents = int(rng.integers(min_count, max_count + 1))
        n_optional = n_parents - len(required)

        selected = list(required)
        if n_optional > 0:
            selected.extend(
                int(v)
                for v in rng.choice(optional, size=n_optional, replace=False)
            )
        rng.shuffle(selected)

        parents[node] = selected
        edge_weights[node] = [float(v) for v in rng.normal(size=len(selected))]
        activations[node] = _sample_cauker_activation(rng)

    # Root-to-base assignment is a bijection: each of the latent_dim generated
    # base series is consumed by exactly one root node.
    source_indices = [-1 for _ in range(num_nodes)]
    for source_idx, root_node in enumerate(root_nodes):
        source_indices[root_node] = int(source_idx)

    source_scales = [
        log_uniform(rng, cfg.source_scale_low, cfg.source_scale_high)
        for _ in range(num_nodes)
    ]
    noise_scales = [
        log_uniform(rng, cfg.noise_scale_low, cfg.noise_scale_high)
        for _ in range(num_nodes)
    ]

    return {
        "topological_order": order,
        "root_nodes": root_nodes,
        "observed_nodes": observed_nodes,
        "parents": parents,
        "edge_weights": edge_weights,
        "activations": activations,
        "source_indices": source_indices,
        "source_scales": source_scales,
        "noise_scales": noise_scales,
    }


def _sample_cauker_activation(rng: np.random.Generator) -> Dict[str, float]:
    """Sample the activation and parameters used by one CauKer SCM node."""
    name = str(rng.choice(["linear", "relu", "sigmoid", "sin", "mod", "leakyrelu"]))
    params: Dict[str, float] = {"name": name}
    if name == "linear":
        params["a"] = float(rng.uniform(0.5, 2.0))
        params["b"] = 0.0
    elif name == "mod":
        params["c"] = float(rng.uniform(1.0, 5.0))
    elif name == "leakyrelu":
        params["alpha"] = float(rng.uniform(0.01, 0.3))
    return params


def _apply_cauker_activation(x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """Apply CauKer's vectorized random activation."""
    name = str(params["name"])
    if name == "linear":
        return float(params.get("a", 1.0)) * x + float(params.get("b", 0.0))
    if name == "relu":
        return np.maximum(0.0, x)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))
    if name == "sin":
        return np.sin(x)
    if name == "mod":
        return np.mod(x, float(params.get("c", 1.0)))
    alpha = float(params.get("alpha", 0.01))
    return np.where(x > 0.0, x, alpha * x)


def apply_cauker_style_scm(
    base_series: np.ndarray,
    cfg: SCMConfig,
    rng: np.random.Generator,
    structure: Optional[Dict[str, object]] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Transform primitive base series U into observed multivariate series X.

    The public function name and signature are kept for the rest of the project,
    but the internal generation now mirrors CauKer: a random DAG is sampled,
    roots are assigned all exogenous base functions one-to-one, and observed
    non-root nodes are produced by random parent mappings plus activations.
    """
    if base_series.ndim != 2:
        raise ValueError("base_series must have shape [T, latent_dim].")
    length, latent_dim = base_series.shape
    if latent_dim != cfg.latent_dim:
        raise ValueError(f"Expected latent_dim={cfg.latent_dim}, got {latent_dim}.")

    if structure is None:
        structure = sample_scm_structure(cfg, rng)

    order = [int(v) for v in structure["topological_order"]]
    parents = structure["parents"]
    edge_weights = structure["edge_weights"]
    activations = structure["activations"]
    source_indices = structure["source_indices"]
    source_scales = structure["source_scales"]
    noise_scales = structure["noise_scales"]
    observed_nodes = [int(v) for v in structure["observed_nodes"]]
    num_nodes = len(parents)
    x = np.zeros((length, num_nodes), dtype=np.float64)

    for node in order:
        parent_nodes = [int(v) for v in parents[node]]
        if len(parent_nodes) == 0:
            src_idx = int(source_indices[node])
            if not (0 <= src_idx < cfg.latent_dim):
                raise ValueError(f"Root node {node} has invalid source index {src_idx}.")
            x[:, node] = float(source_scales[node]) * base_series[:, src_idx]
        else:
            combined = np.stack([x[:, p] for p in parent_nodes], axis=1)
            weights = np.asarray(edge_weights[node], dtype=np.float64)
            values = combined @ weights
            x[:, node] = _apply_cauker_activation(values, activations[node])

        x[:, node] += float(noise_scales[node]) * rng.normal(size=length)

    observed_x = x[:, observed_nodes]
    observed_x = normalize_matrix_columns(observed_x, clip=cfg.clip)
    return observed_x, structure


# =============================================================================
# 4. Dataset generation
# =============================================================================


def generate_base_matrix_from_schedule(
    length: int,
    latent_dim: int,
    schedule_slice: Sequence[str],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, List[str]]:
    """Generate a [T, latent_dim] matrix of primitive base signals."""
    if len(schedule_slice) != latent_dim:
        raise ValueError("schedule_slice must have length latent_dim.")

    u = np.zeros((length, latent_dim), dtype=np.float64)
    names: List[str] = []
    for k, name in enumerate(schedule_slice):
        if name not in GENERATOR_REGISTRY:
            raise KeyError(f"Unknown generator name: {name}")
        u[:, k] = GENERATOR_REGISTRY[name](length, rng)
        names.append(name)
    u = normalize_matrix_columns(u)
    return u, names


def generate_one_sample(
    length: int,
    latent_dim: int,
    n_channels: int,
    schedule_slice: Sequence[str],
    rng: np.random.Generator,
    scm_overrides: Optional[Dict[str, object]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, object]]:
    """Generate one multivariate TSFM pretraining sample."""
    u, names = generate_base_matrix_from_schedule(length, latent_dim, schedule_slice, rng)

    scm_cfg_kwargs = {
        "n_channels": n_channels,
        "latent_dim": latent_dim,
    }
    if scm_overrides:
        scm_cfg_kwargs.update(scm_overrides)
    scm_cfg = SCMConfig(**scm_cfg_kwargs)

    x, structure = apply_cauker_style_scm(u, scm_cfg, rng)
    return x, u, names, structure


def load_generator_weights(weights_json: Optional[str]) -> Dict[str, float]:
    """Load optional JSON weights for the primitive generator portfolio."""
    if weights_json is None:
        return dict(DEFAULT_GENERATOR_WEIGHTS)
    with open(weights_json, "r", encoding="utf-8") as f:
        weights = json.load(f)
    unknown = set(weights.keys()) - set(GENERATOR_REGISTRY.keys())
    if unknown:
        raise ValueError(f"Unknown generator names in weights JSON: {sorted(unknown)}")
    return {str(k): float(v) for k, v in weights.items()}


class OnlineCauKerIterableDataset(IterableDataset):
    """Infinite online multi-base CauKer V2 dataset.

    Each generated synthetic object is a multivariate series

        X in R^{T x P},

    obtained by first drawing ``latent_dim`` primitive base time series from the
    weighted generator portfolio and then passing them through the CauKer-style
    SCM.  Since the current PatchTST-FM training objective is univariate, the
    ``P`` output channels are stored in a local worker buffer and yielded one by
    one as univariate pretraining samples.

    The class deliberately keeps the same public name as the original
    ``OnlineCauKerIterableDataset`` so that the rest of the training script,
    including batch mixing, DDP, AMP, and checkpointing, remains unchanged.
    """

    def __init__(
        self,
        context_length: int = 2048,
        min_length: int = 128,
        max_length: int = 2048,
        num_features: int = 8,
        max_parents: int = 6,
        num_nodes: int = 8,
        seed: int = 42,
        length_sampling: str = "log_uniform",
        weights_json: Optional[str] = None,
    ) -> None:
        super().__init__()
        if min_length < 4:
            raise ValueError("min_length must be at least 4.")
        if max_length > context_length:
            raise ValueError("max_length should be <= context_length for this online setting.")
        if min_length > max_length:
            raise ValueError("min_length cannot exceed max_length.")
        if length_sampling not in {"uniform", "log_uniform"}:
            raise ValueError("length_sampling must be either 'uniform' or 'log_uniform'.")

        self.context_length = int(context_length)
        self.min_length = int(min_length)
        self.max_length = int(max_length)

        # Backward-compatible argument semantics:
        #   cauker_features  -> number of observed SCM channels P.
        #   cauker_num_nodes -> number of primitive base sources K = latent_dim.
        # This lets the old launch file remain valid while changing only the
        # synthetic data source.
        self.n_channels = int(num_features)
        self.latent_dim = int(num_nodes)
        self.max_parents = int(max_parents)
        if self.n_channels <= 0 or self.latent_dim <= 0:
            raise ValueError("num_features and num_nodes must both be positive.")
        if self.max_parents <= 0:
            raise ValueError("max_parents must be positive.")
        if self.latent_dim > self.n_channels * self.max_parents:
            raise ValueError(
                "Cannot route every latent base into the observed SCM nodes: "
                f"num_nodes={self.latent_dim} exceeds "
                f"num_features*max_parents={self.n_channels * self.max_parents}."
            )
        self.seed = int(seed)
        self.length_sampling = str(length_sampling)
        self.generator_weights = load_generator_weights(weights_json)

        self.scm_overrides: Dict[str, object] = {
            "max_parents": self.max_parents,
        }

    def _global_worker_id(self) -> Tuple[int, int, int, int]:
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        info = get_worker_info()
        if info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = info.id, info.num_workers

        global_worker_id = rank * num_workers + worker_id
        global_num_workers = world_size * num_workers
        return rank, worker_id, global_worker_id, global_num_workers

    def _sample_length(self, np_rng: np.random.Generator) -> int:
        if self.length_sampling == "uniform":
            return int(np_rng.integers(self.min_length, self.max_length + 1))

        low, high = math.log(self.min_length), math.log(self.max_length)
        val = int(round(math.exp(np_rng.uniform(low, high))))
        return int(np.clip(val, self.min_length, self.max_length))

    def _sample_generator_schedule(self, np_rng: np.random.Generator) -> List[str]:
        """Sample the primitive generator family for each latent base source.

        For ``latent_dim >= number_of_generator_families``, every synthetic SCM
        draw contains at least one representative from each primitive family.
        This is the intended "multi-base" behavior of CauKer V2.
        """
        return build_weighted_schedule(
            total=self.latent_dim,
            weights=self.generator_weights,
            rng=np_rng,
            ensure_min_one_when_possible=True,
        )

    def _format_sample(self, arr: np.ndarray) -> Dict[str, torch.Tensor]:
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
        arr = np.clip(arr, -1e6, 1e6)

        n = int(arr.shape[0])
        T = self.context_length
        if n > T:
            start = random.randint(0, n - T)
            arr = arr[start : start + T]
            n = T

        pad_len = T - n
        x = np.zeros(T, dtype=np.float32)
        observed = np.zeros(T, dtype=np.bool_)
        padding = np.ones(T, dtype=np.bool_)

        x[pad_len:] = arr
        observed[pad_len:] = np.isfinite(arr)
        padding[pad_len:] = False

        return {
            "x": torch.from_numpy(x),
            "observed_mask": torch.from_numpy(observed),
            "padding_mask": torch.from_numpy(padding),
        }

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        _, _, global_worker_id, _ = self._global_worker_id()
        base_seed = self.seed + 1_000_003 * global_worker_id
        np_rng = np.random.default_rng(base_seed)
        py_rng = random.Random(base_seed)

        local_buffer: List[np.ndarray] = []

        while True:
            if len(local_buffer) == 0:
                length = self._sample_length(np_rng)
                schedule = self._sample_generator_schedule(np_rng)
                try:
                    x_multi, _, _, _ = generate_one_sample(
                        length=length,
                        latent_dim=self.latent_dim,
                        n_channels=self.n_channels,
                        schedule_slice=schedule,
                        rng=np_rng,
                        scm_overrides=self.scm_overrides,
                    )
                    # x_multi: [T, P].  The model is univariate, so yield each
                    # SCM channel as one training series.
                    for j in range(x_multi.shape[1]):
                        local_buffer.append(x_multi[:, j].astype(np.float32, copy=False))
                except Exception:
                    # Random synthetic generators can occasionally produce a
                    # numerically invalid draw.  The infinite iterable simply
                    # rejects that draw and samples a new one.
                    continue

                py_rng.shuffle(local_buffer)

            arr = local_buffer.pop()
            yield self._format_sample(arr)


def discover_arrow_files(root: str) -> List[Path]:
    """Find KernelSynth Arrow files under a folder."""
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Synthetic Arrow root does not exist: {root_path}")

    if root_path.is_file():
        files = [root_path] if root_path.suffix == ".arrow" else []
    else:
        files = sorted(root_path.rglob("*.arrow"))

    if len(files) == 0:
        raise FileNotFoundError(f"No .arrow files found under {root_path}")
    return files


def load_gluonts_arrow_file(path: Path):
    """Load a GluonTS ArrowWriter output file."""
    try:
        from gluonts.dataset.arrow import File
    except Exception as exc:
        raise ImportError(
            "Reading KernelSynth Arrow files requires GluonTS. Install it with:\n"
            "  pip install gluonts"
        ) from exc

    return File.infer(path)

# =============================================================================
# 5. Arrow-rooted CauKer-style SCM dataset
# =============================================================================

def _crop_or_interpolate_1d(
    arr: np.ndarray,
    length: int,
    rng: random.Random,
    min_real_length: int,
    clip: float = 8.0,
) -> np.ndarray:
    """
    Convert one Arrow target to one finite, normalized root-node series of exact length.

    输入:
        arr: 从 Arrow entry 中读取出来的一条原始时间序列
        length: 本次 SCM 需要的统一长度
        rng: Python 随机数生成器，用于 crop 起点
        min_real_length: 太短的序列直接拒绝
        clip: robust_normalize 的裁剪阈值

    输出:
        shape = [length] 的 np.ndarray，可作为 SCM 根节点序列
    """
    y = np.asarray(arr, dtype=np.float64).reshape(-1)

    # 1. 清理 NaN / Inf，避免 SCM 非线性传播后数值爆炸。
    y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
    y = np.clip(y, -1e6, 1e6)

    n = int(y.shape[0])
    if n < int(min_real_length):
        raise ValueError(f"Series too short: got {n}, need >= {min_real_length}.")

    # 2. 如果原始序列足够长，随机 crop 一个长度为 length 的片段。
    if n >= length:
        start = int(rng.randrange(n - length + 1))
        y = y[start : start + length]

    # 3. 如果原始序列短于 length，做一次线性插值拉伸。
    #    这只是兜底逻辑；如果你的 Arrow 序列普遍很长，基本不会触发。
    else:
        old_grid = np.linspace(0.0, 1.0, n, dtype=np.float64)
        new_grid = np.linspace(0.0, 1.0, length, dtype=np.float64)
        y = np.interp(new_grid, old_grid, y)

    # 4. 每个根节点单独 robust normalize，避免某个根节点量纲支配整个 DAG。
    return robust_normalize(y, clip=clip).astype(np.float64, copy=False)


def generate_arrow_rooted_scm(
    *,
    length: int,
    num_features: int,
    num_nodes: int,
    max_parents: int,
    rng: np.random.Generator,
    base_sampler: Callable[[int], np.ndarray],
    source_scale_low: float = 0.6,
    source_scale_high: float = 1.8,
    noise_scale_low: float = 0.01,
    noise_scale_high: float = 0.15,
    clip: float = 8.0,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    CauKer-style SCM whose root nodes are sampled from pre-generated Arrow series.

    数学形式:

        G = (V, E), |V| = num_nodes

    若节点 j 是根节点:

        x_j(t) = a_j u_j(t) + eps_j(t)

    若节点 j 有父节点 Pa(j):

        z_j(t) = sum_{p in Pa(j)} w_{jp} x_p(t)
        x_j(t) = phi_j(z_j(t)) + eps_j(t)

    最后随机抽样 num_features 个节点作为输出:

        X(t) = [x_{j_1}(t), ..., x_{j_P}(t)]

    输出:
        x: shape = [length, num_features]
    """
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive.")
    if not (1 <= num_features <= num_nodes):
        raise ValueError("num_features must satisfy 1 <= num_features <= num_nodes.")

    # -------------------------------------------------------------------------
    # 1. 随机拓扑序，然后只允许当前节点连接到拓扑序中更早的节点。
    #    这样天然保证图是 DAG。
    # -------------------------------------------------------------------------
    order = [int(v) for v in rng.permutation(num_nodes)]

    parents: List[List[int]] = [[] for _ in range(num_nodes)]
    edge_weights: List[List[float]] = [[] for _ in range(num_nodes)]
    activations: List[Dict[str, float]] = [
        {"name": "linear", "a": 1.0, "b": 0.0} for _ in range(num_nodes)
    ]

    max_parents = max(0, int(max_parents))

    for pos, node in enumerate(order):
        possible_parents = order[:pos]
        n_possible = len(possible_parents)

        n_parents = int(rng.integers(0, min(n_possible, max_parents) + 1))

        if n_parents > 0:
            selected = [
                int(v)
                for v in rng.choice(possible_parents, size=n_parents, replace=False)
            ]
            parents[node] = selected
            edge_weights[node] = [float(v) for v in rng.normal(size=n_parents)]
            activations[node] = _sample_cauker_activation(rng)

    # -------------------------------------------------------------------------
    # 2. 沿拓扑序生成每一个节点的时间序列。
    # -------------------------------------------------------------------------
    node_data: Dict[int, np.ndarray] = {}
    root_nodes: List[int] = []

    for node in order:
        parent_nodes = [int(v) for v in parents[node]]

        # 根节点：直接从 Arrow 文件中抽样一条已有时间序列。
        if len(parent_nodes) == 0:
            root_nodes.append(node)
            y = base_sampler(length)
            y = log_uniform(rng, source_scale_low, source_scale_high) * y

        # 非根节点：父节点线性组合 + 随机非线性激活。
        else:
            combined = np.stack([node_data[p] for p in parent_nodes], axis=1)
            weights = np.asarray(edge_weights[node], dtype=np.float64)
            z = combined @ weights
            y = _apply_cauker_activation(z, activations[node])

        # 每个节点加一点噪声，避免完全确定性依赖。
        noise_scale = log_uniform(rng, noise_scale_low, noise_scale_high)
        y = y + noise_scale * rng.normal(size=length)

        # 每个节点生成后都 normalize，防止深层 DAG 数值爆炸。
        node_data[node] = robust_normalize(y, clip=clip)

    # -------------------------------------------------------------------------
    # 3. 像 CauKer 一样，从 DAG 节点中随机抽样 num_features 个节点作为观测序列。
    # -------------------------------------------------------------------------
    chosen_nodes = [
        int(v)
        for v in rng.choice(order, size=num_features, replace=False)
    ]

    x = np.stack([node_data[n] for n in chosen_nodes], axis=1)
    x = normalize_matrix_columns(x, clip=clip)

    structure = {
        "topological_order": order,
        "parents": parents,
        "edge_weights": edge_weights,
        "activations": activations,
        "root_nodes": root_nodes,
        "chosen_nodes": chosen_nodes,
    }

    return x, structure


class ArrowCauKerSCMIterableDataset(IterableDataset):
    """
    Read Arrow root series, pass them through a CauKer-style random DAG,
    then yield each observed channel as one univariate training sample.

    这个模式和普通 arrow 模式的区别是:

        普通 arrow:
            Arrow target -> 直接训练

        arrow_scm:
            Arrow target -> SCM 根节点
                         -> DAG 非线性传播
                         -> 抽样节点输出
                         -> 训练
    """

    def __init__(
        self,
        root: str,
        context_length: int = 2048,
        seed: int = 42,
        min_sample_length: int = 126,
        max_sample_length: int = 2048,
        min_real_length: int = 2,
        max_sample_attempts: int = 32,
        balance_files: bool = True,
        num_features: int = 4,
        max_parents: int = 6,
        num_nodes: int = 18,
        length_sampling: str = "log_uniform",
    ) -> None:
        super().__init__()

        if min_sample_length < 4:
            raise ValueError("min_sample_length must be at least 4.")
        if max_sample_length > context_length:
            raise ValueError("max_sample_length must be <= context_length.")
        if min_sample_length > max_sample_length:
            raise ValueError("min_sample_length cannot exceed max_sample_length.")
        if length_sampling not in {"uniform", "log_uniform"}:
            raise ValueError("length_sampling must be either 'uniform' or 'log_uniform'.")
        if num_features < 1 or num_nodes < num_features:
            raise ValueError("Require 1 <= num_features <= num_nodes for arrow_scm.")

        self.root = str(root)
        self.context_length = int(context_length)
        self.seed = int(seed)

        self.min_sample_length = int(min_sample_length)
        self.max_sample_length = int(max_sample_length)
        self.min_real_length = int(min_real_length)
        self.max_sample_attempts = int(max_sample_attempts)

        self.balance_files = bool(balance_files)

        # 对 arrow_scm:
        #   cauker_features  -> 最终从 DAG 里抽样多少条观测序列
        #   cauker_num_nodes -> DAG 总节点数
        self.num_features = int(num_features)
        self.max_parents = int(max_parents)
        self.num_nodes = int(num_nodes)

        self.length_sampling = str(length_sampling)

        self.arrow_files = discover_arrow_files(self.root)

    def _global_worker_id(self) -> int:
        """
        DDP rank + DataLoader worker id 共同决定全局 worker id。
        这样不同 GPU / worker 读到的随机流不同。
        """
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0

        info = get_worker_info()
        if info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = info.id, info.num_workers

        return rank * num_workers + worker_id

    def _sample_length(self, np_rng: np.random.Generator) -> int:
        """
        抽样本次 SCM 的时间长度。

        注意:
            输出长度 <= context_length。
            后面 format_real_training_sample 会负责左 padding 到 context_length。
        """
        if self.length_sampling == "uniform":
            return int(np_rng.integers(self.min_sample_length, self.max_sample_length + 1))

        low = math.log(self.min_sample_length)
        high = math.log(self.max_sample_length)
        val = int(round(math.exp(np_rng.uniform(low, high))))
        return int(np.clip(val, self.min_sample_length, self.max_sample_length))

    def _draw_path(self, rng: random.Random, file_cycle: List[Path]) -> Path:
        """
        抽样一个 Arrow 文件。

        balance_files=True 时，先 shuffle 文件列表，再循环 pop，
        保证不同 Arrow 文件被比较均匀地使用。
        """
        if self.balance_files:
            if len(file_cycle) == 0:
                file_cycle.extend(self.arrow_files)
                rng.shuffle(file_cycle)
            return file_cycle.pop()

        return self.arrow_files[int(rng.randrange(len(self.arrow_files)))]

    def _sample_raw_from_file(
        self,
        arrow_file,
        rng: random.Random,
    ) -> Optional[np.ndarray]:
        """
        从一个已经打开的 Arrow 文件中抽样一条原始 target 序列。

        和 ArrowSyntheticIterableDataset 一样，优先使用 __getitem__ 随机访问。
        """
        try:
            n = len(arrow_file)
        except Exception:
            n = None

        if n is not None and n > 0 and hasattr(arrow_file, "__getitem__"):
            for _ in range(max(1, self.max_sample_attempts)):
                try:
                    entry = arrow_file[int(rng.randrange(n))]
                    arr = extract_target_from_entry(entry, rng)
                    return np.asarray(arr, dtype=np.float64).reshape(-1)
                except Exception:
                    continue
            return None

        for entry in arrow_file:
            try:
                arr = extract_target_from_entry(entry, rng)
                return np.asarray(arr, dtype=np.float64).reshape(-1)
            except Exception:
                continue

        return None

    def _sample_base_series(
        self,
        *,
        length: int,
        rng: random.Random,
        loaded: Dict[Path, object],
        file_cycle: List[Path],
    ) -> Optional[np.ndarray]:
        """
        抽样一条可用的根节点序列。

        输出 shape = [length]。
        """
        for _ in range(max(1, self.max_sample_attempts)):
            path = self._draw_path(rng, file_cycle)

            if path not in loaded:
                try:
                    loaded[path] = load_gluonts_arrow_file(path)
                except Exception:
                    continue

            raw = self._sample_raw_from_file(loaded[path], rng)
            if raw is None:
                continue

            try:
                return _crop_or_interpolate_1d(
                    raw,
                    length=length,
                    rng=rng,
                    min_real_length=self.min_real_length,
                )
            except Exception:
                continue

        return None

    def _format_sample(
        self,
        arr: np.ndarray,
        rng: random.Random,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert one SCM output channel to the training format.

        Important:
            Here we do NOT call format_real_training_sample again,
            because the Arrow root series has already been cropped/interpolated
            before entering the SCM.

        Input:
            arr: shape = [L], where L <= context_length

        Output:
            x:             [context_length]
            observed_mask: [context_length]
            padding_mask:  [context_length]
        """
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
        arr = np.clip(arr, -1e6, 1e6)

        n = int(arr.shape[0])
        T = int(self.context_length)

        if n > T:
            # 理论上不会发生，因为 L <= max_sample_length <= context_length。
            # 这里保留防御性处理。
            start = rng.randrange(n - T + 1)
            arr = arr[start : start + T]
            n = T

        pad_len = T - n

        x = np.zeros(T, dtype=np.float32)
        observed = np.zeros(T, dtype=np.bool_)
        padding = np.ones(T, dtype=np.bool_)

        # 左 padding，真实序列放在末尾，和你 online_cauker 的 _format_sample 逻辑一致。
        x[pad_len:] = arr
        observed[pad_len:] = np.isfinite(arr)
        padding[pad_len:] = False

        return {
            "x": torch.from_numpy(x),
            "observed_mask": torch.from_numpy(observed),
            "padding_mask": torch.from_numpy(padding),
        }

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_id = self._global_worker_id()

        # 使用一个和普通 arrow / online_cauker 不同的 seed offset。
        base_seed = self.seed + 4_000_037 * worker_id

        py_rng = random.Random(base_seed)
        np_rng = np.random.default_rng(base_seed)

        loaded: Dict[Path, object] = {}
        file_cycle: List[Path] = []
        local_buffer: List[np.ndarray] = []

        while True:
            if len(local_buffer) == 0:
                length = self._sample_length(np_rng)

                def base_sampler(requested_length: int) -> np.ndarray:
                    arr = self._sample_base_series(
                        length=requested_length,
                        rng=py_rng,
                        loaded=loaded,
                        file_cycle=file_cycle,
                    )
                    if arr is None:
                        raise RuntimeError("Failed to sample a valid Arrow root series.")
                    return arr

                try:
                    x_multi, _ = generate_arrow_rooted_scm(
                        length=length,
                        num_features=self.num_features,
                        num_nodes=self.num_nodes,
                        max_parents=self.max_parents,
                        rng=np_rng,
                        base_sampler=base_sampler,
                    )
                except Exception:
                    # Arrow 文件中可能偶尔有坏样本；直接拒绝本次 SCM draw。
                    continue

                # 当前模型是 univariate objective，所以把多通道输出拆成多条样本。
                for j in range(x_multi.shape[1]):
                    local_buffer.append(
                        x_multi[:, j].astype(np.float32, copy=False)
                    )

                py_rng.shuffle(local_buffer)

            arr = local_buffer.pop()
            yield self._format_sample(arr, py_rng)

            
class ArrowSyntheticIterableDataset(IterableDataset):
    """Infinite stream over pre-generated KernelSynth Arrow files.

    Files are sampled uniformly, then one series is sampled from the selected
    file.  Each Arrow record is expected to contain a target-like field, matching
    Chronos kernel-synth.py records written by GluonTS ArrowWriter.
    """

    def __init__(
        self,
        root: str,
        context_length: int = 2048,
        seed: int = 42,
        min_sample_length: int = 126,
        max_sample_length: int = 2048,
        min_real_length: int = 2,
        max_sample_attempts: int = 32,
        balance_files: bool = True,
    ) -> None:
        super().__init__()
        self.root = str(root)
        self.context_length = int(context_length)
        self.seed = int(seed)
        self.min_sample_length = int(min_sample_length)
        self.max_sample_length = int(max_sample_length)
        self.min_real_length = int(min_real_length)
        self.max_sample_attempts = int(max_sample_attempts)
        self.balance_files = bool(balance_files)
        self.arrow_files = discover_arrow_files(self.root)

    def _global_worker_id(self) -> int:
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0

        info = get_worker_info()
        if info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = info.id, info.num_workers

        return rank * num_workers + worker_id

    def _format_entry(self, entry: Dict, rng: random.Random) -> Dict[str, torch.Tensor]:
        arr = extract_target_from_entry(entry, rng)
        return format_real_training_sample(
            arr=arr,
            context_length=self.context_length,
            rng=rng,
            min_real_length=self.min_real_length,
            min_sample_length=self.min_sample_length,
            max_sample_length=self.max_sample_length,
        )

    def _sample_from_file(self, arrow_file, rng: random.Random) -> Optional[Dict[str, torch.Tensor]]:
        try:
            n = len(arrow_file)
        except Exception:
            n = None

        if n is not None and n > 0 and hasattr(arrow_file, "__getitem__"):
            for _ in range(max(1, self.max_sample_attempts)):
                try:
                    entry = arrow_file[int(rng.randrange(n))]
                    return self._format_entry(entry, rng)
                except Exception:
                    continue
            return None

        for entry in arrow_file:
            try:
                return self._format_entry(entry, rng)
            except Exception:
                continue
        return None

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_id = self._global_worker_id()
        rng = random.Random(self.seed + 3_000_017 * worker_id)
        files = list(self.arrow_files)
        loaded: Dict[Path, object] = {}

        while True:
            if self.balance_files:
                order = list(files)
                rng.shuffle(order)
            else:
                order = [files[int(rng.randrange(len(files)))]]

            for path in order:
                if path not in loaded:
                    try:
                        loaded[path] = load_gluonts_arrow_file(path)
                    except Exception:
                        continue

                sample = self._sample_from_file(loaded[path], rng)
                if sample is not None:
                    yield sample


# =============================================================================
