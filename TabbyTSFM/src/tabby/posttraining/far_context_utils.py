"""
Utility functions for computing far-context statistics.
Used in data pipeline (numpy) before data enters the model.

Features must match ContextStatisticsExtractor exactly (same order, same semantics).
"""
import numpy as np


def _autocorr_numpy(z, lag):
    """Compute autocorrelation at a specific lag for z-scored array."""
    if len(z) <= lag:
        return 0.0
    z_c = z - z.mean()
    var = (z_c ** 2).mean()
    if var < 1e-8:
        return 0.0
    cov = (z_c[:-lag] * z_c[lag:]).mean()
    return float(np.clip(cov / var, -1, 1))


def compute_far_stats_numpy(full_target: np.ndarray, context_length: int, n_stats: int = 20) -> np.ndarray:
    """
    Compute statistics from the far-context portion of a time series.
    Far context = full_target[: T - context_length], i.e. everything the encoder won't see.
    
    Features are computed in z-scored space for scale invariance.
    Feature order matches ContextStatisticsExtractor exactly.
    
    Args:
        full_target: 1D numpy array, the complete time series (raw values)
        context_length: how many steps the model's encoder sees (e.g. 2048)
        n_stats: number of statistics to output (10 for backward compat, 20 for full)
        
    Returns:
        np.ndarray of shape [n_stats].
        Returns zeros if far context is empty (series shorter than context_length).
    """
    full_target = np.asarray(full_target, dtype=np.float32).ravel()
    T = len(full_target)
    
    if T <= context_length:
        return np.zeros(n_stats, dtype=np.float32)
    
    far = full_target[:T - context_length]
    
    # Remove NaN/Inf
    valid_mask = np.isfinite(far)
    n_valid = valid_mask.sum()
    
    if n_valid < 2:
        return np.zeros(n_stats, dtype=np.float32)
    
    far_valid = far[valid_mask]
    
    # Z-score normalization (scale-invariant, matching torch extractor's scaled space)
    mean_raw = far_valid.mean()
    std_raw = far_valid.std()
    if std_raw < 1e-8:
        std_raw = 1.0
    z = (far_valid - mean_raw) / std_raw
    
    stats = np.zeros(max(n_stats, 20), dtype=np.float32)
    
    # ---- Original 10 features ----
    # 0: mean (of z-scored)
    stats[0] = np.clip(z.mean(), -10, 10)
    
    # 1: std
    stats[1] = np.clip(z.std(), 0, 10)
    
    # 2: min
    stats[2] = np.clip(z.min(), -10, 10)
    
    # 3: max
    stats[3] = np.clip(z.max(), -10, 10)
    
    # 4: trend (2nd half mean - 1st half mean)
    half = len(z) // 2
    if half > 0:
        stats[4] = np.clip(z[half:].mean() - z[:half].mean(), -5, 5)
    
    # 5: diff_mean
    if len(z) > 1:
        diff = np.diff(z)
        stats[5] = np.clip(diff.mean(), -5, 5)
        # 6: diff_std
        stats[6] = np.clip(diff.std(), 0, 10)
    
    # 7: autocorrelation lag-1
    stats[7] = _autocorr_numpy(z, 1)
    
    # 8: fraction observed
    stats[8] = n_valid / len(far)
    
    # 9: log length (normalized)
    stats[9] = np.log1p(n_valid) / np.log(4001)
    
    # ---- New 10 features (only if n_stats > 10) ----
    if n_stats > 10:
        z_mean = z.mean()
        z_std = max(z.std(), 1e-6)
        z_centered = z - z_mean
        
        # 10: skewness
        m3 = (z_centered ** 3).mean()
        stats[10] = np.clip(m3 / z_std ** 3, -5, 5)
        
        # 11: kurtosis (excess)
        m4 = (z_centered ** 4).mean()
        stats[11] = np.clip(m4 / z_std ** 4 - 3.0, -10, 10)
        
        # 12: Q25
        stats[12] = np.clip(np.percentile(z, 25), -10, 10)
        
        # 13: Q75
        stats[13] = np.clip(np.percentile(z, 75), -10, 10)
        
        # 14: IQR
        stats[14] = np.clip(stats[13] - stats[12], 0, 20)
        
        # 15: autocorr lag-7
        stats[15] = _autocorr_numpy(z, 7)
        
        # 16: autocorr lag-24
        stats[16] = _autocorr_numpy(z, 24)
        
        # 17: zero crossing rate
        if len(z) > 1:
            sign = np.sign(z_centered)
            sign_changes = np.sum(sign[1:] != sign[:-1])
            stats[17] = np.clip(sign_changes / (len(z) - 1), 0, 1)
        
        # 18: energy
        stats[18] = np.clip((z ** 2).mean(), 0, 100)
        
        # 19: range_ratio = (max - min) / std
        range_val = max(0, stats[3] - stats[2])
        stats[19] = np.clip(range_val / max(z_std, 1e-6), 0, 20)
    
    return np.nan_to_num(stats[:n_stats], nan=0.0, posinf=0.0, neginf=0.0)