"""Zero-shot anomaly detector built on Tabby-Pretrain.

The predictor is loaded once, frozen, and only used for forward passes: there is
no per-series fitting, so the detector is genuinely zero-shot.

Scoring happens in representation space. The frozen encoder embeds the series
into per-patch latents (no forecasting), and each patch is scored by how far it
sits from the rest of the series' latent distribution (``latent_metric``).
"""

from __future__ import annotations

import numpy as np


# latent-space outlierness metrics
LATENT_METRICS = ("centroid", "centroid_robust", "mahalanobis_diag", "mahalanobis")


class PatchTSTFM_AD:
    """Zero-shot latent-space anomaly detector.

    Parameters
    ----------
    predictor : object
        A pre-loaded ``PatchTSTFMPredictor``. Constructed once by the caller and
        reused across configs so the checkpoint is not reloaded.
    latent_metric : {"centroid", "centroid_robust", "mahalanobis_diag", "mahalanobis"}
        Latent-space outlierness metric.
    latent_layer : int
        Number of Transformer blocks applied by the predictor after clamping;
        a negative value applies all blocks.
    latent_stride : int, optional
        Step between latent windows. Defaults to a non-overlapping tiling
        (== the model's context length).
    agg : {"mean", "max"}
        Channel aggregation for multivariate series (ignored when univariate).
    eps : float
        Numerical floor.
    """

    def __init__(
        self,
        predictor,
        latent_metric: str = "centroid",
        latent_layer: int = -1,
        latent_stride: int | None = None,
        agg: str = "mean",
        eps: float = 1e-6,
    ):
        self.predictor = predictor
        if latent_metric not in LATENT_METRICS:
            raise ValueError(f"Unknown latent_metric: {latent_metric} "
                             f"(choose from {LATENT_METRICS})")
        self.latent_metric = latent_metric
        self.latent_layer = latent_layer
        self.latent_stride = latent_stride
        if agg not in ("mean", "max"):
            raise ValueError(f"Unknown agg: {agg}")
        self.agg = agg
        self.eps = eps

    def _minmax(self, s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else np.zeros_like(s)

    def _latent_distance(self, P, metric):
        """Per-row outlierness of patch embeddings ``P`` [n_patch, d]."""
        mu = P.mean(axis=0)
        if metric == "centroid":
            return np.linalg.norm(P - mu, axis=1)
        if metric == "centroid_robust":
            med = np.median(P, axis=0)
            mad = np.median(np.abs(P - med), axis=0) + self.eps
            return np.linalg.norm((P - med) / mad, axis=1)
        if metric == "mahalanobis_diag":
            var = P.var(axis=0) + self.eps
            return np.sqrt((((P - mu) ** 2) / var).sum(axis=1))
        if metric == "mahalanobis":
            Pc = P - mu
            d = P.shape[1]
            cov = np.cov(P, rowvar=False)
            if cov.ndim == 0:                # single feature
                cov = cov.reshape(1, 1)
            lam = 0.1                        # Ledoit-Wolf-style shrinkage
            cov = (1 - lam) * cov + lam * (np.trace(cov) / d) * np.eye(d)
            inv = np.linalg.pinv(cov)
            m2 = np.einsum("ij,jk,ik->i", Pc, inv, Pc)
            return np.sqrt(np.clip(m2, 0, None))
        raise ValueError(f"Unknown latent_metric: {metric}")

    def _collect_latent(self, series, stride=None):
        """Embed one channel to per-patch latents (the GPU-bound step).

        The series is tiled into windows of the model's native context length
        ``W = cfg.context_length``; each is embedded and every content patch is
        kept with the timestep span it covers. The encoder right-aligns real
        content inside its fixed ``W`` buffer, so tiling and the patch->timestep
        mapping are anchored on ``W`` and left-pad patches are dropped. The
        returned ``(P, spans)`` are metric independent, so one embedding pass
        feeds every metric (see :meth:`score_latent_variants`). Returns
        ``(P, spans, T, fallback)`` where ``fallback`` is a ready score vector
        for series too short to patch.
        """
        T = len(series)
        L = int(self.predictor.cfg.patch_size)
        W = int(self.predictor.cfg.context_length)
        if T < L:
            mu, sigma = series.mean(), series.std() + self.eps
            return None, None, T, np.abs(series - mu) / sigma

        stride = stride or self.latent_stride or W
        if T >= W:
            starts = list(range(0, T - W + 1, stride))
            if starts[-1] != T - W:
                starts.append(T - W)           # cover the tail
            windows = [series[s: s + W] for s in starts]
            gstarts = list(starts)
            clens = [W] * len(starts)
        else:
            # single short window, right-aligned inside the model buffer
            windows = [series]
            gstarts = [0]
            clens = [T]

        emb = self.predictor.embed(windows, layer=self.latent_layer)  # [n, N, d]
        n, N, _ = emb.shape

        rows = []            # embeddings of content patches
        spans = []           # their (global_lo, global_hi) timestep spans
        for wi in range(n):
            g = gstarts[wi]
            leftpad = W - clens[wi]            # real content is right-aligned
            for p in range(N):
                lo = max(g + p * L - leftpad, 0)
                hi = min(g + (p + 1) * L - leftpad, T)
                if hi <= lo:
                    continue                   # patch sits in the left-pad region
                rows.append(emb[wi, p])
                spans.append((lo, hi))

        return np.asarray(rows), spans, T, None

    def _latent_from(self, P, spans, T, metric):
        """Distance-score cached patch embeddings and broadcast to timesteps."""
        dist = self._latent_distance(P, metric)
        scores = np.zeros(T)
        counts = np.zeros(T)
        for (lo, hi), dv in zip(spans, dist):
            scores[lo:hi] += dv
            counts[lo:hi] += 1
        counts[counts == 0] = 1
        return scores / counts

    def _score_channel(self, series, metric=None):
        """Latent-space outlierness for one channel."""
        P, spans, T, fallback = self._collect_latent(series)
        if fallback is not None:
            return fallback
        return self._latent_from(P, spans, T, metric or self.latent_metric)

    def _combine_channels(self, per_channel):
        if per_channel.shape[1] == 1:
            return per_channel[:, 0]
        if self.agg == "mean":
            return per_channel.mean(axis=1)
        return per_channel.max(axis=1)

    def fit(self, X: np.ndarray, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        T, C = X.shape

        per_channel = np.stack(
            [self._score_channel(X[:, c]) for c in range(C)], axis=1
        )
        self.decision_scores_ = self._minmax(self._combine_channels(per_channel))
        return self

    def score_latent_variants(self, X, metrics, strides=(None,)):
        """Score one series over a ``latent_stride`` x ``latent_metric`` grid.

        The embeddings depend only on the stride, so for a given stride one
        embedding pass feeds every metric. Returns ``{(stride, metric):
        scores[T]}`` (``stride=None`` == the default, non-overlapping tiling),
        each min-max normalised as :meth:`fit` would. This is the fast path
        behind ``--sweep_latent_metric`` / ``--sweep_latent_stride``.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        T, C = X.shape
        out = {}
        for stride in strides:
            collected = [self._collect_latent(X[:, c], stride) for c in range(C)]
            for metric in metrics:
                cols = []
                for c in range(C):
                    P, spans, Tc, fallback = collected[c]
                    cols.append(fallback if fallback is not None
                                else self._latent_from(P, spans, Tc, metric))
                per_channel = np.stack(cols, axis=1)
                out[(stride, metric)] = self._minmax(self._combine_channels(per_channel))
        return out

    def decision_function(self, X=None):
        """Return the precomputed decision scores (API consistency)."""
        return self.decision_scores_
