"""Tabby zero-shot anomaly detection adapters for the TSB-AD benchmark."""

from .detector import LATENT_METRICS, PatchTSTFM_AD

TabbyAnomalyDetector = PatchTSTFM_AD

__all__ = ["LATENT_METRICS", "PatchTSTFM_AD", "TabbyAnomalyDetector"]
