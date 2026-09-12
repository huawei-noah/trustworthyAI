"""Tabby time-series foundation model.

The core package contains the pretrained backbone and preprocessing utilities.
Prompt/post-training components live under :mod:`tabby.posttraining`, while
benchmark entry points stay outside the importable package.
"""

from .models.PatchTSTFM import PatchTSTFM, PatchTSTFMConfig

__all__ = ["PatchTSTFM", "PatchTSTFMConfig"]

__version__ = "0.1.0"
