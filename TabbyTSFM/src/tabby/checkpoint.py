"""Checkpoint resolution and loading for the Tabby backbone.

Two on-disk layouts are supported, and every entry point in the repository goes
through :func:`load_backbone` so they accept the same things:

``config.json`` + ``model.safetensors``
    A Hugging Face snapshot -- what ``save_pretrained`` writes, what
    ``huggingface-cli download`` leaves on disk, and the layout of the published
    Tabby-Pretrain and ``patchtst-fm-r1`` weights.  The tensor names already
    match the bundled architecture, so no key translation is needed; only the
    HF config vocabulary (``d_patch``, ``n_layer``, ``num_quantile``, ...) has
    to be mapped onto the trainer-facing dataclass fields.

``pytorch_model.bin``
    An in-house training checkpoint: a torch payload carrying ``model``,
    ``config`` and ``step``.  Accepted as the file itself, as the step directory
    containing it, or as a parent directory holding a ``latest`` symlink.

A bare ``org/name`` string that does not exist locally is treated as a Hub
repository id and downloaded via ``huggingface_hub``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import torch

from .models.PatchTSTFM import PatchTSTFM, PatchTSTFMConfig

logger = logging.getLogger(__name__)

HF_CONFIG_NAME = "config.json"
HF_WEIGHTS_NAME = "model.safetensors"
TORCH_WEIGHTS_NAME = "pytorch_model.bin"

# HF config.json key -> PatchTSTFMConfig dataclass field.  ``head_dim`` is
# derived from d_model / n_head below; ``mlp_ratio`` and ``eps`` are not stored
# in config.json, so the dataclass defaults stand (mlp_ratio is verified against
# the actual weight shape when the weights are read).
_HF_CONFIG_FIELDS = {
    "context_length": "context_length",
    "d_patch": "patch_size",
    "n_layer": "num_layers",
    "d_model": "d_model",
    "num_quantile": "num_quantiles",
    "pretrain_mask_ratio": "mask_ratio",
    "pretrain_mask_cont": "cpm_blocks",
    "dropout": "dropout",
}

# Probed to confirm mlp_ratio, which no config.json records.
_MLP_PROBE_KEY = "backbone.blocks.0.mlp.layers.0.weight"


def is_hf_snapshot(path) -> bool:
    """True if ``path`` is a directory holding config.json + model.safetensors."""
    path = Path(path)
    return (
        path.is_dir()
        and (path / HF_CONFIG_NAME).is_file()
        and (path / HF_WEIGHTS_NAME).is_file()
    )


def clean_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip DDP ``module.`` and torch.compile ``_orig_mod.`` prefixes."""
    cleaned = {}
    for key, value in state.items():
        while key.startswith("module.") or key.startswith("_orig_mod."):
            if key.startswith("module."):
                key = key[len("module."):]
            if key.startswith("_orig_mod."):
                key = key[len("_orig_mod."):]
        cleaned[key] = value
    return cleaned


def _download_from_hub(repo_id: str) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise FileNotFoundError(
            f"Checkpoint not found locally: {repo_id}. Install huggingface_hub to "
            f"resolve it as a Hub repository id."
        ) from exc
    logger.info("Downloading checkpoint from the Hugging Face Hub: %s", repo_id)
    return Path(snapshot_download(repo_id))


def resolve_checkpoint_path(checkpoint) -> Path:
    """Return the snapshot directory or the ``.bin`` file to load.

    Accepts an HF snapshot directory, a ``pytorch_model.bin``, a step directory,
    a parent directory with a ``latest`` symlink, or a Hub repository id.
    """
    raw = str(checkpoint)
    path = Path(raw).expanduser()

    if not path.exists() and raw.count("/") == 1 and not raw.startswith("."):
        # Validate the download like any other local directory, so a repository
        # laid out differently (sharded weights, say) fails with the same message.
        return resolve_checkpoint_path(_download_from_hub(raw))

    path = path.resolve()

    if path.is_dir():
        if is_hf_snapshot(path):
            return path

        latest = path / "latest"
        if latest.exists():
            path = latest.resolve()
            if is_hf_snapshot(path):
                return path

        if path.is_dir():
            candidate = path / TORCH_WEIGHTS_NAME
            if candidate.is_file():
                return candidate
            raise FileNotFoundError(
                f"Directory {path} is neither a Hugging Face snapshot "
                f"({HF_CONFIG_NAME} + {HF_WEIGHTS_NAME}) nor a step directory "
                f"containing {TORCH_WEIGHTS_NAME}."
            )

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def _config_from_hf(hf: Dict) -> PatchTSTFMConfig:
    """Map an HF config.json onto the trainer-facing dataclass."""
    kwargs = {field: hf[key] for key, field in _HF_CONFIG_FIELDS.items() if key in hf}
    if "n_head" in hf:
        if int(hf["d_model"]) % int(hf["n_head"]):
            raise ValueError(
                f"d_model {hf['d_model']} is not divisible by n_head {hf['n_head']}."
            )
        kwargs["head_dim"] = int(hf["d_model"]) // int(hf["n_head"])

    cfg = PatchTSTFMConfig(**kwargs)

    # n_patch is redundant with context_length / d_patch; disagreement means the
    # config was hand-edited and the patch grid would silently be wrong.
    if "n_patch" in hf and int(hf["n_patch"]) != cfg.num_patches:
        raise ValueError(
            f"config.json n_patch={hf['n_patch']} contradicts "
            f"context_length/d_patch ({cfg.context_length}/{cfg.patch_size} = "
            f"{cfg.num_patches})."
        )
    return cfg


def _config_from_payload(payload: Dict) -> PatchTSTFMConfig:
    """Build the dataclass config from an in-house checkpoint payload."""
    if "config" not in payload:
        raise KeyError("Checkpoint does not contain a 'config' field.")

    raw = dict(payload["config"])
    # Older checkpoints used 'attn_gate' instead of 'attn_gate_type'.
    if "attn_gate_type" not in raw and "attn_gate" in raw:
        raw["attn_gate_type"] = raw.pop("attn_gate")

    valid = set(PatchTSTFMConfig.__dataclass_fields__)
    return PatchTSTFMConfig(**{k: raw[k] for k in valid if k in raw})


def load_hf_snapshot(path: Path) -> Tuple[PatchTSTFM, PatchTSTFMConfig, int]:
    """Load ``config.json`` + ``model.safetensors`` into the bundled backbone."""
    try:
        from safetensors import safe_open
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError(
            "safetensors is required to read a Hugging Face snapshot "
            "(pip install safetensors, or install the 'forecast' extra)."
        ) from exc

    hf = json.loads((path / HF_CONFIG_NAME).read_text(encoding="utf-8"))
    cfg = _config_from_hf(hf)
    model = PatchTSTFM(cfg)

    weights = path / HF_WEIGHTS_NAME
    state = clean_state_dict(load_file(str(weights)))

    probe = state.get(_MLP_PROBE_KEY)
    expected_hidden = int(cfg.mlp_ratio * cfg.d_model)
    if probe is not None and probe.shape[0] != expected_hidden:
        raise RuntimeError(
            f"mlp hidden width {probe.shape[0]} != mlp_ratio * d_model "
            f"({cfg.mlp_ratio} * {cfg.d_model} = {expected_hidden}); "
            f"fix mlp_ratio before loading {weights}."
        )

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Hugging Face snapshot key mismatch in {weights}.\n"
            f"  Missing:    {missing}\n"
            f"  Unexpected: {unexpected}"
        )

    # The training step travels in the safetensors metadata, not in config.json.
    step = -1
    with safe_open(str(weights), framework="pt") as handle:
        metadata = handle.metadata() or {}
    for key in ("pretrain_step", "step"):
        if key in metadata:
            try:
                step = int(metadata[key])
            except (TypeError, ValueError):
                logger.warning("Ignoring non-integer %s=%r", key, metadata[key])
            break

    n_tensors = len(state)
    del state
    logger.info(
        "Loaded HF snapshot %s (%d tensors, step=%d, L=%d d=%d p=%d K=%d T=%d)",
        path, n_tensors, step, cfg.num_layers, cfg.d_model, cfg.patch_size,
        cfg.num_quantiles, cfg.context_length,
    )
    return model, cfg, step


def load_torch_checkpoint(path: Path) -> Tuple[PatchTSTFM, PatchTSTFMConfig, int]:
    """Load an in-house ``pytorch_model.bin`` training checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a dict checkpoint payload, got {type(payload)!r}.")

    cfg = _config_from_payload(payload)
    model = PatchTSTFM(cfg)

    missing, unexpected = model.load_state_dict(
        clean_state_dict(payload["model"]), strict=False
    )
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint key mismatch in {path}.\n"
            f"  Missing:    {missing}\n"
            f"  Unexpected: {unexpected}"
        )

    step = int(payload.get("step", -1))
    del payload
    logger.info("Loaded checkpoint %s at step=%d", path, step)
    return model, cfg, step


def load_backbone(
    checkpoint, device=None
) -> Tuple[PatchTSTFM, PatchTSTFMConfig, int]:
    """Load either checkpoint layout and return ``(model, config, step)``.

    The model is moved to ``device`` (when given) and left in eval mode.
    """
    path = resolve_checkpoint_path(checkpoint)
    if is_hf_snapshot(path):
        model, cfg, step = load_hf_snapshot(path)
    else:
        model, cfg, step = load_torch_checkpoint(path)

    if device is not None:
        model.to(device)
    model.eval()
    return model, cfg, step
