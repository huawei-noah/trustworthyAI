"""Checkpoint-loading checks against a real Hugging Face snapshot.

Every test here runs on the sample snapshot written by
``tests/make_tiny_hf_snapshot.py``: an actual ``config.json`` +
``model.safetensors`` pair produced by ``save_pretrained``, which is the layout
both loading paths accept in production --

  * ``tabby.posttraining.model.load_patchtstfm_backbone`` maps the HF config
    onto the trainer-facing dataclass and loads the weights into the bundled
    ``tabby.models.PatchTSTFM`` adapter (the ``patchtst-fm-r1`` path), and
  * ``transformers``' own ``from_pretrained`` rebuilds the HF model classes.

The snapshot is tiny (~35k parameters) but structurally faithful, so the tests
exercise config mapping, tensor-name agreement, and the resulting forward pass
without shipping release weights.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import sys
import tempfile
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from unittest import SkipTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

_MISSING = ""
try:
    import numpy as np
    import pandas as pd
    import torch
    from safetensors.torch import load_file, save_file

    from tabby.checkpoint import load_backbone, resolve_checkpoint_path
    from tabby.models.PatchTSTFM import PatchTSTFM, PatchTSTFMConfig
    from tabby.models.configuration_patchtst_fm import PatchTSTFMConfig as HFConfig
    from tabby.models.modeling_patchtst_fm import (
        PatchTSTFMForPrediction,
        PatchTSTFMForPretraining,
    )
    from tabby.posttraining.model import _is_hf_snapshot, load_patchtstfm_backbone
    from tabby.utils.input_preprocessing import mask_aware_normalize_for_inference

    from make_tiny_hf_snapshot import TINY_CONFIG, build_tiny_model, write_tiny_snapshot
except ImportError as exc:  # torch / transformers / safetensors not installed
    _MISSING = str(exc)

if _MISSING:  # pragma: no cover - environment guard
    import pytest

    pytest.skip(
        "install the forecast extra to run the checkpoint tests: " + _MISSING,
        allow_module_level=True,
    )


# =============================================================================
# Sample snapshot
# =============================================================================

def _temp_dir(prefix: str) -> Path:
    path = tempfile.mkdtemp(prefix=prefix)
    atexit.register(shutil.rmtree, path, ignore_errors=True)
    return Path(path)


@lru_cache(maxsize=1)
def sample_snapshot() -> Path:
    """Write the sample HF snapshot once and reuse it across tests."""
    return write_tiny_snapshot(_temp_dir("tabby_hf_snapshot_") / "tiny_patchtstfm_hf")


def snapshot_tensors() -> dict:
    return load_file(str(sample_snapshot() / "model.safetensors"))


def snapshot_config() -> dict:
    """The snapshot's config.json -- what the loaders actually read."""
    return json.loads((sample_snapshot() / "config.json").read_text(encoding="utf-8"))


def edited_snapshot(edit) -> Path:
    """Copy the sample snapshot and apply ``edit`` to its state dict."""
    dest = _temp_dir("tabby_hf_snapshot_edit_") / "edited"
    shutil.copytree(sample_snapshot(), dest)
    state = load_file(str(dest / "model.safetensors"))
    edit(state)
    save_file(state, str(dest / "model.safetensors"))
    return dest


def forward_inputs(cfg, batch_size: int = 2, history: int = 40):
    """Build one masked, normalized batch the way inference does.

    Layout is the production one -- ``[left-pad | history | forecast]`` over a
    full ``cfg.context_length`` window -- so the tensors fed to the loaded model
    are the tensors it was trained through.
    """
    total = cfg.context_length
    forecast = 8
    pad = total - history - forecast
    assert pad >= 0, "history + forecast must fit inside the context window"

    generator = torch.Generator().manual_seed(0)
    x = torch.randn(batch_size, total, generator=generator)

    padding_mask = torch.zeros(batch_size, total, dtype=torch.bool)
    padding_mask[:, :pad] = True
    pred_mask = torch.zeros(batch_size, total, dtype=torch.bool)
    pred_mask[:, pad + history:] = True
    observed_mask = ~padding_mask

    x_norm, _, _, union_mask, patch_padding = mask_aware_normalize_for_inference(
        x, observed_mask, pred_mask, padding_mask, cfg
    )
    return x_norm, union_mask, patch_padding


# =============================================================================
# The snapshot itself
# =============================================================================

def test_sample_snapshot_has_the_hugging_face_layout() -> None:
    snapshot = sample_snapshot()
    assert sorted(p.name for p in snapshot.iterdir()) == [
        "config.json",
        "model.safetensors",
    ]
    assert _is_hf_snapshot(snapshot)

    config = snapshot_config()
    assert config["model_type"] == "patchtst_fm"
    for key, value in TINY_CONFIG.items():
        assert config[key] == value, key
    # n_patch is derived, not passed in, and must survive serialization.
    assert config["n_patch"] == TINY_CONFIG["context_length"] // TINY_CONFIG["d_patch"]

    tensors = snapshot_tensors()
    assert tensors
    assert all(name.startswith("backbone.") for name in tensors), sorted(tensors)[:5]


def test_hf_snapshot_detection_rejects_incomplete_directories() -> None:
    snapshot = sample_snapshot()

    config_only = _temp_dir("tabby_cfg_only_")
    shutil.copy(snapshot / "config.json", config_only / "config.json")
    assert not _is_hf_snapshot(config_only)

    weights_only = _temp_dir("tabby_weights_only_")
    shutil.copy(snapshot / "model.safetensors", weights_only / "model.safetensors")
    assert not _is_hf_snapshot(weights_only)

    # An in-house step directory must not be mistaken for an HF snapshot.
    step_dir = _temp_dir("tabby_step_dir_")
    (step_dir / "pytorch_model.bin").write_bytes(b"")
    assert not _is_hf_snapshot(step_dir)
    assert not _is_hf_snapshot(step_dir / "pytorch_model.bin")


# =============================================================================
# tabby.posttraining.model.load_patchtstfm_backbone (the r1 path)
# =============================================================================

def test_backbone_loads_from_the_hf_snapshot() -> None:
    model, cfg, step = load_patchtstfm_backbone(
        str(sample_snapshot()), torch.device("cpu")
    )

    assert isinstance(model, PatchTSTFM)
    assert isinstance(cfg, PatchTSTFMConfig)
    # An HF snapshot carries no training step.
    assert step == -1
    assert not model.training

    # Every HF config key mapped onto its trainer-facing dataclass field.
    hf = snapshot_config()
    assert cfg.context_length == hf["context_length"]
    assert cfg.patch_size == hf["d_patch"]
    assert cfg.num_layers == hf["n_layer"]
    assert cfg.d_model == hf["d_model"]
    assert cfg.num_quantiles == hf["num_quantile"]
    assert cfg.cpm_blocks == hf["pretrain_mask_cont"]
    assert cfg.mask_ratio == hf["pretrain_mask_ratio"]
    # head_dim is not stored; it is derived from d_model / n_head.
    assert cfg.head_dim == hf["d_model"] // hf["n_head"]
    assert cfg.num_heads == hf["n_head"]
    assert cfg.num_patches == hf["n_patch"]

    assert len(model.backbone.blocks) == hf["n_layer"]
    assert all(p.device.type == "cpu" for p in model.parameters())


def test_every_snapshot_tensor_lands_in_the_model() -> None:
    model, _, _ = load_patchtstfm_backbone(str(sample_snapshot()), torch.device("cpu"))
    expected = snapshot_tensors()
    loaded = model.state_dict()

    # No missing and no unexpected keys: the adapter and the snapshot use the
    # same tensor names, so no key translation is needed.
    assert set(loaded) == set(expected)
    for name, tensor in expected.items():
        assert torch.equal(loaded[name], tensor), name


def test_both_loaders_agree_on_the_same_snapshot() -> None:
    """The r1 adapter path and ``from_pretrained`` must read a snapshot alike.

    The two paths build their configs differently -- one maps ``config.json``
    onto the trainer dataclass, the other hands it straight to the HF config --
    so a drift in either mapping shows up as disagreeing weights here.
    """
    adapter, _, _ = load_patchtstfm_backbone(str(sample_snapshot()), torch.device("cpu"))
    hf_model = PatchTSTFMForPrediction.from_pretrained(str(sample_snapshot())).eval()

    by_adapter, by_transformers = adapter.state_dict(), hf_model.state_dict()
    assert set(by_adapter) == set(by_transformers)
    for name in by_adapter:
        assert torch.equal(by_adapter[name], by_transformers[name]), name


def test_loaded_backbone_produces_finite_monotone_quantiles() -> None:
    model, cfg, _ = load_patchtstfm_backbone(str(sample_snapshot()), torch.device("cpu"))
    x_norm, union_mask, patch_padding = forward_inputs(cfg)

    with torch.no_grad():
        q_hat = model(x_norm, union_mask, patch_padding)

    assert q_hat.shape == (x_norm.shape[0], cfg.context_length, cfg.num_quantiles)
    assert torch.isfinite(q_hat).all()
    # The quantile head is monotone by construction; a shape or config mismatch
    # in the loaded weights shows up here first.
    assert (torch.diff(q_hat, dim=-1) >= 0).all()


def test_loaded_backbone_encodes_patch_latents() -> None:
    model, cfg, _ = load_patchtstfm_backbone(str(sample_snapshot()), torch.device("cpu"))
    x_norm, union_mask, patch_padding = forward_inputs(cfg)

    with torch.no_grad():
        latents = model.encode(x_norm, union_mask, patch_padding)
        first_layer = model.encode(x_norm, union_mask, patch_padding, layer=1)

    assert latents.shape == (x_norm.shape[0], cfg.num_patches, cfg.d_model)
    assert torch.isfinite(latents).all()
    # Two blocks are loaded, so stopping after the first must change the result.
    assert not torch.equal(latents, first_layer)


def test_snapshot_with_a_mismatched_mlp_width_is_rejected() -> None:
    def widen_mlp(state):
        name = "backbone.blocks.0.mlp.layers.0.weight"
        state[name] = torch.cat([state[name], state[name][:1]], dim=0)

    try:
        load_patchtstfm_backbone(str(edited_snapshot(widen_mlp)), torch.device("cpu"))
    except RuntimeError as exc:
        assert "mlp" in str(exc)
    else:
        raise AssertionError("an inconsistent mlp hidden width must not load silently")


def test_snapshot_with_a_dropped_tensor_is_rejected() -> None:
    dropped = "backbone.out_layer.layer2.weight"

    try:
        load_patchtstfm_backbone(
            str(edited_snapshot(lambda state: state.pop(dropped))), torch.device("cpu")
        )
    except RuntimeError as exc:
        assert dropped in str(exc)
    else:
        raise AssertionError("an incomplete snapshot must not load silently")


# =============================================================================
# transformers-native from_pretrained
# =============================================================================

def test_config_round_trips_through_from_pretrained() -> None:
    config = HFConfig.from_pretrained(str(sample_snapshot()))

    for key, value in TINY_CONFIG.items():
        assert getattr(config, key) == value, key
    assert config.n_patch == TINY_CONFIG["context_length"] // TINY_CONFIG["d_patch"]
    assert len(config.quantile_levels) == TINY_CONFIG["num_quantile"]
    # attribute_map aliases stay wired up after a round trip.
    assert config.hidden_size == TINY_CONFIG["d_model"]
    assert config.num_hidden_layers == TINY_CONFIG["n_layer"]


def test_prediction_model_round_trips_through_from_pretrained() -> None:
    source = build_tiny_model()
    reloaded = PatchTSTFMForPrediction.from_pretrained(str(sample_snapshot())).eval()

    saved, restored = source.state_dict(), reloaded.state_dict()
    assert set(saved) == set(restored)
    for name in saved:
        assert torch.equal(saved[name], restored[name]), name

    past_values = torch.randn(2, 32, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        before = source(past_values=past_values, prediction_length=8)
        after = reloaded(past_values=past_values, prediction_length=8)

    # (batch, horizon, channels) and (batch, quantiles, horizon, channels).
    assert before.prediction_outputs.shape == (2, 8, 1)
    assert before.quantile_outputs.shape == (2, TINY_CONFIG["num_quantile"], 8, 1)
    assert torch.isfinite(before.prediction_outputs).all()
    assert torch.equal(before.prediction_outputs, after.prediction_outputs)
    assert torch.equal(before.quantile_outputs, after.quantile_outputs)


def test_pretraining_model_loads_the_same_snapshot() -> None:
    """``ForPretraining`` and ``ForPrediction`` share the ``backbone.*`` names."""
    model = PatchTSTFMForPretraining.from_pretrained(str(sample_snapshot())).eval()

    expected = snapshot_tensors()
    loaded = model.state_dict()
    assert set(loaded) == set(expected)
    for name, tensor in expected.items():
        assert torch.equal(loaded[name], tensor), name


def test_bare_backbone_class_cannot_absorb_a_wrapper_snapshot() -> None:
    """A snapshot stores wrapper-level ``backbone.*`` names.

    ``PatchTSTFMModel`` *is* the wrapped module, so its own parameters are
    unprefixed and ``from_pretrained`` matches none of them -- transformers only
    warns, handing back a randomly initialized model.  This is why every entry
    point goes through :mod:`tabby.checkpoint`, which treats a key mismatch as an
    error.  A published ``config.json`` should therefore name one of the wrapper
    classes in ``architectures``, not ``PatchTSTFMModel``.
    """
    from tabby.models.modeling_patchtst_fm import PatchTSTFMModel

    expected = snapshot_tensors()
    assert all(name.startswith("backbone.") for name in expected)

    bare = PatchTSTFMModel.from_pretrained(str(sample_snapshot())).eval()
    assert not set(bare.state_dict()) & set(expected), "prefixes unexpectedly agree"

    # The supported paths load every tensor from that same snapshot.
    wrapper = PatchTSTFMForPrediction.from_pretrained(str(sample_snapshot()))
    adapter, _, _ = load_backbone(sample_snapshot(), torch.device("cpu"))
    for model in (wrapper, adapter):
        assert set(model.state_dict()) == set(expected)


def test_hf_prediction_and_trainer_adapter_share_state_dict_keys() -> None:
    """The interchange contract documented in ``tabby/models/PatchTSTFM.py``.

    Checkpoints written by the pretraining recipe load into the HF prediction
    class and vice versa, which only holds while both expose the same names.
    """
    _, cfg, _ = load_patchtstfm_backbone(str(sample_snapshot()), torch.device("cpu"))
    adapter = PatchTSTFM(cfg)
    hf_model = PatchTSTFMForPrediction(cfg.to_hf_config())

    assert set(adapter.state_dict()) == set(hf_model.state_dict())
    adapter_shapes = {k: tuple(v.shape) for k, v in adapter.state_dict().items()}
    hf_shapes = {k: tuple(v.shape) for k, v in hf_model.state_dict().items()}
    assert adapter_shapes == hf_shapes

    missing, unexpected = adapter.load_state_dict(snapshot_tensors(), strict=True)
    assert not missing and not unexpected


# =============================================================================
# tabby.checkpoint: path resolution across the supported layouts
# =============================================================================

def write_step_dir(parent: Path, step: int = 10) -> Path:
    """Write a step directory the way ``recipes/pretrain/train.py`` does.

    That recipe saves ``pytorch_model.bin`` *and* a ``config.json`` holding the
    trainer dataclass, so a config.json on its own must never make a step
    directory look like a Hugging Face snapshot.
    """
    _, cfg, _ = load_patchtstfm_backbone(str(sample_snapshot()), torch.device("cpu"))
    model = PatchTSTFM(cfg)

    step_dir = parent / f"step_{step:07d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": {},
            "step": step,
            "config": asdict(cfg),
            "args": {},
        },
        step_dir / "pytorch_model.bin",
    )
    (step_dir / "config.json").write_text(json.dumps(asdict(cfg)), encoding="utf-8")
    return step_dir


def test_resolve_accepts_every_supported_layout() -> None:
    snapshot = sample_snapshot()
    # A Hugging Face snapshot resolves to the directory itself.
    assert resolve_checkpoint_path(snapshot) == snapshot.resolve()

    # A step directory resolves to the .bin inside it; so does the .bin itself.
    step_dir = write_step_dir(_temp_dir("tabby_run_"))
    weights = (step_dir / "pytorch_model.bin").resolve()
    assert resolve_checkpoint_path(step_dir) == weights
    assert resolve_checkpoint_path(step_dir / "pytorch_model.bin") == weights

    # A run directory is entered through its `latest` pointer.  Real runs make
    # that a symlink, which needs extra privileges on Windows, so the test uses
    # a plain directory -- `resolve_checkpoint_path` follows both the same way.
    run_dir = _temp_dir("tabby_latest_")
    write_step_dir(run_dir).rename(run_dir / "latest")
    assert resolve_checkpoint_path(run_dir) == (
        run_dir / "latest" / "pytorch_model.bin"
    ).resolve()

    # `latest` may also point at a Hugging Face snapshot.
    hf_run = _temp_dir("tabby_latest_hf_")
    shutil.copytree(snapshot, hf_run / "latest")
    assert resolve_checkpoint_path(hf_run) == (hf_run / "latest").resolve()


def test_resolve_rejects_unusable_locations() -> None:
    try:
        resolve_checkpoint_path(_temp_dir("tabby_empty_"))
    except FileNotFoundError as exc:
        # The message names both layouts, since either one would have worked.
        assert "model.safetensors" in str(exc) and "pytorch_model.bin" in str(exc)
    else:
        raise AssertionError("an empty directory is not a checkpoint")

    try:
        resolve_checkpoint_path(_temp_dir("tabby_absent_") / "absent.bin")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("a missing file is not a checkpoint")


# =============================================================================
# tabby.checkpoint: both formats through one entry point
# =============================================================================

def test_in_house_checkpoint_round_trips_through_load_backbone() -> None:
    """The .bin path must keep working now that both formats share a loader."""
    step_dir = write_step_dir(_temp_dir("tabby_bin_"), step=4242)
    model, cfg, step = load_backbone(step_dir, torch.device("cpu"))

    assert step == 4242
    assert not model.training
    assert cfg.context_length == TINY_CONFIG["context_length"]
    assert cfg.num_layers == TINY_CONFIG["n_layer"]

    saved = torch.load(
        step_dir / "pytorch_model.bin", map_location="cpu", weights_only=False
    )["model"]
    loaded = model.state_dict()
    assert set(loaded) == set(saved)
    for name, tensor in saved.items():
        assert torch.equal(loaded[name], tensor), name


def test_training_step_is_read_from_safetensors_metadata() -> None:
    """Published snapshots carry their training step in safetensors metadata."""
    _, _, step = load_backbone(sample_snapshot(), torch.device("cpu"))
    # save_pretrained stores no step, so the sample snapshot reports "unknown".
    assert step == -1

    stamped = _temp_dir("tabby_stamped_") / "stamped"
    shutil.copytree(sample_snapshot(), stamped)
    save_file(
        snapshot_tensors(),
        str(stamped / "model.safetensors"),
        metadata={"format": "pt", "pretrain_step": "175000"},
    )
    _, _, stamped_step = load_backbone(stamped, torch.device("cpu"))
    assert stamped_step == 175000


def test_contradictory_config_values_are_rejected() -> None:
    def rewritten(**changes) -> Path:
        dest = _temp_dir("tabby_badcfg_") / "bad"
        shutil.copytree(sample_snapshot(), dest)
        config = snapshot_config()
        config.update(changes)
        (dest / "config.json").write_text(json.dumps(config), encoding="utf-8")
        return dest

    # n_patch is redundant with context_length / d_patch; a disagreement would
    # silently reshape the patch grid.
    try:
        load_backbone(rewritten(n_patch=999), torch.device("cpu"))
    except ValueError as exc:
        assert "n_patch" in str(exc)
    else:
        raise AssertionError("a contradictory n_patch must not load")

    try:
        load_backbone(rewritten(n_head=7), torch.device("cpu"))
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("d_model must be divisible by n_head")


# =============================================================================
# The GluonTS predictor
# =============================================================================

def test_predictor_loads_an_hf_snapshot_and_forecasts() -> None:
    from tabby.predictor import PatchTSTFMPredictor

    predictor = PatchTSTFMPredictor(
        checkpoint=str(sample_snapshot()),
        prediction_length=8,
        batch_size=2,
        device="cpu",
        precision="fp32",
    )
    assert predictor.cfg.context_length == TINY_CONFIG["context_length"]

    history = np.sin(np.arange(40) / 6.0).astype("float32")
    items = [
        {"start": pd.Period("2024-01-01", freq="h"), "target": history, "item_id": "a"},
        {"start": pd.Period("2024-01-01", freq="h"), "target": history[:20], "item_id": "b"},
    ]
    forecasts = predictor.predict(items)

    # One forecast per input series, including the short one that needs padding.
    assert len(forecasts) == 2
    for forecast in forecasts:
        median = forecast.quantile(0.5)
        assert median.shape == (8,)
        assert np.isfinite(median).all()
        # Quantiles stay ordered after denormalization.
        assert (forecast.quantile(0.9) >= forecast.quantile(0.1) - 1e-6).all()


# =============================================================================
# The published release checkpoint (opt-in)
# =============================================================================

def test_release_checkpoint_loads() -> None:
    """Load the real Tabby-Pretrain weights when a path is configured.

    Set ``TABBY_TEST_CHECKPOINT`` to a downloaded snapshot directory to run it;
    the repository ships no weights, so it is skipped by default.
    """
    checkpoint = os.environ.get("TABBY_TEST_CHECKPOINT")
    if not checkpoint:
        raise SkipTest("set TABBY_TEST_CHECKPOINT to a snapshot directory")

    model, cfg, step = load_backbone(checkpoint, torch.device("cpu"))
    assert step >= 0
    assert not model.training
    assert cfg.num_patches == cfg.context_length // cfg.patch_size
    assert len(model.backbone.blocks) == cfg.num_layers

    x_norm, union_mask, patch_padding = forward_inputs(
        cfg, batch_size=1, history=cfg.context_length - 64
    )
    with torch.no_grad():
        q_hat = model(x_norm, union_mask, patch_padding)

    assert q_hat.shape == (1, cfg.context_length, cfg.num_quantiles)
    assert torch.isfinite(q_hat).all()
    assert (torch.diff(q_hat, dim=-1) >= 0).all()


if __name__ == "__main__":
    checks = [
        value for name, value in sorted(globals().items()) if name.startswith("test_")
    ]
    for check in checks:
        try:
            check()
        except SkipTest as reason:
            print(f"[SKIP] {check.__name__}: {reason}")
        else:
            print(f"[OK] {check.__name__}")
