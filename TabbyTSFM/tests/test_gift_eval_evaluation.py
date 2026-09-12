"""Checks that the GIFT-Eval evaluation script runs off a safetensors snapshot.

``benchmarks/forecasting/gift_eval/evaluate.py`` is a script rather than an
importable package, so it is loaded here by path.  Its ``build_model`` is the
whole checkpoint-facing surface: it resolves ``--pretrain_ckpt``, wraps the
backbone in ``PromptedPatchTSTFM`` and is what every evaluated configuration
then calls.  Those tests need no benchmark data and always run.

The end-to-end test drives the real CLI over one real GIFT-Eval configuration.
It needs the external dataset, so it is opt-in:

    GIFT_EVAL=/path/to/gift-eval/data \\
    TABBY_TEST_GIFT_EVAL_REPO=/path/to/gift-eval \\
    python -m pytest tests/test_gift_eval_evaluation.py

It deliberately runs on the generated tiny snapshot rather than release weights:
the point is that the script completes and writes finite metrics, not that the
metrics are good.
"""

from __future__ import annotations

import atexit
import csv
import importlib.util
import math
import os
import shutil
import subprocess
import sys
import tempfile
import types
from functools import lru_cache
from pathlib import Path
from unittest import SkipTest

ROOT = Path(__file__).resolve().parents[1]
EVALUATE_SCRIPT = ROOT / "benchmarks" / "forecasting" / "gift_eval" / "evaluate.py"
sys.path.insert(0, str(Path(__file__).resolve().parent))

_MISSING = ""
try:
    import torch
    from safetensors.torch import load_file, save_file

    from make_tiny_hf_snapshot import GIFT_EVAL_CONFIG, write_tiny_snapshot
except ImportError as exc:
    _MISSING = str(exc)

if _MISSING:  # pragma: no cover - environment guard
    import pytest

    pytest.skip(
        "install the forecast extra to run the GIFT-Eval tests: " + _MISSING,
        allow_module_level=True,
    )


# The zero-shot path fixes prediction_length at 96 and the wrapper requires
# context_length + prediction_length to fit inside the backbone window.
CONTEXT_LENGTH = 128


def _temp_dir(prefix: str) -> Path:
    path = tempfile.mkdtemp(prefix=prefix)
    atexit.register(shutil.rmtree, path, ignore_errors=True)
    return Path(path)


@lru_cache(maxsize=1)
def sample_snapshot() -> Path:
    """A snapshot wide enough for the GIFT-Eval prompt wrapper."""
    return write_tiny_snapshot(
        _temp_dir("tabby_gift_snapshot_") / "tiny_patchtstfm_hf",
        context_length=GIFT_EVAL_CONFIG["context_length"],
    )


@lru_cache(maxsize=1)
def evaluate_module():
    """Load evaluate.py by path, the way a script with no package is loaded."""
    spec = importlib.util.spec_from_file_location(
        "tabby_gift_eval_evaluate", EVALUATE_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def zero_shot_args(checkpoint: Path) -> types.SimpleNamespace:
    """The subset of the CLI namespace that ``build_model`` reads in zs mode."""
    return types.SimpleNamespace(
        mode="zs",
        ckpt=None,
        pretrain_ckpt=str(checkpoint),
        context_length=CONTEXT_LENGTH,
        norm_mode="inhouse_asinh",
        min_forecast_span=0,
    )


# =============================================================================
# build_model: the script's checkpoint-facing surface
# =============================================================================

def test_evaluate_script_exposes_the_expected_entry_points() -> None:
    module = evaluate_module()
    for name in ("build_model", "main", "PatchTSTFMGiftPredictor"):
        assert hasattr(module, name), name
    # The GIFT-Eval quantile grid the predictor scores against.
    assert module.GIFT_EVAL_Q == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def test_build_model_loads_a_safetensors_snapshot() -> None:
    module = evaluate_module()
    model, cfg, model_name = module.build_model(
        zero_shot_args(sample_snapshot()), torch.device("cpu")
    )

    assert model_name == "patchtstfm_zs"
    assert cfg.context_length == CONTEXT_LENGTH
    assert not model.training

    # Every tensor in model.safetensors reached the wrapped backbone: a partial
    # load would otherwise evaluate a half-random model and report plausible
    # numbers for it.
    expected = load_file(str(sample_snapshot() / "model.safetensors"))
    loaded = model.backbone.state_dict()
    assert set(expected) <= set(loaded), sorted(set(expected) - set(loaded))[:5]
    for name, tensor in expected.items():
        assert torch.equal(loaded[name], tensor), name

    # Zero-shot means the backbone contributes no gradients.
    assert not any(p.requires_grad for p in model.backbone.parameters())


def test_build_model_smoke_forward_matches_the_scripts_own_assertion() -> None:
    """Reproduce the shape check main() runs before touching any dataset."""
    module = evaluate_module()
    model, _, _ = module.build_model(
        zero_shot_args(sample_snapshot()), torch.device("cpu")
    )

    with torch.no_grad():
        out = model(context=torch.randn(2, CONTEXT_LENGTH), prediction_length=8)

    quantile_preds = out["quantile_preds"]
    assert quantile_preds.shape == (2, model.num_quantiles, 8)
    assert torch.isfinite(quantile_preds).all()
    # Quantiles are monotone, so the scored 0.1/0.9 pair cannot cross.
    assert (torch.diff(quantile_preds, dim=1) >= 0).all()


def test_build_model_rejects_an_incomplete_snapshot() -> None:
    """A truncated checkpoint must stop the run, not evaluate random weights."""
    broken = _temp_dir("tabby_gift_broken_") / "broken"
    shutil.copytree(sample_snapshot(), broken)
    state = load_file(str(broken / "model.safetensors"))
    state.pop("backbone.out_layer.layer2.weight")
    save_file(state, str(broken / "model.safetensors"))

    try:
        evaluate_module().build_model(zero_shot_args(broken), torch.device("cpu"))
    except RuntimeError as exc:
        assert "out_layer.layer2.weight" in str(exc)
    else:
        raise AssertionError("an incomplete snapshot must not be evaluated")


def test_prompt_mode_requires_a_prompt_checkpoint() -> None:
    args = zero_shot_args(sample_snapshot())
    args.mode = "prompt"
    args.ckpt = None

    try:
        evaluate_module().build_model(args, torch.device("cpu"))
    except SystemExit as exc:
        assert "--ckpt" in str(exc)
    else:
        raise AssertionError("--mode prompt without --ckpt must abort")


# =============================================================================
# The real CLI over a real GIFT-Eval configuration (opt-in)
# =============================================================================

def test_evaluation_script_runs_end_to_end() -> None:
    data_root = os.environ.get("GIFT_EVAL")
    repo = os.environ.get("TABBY_TEST_GIFT_EVAL_REPO")
    if not data_root or not repo:
        raise SkipTest(
            "set GIFT_EVAL and TABBY_TEST_GIFT_EVAL_REPO to run the CLI test"
        )

    dataset = os.environ.get("TABBY_TEST_GIFT_EVAL_DATASET", "us_births/D")
    work_dir = _temp_dir("tabby_gift_cli_")
    properties = (
        EVALUATE_SCRIPT.parent / "dataset_properties.json"
    )

    completed = subprocess.run(
        [
            sys.executable, str(EVALUATE_SCRIPT),
            "--mode", "zs",
            "--pretrain_ckpt", str(sample_snapshot()),
            "--gift_eval_repo", repo,
            "--dataset_properties", str(properties),
            "--datasets", dataset,
            "--device", "cpu",
            "--precision", "fp32",
            "--batch_size", "16",
            "--context_length", str(CONTEXT_LENGTH),
            "--out_csv", "cli_test",
        ],
        cwd=work_dir,
        env={**os.environ, "GIFT_EVAL": data_root, "PYTHONIOENCODING": "utf-8"},
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout[-3000:] + completed.stderr[-3000:]
    assert "[smoke] dummy predict OK" in completed.stdout

    results = work_dir / "results" / "cli_test.csv"
    assert results.is_file(), completed.stdout[-2000:]
    with results.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    # A configuration that errors is logged and skipped, so an empty CSV means
    # the run failed even though the process exited cleanly.
    assert rows, completed.stdout[-3000:]
    for row in rows:
        assert row["model"] == "patchtstfm_zs"
        assert int(row["context_length"]) == CONTEXT_LENGTH
        for metric in ("MASE[0.5]", "mean_weighted_sum_quantile_loss"):
            assert math.isfinite(float(row[metric])), (metric, row)


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
