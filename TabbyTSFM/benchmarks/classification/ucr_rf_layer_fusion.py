#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen-feature UCR evaluation with multi-layer feature fusion.

The script follows a Mantis-style evaluation protocol:

1. keep the pretrained encoder completely frozen;
2. resize each UCR series to a fixed length (512 by default; the paper run uses 1024);
3. extract representations from every requested Transformer block;
4. concatenate (or average) the layer representations into one embedding;
5. fit a Random Forest on the official TRAIN split;
6. report accuracy once on the untouched official TEST split.

The forecasting quantile head is intentionally not used.  Since PatchTST-FM
has no CLS token, patch tokens are aggregated with padding-aware mean pooling
(or mean+max pooling when requested).

Install the Tabby package first (for example, ``pip install -e .`` from the
release root), then run this file from anywhere.

Example
-------
python benchmarks/classification/ucr_rf_layer_fusion.py \
    --checkpoint /path/to/step_0165000 \
    --ucr_root /path/to/UCRArchive_2018 \
    --output_dir ./results/ucr_rf_final_step165k \
    --device cuda:6 \
    --precision bf16 \
    --batch_size 512 \
    --resize_length 1024 \
    --layers all \
    --layer_fusion concat \
    --n_estimators 400 \
    --skip_existing
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import re
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from tabby.checkpoint import load_backbone, resolve_checkpoint_path


LOGGER = logging.getLogger("Tabby-UCR-RF-LayerFusion")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


# ---------------------------------------------------------------------------
# Checkpoint handling
# ---------------------------------------------------------------------------


def load_model(
    checkpoint: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, Any, int, Path]:
    """Load PatchTST-FM using the same convention as forecasting evaluation.

    Accepts every layout :mod:`tabby.checkpoint` supports: a Hugging Face
    snapshot directory (``config.json`` + ``model.safetensors``), an in-house
    ``pytorch_model.bin``, the step directory holding one, a parent directory
    with a ``latest`` symlink, or a Hub repository id.
    """
    ckpt_path = resolve_checkpoint_path(checkpoint)
    LOGGER.info("Loading checkpoint: %s", ckpt_path)

    model, config, step = load_backbone(ckpt_path, device)
    model.requires_grad_(False)

    LOGGER.info(
        "Loaded step=%s | context=%d | patch=%d | layers=%d | d_model=%d",
        step,
        config.context_length,
        config.patch_size,
        config.num_layers,
        config.d_model,
    )
    return model, config, step, ckpt_path


# ---------------------------------------------------------------------------
# UCR loading and preprocessing
# ---------------------------------------------------------------------------


_TRAIN_PATTERN = re.compile(r"^(?P<name>.+)_TRAIN\.(?P<ext>tsv|txt)$", re.IGNORECASE)
_UCR_ADJUSTED_DIR_NAME = "Missing_value_and_variable_length_datasets_adjusted"


def discover_ucr_datasets(ucr_root: str) -> Dict[str, Tuple[Path, Path]]:
    """Find 128 tasks and prefer official adjusted versions when available."""
    root = Path(ucr_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"UCR root is not a directory: {root}")

    found: Dict[str, Tuple[Path, Path]] = {}
    adjusted_root = root / _UCR_ADJUSTED_DIR_NAME

    # First collect the original 128 benchmark tasks.
    for train_path in sorted(root.rglob("*_TRAIN.*")):
        if adjusted_root in train_path.parents:
            continue

        match = _TRAIN_PATTERN.match(train_path.name)
        if match is None:
            continue
        dataset = match.group("name")
        extension = match.group("ext")
        test_path = train_path.with_name(f"{dataset}_TEST.{extension}")
        if not test_path.is_file():
            LOGGER.warning("Ignoring %s because %s is missing.", train_path, test_path.name)
            continue

        # Prefer the modern TSV copy if both TSV and TXT versions exist.
        if dataset not in found or train_path.suffix.lower() == ".tsv":
            found[dataset] = (train_path, test_path)

    # Then override the 15 missing-value/variable-length tasks with the
    # processed versions supplied by the official archive.
    adjusted_names: List[str] = []
    if adjusted_root.is_dir():
        for train_path in sorted(adjusted_root.rglob("*_TRAIN.*")):
            match = _TRAIN_PATTERN.match(train_path.name)
            if match is None:
                continue
            dataset = match.group("name")
            extension = match.group("ext")
            test_path = train_path.with_name(f"{dataset}_TEST.{extension}")
            if not test_path.is_file():
                LOGGER.warning(
                    "Ignoring adjusted %s because %s is missing.",
                    train_path,
                    test_path.name,
                )
                continue
            if dataset not in found:
                raise RuntimeError(
                    f"Adjusted UCR dataset {dataset!r} has no matching original task."
                )
            found[dataset] = (train_path, test_path)
            adjusted_names.append(dataset)

    if not found:
        raise FileNotFoundError(
            f"No UCR *_TRAIN.tsv (or .txt) files were found below {root}."
        )
    LOGGER.info(
        "Discovered %d UCR tasks; using official adjusted versions for %d tasks: %s",
        len(found),
        len(adjusted_names),
        ", ".join(sorted(adjusted_names)),
    )
    return dict(sorted(found.items()))


def parse_float(token: str) -> float:
    """Parse UCR values while treating common missing tokens as NaN."""
    token = token.strip()
    if token == "" or token.lower() in {"nan", "na", "null", "none", "?"}:
        return float("nan")
    return float(token)


def read_ucr_file(path: Path) -> Tuple[List[np.ndarray], np.ndarray]:
    """Read a UCR TRAIN/TEST file without assuming equal row lengths."""
    series: List[np.ndarray] = []
    labels: List[str] = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            fields = next(csv.reader([line], delimiter="\t"))
            if len(fields) == 1:
                # Old .txt archives are whitespace-separated.
                fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"Malformed row {line_number} in {path}: {line!r}")

            labels.append(fields[0].strip())
            try:
                values = np.asarray([parse_float(value) for value in fields[1:]], dtype=np.float32)
            except ValueError as error:
                raise ValueError(
                    f"Cannot parse numeric value at {path}:{line_number}."
                ) from error
            series.append(values)

    if not series:
        raise ValueError(f"No examples found in {path}.")
    return series, np.asarray(labels, dtype=object)


def trim_and_fill_missing(values: np.ndarray) -> np.ndarray:
    """Trim NaN padding and linearly fill internal missing observations."""
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = np.isfinite(x)
    finite_indices = np.flatnonzero(finite)
    if finite_indices.size == 0:
        raise ValueError("Encountered an all-missing time series.")

    # Trailing NaNs in variable-length UCR files denote padding, not samples.
    first, last = int(finite_indices[0]), int(finite_indices[-1])
    x = x[first : last + 1]
    finite = np.isfinite(x)
    if finite.all():
        return x

    grid = np.arange(x.size, dtype=np.float64)
    known = np.flatnonzero(finite)
    if known.size == 1:
        return np.full(x.shape, x[known[0]], dtype=np.float32)
    filled = np.interp(grid, known.astype(np.float64), x[known].astype(np.float64))
    return filled.astype(np.float32)


def linear_resize(values: np.ndarray, target_length: int) -> np.ndarray:
    """Linearly resample a finite 1D sequence to target_length."""
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    if target_length <= 0:
        raise ValueError(f"target_length must be positive; got {target_length}.")
    if x.size == target_length:
        return x.copy()
    if x.size == 1:
        return np.full(target_length, x[0], dtype=np.float32)

    old_grid = np.linspace(0.0, 1.0, num=x.size, dtype=np.float64)
    new_grid = np.linspace(0.0, 1.0, num=target_length, dtype=np.float64)
    return np.interp(new_grid, old_grid, x.astype(np.float64)).astype(np.float32)


def prepare_split(
    raw_series: Sequence[np.ndarray],
    resize_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Clean and resize a UCR split; return [B, resize_length] and raw lengths."""
    prepared: List[np.ndarray] = []
    lengths: List[int] = []
    for values in raw_series:
        clean = trim_and_fill_missing(values)
        lengths.append(int(clean.size))
        prepared.append(linear_resize(clean, resize_length))
    return np.stack(prepared, axis=0).astype(np.float32), np.asarray(lengths, dtype=np.int64)


def encode_labels(
    train_labels: np.ndarray,
    test_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Fit the label mapping on TRAIN and verify TEST has no unknown class."""
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_labels.astype(str))
    unknown = sorted(set(test_labels.astype(str)) - set(encoder.classes_.tolist()))
    if unknown:
        raise ValueError(f"TEST contains classes absent from TRAIN: {unknown}")
    y_test = encoder.transform(test_labels.astype(str))
    return y_train.astype(np.int64), y_test.astype(np.int64), encoder


# ---------------------------------------------------------------------------
# Frozen multi-layer feature extraction
# ---------------------------------------------------------------------------


def resolve_layers(layer_tokens: Sequence[str], num_layers: int) -> List[int]:
    """Parse --layers all, space-separated integers, or comma-separated lists."""
    flattened: List[str] = []
    for token in layer_tokens:
        flattened.extend(part for part in token.split(",") if part)

    if len(flattened) == 1 and flattened[0].lower() == "all":
        return list(range(1, num_layers + 1))
    if any(token.lower() == "all" for token in flattened):
        raise ValueError("Use either '--layers all' or explicit layer numbers, not both.")

    try:
        layers = sorted(set(int(token) for token in flattened))
    except ValueError as error:
        raise ValueError(f"Invalid --layers value: {layer_tokens}") from error
    invalid = [layer for layer in layers if layer < 1 or layer > num_layers]
    if invalid:
        raise ValueError(f"Layers must lie in [1, {num_layers}]; got {invalid}.")
    if not layers:
        raise ValueError("At least one layer must be requested.")
    return layers


def build_visible_batch(
    resized_series: np.ndarray,
    context_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-align fully visible classification series in model context."""
    if resized_series.ndim != 2:
        raise ValueError(f"Expected [B, L] input, got {resized_series.shape}.")
    batch_size, length = resized_series.shape
    if length > context_length:
        raise ValueError(
            f"resize_length={length} exceeds checkpoint context_length={context_length}."
        )

    start = context_length - length
    x = np.zeros((batch_size, context_length), dtype=np.float32)
    observed = np.zeros((batch_size, context_length), dtype=np.bool_)
    padding = np.ones((batch_size, context_length), dtype=np.bool_)
    prediction = np.zeros((batch_size, context_length), dtype=np.bool_)

    x[:, start:] = resized_series
    observed[:, start:] = True
    padding[:, start:] = False
    return (
        torch.from_numpy(x),
        torch.from_numpy(observed),
        torch.from_numpy(padding),
        torch.from_numpy(prediction),
    )


def masked_pool(
    hidden: torch.Tensor,
    patch_padding: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    """Aggregate only non-padding patch tokens."""
    valid = ~patch_padding.bool()  # [B, N]
    count = valid.sum(dim=1, keepdim=True).clamp_min(1).to(hidden.dtype)
    mean = (hidden * valid.unsqueeze(-1).to(hidden.dtype)).sum(dim=1) / count
    if pooling == "mean":
        return mean
    if pooling != "mean_max":
        raise ValueError(f"Unknown pooling mode: {pooling}")

    neg_inf = torch.finfo(hidden.dtype).min
    maximum = hidden.masked_fill(~valid.unsqueeze(-1), neg_inf).max(dim=1).values
    return torch.cat([mean, maximum], dim=-1)


def fuse_layer_features(
    layer_features: Sequence[torch.Tensor],
    fusion: str,
) -> torch.Tensor:
    """Fuse pooled layer representations without using labels.

    ``concat`` preserves the full information from every layer and is the
    default. ``mean`` is a lower-dimensional ablation; it is valid because all
    Transformer blocks have the same hidden width.
    """
    if not layer_features:
        raise ValueError("No layer representations were collected for fusion.")
    if fusion == "concat":
        return torch.cat(list(layer_features), dim=-1)
    if fusion == "mean":
        return torch.stack(list(layer_features), dim=0).mean(dim=0)
    raise ValueError(f"Unknown layer fusion mode: {fusion}")


@torch.inference_mode()
def extract_fused_features(
    *,
    model: torch.nn.Module,
    config: Any,
    resized_series: np.ndarray,
    layers: Sequence[int],
    batch_size: int,
    device: torch.device,
    precision: str,
    pooling: str,
    layer_fusion: str,
    normalize_function: Any,
) -> np.ndarray:
    """Extract one padding-aware embedding formed from all requested layers."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    requested = set(layers)
    maximum_layer = max(requested)
    chunks: List[np.ndarray] = []

    if precision == "bf16":
        amp_dtype = torch.bfloat16
    elif precision == "fp16":
        amp_dtype = torch.float16
    elif precision == "fp32":
        amp_dtype = torch.float32
    else:
        raise ValueError(f"Unknown precision: {precision}")
    use_amp = device.type == "cuda" and precision != "fp32"

    for start in range(0, resized_series.shape[0], batch_size):
        stop = min(start + batch_size, resized_series.shape[0])
        x, observed, padding, prediction = build_visible_batch(
            resized_series[start:stop],
            context_length=config.context_length,
        )
        x = x.to(device=device, dtype=torch.float32, non_blocking=True)
        observed = observed.to(device=device, dtype=torch.bool, non_blocking=True)
        padding = padding.to(device=device, dtype=torch.bool, non_blocking=True)
        prediction = prediction.to(device=device, dtype=torch.bool, non_blocking=True)

        x_norm, _, _, union_mask, patch_padding = normalize_function(
            x=x,
            observed_mask=observed,
            pred_mask=prediction,
            padding_mask=padding,
            cfg=config,
        )

        amp_context = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if use_amp
            else nullcontext()
        )
        with amp_context:
            batch, total_length = x_norm.shape
            patch_length = config.patch_size
            num_patches = config.num_patches
            if total_length != config.context_length:
                raise RuntimeError(
                    f"Normalized input has length {total_length}, expected {config.context_length}."
                )

            backbone = model.backbone
            value_patch = x_norm.reshape(batch, num_patches, patch_length)
            mask_patch = union_mask.reshape(batch, num_patches, patch_length).to(x_norm.dtype)

            # Exactly mirror PatchTSTFM.forward up to the Transformer blocks.
            hidden = backbone.in_layer(
                torch.cat([value_patch, 1.0 - mask_patch], dim=-1)
            )
            hidden = backbone.pos_embed(hidden)
            attention_mask = ~patch_padding.bool()[:, None, None, :]

            pooled_layers: List[torch.Tensor] = []
            for layer_index, block in enumerate(backbone.blocks, start=1):
                hidden = block(hidden, attention_mask)
                if layer_index in requested:
                    pooled_layers.append(masked_pool(hidden, patch_padding, pooling))
                if layer_index >= maximum_layer:
                    break

            if len(pooled_layers) != len(layers):
                raise RuntimeError(
                    f"Collected {len(pooled_layers)} layers but expected {len(layers)}."
                )
            fused = fuse_layer_features(pooled_layers, layer_fusion)
            fused_np = fused.float().cpu().numpy()
            if not np.isfinite(fused_np).all():
                raise FloatingPointError("Non-finite values found in fused features.")
            chunks.append(fused_np)

        if start == 0 or stop == resized_series.shape[0] or stop % (10 * batch_size) == 0:
            LOGGER.info("Feature extraction: %d/%d", stop, resized_series.shape[0])

    features = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    expected_per_layer = config.d_model * (2 if pooling == "mean_max" else 1)
    expected_dim = (
        expected_per_layer * len(layers)
        if layer_fusion == "concat"
        else expected_per_layer
    )
    if features.shape != (resized_series.shape[0], expected_dim):
        raise RuntimeError(
            f"Unexpected fused feature shape {features.shape}; expected "
            f"({resized_series.shape[0]}, {expected_dim})."
        )
    return features


# ---------------------------------------------------------------------------
# Random Forest evaluation
# ---------------------------------------------------------------------------


def make_random_forest(
    *,
    n_estimators: int,
    random_state: int,
    n_jobs: int,
) -> RandomForestClassifier:
    """Mantis-style RF: 200 trees by default and unrestricted depth."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight=None,
    )


def evaluate_one_dataset(
    *,
    dataset: str,
    train_path: Path,
    test_path: Path,
    model: torch.nn.Module,
    config: Any,
    candidate_layers: Sequence[int],
    args: argparse.Namespace,
    device: torch.device,
    normalize_function: Any,
) -> Dict[str, Any]:
    """Run the full leakage-safe protocol for one UCR dataset."""
    started = time.perf_counter()
    raw_train, train_labels = read_ucr_file(train_path)
    raw_test, test_labels = read_ucr_file(test_path)
    x_train, train_lengths = prepare_split(raw_train, args.resize_length)
    x_test, test_lengths = prepare_split(raw_test, args.resize_length)
    y_train, y_test, label_encoder = encode_labels(train_labels, test_labels)

    extraction_layers = list(candidate_layers)

    LOGGER.info(
        "%s | train=%d test=%d classes=%d raw_length=[%d, %d] resize=%d",
        dataset,
        len(y_train),
        len(y_test),
        len(label_encoder.classes_),
        int(min(train_lengths.min(), test_lengths.min())),
        int(max(train_lengths.max(), test_lengths.max())),
        args.resize_length,
    )

    train_started = time.perf_counter()
    train_features = extract_fused_features(
        model=model,
        config=config,
        resized_series=x_train,
        layers=extraction_layers,
        batch_size=args.batch_size,
        device=device,
        precision=args.precision,
        pooling=args.pooling,
        layer_fusion=args.layer_fusion,
        normalize_function=normalize_function,
    )
    train_feature_seconds = time.perf_counter() - train_started

    classifier = make_random_forest(
        n_estimators=args.n_estimators,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )
    fit_started = time.perf_counter()
    classifier.fit(train_features, y_train)
    rf_fit_seconds = time.perf_counter() - fit_started

    test_started = time.perf_counter()
    test_features = extract_fused_features(
        model=model,
        config=config,
        resized_series=x_test,
        layers=extraction_layers,
        batch_size=args.batch_size,
        device=device,
        precision=args.precision,
        pooling=args.pooling,
        layer_fusion=args.layer_fusion,
        normalize_function=normalize_function,
    )
    test_feature_seconds = time.perf_counter() - test_started

    predictions = classifier.predict(test_features)
    test_accuracy = float(accuracy_score(y_test, predictions))

    result = {
        "dataset": dataset,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_classes": int(len(label_encoder.classes_)),
        "raw_length_min": int(min(train_lengths.min(), test_lengths.min())),
        "raw_length_max": int(max(train_lengths.max(), test_lengths.max())),
        "resize_length": int(args.resize_length),
        "pooling": args.pooling,
        "layer_fusion": args.layer_fusion,
        "fused_layers": ",".join(map(str, extraction_layers)),
        "feature_dim": int(train_features.shape[1]),
        "test_accuracy": test_accuracy,
        "train_feature_seconds": train_feature_seconds,
        "test_feature_seconds": test_feature_seconds,
        "rf_fit_seconds": rf_fit_seconds,
        "total_seconds": time.perf_counter() - started,
    }
    LOGGER.info(
        "%s DONE | fused layers=%s | feature_dim=%d | TEST accuracy=%.5f",
        dataset,
        extraction_layers,
        train_features.shape[1],
        test_accuracy,
    )
    return result


# ---------------------------------------------------------------------------
# Result persistence and CLI
# ---------------------------------------------------------------------------


def write_results(
    output_dir: Path,
    results: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
) -> None:
    """Rewrite small CSV files after each dataset for interruption safety."""
    output_dir.mkdir(parents=True, exist_ok=True)

    result_frame = pd.DataFrame(results)
    if result_frame.empty:
        result_frame = pd.DataFrame(columns=["dataset", "test_accuracy"])
    else:
        result_frame = result_frame.sort_values("dataset")
    result_frame.to_csv(output_dir / "per_dataset.csv", index=False)

    failure_frame = pd.DataFrame(failures)
    if failure_frame.empty:
        failure_frame = pd.DataFrame(columns=["dataset", "error_type", "error"])
    failure_frame.to_csv(output_dir / "failures.csv", index=False)


def load_existing_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    try:
        return pd.read_csv(path).to_dict(orient="records")
    except pd.errors.EmptyDataError:
        return []


def build_summary(
    *,
    results: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    step: int,
    checkpoint: Path,
    config: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    accuracies = np.asarray([float(row["test_accuracy"]) for row in results], dtype=np.float64)
    if accuracies.size:
        metrics = {
            "mean_test_accuracy": float(accuracies.mean()),
            "median_test_accuracy": float(np.median(accuracies)),
            "std_test_accuracy": float(accuracies.std(ddof=0)),
            "min_test_accuracy": float(accuracies.min()),
            "max_test_accuracy": float(accuracies.max()),
        }
    else:
        metrics = {
            "mean_test_accuracy": None,
            "median_test_accuracy": None,
            "std_test_accuracy": None,
            "min_test_accuracy": None,
            "max_test_accuracy": None,
        }
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_step": step,
        "model_config": asdict(config),
        "evaluation_config": {
            key: value
            for key, value in vars(args).items()
            if isinstance(value, (str, int, float, bool, type(None), list))
        },
        "num_completed_datasets": len(results),
        "num_failed_datasets": len(failures),
        **metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fused frozen PatchTST-FM layer features on UCR with Random Forest."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--ucr_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Optional exact dataset names; default evaluates every discovered UCR dataset.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Debug: cap dataset count.")

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--resize_length",
        type=int,
        default=512,
        help="Mantis-style interpolation length before left-padding to model context.",
    )
    parser.add_argument("--pooling", choices=["mean", "mean_max"], default="mean")
    parser.add_argument(
        "--layers",
        type=str,
        nargs="+",
        default=["all"],
        help="Blocks to fuse, e.g. '--layers 4 8 12 16 20', or 'all'.",
    )
    parser.add_argument(
        "--layer_fusion",
        choices=["concat", "mean"],
        default="concat",
        help="concat preserves every layer; mean is a compact ablation.",
    )
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Abort on the first dataset error instead of recording failures.csv.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace, config: Any) -> None:
    if args.resize_length <= 0:
        raise ValueError("--resize_length must be positive.")
    if args.resize_length > config.context_length:
        raise ValueError(
            f"--resize_length={args.resize_length} exceeds checkpoint context_length="
            f"{config.context_length}."
        )
    if args.resize_length % config.patch_size != 0:
        raise ValueError(
            f"--resize_length={args.resize_length} must be divisible by patch_size="
            f"{config.patch_size} so the visible interval aligns with complete patches."
        )
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.n_estimators <= 0:
        raise ValueError("--n_estimators must be positive.")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {args.device}")
        device = torch.device(args.device)
    else:
        device = torch.device(args.device)
        if args.precision != "fp32":
            LOGGER.warning("CPU evaluation uses fp32; overriding --precision %s.", args.precision)
            args.precision = "fp32"

    from tabby.utils.input_preprocessing import mask_aware_normalize_for_inference

    normalize_function = mask_aware_normalize_for_inference
    model, config, step, checkpoint_path = load_model(args.checkpoint, device)
    validate_args(args, config)
    candidate_layers = resolve_layers(args.layers, config.num_layers)

    datasets = discover_ucr_datasets(args.ucr_root)
    if args.datasets is not None:
        missing = sorted(set(args.datasets) - set(datasets))
        if missing:
            raise KeyError(
                f"Requested datasets not found: {missing}. "
                f"Discovered {len(datasets)} datasets below {args.ucr_root}."
            )
        datasets = {name: datasets[name] for name in args.datasets}
    if args.limit is not None:
        datasets = dict(list(datasets.items())[: args.limit])

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    if args.skip_existing:
        results = load_existing_rows(output_dir / "per_dataset.csv")
        failures = load_existing_rows(output_dir / "failures.csv")
    completed = {str(row["dataset"]) for row in results}

    run_config = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": step,
        "model_config": asdict(config),
        "candidate_layers": candidate_layers,
        "arguments": vars(args),
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2, ensure_ascii=False)

    LOGGER.info(
        "Starting UCR evaluation | datasets=%d | layers=%s | fusion=%s | pooling=%s",
        len(datasets),
        candidate_layers,
        args.layer_fusion,
        args.pooling,
    )

    for index, (dataset, (train_path, test_path)) in enumerate(datasets.items(), start=1):
        if dataset in completed:
            LOGGER.info("[%d/%d] Skipping completed dataset %s.", index, len(datasets), dataset)
            continue

        LOGGER.info("%s", "=" * 100)
        LOGGER.info("[%d/%d] %s", index, len(datasets), dataset)
        LOGGER.info("%s", "=" * 100)
        try:
            result = evaluate_one_dataset(
                dataset=dataset,
                train_path=train_path,
                test_path=test_path,
                model=model,
                config=config,
                candidate_layers=candidate_layers,
                args=args,
                device=device,
                normalize_function=normalize_function,
            )
            results.append(result)
            # Remove an old failure for a dataset that succeeded on resume.
            failures = [row for row in failures if str(row.get("dataset")) != dataset]
        except Exception as error:
            LOGGER.exception("FAILED on %s: %s", dataset, error)
            failures = [row for row in failures if str(row.get("dataset")) != dataset]
            failures.append(
                {
                    "dataset": dataset,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            if args.fail_fast:
                raise
        finally:
            write_results(output_dir, results, failures)
            summary = build_summary(
                results=results,
                failures=failures,
                step=step,
                checkpoint=checkpoint_path,
                config=config,
                args=args,
            )
            with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, ensure_ascii=False)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary = build_summary(
        results=results,
        failures=failures,
        step=step,
        checkpoint=checkpoint_path,
        config=config,
        args=args,
    )
    LOGGER.info(
        "Finished | completed=%d failed=%d mean UCR accuracy=%s | results=%s",
        summary["num_completed_datasets"],
        summary["num_failed_datasets"],
        "n/a"
        if summary["mean_test_accuracy"] is None
        else f'{summary["mean_test_accuracy"]:.5f}',
        output_dir,
    )


if __name__ == "__main__":
    main()
