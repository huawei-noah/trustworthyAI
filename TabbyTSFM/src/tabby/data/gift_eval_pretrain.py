#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Legacy optional GIFT-Eval-Pretrain streaming utilities.

This module is retained only to keep older research commands readable. It is
not exported from :mod:`tabby.data`, and the Tabby-Pretrain 165K release recipe
sets ``--gift_ratio 0`` because that checkpoint was trained without GIFT data.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info


# 3. GIFT-Eval-Pretrain real-data loader
# =============================================================================

def discover_hf_dataset_dirs(root: str) -> List[Path]:
    """Find HuggingFace load_from_disk dataset folders under ``root``.

    Expected example:
        /path/to/gift-eval-pretrain/largest_2019/
            data-00000-of-00008.arrow
            state.json
            dataset_info.json
    """
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Real data root does not exist: {root_path}")

    candidates: List[Path] = []

    if (root_path / "state.json").exists() and len(list(root_path.glob("*.arrow"))) > 0:
        candidates.append(root_path)

    for state_file in root_path.rglob("state.json"):
        ds_dir = state_file.parent
        if len(list(ds_dir.glob("*.arrow"))) > 0:
            candidates.append(ds_dir)

    seen = set()
    unique: List[Path] = []
    for p in candidates:
        key = str(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)

    if len(unique) == 0:
        raise FileNotFoundError(
            f"No HuggingFace dataset folders found under {root_path}. "
            "Expected subfolders containing state.json and data-*.arrow."
        )
    return unique


def load_hf_dataset_from_disk(path: Path):
    """Load one HuggingFace dataset directory."""
    try:
        from datasets import load_from_disk
    except Exception as exc:
        raise ImportError(
            "The real-data loader requires HuggingFace datasets. Install it with:\n"
            "  pip install datasets"
        ) from exc

    ds = load_from_disk(str(path))

    # load_from_disk may return Dataset or DatasetDict.
    if hasattr(ds, "keys") and not hasattr(ds, "column_names"):
        if "train" in ds:
            ds = ds["train"]
        else:
            ds = ds[list(ds.keys())[0]]
    return ds


def choose_univariate_from_target(target: object, rng: random.Random) -> np.ndarray:
    """Convert a target field into one univariate float32 series.

    Handles common cases:
      - [T]
      - [C, T]
      - [T, C]
    """
    arr = np.asarray(target)
    if arr.dtype == object:
        arr = np.asarray(target, dtype=np.float32)
    else:
        arr = arr.astype(np.float32, copy=False)

    if arr.ndim == 0:
        return arr.reshape(1).astype(np.float32, copy=False)
    if arr.ndim == 1:
        return arr.astype(np.float32, copy=False)
    if arr.ndim == 2:
        if arr.shape[0] <= arr.shape[1]:
            c = rng.randrange(arr.shape[0])
            return arr[c].astype(np.float32, copy=False)
        c = rng.randrange(arr.shape[1])
        return arr[:, c].astype(np.float32, copy=False)
    return arr.reshape(-1).astype(np.float32, copy=False)


def extract_target_from_entry(entry: Dict, rng: random.Random) -> np.ndarray:
    """Extract one univariate target from a GIFT-Eval-Pretrain row."""
    preferred_keys = ["target", "values", "value", "series", "data", "past_values"]
    target = None
    for key in preferred_keys:
        if key in entry:
            target = entry[key]
            break

    if target is None:
        for key, value in entry.items():
            try:
                arr = np.asarray(value)
            except Exception:
                continue
            if arr.ndim >= 1 and arr.size >= 2 and np.issubdtype(arr.dtype, np.number):
                target = value
                break

    if target is None:
        raise KeyError(f"Cannot find target-like field. Available keys: {list(entry.keys())}")

    return choose_univariate_from_target(target, rng)

def format_real_training_sample(
    arr: np.ndarray,
    context_length: int,
    rng: random.Random,
    min_real_length: int = 2,
    min_sample_length: int = 96,
    max_sample_length: int = 8192,
    max_crop_attempts: int = 8,
) -> Dict[str, torch.Tensor]:
    """
    Fast random-crop formatter.

    Key optimization:
        Old version:
            clean the full series, then crop.

        New version:
            crop a small window first, then clean only that window.

    Output semantics are unchanged:
        x             : FloatTensor[T]
        observed_mask : BoolTensor[T]
        padding_mask  : BoolTensor[T]
    """
    # ------------------------------------------------------------
    # 1. Basic hyperparameter checks.
    # ------------------------------------------------------------
    T = int(context_length)
    min_sample_length = int(min_sample_length)
    max_sample_length = int(max_sample_length)

    if min_sample_length <= 0:
        raise ValueError("min_sample_length must be positive")

    if max_sample_length < min_sample_length:
        raise ValueError("max_sample_length must be >= min_sample_length")

    if max_sample_length > T:
        raise ValueError("max_sample_length must be <= context_length")

    # ------------------------------------------------------------
    # 2. Convert arr to a one-dimensional view.
    #
    # In the current pipeline, arr is already usually a 1D np.ndarray
    # returned by extract_target_from_entry(...).
    # ------------------------------------------------------------
    arr = np.asarray(arr).reshape(-1)

    n = int(arr.shape[0])
    if n == 0:
        raise ValueError("empty series")

    if n < min_sample_length:
        raise ValueError("series shorter than min_sample_length")

    # ------------------------------------------------------------
    # 3. Rejection sampling over crop windows.
    #
    # We avoid scanning the full sequence for finite values.
    # Instead, we sample a crop and check only that crop.
    # ------------------------------------------------------------
    sample_high = min(max_sample_length, n)

    last_error = None

    for _ in range(max(1, int(max_crop_attempts))):
        sample_length = rng.randint(min_sample_length, sample_high)
        start = rng.randint(0, n - sample_length) if n > sample_length else 0

        # Important:
        #     Only this small slice is converted to float32 and cleaned.
        arr_slice = np.asarray(
            arr[start:start + sample_length],
            dtype=np.float32,
        ).reshape(-1)

        finite_slice = np.isfinite(arr_slice)

        if int(finite_slice.sum()) < min_real_length:
            last_error = "cropped sample has too few finite observations"
            continue

        # --------------------------------------------------------
        # 4. Clean only the cropped window.
        # --------------------------------------------------------
        arr_slice = np.where(
            finite_slice,
            arr_slice,
            0.0,
        ).astype(np.float32, copy=False)

        arr_slice = np.nan_to_num(
            arr_slice,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )

        arr_slice = np.clip(
            arr_slice,
            -1e6,
            1e6,
        )

        # --------------------------------------------------------
        # 5. Left-pad to context_length.
        # --------------------------------------------------------
        pad_len = T - sample_length

        x = np.zeros(T, dtype=np.float32)
        obs = np.zeros(T, dtype=np.bool_)
        pad = np.ones(T, dtype=np.bool_)

        x[pad_len:] = arr_slice
        obs[pad_len:] = finite_slice
        pad[pad_len:] = False

        return {
            "x": torch.from_numpy(x),
            "observed_mask": torch.from_numpy(obs),
            "padding_mask": torch.from_numpy(pad),
        }

    raise ValueError(last_error or "failed to sample a valid crop")

class GiftEvalPretrainRealIterableDataset(IterableDataset):
    """Infinite domain-balanced stream over downloaded GIFT-Eval-Pretrain folders.

    Each discovered HuggingFace dataset directory is treated as one domain.  The
    iterator samples domains uniformly, then samples an example inside the chosen
    domain, so very large datasets do not dominate the real-data stream.
    """
    def __init__(
        self,
        root: str,
        context_length: int = 8192,
        seed: int = 42,
        min_real_length: int = 2,
        min_sample_length: int = 96,
        max_sample_length: int = 8192,
        shuffle_datasets: bool = True,
        shuffle_examples: bool = True,
        domain_balance: bool = True,
        max_sample_attempts: int = 32,
    ) -> None:
        super().__init__()
        self.root = str(root)
        self.context_length = int(context_length)
        self.seed = int(seed)
        self.min_real_length = int(min_real_length)
        self.min_sample_length = int(min_sample_length)
        self.max_sample_length = int(max_sample_length)
        self.shuffle_datasets = bool(shuffle_datasets)
        self.shuffle_examples = bool(shuffle_examples)
        self.domain_balance = bool(domain_balance)
        self.max_sample_attempts = int(max_sample_attempts)

        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.min_real_length <= 0:
            raise ValueError("min_real_length must be positive")
        if self.min_sample_length <= 0:
            raise ValueError("min_sample_length must be positive")
        if self.max_sample_length < self.min_sample_length:
            raise ValueError("max_sample_length must be >= min_sample_length")
        if self.max_sample_length > self.context_length:
            raise ValueError("max_sample_length must be <= context_length")
        if self.max_sample_attempts <= 0:
            raise ValueError("max_sample_attempts must be positive")

        self.dataset_dirs = discover_hf_dataset_dirs(self.root)

    def _global_worker_info(self) -> Tuple[int, int]:
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
        return global_worker_id, global_num_workers

    def _format_entry(self, entry: Dict, rng: random.Random) -> Dict[str, torch.Tensor]:
        arr = extract_target_from_entry(entry, rng)
        return format_real_training_sample(
            arr=arr,
            context_length=self.context_length,
            rng=rng,
            min_real_length=self.min_real_length,
            min_sample_length=self.min_sample_length,
            max_sample_length=self.max_sample_length,
            max_crop_attempts=self.max_sample_attempts,
        )

    def _sample_indexable_dataset(
        self,
        ds,
        rng: random.Random,
    ) -> Optional[Dict[str, torch.Tensor]]:
        n = len(ds)
        if n <= 0:
            return None

        for _ in range(max(1, self.max_sample_attempts)):
            try:
                entry = ds[int(rng.randrange(n))]
                return self._format_entry(entry, rng)
            except Exception:
                continue
        return None

    def _next_iterable_dataset_sample(
        self,
        ds,
        iterator_state: Dict[Path, Iterator],
        ds_dir: Path,
        rng: random.Random,
    ) -> Optional[Dict[str, torch.Tensor]]:
        for _ in range(max(1, self.max_sample_attempts)):
            iterator = iterator_state.get(ds_dir)
            if iterator is None:
                iterator = iter(ds)
                iterator_state[ds_dir] = iterator

            try:
                entry = next(iterator)
            except StopIteration:
                iterator = iter(ds)
                iterator_state[ds_dir] = iterator
                try:
                    entry = next(iterator)
                except StopIteration:
                    return None

            try:
                return self._format_entry(entry, rng)
            except Exception:
                continue
        return None

    def _domain_balanced_iter(
        self,
        dirs: List[Path],
        rng: random.Random,
    ) -> Iterator[Dict[str, torch.Tensor]]:
        datasets: Dict[Path, object] = {}
        lengths: Dict[Path, Optional[int]] = {}
        iterators: Dict[Path, Iterator] = {}
        active_dirs = list(dirs)

        while len(active_dirs) > 0:
            domain_order = list(active_dirs)
            if self.shuffle_datasets:
                rng.shuffle(domain_order)

            for ds_dir in domain_order:
                if ds_dir not in datasets:
                    try:
                        ds = load_hf_dataset_from_disk(ds_dir)
                    except Exception:
                        if ds_dir in active_dirs:
                            active_dirs.remove(ds_dir)
                        continue

                    datasets[ds_dir] = ds
                    try:
                        lengths[ds_dir] = len(ds)
                    except Exception:
                        lengths[ds_dir] = None

                ds = datasets[ds_dir]
                if lengths[ds_dir] is not None:
                    sample = self._sample_indexable_dataset(ds, rng)
                else:
                    sample = self._next_iterable_dataset_sample(ds, iterators, ds_dir, rng)

                if sample is not None:
                    yield sample

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        global_worker_id, global_num_workers = self._global_worker_info()
        rng = random.Random(self.seed + 2_000_003 * global_worker_id)

        dirs = list(self.dataset_dirs)
        if self.domain_balance:
            yield from self._domain_balanced_iter(dirs, rng)
            return

        epoch = 0

        while True:
            if self.shuffle_datasets:
                rng.shuffle(dirs)

            for ds_idx, ds_dir in enumerate(dirs):
                # Directory-level sharding if enough dataset directories exist.
                if len(dirs) >= global_num_workers and ds_idx % global_num_workers != global_worker_id:
                    continue

                try:
                    ds = load_hf_dataset_from_disk(ds_dir)
                except Exception:
                    continue

                try:
                    n = len(ds)
                except Exception:
                    n = None

                if n is not None:
                    # Example-level sharding if there are fewer dirs than workers.
                    if len(dirs) < global_num_workers:
                        indices = list(range(global_worker_id, n, global_num_workers))
                    else:
                        indices = list(range(n))
                    if self.shuffle_examples:
                        rng.shuffle(indices)

                    for i in indices:
                        try:
                            entry = ds[int(i)]
                            arr = extract_target_from_entry(entry, rng)
                            yield format_real_training_sample(
                                arr=arr,
                                context_length=self.context_length,
                                rng=rng,
                                min_real_length=self.min_real_length,
                                min_sample_length=self.min_sample_length,
                                max_sample_length=self.max_sample_length,
                                max_crop_attempts=self.max_sample_attempts,
                            )
                        except Exception:
                            continue
                else:
                    for i, entry in enumerate(ds):
                        if len(dirs) < global_num_workers and i % global_num_workers != global_worker_id:
                            continue
                        try:
                            arr = extract_target_from_entry(entry, rng)
                            yield format_real_training_sample(
                                arr=arr,
                                context_length=self.context_length,
                                rng=rng,
                                min_real_length=self.min_real_length,
                                min_sample_length=self.min_sample_length,
                                max_sample_length=self.max_sample_length,
                                max_crop_attempts=self.max_sample_attempts,
                            )
                        except Exception:
                            continue

            epoch += 1
            rng.seed(self.seed + 2_000_003 * global_worker_id + 97 * epoch)


def concat_mixed_batches(parts: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Concatenate real and synthetic batches and shuffle inside the micro-batch."""
    if len(parts) == 1:
        batch = parts[0]
    else:
        batch = {
            "x": torch.cat([p["x"] for p in parts], dim=0),
            "observed_mask": torch.cat([p["observed_mask"] for p in parts], dim=0),
            "padding_mask": torch.cat([p["padding_mask"] for p in parts], dim=0),
        }

    perm = torch.randperm(batch["x"].shape[0])
    return {
        "x": batch["x"][perm],
        "observed_mask": batch["observed_mask"][perm],
        "padding_mask": batch["padding_mask"][perm],
    }

# =============================================================================
