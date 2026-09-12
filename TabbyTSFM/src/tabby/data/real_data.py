#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BLAST real-data streaming utilities for PatchTST-FM pretraining.

This module provides the BLAST stream used by the mixed-data trainer. It exposes
``BlastRealIterableDataset`` and preserves the trainer's sample contract::

    {
        "x": FloatTensor[context_length],
        "observed_mask": BoolTensor[context_length],
        "padding_mask": BoolTensor[context_length],
    }

The downloaded BLAST training set consists of raw float32 memmap shards named
``data_0_99.dat``, ``data_1_99.dat``, ..., rather than HuggingFace Arrow
datasets.  Every record contains 4096 values.  Short records are right-padded
with NaNs by the official BLAST preprocessing code.

The implementation opens one shard at a time with ``numpy.memmap`` and shuffles
small contiguous blocks.  This keeps memory use bounded and avoids the severe
I/O penalty of fully random 16-KiB reads across the roughly 300-GB corpus.
Blocks are deterministically partitioned across DDP ranks and DataLoader
workers, so workers do not duplicate records within an epoch.
"""

from __future__ import annotations

import random
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info


DEFAULT_BLAST_RECORD_LENGTH = 4096
DEFAULT_SHUFFLE_BLOCK_ROWS = 2048


@dataclass(frozen=True)
class _BlastShard:
    """Metadata for one raw BLAST memmap shard."""

    path: Path
    num_rows: int


def _natural_path_key(path: Path) -> Tuple:
    """Return a natural-sort key (``data_2`` precedes ``data_10``)."""

    pieces = re.split(r"(\d+)", path.name)
    return tuple(int(piece) if piece.isdigit() else piece.lower() for piece in pieces)


def discover_blast_shards(root: str) -> List[Path]:
    """Discover the raw ``.dat`` files in a downloaded BLAST training set.

    Supported roots include both of the common forms::

        /path/to/BLAST/train
        /path/to/BLAST

    The second form is resolved to its ``train`` child.  Files from ``valid``
    are never mixed into training when a ``train`` directory is present.
    """

    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"BLAST data root does not exist: {root_path}")

    if root_path.is_file():
        if root_path.suffix.lower() != ".dat":
            raise ValueError(f"Expected a BLAST .dat file, got: {root_path}")
        return [root_path]

    # Prefer the explicit training directory when the caller supplies the
    # repository root.  This prevents accidental train/validation leakage.
    search_root = root_path / "train" if (root_path / "train").is_dir() else root_path

    shards = list(search_root.glob("data*.dat"))
    if not shards:
        # A recursive fallback supports an extra HuggingFace cache/repository
        # directory level while still remaining under the selected train root.
        shards = list(search_root.rglob("data*.dat"))

    shards = sorted({path.resolve() for path in shards if path.is_file()}, key=_natural_path_key)
    if not shards:
        raise FileNotFoundError(
            f"No BLAST memmap shards found under {search_root}. Expected files "
            "such as data_0_99.dat."
        )
    return shards


def discover_hf_dataset_dirs(root: str) -> List[Path]:
    """Compatibility wrapper for the old helper name.

    The former loader returned HuggingFace dataset directories.  BLAST uses
    raw memmap files, so this function now returns the discovered shard paths.
    """

    return discover_blast_shards(root)


def _row_count(path: Path, record_length: int) -> int:
    """Infer a shard's number of fixed-width float32 rows from its byte size."""

    record_length = int(record_length)
    if record_length <= 0:
        raise ValueError("record_length must be positive")

    row_nbytes = record_length * np.dtype(np.float32).itemsize
    file_nbytes = path.stat().st_size
    if file_nbytes <= 0:
        raise ValueError(f"BLAST shard is empty: {path}")
    if file_nbytes % row_nbytes != 0:
        raise ValueError(
            f"BLAST shard size is incompatible with float32 records of length "
            f"{record_length}: {path} has {file_nbytes} bytes, but one row uses "
            f"{row_nbytes} bytes. The download may be incomplete or corrupt."
        )
    return file_nbytes // row_nbytes


def open_blast_shard(
    path: Path,
    record_length: int = DEFAULT_BLAST_RECORD_LENGTH,
) -> np.memmap:
    """Open one BLAST shard read-only without loading it into RAM."""

    path = Path(path).expanduser().resolve()
    num_rows = _row_count(path, record_length)
    return np.memmap(
        str(path),
        dtype=np.float32,
        mode="r",
        shape=(num_rows, int(record_length)),
        order="C",
    )


def load_hf_dataset_from_disk(path: Path):
    """Compatibility wrapper returning a BLAST shard as an indexable memmap.

    The name is retained so imports from the previous ``real_data.py`` do not
    fail.  BLAST itself does not require the ``datasets`` package.
    """

    path = Path(path)
    if path.is_dir():
        shards = discover_blast_shards(str(path))
        if len(shards) != 1:
            raise ValueError(
                "load_hf_dataset_from_disk expects one BLAST shard; "
                f"{path} contains {len(shards)} shards"
            )
        path = shards[0]
    return open_blast_shard(path)


def choose_univariate_from_target(target: object, rng: random.Random) -> np.ndarray:
    """Convert a target-like object into one univariate float32 series.

    This helper is preserved from the old loader for import compatibility and
    also makes the BLAST formatter tolerant of one- or two-dimensional arrays.
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
            channel = rng.randrange(arr.shape[0])
            return arr[channel].astype(np.float32, copy=False)
        channel = rng.randrange(arr.shape[1])
        return arr[:, channel].astype(np.float32, copy=False)
    return arr.reshape(-1).astype(np.float32, copy=False)


def extract_target_from_entry(entry: object, rng: random.Random) -> np.ndarray:
    """Extract one univariate series from either a BLAST row or a legacy row."""

    if not isinstance(entry, Mapping):
        return choose_univariate_from_target(entry, rng)

    preferred_keys = ["target", "values", "value", "series", "data", "past_values"]
    target = None
    for key in preferred_keys:
        if key in entry:
            target = entry[key]
            break

    if target is None:
        for value in entry.values():
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


def _trim_blast_right_padding(arr: np.ndarray) -> np.ndarray:
    """Remove BLAST's trailing NaN padding while preserving internal missingness."""

    arr = np.asarray(arr).reshape(-1)
    finite = np.isfinite(arr)
    if not bool(finite.any()):
        raise ValueError("BLAST record contains no finite observations")

    # If the final finite value occurs at index j, argmax on the reversed mask
    # is len(arr)-1-j, hence the exclusive prefix end is len(arr)-that_offset.
    valid_end = arr.shape[0] - int(np.argmax(finite[::-1]))
    return arr[:valid_end]


def format_real_training_sample(
    arr: np.ndarray,
    context_length: int,
    rng: random.Random,
    min_real_length: int = 2,
    min_sample_length: int = 96,
    max_sample_length: int = 4096,
    max_crop_attempts: int = 8,
) -> Dict[str, torch.Tensor]:
    """Randomly crop a series and left-pad it to the model context length.

    Only the selected crop is converted and cleaned.  This preserves the old
    loader's exact output keys, shapes, dtypes, and mask semantics.
    """

    context_length = int(context_length)
    min_real_length = int(min_real_length)
    min_sample_length = int(min_sample_length)
    max_sample_length = int(max_sample_length)

    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if min_real_length <= 0:
        raise ValueError("min_real_length must be positive")
    if min_sample_length <= 0:
        raise ValueError("min_sample_length must be positive")
    if max_sample_length < min_sample_length:
        raise ValueError("max_sample_length must be >= min_sample_length")
    if max_sample_length > context_length:
        raise ValueError("max_sample_length must be <= context_length")

    arr = np.asarray(arr).reshape(-1)
    num_values = int(arr.shape[0])
    if num_values < min_sample_length:
        raise ValueError(
            f"series length {num_values} is shorter than min_sample_length "
            f"{min_sample_length}"
        )

    sample_high = min(max_sample_length, num_values)
    last_error = "failed to sample a valid crop"

    for _ in range(max(1, int(max_crop_attempts))):
        sample_length = rng.randint(min_sample_length, sample_high)
        start = rng.randint(0, num_values - sample_length) if num_values > sample_length else 0

        arr_slice = np.asarray(
            arr[start : start + sample_length],
            dtype=np.float32,
        ).reshape(-1)
        finite_slice = np.isfinite(arr_slice)

        if int(finite_slice.sum()) < min_real_length:
            last_error = "cropped sample has too few finite observations"
            continue

        clean_slice = np.where(finite_slice, arr_slice, 0.0).astype(np.float32, copy=False)
        clean_slice = np.nan_to_num(
            clean_slice,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )
        clean_slice = np.clip(clean_slice, -1e6, 1e6)

        pad_length = context_length - sample_length
        x = np.zeros(context_length, dtype=np.float32)
        observed = np.zeros(context_length, dtype=np.bool_)
        padding = np.ones(context_length, dtype=np.bool_)

        x[pad_length:] = clean_slice
        observed[pad_length:] = finite_slice
        padding[pad_length:] = False

        return {
            "x": torch.from_numpy(x),
            "observed_mask": torch.from_numpy(observed),
            "padding_mask": torch.from_numpy(padding),
        }

    raise ValueError(last_error)


class BlastRealIterableDataset(IterableDataset):
    """Infinite, worker-sharded stream over the BLAST training memmaps.

    The class name and the first ten constructor arguments intentionally match
    the previous GIFT-Eval loader.  ``domain_balance`` is retained as a no-op
    compatibility argument: the released BLAST corpus is already the result of
    pattern-balanced sampling, so balancing its storage shards as if they were
    semantic domains would be incorrect.

    Args:
        root: BLAST ``train`` directory, or its parent containing ``train``.
        context_length: Fixed length of each returned tensor.
        seed: Base seed shared across DDP ranks and DataLoader workers.
        min_real_length: Minimum number of finite points in a sampled crop.
        min_sample_length: Minimum random-crop length.
        max_sample_length: Maximum random-crop length.
        shuffle_datasets: Shuffle the BLAST shard order each epoch.
        shuffle_examples: Shuffle blocks and rows within each local block.
        domain_balance: Compatibility-only; BLAST is already balanced.
        max_sample_attempts: Maximum crop attempts for a selected record.
        blast_record_length: Fixed raw record width; official BLAST uses 4096.
        shuffle_block_rows: Rows per locality-preserving shuffle block.
    """

    def __init__(
        self,
        root: str,
        context_length: int = 8192,
        seed: int = 42,
        min_real_length: int = 2,
        min_sample_length: int = 96,
        max_sample_length: int = 4096,
        shuffle_datasets: bool = True,
        shuffle_examples: bool = True,
        domain_balance: bool = True,
        max_sample_attempts: int = 32,
        blast_record_length: int = DEFAULT_BLAST_RECORD_LENGTH,
        shuffle_block_rows: int = DEFAULT_SHUFFLE_BLOCK_ROWS,
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
        self.blast_record_length = int(blast_record_length)
        self.shuffle_block_rows = int(shuffle_block_rows)

        self._validate_configuration()

        # Keep the old attribute name because downstream diagnostics sometimes
        # print ``dataset.dataset_dirs``.
        self.dataset_dirs = discover_blast_shards(self.root)
        self.shards = [
            _BlastShard(path=path, num_rows=_row_count(path, self.blast_record_length))
            for path in self.dataset_dirs
        ]
        self.total_rows = sum(shard.num_rows for shard in self.shards)
        if self.total_rows <= 0:
            raise ValueError(f"BLAST root contains no records: {self.root}")

    def _validate_configuration(self) -> None:
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
        if self.blast_record_length <= 0:
            raise ValueError("blast_record_length must be positive")
        if self.shuffle_block_rows <= 0:
            raise ValueError("shuffle_block_rows must be positive")

    def _global_worker_info(self) -> Tuple[int, int]:
        """Combine the DDP rank and DataLoader worker into one worker index."""

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        worker = get_worker_info()
        if worker is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker.id, worker.num_workers

        global_worker_id = rank * num_workers + worker_id
        global_num_workers = world_size * num_workers
        return global_worker_id, global_num_workers

    def _format_entry(self, entry: object, rng: random.Random) -> Dict[str, torch.Tensor]:
        arr = extract_target_from_entry(entry, rng)
        arr = _trim_blast_right_padding(arr)
        return format_real_training_sample(
            arr=arr,
            context_length=self.context_length,
            rng=rng,
            min_real_length=self.min_real_length,
            min_sample_length=self.min_sample_length,
            max_sample_length=self.max_sample_length,
            max_crop_attempts=self.max_sample_attempts,
        )

    def _epoch_shard_order(self, epoch: int) -> List[int]:
        order = list(range(len(self.shards)))
        if self.shuffle_datasets:
            rng = random.Random(self.seed + 104_729 * int(epoch))
            rng.shuffle(order)
        return order

    def _epoch_block_order(self, epoch: int, shard_index: int, num_blocks: int) -> List[int]:
        order = list(range(num_blocks))
        if self.shuffle_examples:
            # This seed must not depend on worker id.  Every worker must see the
            # same logical block order before modulo-based worker partitioning.
            rng = random.Random(
                self.seed
                + 1_000_003 * int(epoch)
                + 9_176 * int(shard_index)
            )
            rng.shuffle(order)
        return order

    def _block_row_order(
        self,
        epoch: int,
        shard_index: int,
        block_index: int,
        start: int,
        stop: int,
    ) -> List[int]:
        rows = list(range(start, stop))
        if self.shuffle_examples:
            rng = random.Random(
                self.seed
                + 15_485_863 * int(epoch)
                + 32_452_843 * int(shard_index)
                + 49_979_687 * int(block_index)
            )
            rng.shuffle(rows)
        return rows

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        global_worker_id, global_num_workers = self._global_worker_info()
        crop_rng = random.Random(self.seed + 2_000_003 * global_worker_id)
        epoch = 0

        while True:
            # Ensure there are normally at least as many blocks as workers,
            # while respecting the configured upper bound for I/O locality.
            rows_per_worker = max(1, self.total_rows // global_num_workers)
            block_rows = min(self.shuffle_block_rows, rows_per_worker)

            logical_block_index = 0
            yielded_this_epoch = 0

            for shard_index in self._epoch_shard_order(epoch):
                shard = self.shards[shard_index]
                num_blocks = (shard.num_rows + block_rows - 1) // block_rows
                assigned_blocks: List[int] = []

                for block_index in self._epoch_block_order(epoch, shard_index, num_blocks):
                    if logical_block_index % global_num_workers == global_worker_id:
                        assigned_blocks.append(block_index)
                    logical_block_index += 1

                if not assigned_blocks:
                    continue

                data = open_blast_shard(shard.path, self.blast_record_length)
                try:
                    for block_index in assigned_blocks:
                        start = block_index * block_rows
                        stop = min(start + block_rows, shard.num_rows)
                        for row_index in self._block_row_order(
                            epoch,
                            shard_index,
                            block_index,
                            start,
                            stop,
                        ):
                            try:
                                sample = self._format_entry(data[row_index], crop_rng)
                            except (KeyError, TypeError, ValueError, OverflowError):
                                # A malformed/all-missing record must not stop a
                                # multi-day pretraining job; skip it and continue.
                                continue
                            yielded_this_epoch += 1
                            yield sample
                finally:
                    # Do not retain hundreds of open 3-GB mappings per worker.
                    mmap_object = getattr(data, "_mmap", None)
                    if mmap_object is not None:
                        mmap_object.close()
                    del data

            if yielded_this_epoch == 0:
                raise RuntimeError(
                    "This DDP/DataLoader worker received no valid BLAST samples. "
                    "Check that the .dat files are complete, reduce the number of "
                    "workers, or lower min_sample_length."
                )

            epoch += 1
            crop_rng.seed(self.seed + 2_000_003 * global_worker_id + 97 * epoch)


# Backward-compatible alias for older internal scripts that used this class
# name while reading BLAST memmaps. New code should use
# BlastRealIterableDataset.
GiftEvalPretrainRealIterableDataset = BlastRealIterableDataset


def concat_mixed_batches(parts: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Concatenate real/synthetic batches and shuffle inside the micro-batch."""
    if len(parts) == 0:
        raise ValueError("parts must contain at least one batch")

    if len(parts) == 1:
        batch = parts[0]
    else:
        batch = {
            "x": torch.cat([part["x"] for part in parts], dim=0),
            "observed_mask": torch.cat([part["observed_mask"] for part in parts], dim=0),
            "padding_mask": torch.cat([part["padding_mask"] for part in parts], dim=0),
            "source_id": torch.cat([part["source_id"] for part in parts], dim=0),
        }

    permutation = torch.randperm(batch["x"].shape[0])
    return {
        "x": batch["x"][permutation],
        "observed_mask": batch["observed_mask"][permutation],
        "padding_mask": batch["padding_mask"][permutation],
        "source_id": batch["source_id"][permutation],
    }

__all__ = [
    "BlastRealIterableDataset",
    "GiftEvalPretrainRealIterableDataset",
    "choose_univariate_from_target",
    "concat_mixed_batches",
    "discover_blast_shards",
    "discover_hf_dataset_dirs",
    "extract_target_from_entry",
    "format_real_training_sample",
    "load_hf_dataset_from_disk",
    "open_blast_shard",
]
