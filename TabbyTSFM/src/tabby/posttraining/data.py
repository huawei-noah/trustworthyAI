"""
Data pipeline for prompt tuning on the GIFT-Eval training split.

Two properties are worth calling out because they are easy to get wrong:

  1. VAL SAMPLING. A single global min_past (the minimum over all tasks' val
     slice lengths) lets one ultra-short dataset drag min_past toward 0 for every
     dataset, so validation windows land inside the training region. Validation
     then measures fit, improves monotonically, and prefers late memorising
     checkpoints. ValBoundarySampler instead derives each series' own train/val
     boundary from its val slice length and only samples prediction windows past
     it.

  2. drop_last=(split=="train"): validation keeps its final partial batch.

Train/Val split strategy (time-based, zero overlap for long series):

  Original series:  [=================== 100% ===================]
  
  Step 1 - Official test cutoff:
    Available:      [===== training-only prefix =====]
    Test (hidden):                                  [test window]

  The available prefix ends at the earlier of 90% of the series or the start of
  the benchmark's absolute test window. This matters for short series, where the
  official test window may be longer than the final 10%.
  
  Step 2 - Train/Val time split (inside the available prefix):
    Train target:   [======= first 89% of prefix =======]
    Val:            [.... context ....][= final 11% =]
                     borrowed from      val predictions
                     train (read-only)  (zero overlap with train targets)
  
  For long series (T > context_length): val uses min_past=context_length
    → validation predictions are after the train target boundary, zero overlap.
  For short series (T < context_length): val uses adaptive min_past
    → partial overlap unavoidable, but maximally separated.
"""
import os
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import random

from dotenv import find_dotenv, load_dotenv

from gluonts.dataset import DataEntry
from gluonts.dataset.common import ProcessDataEntry

try:
    from gluonts.transform import InstanceSplitter
    from gluonts.transform.sampler import ExpectedNumInstanceSampler
except Exception as e:
    raise ImportError(
        "Cannot import GluonTS InstanceSplitter / ExpectedNumInstanceSampler. "
        "Please check your gluonts installation/version."
    ) from e


def _worker_init_fn(worker_id):
    """Fixed seed per worker for reproducibility."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def itemize_start(entry: DataEntry) -> DataEntry:
    if "start" in entry and hasattr(entry["start"], "item"):
        entry["start"] = entry["start"].item()
    return entry


def _as_DT(y: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Normalise a 2-D target to [D, T]; returns (y_DT, transposed_flag)."""
    if y.ndim != 2:
        raise ValueError(f"_as_DT expects 2D, got {y.ndim}D shape={y.shape}")
    if y.shape[0] > y.shape[1]:
        return np.transpose(y, (1, 0)), True
    return y, False


# GIFT-Eval short-term eval horizons, verbatim from the official benchmark
# (gift-eval src/gift_eval/data.py; paper arXiv:2410.10393 Table 13).
# Per-dataset eval pred_len = MAP[freq] * term_multiplier (short term = x1).
M4_PRED_LENGTH_MAP = {"A": 6, "Q": 8, "M": 18, "W": 13, "D": 14, "H": 48}
PRED_LENGTH_MAP = {"M": 12, "W": 8, "D": 30, "H": 48, "T": 48, "S": 60}


def gift_eval_short_horizon(task_name: str, freq: str) -> Optional[int]:
    """Official GIFT-Eval short-term horizon for this task; None if freq is unknown."""
    base = str(freq).lstrip("0123456789").split("-")[0].upper()
    # Normalise old/new pandas offset aliases (MIN->T, ME/MS->M, YE/Y->A, QE->Q)
    base = {"MIN": "T", "ME": "M", "MS": "M", "YE": "A", "Y": "A", "QE": "Q"}.get(base, base)
    table = M4_PRED_LENGTH_MAP if "m4" in task_name.lower() else PRED_LENGTH_MAP
    return table.get(base)



# ---------------------------------------------------------------------------
# Length of the official GIFT-Eval test window (windows x prediction_length) per task,
# taking the longest of the terms that are actually evaluated. Measured from the official
# Dataset objects rather than assumed.
#
# Why it matters: the train/val cut is a fixed 90% fraction, while the test window has an
# absolute length. On short series 10% is not enough, so the training region can overlap
# the official test window. The public post-training recipe always applies
#     cutoff = min(int(0.9*T), T - TEST_LEN[task])
# Tasks absent from the table keep the plain 0.9T cut. This is a release invariant:
# prompt/post-training data must not include any official GIFT-Eval test window.
# ---------------------------------------------------------------------------
GIFTEVAL_TEST_LEN = {
    "LOOP_SEATTLE/5T": 10800, "LOOP_SEATTLE/H": 1440, "bitbrains_rnd/5T": 1440,
    "bizitobs_l2c/5T": 3600, "bizitobs_application": 1200, "solar/10T": 5760,
    "solar/H": 1440, "SZ_TAXI/15T": 720, "us_births/D": 600, "M_DENSE/H": 2160,
    "jena_weather/H": 1440, "jena_weather/10T": 5760, "ett2/15T": 7200,
    "bizitobs_service": 1200, "ett2/H": 2160, "ett1/15T": 7200, "ett1/H": 2160,
    "electricity/H": 3840, "electricity/15T": 14400, "kdd_cup_2018_with_missing/H": 1440,
    "bitbrains_fast_storage/5T": 1440, "electricity/D": 150, "electricity/W": 24,
    "m4_monthly": 18, "m4_quarterly": 8, "m4_yearly": 6, "m4_daily": 14,
    "m4_hourly": 48, "m4_weekly": 13, "solar/D": 60, "kdd_cup_2018_with_missing/D": 60,
    "bitbrains_fast_storage/H": 96, "bitbrains_rnd/H": 96, "bizitobs_l2c/H": 720,
    "car_parts_with_missing": 12, "covid_deaths": 30, "ett1/D": 90,
    "hierarchical_sales/D": 210, "hierarchical_sales/W": 32, "hospital": 12,
    "LOOP_SEATTLE/D": 60, "M_DENSE/D": 90, "restaurant": 30, "SZ_TAXI/H": 96,
    "temperature_rain_with_missing": 90, "us_births/W": 112,
    # The two saugeenday entries are easy to miss because the evaluation key spells the
    # dataset "saugeen"; without the alias they look like train-only tasks and drop out of
    # this table, leaving part of the validation tail inside the official test window.
    "saugeenday/M": 84, "saugeenday/D": 600,
}
STRICT_TEST_CUT = True
print(f"[test-cut] STRICT_TEST_CUT=ON "
      f"(official test windows excluded for {len(GIFTEVAL_TEST_LEN)} tasks)")

@dataclass
class TaskSpec:
    name: str
    series_fraction: Optional[float] = None
    max_series: Optional[int] = None
    weight: Optional[float] = None
    freq_override: Optional[str] = None
    storage_env_var: Optional[str] = None


class GiftEvalPostTrainingSeriesDataset(Dataset):
    """
    Map-style Dataset with time-based train/val split.
    All series appear in both train and val pools.
    Split is done by time truncation in __getitem__.
    """

    TRAIN_END_RATIO = 0.89   # ≈ 80% of original (available = 90% of original)

    def __init__(
        self,
        tasks: Sequence[TaskSpec],
        split: str = "train",
        val_fraction: float = 0.1,
        to_univariate: bool = True,
        storage_env_var: str = "GIFT_EVAL",
        seed: int = 0,
        context_length: int = 2048,
        prediction_length: int = 64,
        horizon_mask_tasks: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        dotenv_path = find_dotenv(usecwd=True)
        if dotenv_path:
            load_dotenv(dotenv_path)

        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")

        self.split = split
        self.val_fraction = float(val_fraction)
        self.to_univariate = bool(to_univariate)
        self.seed = int(seed)
        self.context_length = int(context_length)
        self.prediction_length = int(prediction_length)
        self.default_storage_env_var = storage_env_var
        # Report the release invariant again at construction time so every log records it.
        if split == "train":
            print("[test-cut/effective] STRICT_TEST_CUT=ON (release invariant)")
        self.gifteval_prefix_ratio = 0.9
        # PER-TASK HORIZON MASK: tasks in this set get future positions beyond
        # their true GIFT-Eval short-term horizon loss-masked (NaN) in collate,
        # so a unified prediction_length trains each task on its real horizon.
        # Long tasks must NOT be listed (their eval horizons exceed pred_len).
        self.horizon_mask_tasks = set(horizon_mask_tasks) if horizon_mask_tasks else set()

        # per-task storage
        self._specs: List[TaskSpec] = []
        self._hf: List[datasets.Dataset] = []
        self._process: List[ProcessDataEntry] = []
        self._names: List[str] = []
        self._series_pool: List[np.ndarray] = []
        self._task_example_count: List[int] = []
        self._env_vars: List[str] = []
        self._train_pred_len: List[Optional[int]] = []   # per-task horizon cap (None = uncapped)
        self._task_D: List[int] = []                      # variate count per task (used for bucketing)

        # Track minimum val series length for adaptive min_past
        self._min_val_series_len: int = 10**9

        # flat index: (task_id, row_idx, dim)
        self.index: List[Tuple[int, int, int]] = []

        rng = np.random.default_rng(self.seed)

        for spec in tasks:
            env_var = spec.storage_env_var or self.default_storage_env_var
            root = Path(os.getenv(env_var, ""))
            if not root.exists():
                print(f"[GiftEvalDataset] env {env_var} invalid or path not exists: {root}")
                continue

            ds_path = root / spec.name
            if not ds_path.exists():
                print(f"[GiftDataset] skip {ds_path} (not exists)")
                continue

            hf = datasets.load_from_disk(str(ds_path)).with_format("numpy")

            if spec.series_fraction is not None:
                n0 = len(hf)
                k = max(1, int(math.floor(n0 * float(spec.series_fraction))))
                hf = hf.shuffle(seed=self.seed).select(range(k))
            if spec.max_series is not None:
                hf = hf.select(range(min(int(spec.max_series), len(hf))))

            n = len(hf)
            if n < 1:
                continue

            if spec.freq_override is not None:
                freq = spec.freq_override
            else:
                if "freq" not in hf.features:
                    raise KeyError(f"Task {spec.name} has no 'freq' field.")
                freq = hf[0]["freq"]

            y0 = hf[0]["target"]
            one_dim = (getattr(y0, "ndim", 1) == 1)
            process = ProcessDataEntry(freq=freq, one_dim_target=one_dim)

            pool = np.arange(n)

            if getattr(y0, "ndim", 1) == 1:
                D = 1
            elif y0.ndim == 2:
                yDT, _ = _as_DT(np.asarray(y0))
                D = int(yDT.shape[0])
            else:
                raise ValueError(f"Unsupported target ndim={y0.ndim} in task {spec.name}")

            # Estimate minimum series length for this task (sample first few)
            check_n = min(n, 10)
            for ci in range(check_n):
                yi = np.asarray(hf[ci]["target"])
                if yi.ndim == 2:
                    yi_dt, _ = _as_DT(yi)
                    series_len = yi_dt.shape[1]
                else:
                    series_len = yi.shape[0]
                # Compute val series length
                T_avail = int(series_len * self.gifteval_prefix_ratio) if env_var == "GIFT_EVAL" else series_len
                T_train = int(T_avail * self.TRAIN_END_RATIO)
                val_start = max(0, T_train - self.context_length)
                val_len = T_avail - val_start
                self._min_val_series_len = min(self._min_val_series_len, val_len)

            # register task
            task_id = len(self._hf)
            self._specs.append(spec)
            self._hf.append(hf)
            self._process.append(process)
            self._names.append(spec.name)
            self._series_pool.append(pool)
            self._env_vars.append(env_var)
            self._task_D.append(int(D))

            # per-task horizon cap (only for tasks listed in horizon_mask_tasks)
            tpl = None
            if spec.name in self.horizon_mask_tasks:
                h = gift_eval_short_horizon(spec.name, freq)
                if h is None:
                    print(f"[HorizonMask] WARNING {spec.name}: unrecognised freq '{freq}', no cap applied")
                elif h < self.prediction_length:
                    tpl = int(h)
            self._train_pred_len.append(tpl)

            start_len = len(self.index)
            if (not self.to_univariate) or D == 1:
                for row_idx in pool:
                    self.index.append((task_id, int(row_idx), -1))
            else:
                for row_idx in pool:
                    for d in range(D):
                        self.index.append((task_id, int(row_idx), d))

            self._task_example_count.append(len(self.index) - start_len)

        if len(self._hf) == 0 or len(self.index) == 0:
            raise ValueError("No data loaded. Check task list & GIFT_EVAL path.")

        self._task_example_count = [max(1, c) for c in self._task_example_count]

        # Compute safe val min_past
        self.val_min_past = min(
            self.context_length,
            max(0, self._min_val_series_len - self.prediction_length - 1)
        )

        n_series = sum(len(p) for p in self._series_pool)
        overlap_status = "ZERO" if self.val_min_past >= self.context_length else f"PARTIAL (min_past={self.val_min_past})"
        print(f"[DataLoader] split={self.split}, tasks={len(self._hf)}, "
              f"total_series={n_series}, total_items={len(self.index)}, "
              f"val_overlap={overlap_status}")
        if not self.to_univariate:
            mv = {self._names[t]: self._task_D[t] for t in range(len(self._names)) if self._task_D[t] > 1}
            print(f"[mv] split={self.split}: to_univariate=False -> {len(mv)} multivariate task(s) kept whole "
                  f"(one item = one [D,T] series; task sampling mass still counts rows x D): {mv}")
        if self.horizon_mask_tasks:
            capped = {self._names[t]: self._train_pred_len[t]
                      for t in range(len(self._names)) if self._train_pred_len[t] is not None}
            missing = self.horizon_mask_tasks - set(self._names)
            print(f"[HorizonMask] split={self.split}, capped {len(capped)} tasks: {capped}")
            if missing:
                print(f"[HorizonMask] WARNING listed but not loaded: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> DataEntry:
        task_id, row_idx, dim = self.index[i]

        hf = self._hf[task_id]
        process = self._process[task_id]
        name = self._names[task_id]

        row = hf[row_idx]
        entry: DataEntry = dict(row)
        entry.setdefault("item_id", f"{name}_{row_idx}")

        entry = itemize_start(entry)
        entry = process(entry)

        # univariate extraction
        y = np.asarray(entry["target"])
        if dim >= 0:
            if y.ndim != 2:
                raise ValueError(f"Expect 2D target when dim>=0, got {y.ndim}D shape={y.shape}")
            yDT, _ = _as_DT(y)
            entry["target"] = yDT[dim]
            entry["item_id"] = f"{entry['item_id']}_dim{dim}"
        else:
            if y.ndim == 2 and self.to_univariate:
                yDT, _ = _as_DT(y)
                entry["target"] = yDT[0]
                entry["item_id"] = f"{entry['item_id']}_dim0"
            elif y.ndim == 2:
                # Multivariate training: keep the block as [D, T] (gluonts puts variates
                # first). The downstream InstanceSplitter slices along the last axis and
                # yields [T, D] (time first).
                yDT, _ = _as_DT(y)
                entry["target"] = np.ascontiguousarray(yDT)

        # -------- Step 1: Cut last 10% for test --------
        # Always slice along the last (time) axis: identical to the 1-D path, and correct for [D,T].
        env_var = self._env_vars[task_id]
        if env_var == "GIFT_EVAL" and self.gifteval_prefix_ratio < 1.0:
            y1 = np.asarray(entry["target"])
            T = y1.shape[-1]
            cutoff = max(1, int(T * self.gifteval_prefix_ratio))
            if STRICT_TEST_CUT:
                # A 10% fraction does not cover the test window on short series; tighten by absolute length.
                tl = GIFTEVAL_TEST_LEN.get(name)
                if tl:
                    cutoff = max(1, min(cutoff, T - int(tl)))
            if cutoff < T:
                entry["target"] = y1[..., :cutoff]

        # -------- Step 2: Time-based train/val split --------
        y2 = np.asarray(entry["target"])
        T_avail = y2.shape[-1]
        T_train = max(1, int(T_avail * self.TRAIN_END_RATIO))

        if self.split == "train":
            # Train: only first ~80% of original
            entry["target"] = y2[..., :T_train]
        else:
            # Val: [T_train - context_length, T_avail]
            # Context borrows from train region, predictions in val region
            val_start = max(0, T_train - self.context_length)
            entry["target"] = y2[..., val_start:T_avail]

        entry["task_name"] = name
        entry["_task_id"] = int(task_id)   # passthrough for neighbor_fill
        tpl = self._train_pred_len[task_id]
        entry["_train_pred_len"] = int(tpl) if tpl is not None else 0   # 0 = uncapped
        return entry

    def make_task_sampler(
        self,
        temperature_alpha: float = 0.5,
        epoch_size: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> WeightedRandomSampler:
        """Task temperature sampling: mass_i = N_i^alpha."""
        alpha = float(temperature_alpha)
        if epoch_size is None:
            epoch_size = len(self)

        task_mass = []
        for tid, spec in enumerate(self._specs):
            Ni = float(self._task_example_count[tid])
            if not self.to_univariate:
                # An index entry counts rows, not variates. Multiply D back in so the
                # per-task sampling mass matches the univariate arm (N_i = rows x D).
                Ni *= float(self._task_D[tid])
            if spec.weight is not None:
                mass = float(spec.weight)
            else:
                mass = Ni ** alpha
            task_mass.append(mass)

        task_mass = np.asarray(task_mass, dtype=np.float64)

        weights = np.empty(len(self.index), dtype=np.float64)
        for j, (tid, _row, _dim) in enumerate(self.index):
            weights[j] = task_mass[tid] / float(self._task_example_count[tid])

        w = torch.as_tensor(weights, dtype=torch.double)
        return WeightedRandomSampler(
            weights=w,
            num_samples=int(epoch_size),
            replacement=True,
            generator=generator,
        )


# =====================================================================
# Collate function (InstanceSplitter)
# =====================================================================
from gluonts.transform.sampler import InstanceSampler, NumInstanceSampler


class ValBoundarySampler(InstanceSampler):
    """
    Per-series val sampler: min_past = this series' OWN train/val boundary, so the
    prediction window is guaranteed to start in the val region.

    The val slice built in __getitem__ is y[val_start : T_avail] with the train/val
    boundary at (T_train - val_start) inside the slice. That boundary is recoverable
    from the slice length alone:
      - long series  (val_start > 0):  boundary = context_length
      - short series (val_start == 0): boundary = int(TRAIN_END_RATIO * slice_len)
    both cases collapse to  min(context_length, int(train_end_ratio * slice_len))
    (off-by-one from double int() flooring is possible; a <=1-point overlap is
    negligible).

    Series whose val region is shorter than min_future yield no val instances
    (correct: better to drop them from val than to pollute the signal). Note this
    changes which datasets contribute to val loss — val losses are NOT directly
    comparable with runs using the old global-min_past sampler.
    """
    N: int = 4
    context_length: int = 4096
    train_end_ratio: float = 0.89
    deterministic: bool = True

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)              # a = min_past, b = T - min_future
        T = ts.shape[self.axis]
        boundary = min(self.context_length, int(T * self.train_end_ratio))
        a = max(a, boundary)
        if b < a:
            return np.array([], dtype=int)       # val region too short for this series
        if self.deterministic:
            # FIXED windows, evenly spread over the val region: the val set is now
            # identical across epochs and runs, so a new val low is a REAL
            # improvement, not per-epoch resampling luck. (Random per-epoch windows
            # had ~1e-2 noise, drowning ~1e-2 true gains and letting one lucky
            # early epoch block all later checkpoints.)
            return np.unique(np.linspace(a, b, num=self.N).round().astype(int))
        return np.random.randint(a, b + 1, size=self.N)


class InstanceCollateFn:
    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        num_instances_per_series: int = 1,
        allow_padding: bool = True,
        compute_far_stats: bool = False,
        n_stats: int = 10,
        min_past: int = 0,
        min_future: Optional[int] = None,
        val_boundary: bool = False,
        train_end_ratio: float = 0.89,
        long_context_length: int = 0,
        mv_mode: bool = False,
        mv_row_budget: int = 0,
        mv_min_items: int = 2,
    ):
        self.context_length = int(context_length)
        self.prediction_length = int(prediction_length)
        self.compute_far_stats = compute_far_stats
        self.n_stats = n_stats
        # min_future < prediction_length enables FUTURE PADDING: windows whose
        # future is shorter than prediction_length are right-padded with NaN and
        # the padded positions are loss-masked (future_observed=False). This lets
        # short series (e.g. m4_yearly, car_parts) contribute training windows
        # under a unified prediction_length.
        self.min_future = int(min_future) if min_future is not None else int(prediction_length)

        # Long-history prompt (0 = off). A single splitter is used with
        # past_length = max(C, long); the backbone then takes the last C steps. The window
        # always ends at the forecast start, so that tail is element-wise identical to
        # sampling with C directly, and the set of sampled windows is unchanged.
        self.long_context_length = int(long_context_length or 0)
        self.past_length = max(int(context_length), self.long_context_length)

        # The window count varies with V (see mv_batch_plan), so splitters are cached by N.
        # The univariate path only ever uses N = num_instances.
        self.num_instances = int(num_instances_per_series)
        self.val_boundary = bool(val_boundary)
        self.min_past = int(min_past)
        self.train_end_ratio = float(train_end_ratio)
        self.mv_mode = bool(mv_mode)
        self.mv_row_budget = int(mv_row_budget or 0)
        self.mv_min_items = int(mv_min_items)
        self._splitters: Dict[int, InstanceSplitter] = {}
        self.splitter = self._splitter_for(self.num_instances)

        self.allow_padding = bool(allow_padding)

    def _splitter_for(self, n: int) -> InstanceSplitter:
        n = max(1, int(n))
        if n in self._splitters:
            return self._splitters[n]
        if self.val_boundary:
            # per-series boundary: prediction windows are forced past each series'
            # own train/val boundary (replaces the buggy global min_past)
            sampler = ValBoundarySampler(
                N=n,
                min_past=self.min_past,
                min_future=int(self.min_future),
                context_length=int(self.context_length),
                train_end_ratio=self.train_end_ratio,
            )
        else:
            sampler = NumInstanceSampler(
                N=n,
                min_past=self.min_past,
                min_future=int(self.min_future)
            )
        sp = InstanceSplitter(
            target_field="target",
            is_pad_field="is_pad",
            start_field="start",
            forecast_start_field="forecast_start",
            instance_sampler=sampler,
            past_length=self.past_length,
            future_length=self.prediction_length,
            time_series_fields=[],
            dummy_value=0.0,
        )
        self._splitters[n] = sp
        return sp

    def __call__(self, batch):
        splitter = self.splitter
        v_batch = 1
        if self.mv_mode and len(batch) > 0:
            y0 = np.asarray(batch[0].get("target"))
            v_batch = int(y0.shape[0]) if y0.ndim == 2 else 1
            # Same source as TaskBucketBatchSampler: it fills buckets by item count, this
            # reads the window count from the same plan.
            splitter = self._splitter_for(
                mv_batch_plan(v_batch, self.mv_row_budget or self.num_instances,
                              self.num_instances, self.mv_min_items)[1])
        instances = list(splitter(batch, is_train=True))
        if len(instances) == 0:
            return {}

        past_list, future_list, pad_list, far_list = [], [], [], []
        task_id_list = []
        kept_items = set()          # surviving source items (not windows); numerator of mv_w

        for ins in instances:
            # A 2-D target leaves InstanceSplitter as [T, V] (time first, gluonts
            # convention; past_is_pad stays 1-D). A 1-D target is reshaped to [T, 1] so the
            # univariate tensor shapes are unchanged.
            past = np.asarray(ins["past_target"], dtype=np.float32)
            fut = np.asarray(ins["future_target"], dtype=np.float32)
            past = past.reshape(-1, 1) if past.ndim == 1 else past
            fut = fut.reshape(-1, 1) if fut.ndim == 1 else fut
            pad = np.asarray(ins["past_is_pad"], dtype=np.bool_)

            # FUTURE PADDING: right-pad short futures with NaN → future_observed
            # (isfinite) marks them invalid → Chronos-2 loss masks them out.
            if 0 < fut.shape[0] < self.prediction_length:
                pad_len = self.prediction_length - fut.shape[0]
                fut = np.concatenate(
                    [fut, np.full((pad_len, fut.shape[1]), np.nan, dtype=np.float32)], axis=0
                )

            # PER-TASK HORIZON MASK: NaN-out positions beyond the task's true
            # GIFT-Eval horizon → loss & val only see the horizon the task is
            # actually evaluated at, so the train/val target matches the test horizon.
            tpl = int(ins.get("_train_pred_len", 0) or 0)
            if 0 < tpl < fut.shape[0]:
                fut = fut.copy()          # avoid mutating the shared underlying array
                fut[tpl:, :] = np.nan

            if past.shape[0] != self.past_length or fut.shape[0] != self.prediction_length:
                continue

            past_list.append(past)
            future_list.append(fut)
            pad_list.append(pad)
            task_id_list.append(int(ins.get("_task_id", -1)))
            kept_items.add(str(ins.get("item_id", len(kept_items))))

            if self.compute_far_stats:
                _ft = np.asarray(ins.get("target", []), dtype=np.float32)
                if _ft.ndim > 1:
                    # Under multivariate targets .ravel() would concatenate variates into
                    # one bogus series, so far_stats must be computed per variate before it
                    # can be used here. Refuse loudly rather than return a wrong statistic.
                    raise NotImplementedError(
                        f"far_stats does not support multivariate targets (shape={_ft.shape}); "
                        f"compute it per variate and align by row instead of using ravel()")
                full_target = _ft.ravel()
                from tabby.posttraining.far_context_utils import compute_far_stats_numpy
                far = compute_far_stats_numpy(full_target, self.context_length, n_stats=self.n_stats)
                far_list.append(far)

        if len(past_list) == 0:
            return {}

        # All items in a batch must share the same V (np.stack would fail anyway; this
        # gives a readable reason). TaskBucketBatchSampler normally guarantees it.
        vs = {p.shape[1] for p in past_list}
        if len(vs) != 1:
            raise ValueError(f"mixed variate counts in one batch {sorted(vs)}; multivariate "
                             f"training must bucket by task (built into make_posttraining_loader_dataset)")

        past = torch.from_numpy(np.stack(past_list, axis=0))
        future = torch.from_numpy(np.stack(future_list, axis=0))
        past_is_pad = torch.from_numpy(np.stack(pad_list, axis=0))

        # Long-history prompt: past is past_length wide; the backbone takes the last C steps
        # while the prompt generator sees the whole window.
        long_past = None; long_past_is_pad = None
        if self.long_context_length > self.context_length:
            long_past = past
            long_past_is_pad = past_is_pad
            past = past[:, -self.context_length:, :]
            past_is_pad = past_is_pad[:, -self.context_length:]

        past_observed = torch.isfinite(past) & (~past_is_pad.unsqueeze(-1))
        future_observed = torch.isfinite(future)

        past = torch.nan_to_num(past, nan=0.0, posinf=0.0, neginf=0.0)
        future = torch.nan_to_num(future, nan=0.0, posinf=0.0, neginf=0.0)

        result = {
            "past_target": past,
            "past_observed_target": past_observed,
            "past_is_pad": past_is_pad,
            "future_target": future,
            "future_observed_target": future_observed,
            "task_ids": torch.tensor(task_id_list, dtype=torch.long),
        }

        if self.mv_mode:
            # ---- per-batch gradient weight mv_w ----
            # The loss averages over rows within a batch while batches are weighted equally.
            # V-buckets hold different numbers of items, so without a weight the tasks with
            # many variates would take a gradient share inflated by roughly V.
            # See mv_batch_weight for why the train and val formulas differ.
            # NOTE: the training loop must actually apply mv_w, e.g. (loss * w).backward()
            # followed by dividing the accumulated gradient by the sum of weights.
            n_items = max(1, len(kept_items))
            v_out = int(past.shape[-1])
            result["n_items"] = torch.tensor(n_items, dtype=torch.long)
            result["mv_V"] = torch.tensor(v_out, dtype=torch.long)
            result["mv_w"] = torch.tensor(
                mv_batch_weight(n_items, v_out,
                                self.mv_row_budget or self.num_instances,
                                self.val_boundary), dtype=torch.float32)

        if long_past is not None:
            long_observed = torch.isfinite(long_past) & (~long_past_is_pad.unsqueeze(-1))
            long_past = torch.nan_to_num(long_past, nan=0.0, posinf=0.0, neginf=0.0)
            result["long_past_target"] = long_past
            result["long_past_observed_target"] = long_observed
            result["long_past_is_pad"] = long_past_is_pad

        if self.compute_far_stats and len(far_list) > 0:
            result["far_stats"] = torch.from_numpy(np.stack(far_list, axis=0)).float()

        return result


def make_instance_collate_fn(
    context_length: int,
    prediction_length: int,
    num_instances_per_series: int = 1,
    allow_padding: bool = True,
    device=None,
    compute_far_stats: bool = False,
    n_stats: int = 10,
    min_past: int = 0,
    min_future: Optional[int] = None,
    val_boundary: bool = False,
    train_end_ratio: float = 0.89,
    long_context_length: int = 0,
    mv_mode: bool = False,
    mv_row_budget: int = 0,
    mv_min_items: int = 2,
):
    return InstanceCollateFn(
        context_length=context_length,
        prediction_length=prediction_length,
        num_instances_per_series=num_instances_per_series,
        allow_padding=allow_padding,
        compute_far_stats=compute_far_stats,
        n_stats=n_stats,
        min_past=min_past,
        min_future=min_future,
        val_boundary=val_boundary,
        train_end_ratio=train_end_ratio,
        long_context_length=long_context_length,
        mv_mode=mv_mode,
        mv_row_budget=mv_row_budget,
        mv_min_items=mv_min_items,
    )


# =====================================================================
# Multivariate training: batch sampler that buckets by task
# =====================================================================
def mv_batch_plan(v: int, batch_size: int, num_instances: int, min_items: int = 2):
    """Given V, return (items per batch, windows per item). Single source of truth.

    The sampler (how many series go into a batch) and the collate function (how many
    windows are cut per series) must derive both numbers from this one function, otherwise
    the batch row count and item count disagree and the mv_w weight below is wrong.

      R = batch_size * num_instances, the per-batch row budget of the univariate arm
          (a row = one window of one variate)
      items = max(min_items, batch_size // V)      # at least min_items independent series
      ninst = min(num_instances, max(1, R // (items × V)))

    Two properties:
      * at V=1 this is exactly the univariate plan (batch_size, num_instances), rows = R;
      * rows = items * ninst * V <= R, which also removes the memory spike that a naive
        plan produces for tasks with many variates.
    min_items exists so that a high-V bucket does not end up with a single series per
    batch: it trades windows-per-series for independent-series-per-step at a fixed row
    budget.
    """
    v = max(1, int(v))
    R = int(batch_size) * int(num_instances)
    items = max(1, int(batch_size) // v)
    # Raise to min_items, but never break the row budget: for large V, R // v may leave
    # room for a single item only. One item = all variates of one series and is atomic, so
    # when V > R that item alone exceeds the budget. That is a floor set by the data.
    items = min(max(items, int(min_items)), max(1, R // v))
    ninst = min(int(num_instances), max(1, R // max(1, items * v)))
    return items, ninst


def mv_batch_weight(n_items: int, v: int, batch_size: int, is_val: bool) -> float:
    """Weight that a multivariate batch should carry in the loss.

    The train and val formulas differ because the univariate composition is set by
    different mechanisms on the two sides:
      train: composition comes from WeightedRandomSampler, where P(task t) is already
             proportional to mass_t. It is enough to make the per-item weight equal across
             V, i.e. w = n_items / row_budget (= 1 in the univariate case).
      val:   composition comes from full enumeration. In the univariate arm each variate is
             its own item, so a task's share is proportional to rows * D; a multivariate
             item is a whole [V, T] block, so V must be multiplied back in:
             w = n_items * V / row_budget.
    With this weight, neither the bucketing nor the windows-per-item affect the task mix.
    """
    return float(n_items) * (float(max(1, v)) if is_val else 1.0) / float(max(1, batch_size))


def mv_mix_preflight(ds, batch_size: int, num_instances: int, min_items: int = 2,
                     temperature_alpha: float = 0.5, n_draw: int = 60000,
                     tol_pp: float = 2.0, seed: int = 12345):
    """Pre-flight check of the task mixture (no GPU, no training, a few seconds).

    It draws n_draw samples through the real make_task_sampler, the real
    TaskBucketBatchSampler and the real mv_batch_weight, sums mv_w per task to get each
    task's gradient share, and compares that against P(task) of the univariate arm. The
    check runs the code it is checking rather than a re-derived formula.

    A broken mixture raises no error during training; it only shows up after a full run
    plus a full evaluation. This moves the same signal to a few seconds before launch.

    Returns (ok, report). On ok=False the caller should abort, not merely warn.
    """
    gen = torch.Generator(); gen.manual_seed(int(seed))
    src = ds.make_task_sampler(temperature_alpha=temperature_alpha,
                               epoch_size=int(n_draw), generator=gen)
    target = {}                                  # P(task) of the univariate arm = goal
    w = np.asarray(src.weights, dtype=np.float64)
    for j, (tid, _r, _d) in enumerate(ds.index):
        target[tid] = target.get(tid, 0.0) + float(w[j])
    zt = sum(target.values()) or 1.0
    target = {k: v / zt for k, v in target.items()}

    bsam = TaskBucketBatchSampler(
        src, task_of=lambda i: ds.index[i][0], variates_of=lambda t: ds._task_D[t],
        batch_size=batch_size, drop_last=True,
        num_instances=num_instances, min_items=min_items)
    got, unweighted = {}, {}
    for blist in bsam:
        tid = ds.index[blist[0]][0]
        got[tid] = got.get(tid, 0.0) + mv_batch_weight(
            len(blist), ds._task_D[tid], batch_size, is_val=False)
        unweighted[tid] = unweighted.get(tid, 0.0) + 1.0
    zg = sum(got.values()) or 1.0
    zu = sum(unweighted.values()) or 1.0

    worst, worst_t, worst_raw = 0.0, None, 0.0
    for tid, tgt in target.items():
        d = abs(got.get(tid, 0.0) / zg - tgt) * 100.0
        if d > worst:
            worst, worst_t = d, tid
        worst_raw = max(worst_raw, abs(unweighted.get(tid, 0.0) / zu - tgt) * 100.0)

    lines = [f"[mv/preflight] task-mixture self-check ({n_draw} draws, tolerance {tol_pp}pp):"]
    for tid in sorted(target, key=lambda t: -target[t])[:6]:
        lines.append(f"    {ds._names[tid]:<26} V={ds._task_D[tid]:<3} "
                     f"uv={100 * target[tid]:5.2f}%  weighted={100 * got.get(tid, 0.0) / zg:5.2f}%  "
                     f"(unweighted={100 * unweighted.get(tid, 0.0) / zu:5.2f}%)")
    lines.append(f"    max deviation {worst:.2f}pp (at {ds._names[worst_t] if worst_t is not None else '?'}); "
                 f"without the weight it would be {worst_raw:.2f}pp")
    ok = worst <= tol_pp
    if not ok:
        lines.append(f"[mv/preflight] FAILED: outside tolerance, do not launch. Check whether the "
                     f"training loop applies mv_w, whether make_task_sampler was also changed "
                     f"(double correction), and whether the plan and the weight share a source.")
    else:
        lines.append("[mv/preflight] OK: matches the univariate arm")
    return ok, "\n".join(lines)


def mv_audit_str(seen, items, wsum):
    """Report the mixture per epoch as {V: batches / items / grad share % / item share %}.

    Reading rule: the gradient share must agree with the item share (the latter being the
    row share of the univariate arm). A disagreement means the mixture has drifted."""
    zw = sum(wsum.values()) or 1.0
    zi = sum(items.values()) or 1
    return {v: (f"{seen.get(v, 0)}b/{items.get(v, 0)}it/"
                f"grad{100 * wsum.get(v, 0.0) / zw:.1f}%/item{100 * items.get(v, 0) / zi:.1f}%")
            for v in sorted(seen)}


class TaskBucketBatchSampler(torch.utils.data.Sampler):
    """Group a stream of item indices into batches that share the same V (variate attention
    requires a single V per batch). Installed only when to_univariate=False; the univariate
    path never goes through here.

      source       index stream: train = WeightedRandomSampler (re-drawn each iteration,
                   RNG owned by its generator), val = a deterministic list
      task_of      idx -> task_id;variates_of  task_id -> V
      batch_size   row budget. The items-per-batch and windows-per-item of each V-bucket
                   come from mv_batch_plan; memory is still budgeted as windows x ctx.
      drop_last    train drops the ragged end-of-epoch batch of each bucket (same meaning as
                   drop_last=True in the univariate path); val keeps it.

    Read before modifying: this sampler only groups indices into batches. It does not, and
    must not, correct the task mixture. If the number of batches scales as
    picks / items_of_V while every batch carries weight 1, tasks with many variates get a
    gradient share inflated by roughly V. The correction belongs in the mv_w weight emitted
    by InstanceCollateFn, not in the mass computed by make_task_sampler. Applying it in both
    places overcorrects by a factor of V; the `Ni *= D` in make_task_sampler is correct.

    The bucket key is V, not the task: all V=1 tasks share one bucket and still mix freely,
    so when the data contains no V>1 task this sampler is the identity on `source` and the
    batches are identical to the univariate path. Ordering is preserved - items enter their
    bucket in draw order and a bucket is emitted as soon as it is full - so the task
    sampling distribution equals that of `source`.
    """

    def __init__(self, source, task_of, variates_of, batch_size: int, drop_last: bool,
                 num_instances: int = 1, min_items: int = 2):
        self.source = source
        self.task_of = task_of
        self.variates_of = variates_of
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.num_instances = int(num_instances)
        self.min_items = int(min_items)
        self._v_cache: Dict[int, int] = {}

    def v_of(self, task_id: int) -> int:
        if task_id not in self._v_cache:
            self._v_cache[task_id] = max(1, int(self.variates_of(task_id)))
        return self._v_cache[task_id]

    def bs_of(self, task_id: int) -> int:
        return mv_batch_plan(self.v_of(task_id), self.batch_size,
                             self.num_instances, self.min_items)[0]

    def plan_of_v(self, v: int):
        return mv_batch_plan(v, self.batch_size, self.num_instances, self.min_items)

    def __iter__(self) -> Iterator[List[int]]:
        buckets: Dict[int, List[int]] = {}          # key = V
        for idx in self.source:
            idx = int(idx)
            v = self.v_of(self.task_of(idx))
            b = buckets.setdefault(v, [])
            b.append(idx)
            if len(b) >= self.plan_of_v(v)[0]:
                yield b
                buckets[v] = []
        if not self.drop_last:
            for v in list(buckets.keys()):
                if buckets[v]:
                    yield buckets[v]

    def __len__(self) -> int:
        # Upper-bound estimate: the true batch count depends on which tasks are drawn.
        # DataLoader only uses it for len(); the training loop does not rely on it.
        return max(1, len(self.source) // 1)


# =====================================================================
# DataLoader factory
# =====================================================================
def make_posttraining_loader_dataset(
    tasks: Sequence[TaskSpec],
    context_length: int,
    prediction_length: int,
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    temperature_alpha: float = 0.5,
    epoch_size: Optional[int] = None,
    use_sampler: bool = True,
    allow_padding: bool = True,
    num_instances_per_series: int = 1,
    val_fraction: float = 0.1,
    to_univariate: bool = True,
    device: Optional[str] = None,
    compute_far_stats: bool = False,
    n_stats: int = 10,
    seed: int = 0,
    storage_env_var: str = "GIFT_EVAL",
    min_future: Optional[int] = None,
    long_context_length: int = 0,
    horizon_mask_tasks: Optional[Sequence[str]] = None,
    mv_min_items: int = 2,
) -> Tuple[GiftEvalPostTrainingSeriesDataset, DataLoader]:
    ds = GiftEvalPostTrainingSeriesDataset(
        tasks=tasks,
        split=split,
        val_fraction=val_fraction,
        to_univariate=to_univariate,
        storage_env_var=storage_env_var,
        seed=seed,
        context_length=context_length,
        prediction_length=prediction_length,
        horizon_mask_tasks=horizon_mask_tasks,
    )

    # Val uses a PER-SERIES boundary sampler (fix for the global-min_past bug):
    # every series' prediction windows start past its own train/val boundary,
    # so long datasets keep zero overlap even when trained jointly with ultra-short
    # datasets, and short datasets stop validating inside their train region.
    if split == "val":
        print(f"[Val collate] ValBoundarySampler: per-series min_past = "
              f"min({context_length}, {ds.TRAIN_END_RATIO} * val_slice_len) "
              f"(legacy global val_min_past would have been {ds.val_min_past})")

    collate_fn = make_instance_collate_fn(
        context_length=context_length,
        prediction_length=prediction_length,
        num_instances_per_series=num_instances_per_series,
        allow_padding=allow_padding,
        device=None,
        compute_far_stats=compute_far_stats,
        n_stats=n_stats,
        min_past=0,
        min_future=min_future,
        val_boundary=(split == "val"),
        train_end_ratio=ds.TRAIN_END_RATIO,
        long_context_length=long_context_length,
        mv_mode=(not to_univariate),
        mv_row_budget=batch_size,
        mv_min_items=mv_min_items,
    )

    sampler = None
    shuffle = False

    if use_sampler:
        gen = torch.Generator()
        gen.manual_seed(seed)
        sampler = ds.make_task_sampler(
            temperature_alpha=temperature_alpha,
            epoch_size=epoch_size,
            generator=gen,
        )
        shuffle = False
    elif split == "val":
        # DETERMINISTIC val: enumerate every series exactly once (tiled up to a
        # minimum of 4 batches for tiny datasets). Combined with the linspace
        # windows in ValBoundarySampler, the val set is byte-identical across
        # epochs — val losses are directly comparable and "new low" is meaningful.
        # (The old WeightedRandomSampler redrew series every epoch → ~1e-2 val
        # noise that stalled checkpoint selection on one lucky early epoch.)
        idx = list(range(len(ds)))
        min_needed = batch_size * 4
        if len(idx) < min_needed:
            reps = (min_needed + len(idx) - 1) // len(idx)
            idx = (idx * reps)[:min_needed]
        sampler = idx
        shuffle = False
    else:
        sampler = None
        shuffle = False

    if not to_univariate:
        # Multivariate training: bucket by task, since batches cannot mix different V.
        # The index stream is the same sampler as above; bucketing only groups it.
        source = sampler if sampler is not None else list(range(len(ds)))
        batch_sampler = TaskBucketBatchSampler(
            source, task_of=lambda i: ds.index[i][0],
            variates_of=lambda t: ds._task_D[t],
            batch_size=batch_size, drop_last=(split == "train"),
            num_instances=num_instances_per_series, min_items=mv_min_items)
        bs_plan = {}
        for t in range(len(ds._names)):
            if ds._task_D[t] > 1:
                it, ni = batch_sampler.plan_of_v(ds._task_D[t])
                bs_plan[ds._names[t]] = f"{it}items x {ni}win x V{ds._task_D[t]} = {it * ni * ds._task_D[t]} rows"
        _it0, _ni0 = mv_batch_plan(1, batch_size, num_instances_per_series, mv_min_items)
        print(f"[mv] split={split}: TaskBucketBatchSampler row budget {batch_size}x{num_instances_per_series}"
              f"={batch_size * num_instances_per_series} rows (uv bucket = {_it0} items x {_ni0} win), "
              f"min_items={mv_min_items}; per-task plan: {bs_plan}")
        print(f"[mv] split={split}: gradient weight mv_w = n_items"
              f"{' * V' if split == 'val' else ''}/{batch_size} enabled; the training loop must "
              f"apply it, otherwise high-V tasks take a gradient share inflated by about V")
        loader = DataLoader(
            ds,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=(device is not None and str(device).startswith("cuda")),
            persistent_workers=(num_workers > 0),
            collate_fn=collate_fn,
            worker_init_fn=_worker_init_fn,
            generator=torch.Generator().manual_seed(seed),
        )
        return ds, loader

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        shuffle=shuffle,
        pin_memory=(device is not None and str(device).startswith("cuda")),
        persistent_workers=(num_workers > 0),
        collate_fn=collate_fn,
        drop_last=(split == "train"),   # val keeps its final partial batch
        worker_init_fn=_worker_init_fn,
        generator=torch.Generator().manual_seed(seed),
    )

    return ds, loader


# Legacy aliases keep old research imports readable without changing data logic.
GiftEvalPretrainSeriesDataset = GiftEvalPostTrainingSeriesDataset
make_pretrain_loader_dataset = make_posttraining_loader_dataset
