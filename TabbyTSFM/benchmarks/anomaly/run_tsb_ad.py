#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Zero-shot Tabby latent anomaly detection on TSB-AD.

The foundation model is loaded ONCE per worker and reused for every config and
every series -- there is no per-series fitting, so the detector is zero-shot.

Scoring embeds each series with the frozen encoder and scores every patch by its
distance from the series' latent distribution (no forecasting). The default run
sweeps the latent metric (and, optionally, the window stride) on the Tuning split
and writes ``best_config.json``. ``--final`` then locks that config and evaluates
the full Eva split for TSB-AD-U and TSB-AD-M, reporting mean and median VUS-PR.

Tracks
------
The metric sweep runs on the U-Tuning split. ``--final`` evaluates both the
univariate (TSB-AD-U) and multivariate (TSB-AD-M) Eva splits by default;
``--skip_multivariate`` (alias ``--univariate_only``) restricts it to the U track
and never touches the M data dir / file lists.

Parallelism
-----------
Scoring a single series underfills a large GPU, and the per-series VUS-PR metric
is a heavy CPU cost. Both are recovered by data-parallelism over series:
On CUDA, ``--workers N`` spins up N worker processes, each holding its own
predictor replica on the target device. ``--workers 1`` (default), and every CPU
run, use the in-process serial path.

Examples
--------
# latent metric sweep on the Tuning split, 8-way data-parallel on one big GPU
python benchmarks/anomaly/run_tsb_ad.py --checkpoint /path/to/step_0165000 \
    --device cuda:0 --workers 8

# sweep a specific set of metrics and window strides
python benchmarks/anomaly/run_tsb_ad.py --checkpoint /path/to/step_0165000 --workers 8 \
    --sweep_latent_metric centroid,mahalanobis_diag --sweep_latent_stride 256,512

# final U-Eva run with the released tuned configuration
python benchmarks/anomaly/run_tsb_ad.py --checkpoint /path/to/step_0165000 \
    --final --skip_multivariate \
    --config_json benchmarks/anomaly/configs/tabby_tsb_u.json --workers 8
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
try:
    from .detector import PatchTSTFM_AD
except ImportError:  # direct execution: python benchmarks/anomaly/run_tsb_ad.py
    from detector import PatchTSTFM_AD

LATENT_METRIC_GRID = ["centroid", "centroid_robust", "mahalanobis_diag", "mahalanobis"]

DEFAULT_CONFIG = {
    "latent_metric": "centroid",
    "latent_stride": None,
    "agg": "mean",
}

# Per-process predictor replica and run-wide latent settings, populated once per
# worker by ``_init_worker`` (parallel) or in the main process by ``main`` (serial).
_PREDICTOR = None
_LATENT_METRIC = "centroid"
_LATENT_LAYER = -1
_LATENT_STRIDE = None


# =============================================================================
# Data + single-series scoring (runs inside whichever process owns _PREDICTOR)
# =============================================================================

def list_series_names(list_csv):
    """Return the ordered list of file names named in ``list_csv``."""
    return list(pd.read_csv(list_csv)["file_name"].values)


@functools.lru_cache(maxsize=64)
def _load_series(data_dir, fname):
    """Load one series -> (data[T, C], label[T]). Cached per worker.

    Reading a CSV is far cheaper than the GPU embed + VUS-PR metric, and the LRU
    cache lets a worker reuse a series across the metrics of a sweep without
    re-reading it. Cached arrays are never mutated by the scorer, so sharing is
    safe.
    """
    df = pd.read_csv(Path(data_dir) / fname).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df["Label"].astype(int).to_numpy()
    return data, label


def _make_clf(agg):
    """Build a latent scorer from the run-wide worker settings."""
    return PatchTSTFM_AD(
        predictor=_PREDICTOR,
        latent_metric=_LATENT_METRIC,
        latent_layer=_LATENT_LAYER,
        latent_stride=_LATENT_STRIDE,
        agg=agg,
    )


def _score_one(fname, data_dir, agg):
    """Score a single series; return a per-series metrics row.

    A single bad series yields a NaN row (with the error recorded) instead of
    killing the run.
    """
    try:
        data, label = _load_series(data_dir, fname)
        clf = _make_clf(agg)
        clf.fit(data)
        # slidingWindow matches the TSB-AD leaderboard convention.
        slidingWindow = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
        m = get_metrics(clf.decision_scores_, label, slidingWindow=slidingWindow)
        return {"file_name": fname, "VUS-PR": m["VUS-PR"], "AUC-PR": m["AUC-PR"],
                "VUS-ROC": m["VUS-ROC"], "error": ""}
    except Exception as e:
        print(f"  [skip] {fname}: {type(e).__name__}: {e}", flush=True)
        return {"file_name": fname, "VUS-PR": np.nan, "AUC-PR": np.nan,
                "VUS-ROC": np.nan, "error": f"{type(e).__name__}: {e}"}
    finally:
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass


def _score_task(task):
    """Pool entry point: (job_id, fname, data_dir, agg)."""
    job_id, fname, data_dir, agg = task
    return job_id, _score_one(fname, data_dir, agg)


def _score_one_latent_variants(fname, data_dir, strides, metrics, agg):
    """Score one series over a ``latent_stride`` x ``latent_metric`` grid.

    Embeddings depend only on the stride, so for each stride one embedding pass
    per series feeds every metric -- the fast path behind ``--sweep_latent_metric``.
    Returns ``{(stride, metric): metrics_row}``.
    """
    keys = [(st, m) for st in strides for m in metrics]
    try:
        data, label = _load_series(data_dir, fname)
        clf = _make_clf(agg)
        variants = clf.score_latent_variants(data, metrics, strides=strides)
        slidingWindow = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
        rows = {}
        for key, scores in variants.items():
            m = get_metrics(scores, label, slidingWindow=slidingWindow)
            rows[key] = {"file_name": fname, "VUS-PR": m["VUS-PR"],
                         "AUC-PR": m["AUC-PR"], "VUS-ROC": m["VUS-ROC"], "error": ""}
        return rows
    except Exception as e:
        print(f"  [skip] {fname}: {type(e).__name__}: {e}", flush=True)
        return {key: {"file_name": fname, "VUS-PR": np.nan, "AUC-PR": np.nan,
                      "VUS-ROC": np.nan, "error": f"{type(e).__name__}: {e}"}
                for key in keys}
    finally:
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass


def _score_task_latentsweep(task):
    """Pool entry point: (job_id, fname, data_dir, strides, metrics, agg)."""
    job_id, fname, data_dir, strides, metrics, agg = task
    return job_id, _score_one_latent_variants(fname, data_dir, strides, metrics, agg)


# =============================================================================
# Worker pool
# =============================================================================

def _init_worker(checkpoint, batch_size, device, precision, threads,
                 latent_metric="centroid", latent_layer=-1, latent_stride=None):
    """Load one predictor replica per worker (deferred import keeps --help fast)."""
    global _PREDICTOR, _LATENT_METRIC, _LATENT_LAYER, _LATENT_STRIDE
    import torch

    _LATENT_METRIC = latent_metric
    _LATENT_LAYER = latent_layer
    _LATENT_STRIDE = latent_stride
    # Avoid CPU oversubscription when several workers run get_metrics at once.
    if threads and threads > 0:
        torch.set_num_threads(threads)
    try:
        from .predictor import PatchTSTFMPredictor
    except ImportError:  # direct execution from the repository checkout
        from predictor import PatchTSTFMPredictor
    _PREDICTOR = PatchTSTFMPredictor(
        checkpoint=checkpoint,
        batch_size=batch_size,
        device=device,
        precision=precision,
    )
    # Fail loudly if a cuda device was requested but the predictor silently fell
    # back to CPU (old driver / mismatched torch build). Eight CPU workers look
    # like "parallel" but never touch the GPU -- exactly the case that wastes a run.
    resolved = str(getattr(_PREDICTOR, "device", "cpu"))
    print(f"[worker pid={os.getpid()}] predictor on {resolved} "
          f"(latent_metric={_LATENT_METRIC})", flush=True)
    if device.startswith("cuda") and not resolved.startswith("cuda"):
        raise RuntimeError(
            f"Requested device={device!r} but the predictor resolved to {resolved!r}: "
            "torch.cuda.is_available() is False in this worker. This is almost always "
            "a PyTorch <-> NVIDIA-driver mismatch. Install a torch build matching the "
            "driver's CUDA version and retry (or pass --device cpu deliberately)."
        )


class _Executor:
    """Dispatches score tasks either in-process (serial) or across a pool."""

    def __init__(self, pool=None):
        self.pool = pool

    def map(self, tasks, fn=_score_task):
        tasks = list(tasks)
        if not tasks:
            return []
        if self.pool is None:
            return [fn(t) for t in tasks]
        # chunksize=1 load-balances series of very uneven length/cost.
        return list(self.pool.map(fn, tasks, chunksize=1))


# =============================================================================
# Latent metric sweep (one embedding pass per stride feeds every metric)
# =============================================================================

def run_latent_sweep(executor, args, out_dir):
    """Sweep latent ``metric`` x ``stride`` on the U-Tuning split.

    Embeddings depend only on the stride, so each stride costs one embedding pass
    per series and every metric is evaluated from it. Writes a per-config summary
    and the winning config to ``best_config.json`` for ``--final``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = tuple(m.strip() for m in args.sweep_latent_metric.split(",") if m.strip())
    if not metrics:
        metrics = tuple(LATENT_METRIC_GRID)
    strides = tuple(int(s) for s in args.sweep_latent_stride.split(",") if s.strip())
    if not strides:
        strides = (None,)                       # default: non-overlapping tiling
    fnames = list_series_names(args.dev_u)
    print(f"latent metric x stride sweep over {len(fnames)} U-Tuning series; "
          f"metrics: {list(metrics)}; strides: {list(strides)} "
          f"(None = non-overlapping)")

    tasks = [(0, fname, args.data_u, strides, metrics, "mean") for fname in fnames]
    t0 = time.time()
    results = executor.map(tasks, fn=_score_task_latentsweep)
    dt = time.time() - t0

    per_key = {(st, m): [] for st in strides for m in metrics}
    for _job_id, rows in results:
        for key, row in rows.items():
            per_key[key].append(row)

    summaries = []
    for (st, m), rows in per_key.items():
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / f"latentsweep_{m}_stride{st}.csv", index=False)
        summaries.append({
            "stage": "latent_sweep", "latent_metric": m, "latent_stride": st,
            "n_series": int(len(df)), "VUS_PR_mean": df["VUS-PR"].mean(),
            "VUS_PR_median": df["VUS-PR"].median(), "VUS_PR_std": df["VUS-PR"].std(),
            "seconds": dt,
        })
    summ = pd.DataFrame(summaries).sort_values("VUS_PR_mean", ascending=False)
    summ.to_csv(out_dir / "latent_sweep_summary.csv", index=False)
    print(summ.to_string(index=False))

    best = max(summaries, key=lambda s: (-1 if np.isnan(s["VUS_PR_mean"])
                                         else s["VUS_PR_mean"]))
    final_config = {
        "latent_metric": best["latent_metric"],
        "latent_stride": best["latent_stride"],
        "agg": DEFAULT_CONFIG["agg"],
    }
    (out_dir / "best_config.json").write_text(json.dumps(final_config, indent=2))
    print(f"\nBest: latent_metric={best['latent_metric']}, "
          f"latent_stride={best['latent_stride']} "
          f"(mean VUS-PR {best['VUS_PR_mean']:.4f})")
    print("Selected config:", final_config)


# =============================================================================
# Final evaluation
# =============================================================================

def _resolve_final_config(args, out_dir):
    """Pick the config for --final: --config_json > best_config.json > default."""
    if args.config_json:
        cfg = json.loads(Path(args.config_json).read_text())
        print(f"Loaded config from --config_json ({args.config_json}):", cfg)
        return cfg
    best_path = out_dir / "best_config.json"
    if best_path.exists() and not args.use_default_config:
        cfg = json.loads(best_path.read_text())
        print("Loaded best_config:", cfg)
        return cfg
    print("No tuned best_config.json found (or --use_default_config set); "
          "using placeholder DEFAULT_CONFIG:", DEFAULT_CONFIG)
    return dict(DEFAULT_CONFIG)


def run_final(executor, args, out_dir, results_dir, agg):
    results_dir.mkdir(parents=True, exist_ok=True)
    # Both splits are dispatched together so the workers stay saturated.
    jobs = [
        {"tag": "final_U", "agg": "mean",
         "fnames": list_series_names(args.eval_u), "data_dir": args.data_u},
    ]
    if args.skip_multivariate:
        print("--skip_multivariate: evaluating the TSB-AD-U Eva split only.")
    else:
        jobs.append(
            {"tag": "final_M", "agg": agg,
             "fnames": list_series_names(args.eval_m), "data_dir": args.data_m}
        )
    tasks = []
    for ji, job in enumerate(jobs):
        for fname in job["fnames"]:
            tasks.append((ji, fname, job["data_dir"], job["agg"]))
    results = executor.map(tasks)
    per_job = [[] for _ in jobs]
    for ji, row in results:
        per_job[ji].append(row)

    for job, rows in zip(jobs, per_job):
        df = pd.DataFrame(rows)
        tag = "U" if job["tag"] == "final_U" else "M"
        df.to_csv(results_dir / f"final_{tag}.csv", index=False)
        print(f"{tag} | mean VUS-PR: {df['VUS-PR'].mean():.4f} | "
              f"median VUS-PR: {df['VUS-PR'].median():.4f} | n={len(df)}")


# =============================================================================
# Entry point
# =============================================================================

def _assert_cuda_usable(device):
    """Fail fast in the parent if a cuda device was asked for but is unusable.

    Cheaper than discovering it 8 workers deep, and turns the silent CPU fallback
    into an actionable message.
    """
    if not device.startswith("cuda"):
        return
    import torch
    if torch.cuda.is_available():
        return
    raise SystemExit(
        f"\n--device {device} requested but torch.cuda.is_available() is False.\n"
        "This is almost always a PyTorch <-> NVIDIA-driver mismatch (torch built for "
        "a newer CUDA than the driver supports). Diagnose:\n"
        "  python -c \"import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())\"\n"
        "  nvidia-smi --query-gpu=driver_version --format=csv\n"
        "Then install a torch build matching the driver (e.g. a CUDA 12.4 driver):\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
        "Or pass --device cpu to run on CPU on purpose.\n"
    )


def main():
    ap = argparse.ArgumentParser(description="PatchTST-FM zero-shot latent ablation / eval")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dev_u", default="../Datasets/File_List/TSB-AD-U-Tuning.csv")
    ap.add_argument("--eval_u", default="../Datasets/File_List/TSB-AD-U-Eva.csv")
    ap.add_argument("--eval_m", default="../Datasets/File_List/TSB-AD-M-Eva.csv")
    ap.add_argument("--data_u", default="../Datasets/TSB-AD-U")
    ap.add_argument("--data_m", default="../Datasets/TSB-AD-M")
    ap.add_argument("--out_dir", default="results/ablation")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--latent_metric", default="centroid",
                    choices=LATENT_METRIC_GRID,
                    help="Latent-space outlierness metric for a single-config --final "
                         "run (overridden by best_config.json / --config_json).")
    ap.add_argument("--latent_layer", type=int, default=-1,
                    help="Number of Transformer blocks applied before reading the "
                         "latent (negative = all; non-negative values are clamped "
                         "to at least one block).")
    ap.add_argument("--latent_stride", type=int, default=0,
                    help="Step between latent windows (0 = the default non-overlapping "
                         "tiling). Smaller strides embed each timestep from several "
                         "context alignments and average them (denoises the score, "
                         "more forward passes).")
    ap.add_argument("--sweep_latent_metric", default="",
                    help="Comma list of latent metrics to sweep on the U-Tuning split "
                         "(e.g. 'centroid,centroid_robust,mahalanobis_diag,mahalanobis'). "
                         "Empty = the full metric grid. One embedding pass per series "
                         "per stride feeds every metric.")
    ap.add_argument("--sweep_latent_stride", default="",
                    help="Optional comma list of latent strides to cross with the "
                         "metric sweep (e.g. '256,512'; empty = the default "
                         "non-overlapping tiling). Each stride costs one embedding "
                         "pass per series.")
    ap.add_argument("--workers", type=int, default=1,
                    help="Data-parallel worker processes, each with its own predictor "
                         "replica on --device. 1 = in-process serial. On a GPU much "
                         "larger than the model (e.g. an H200) try 4-8.")
    ap.add_argument("--skip_multivariate", "--univariate_only", action="store_true",
                    dest="skip_multivariate",
                    help="Restrict --final to the univariate TSB-AD-U Eva split; the M "
                         "data dir / file lists are never read.")
    ap.add_argument("--final", action="store_true",
                    help="Skip the sweep; run best_config.json (or the placeholder "
                         "DEFAULT_CONFIG) on the Eva split.")
    ap.add_argument("--config_json", default=None,
                    help="Explicit config JSON for --final (overrides best_config.json).")
    ap.add_argument("--use_default_config", action="store_true",
                    help="Force --final to use the placeholder DEFAULT_CONFIG even if "
                         "best_config.json exists.")
    args = ap.parse_args()

    # Catch a torch/driver mismatch now, before spawning workers, so a cuda run
    # can't silently degrade to CPU.
    _assert_cuda_usable(args.device)

    out_dir = Path(args.out_dir)

    # --final locks the latent config before the workers load their replicas
    # (latent_metric / latent_stride are run-wide worker settings).
    final_agg = DEFAULT_CONFIG["agg"]
    if args.final:
        cfg = _resolve_final_config(args, out_dir)
        args.latent_metric = cfg.get("latent_metric", args.latent_metric)
        args.latent_stride = cfg.get("latent_stride", args.latent_stride) or 0
        final_agg = cfg.get("agg", DEFAULT_CONFIG["agg"])

    latent_stride = args.latent_stride or None   # 0 -> None (default tiling)
    init_args = (args.checkpoint, args.batch_size, args.device, args.precision,
                 None, args.latent_metric, args.latent_layer, latent_stride)

    def _dispatch(executor):
        if args.final:
            run_final(executor, args, out_dir, Path(args.results_dir), final_agg)
        else:
            run_latent_sweep(executor, args, out_dir)

    if args.workers > 1 and args.device.startswith("cuda"):
        import multiprocessing as mp
        cpu = os.cpu_count() or 1
        threads = max(1, cpu // args.workers)
        # 'spawn' is required: forking a CUDA-initialized process is unsafe.
        ctx = mp.get_context("spawn")
        print(f"Launching {args.workers} workers on {args.device} "
              f"({threads} CPU thread(s) each) ...")
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=init_args[:4] + (threads,) + init_args[5:],
        ) as pool:
            _dispatch(_Executor(pool))
        return

    # --- Serial path: load one predictor in the main process. ---
    if args.workers > 1:
        print("--workers > 1 requested but --device is not CUDA; running serially.")
    _init_worker(*(init_args[:4] + (0,) + init_args[5:]))
    if args.device.startswith("cuda"):
        print("Tip: pass --workers 8 to data-parallelise across your GPU.")
    _dispatch(_Executor(pool=None))


if __name__ == "__main__":
    main()
