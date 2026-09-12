#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate PatchTST-FM (zero-shot or prompt-tuned) on the TIME benchmark.

This is the out-of-pool counterpart of evaluate.py: TIME's tasks are disjoint from
the GIFT-Eval training pool, so it measures whether a prompt generalises beyond
the data it was tuned on rather than fitting it.

Structure:
  * data iteration / windowing / result files come from the benchmark's own
    `timebench` package, so its `compute_local_leaderboard` aggregates the output
    directly;
  * model construction and the forward pass are the same as in evaluate.py: BOTH
    modes go through the PromptedPatchTSTFM wrapper (zero-shot = prompt_len 0,
    which is bit-for-bit the bare adapter forward). Sharing one predict path is
    what makes the prompt-vs-zero-shot delta meaningful.

Protocol notes:
  * each variate is forecast independently, so absolute values are not aligned
    with the public leaderboard; the comparison of interest is zero-shot vs
    prompted under identical settings;
  * TIME inputs are intentionally right-truncated to ``--context_length`` before
    prediction. The release protocol uses the default 4000 in both zero-shot and
    prompt modes, even when the prompt checkpoint was trained with a wider
    configured context;
  * quantiles are TIME's official nine; the Seasonal-Naive normalised aggregate
    is computed by the benchmark's leaderboard step.

Requires the TIME benchmark repository, which is NOT included here:
    export TIME_REPO=/path/to/TIME        # the checkout containing src/timebench

Usage:
  zero-shot:
    TIME_REPO=... python benchmarks/forecasting/time/evaluate.py \
        --mode zs --pretrain_ckpt <dir> --context_length 4000 \
        --time_data <dir> --datasets all_datasets \
        --output_dir time_output/zs
  prompted:
    TIME_REPO=... python benchmarks/forecasting/time/evaluate.py \
        --mode prompt --ckpt <prompt.pt> --pretrain_ckpt <dir> \
        --context_length 4000 --time_data <dir> --datasets all_datasets \
        --output_dir time_output/prompted
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# The TIME benchmark package (`timebench`) is not vendored here: it belongs to the
# benchmark authors. Point TIME_REPO at a checkout containing src/timebench, or put
# that directory on PYTHONPATH yourself.
_TIME_REPO = os.environ.get("TIME_REPO", "")
if _TIME_REPO:
    _src = Path(_TIME_REPO).expanduser().resolve() / "src"
    if not (_src / "timebench").is_dir():
        sys.exit(f"[ABORT] TIME_REPO={_TIME_REPO} does not contain src/timebench")
    sys.path.insert(0, str(_src))

from gluonts.time_feature import get_seasonality
try:
    from timebench.evaluation.saver import save_window_predictions
    from timebench.evaluation.utils import get_available_terms
    from timebench.evaluation.data import Dataset, get_dataset_settings, load_dataset_config
except ImportError as e:
    sys.exit(f"[ABORT] cannot import the TIME benchmark package: {e}\n"
             f"        Set TIME_REPO=/path/to/TIME (the checkout containing src/timebench).")

Q_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def visible_ctx(t: np.ndarray, ctx_len: int) -> np.ndarray:
    t = np.asarray(t, dtype=np.float32)
    if t.ndim > 1:
        t = t.reshape(-1)          # defensive; variates are split into rows upstream
    if len(t) > ctx_len:
        t = t[-ctx_len:]
    return t


def build_model(args, device):
    """Same as evaluate.py's build_model, duplicated rather than imported so this
    file does not pull in the GIFT-Eval dependencies."""
    from tabby.posttraining.model import (
        PatchTSTFMPromptCFG, PromptedPatchTSTFM, load_patchtstfm_backbone,
    )
    backbone, model_cfg, step = load_patchtstfm_backbone(args.pretrain_ckpt, device)
    print(f"Backbone loaded: step={step}")

    if args.mode == "zs":
        cfg = PatchTSTFMPromptCFG(
            prompt_len=0, context_aware=False,
            prediction_length=96, context_length=args.context_length,
            norm_mode=args.norm_mode, min_forecast_span=args.min_forecast_span)
        model = PromptedPatchTSTFM(backbone, cfg).to(device)
        model.freeze_base()
        model.eval()
        return model

    if not args.ckpt:
        sys.exit("[ABORT] --mode prompt requires --ckpt")
    print(f"Loading prompt checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    c = ckpt.get("config", {})
    cfg = PatchTSTFMPromptCFG(
        prompt_len=int(c.get("prompt_len", 10)),
        prompt_init_mode=c.get("prompt_init_mode", "anchor_delta"),
        prompt_gate_init=float(c.get("prompt_gate_init", -3.0)),
        context_aware=bool(c.get("context_aware", False)),
        n_stats=int(c.get("n_stats", 10)),
        ctx_rank=int(c.get("ctx_rank", 4)),
        ctx_hidden=int(c.get("ctx_hidden", 32)),
        ctx_mlp_layers=int(c.get("ctx_mlp_layers", 2)),
        ctx_gate_init=float(c.get("ctx_gate_init", -3.0)),
        segmented=bool(c.get("segmented", False)),
        max_segments=int(c.get("max_segments", 16)),
        seg_attn_dim=int(c.get("seg_attn_dim", 64)),
        seg_attn_heads=int(c.get("seg_attn_heads", 4)),
        query_from_base=bool(c.get("query_from_base", True)),
        seg_mode=c.get("seg_mode", "fixed_len"),
        seg_fixed_len=int(c.get("seg_fixed_len", 256)),
        min_seg_len=int(c.get("min_seg_len", 48)),
        seg_relative_pos=bool(c.get("seg_relative_pos", True)),
        prediction_length=int(c.get("prediction_length", 96)),
        season_period=int(c.get("season_period", 1)),
        context_length=int(c.get("context_length", 4096)),
        norm_mode=c.get("norm_mode", "inhouse_asinh"),
        min_forecast_span=int(c.get("min_forecast_span", 0)),
    )
    # Keep the prompt architecture anchored to the training configuration. The
    # TIME input window is independently capped by ``args.context_length`` below;
    # the 4000-point release protocol deliberately does not emit a mismatch warning.
    model = PromptedPatchTSTFM(backbone, cfg).to(device)
    model.freeze_base()
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    real_missing = [k for k in missing if not k.startswith("backbone.")]
    if real_missing:
        raise RuntimeError(f"Prompt checkpoint missing keys: {real_missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")
    print(f"Loaded: epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss', 'N/A')}")
    model.eval()
    return model


def make_qidx(model):
    mq = np.asarray(model.quantile_levels, dtype=np.float64)
    idx = []
    for q in Q_LEVELS:
        i = int(np.argmin(np.abs(mq - q)))
        if abs(mq[i] - q) > 1e-6:
            raise ValueError(f"quantile {q} is not on the model grid (nearest {mq[i]})")
        idx.append(i)
    return idx


@torch.no_grad()
def predict_rows(model, contexts, pred_len, q_indices, use_amp):
    """contexts: list of 1-D arrays -> (N, 9, pred_len). NaN padding and the leading
    padding flag follow the same convention as the GIFT-Eval predictor."""
    device = model.device
    max_len = max(len(t) for t in contexts)
    B = len(contexts)
    ctx = np.full((B, max_len), np.nan, dtype=np.float32)
    is_pad = np.zeros((B, max_len), dtype=np.bool_)
    for i, t in enumerate(contexts):
        ctx[i, -len(t):] = t
        is_pad[i, :max_len - len(t)] = True
    ctx_t = torch.tensor(ctx, device=device)
    mask = torch.isnan(ctx_t).logical_not().float()
    pad_t = torch.tensor(is_pad, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        out = model(context=ctx_t, context_mask=mask, past_is_pad=pad_t,
                    prediction_length=pred_len)
    qp = out["quantile_preds"].float().cpu().numpy()          # [B, K, H]
    return qp[:, q_indices, :pred_len]                        # (N, 9, pred)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["zs", "prompt"], required=True)
    p.add_argument("--ckpt", type=str, default=None, help="required for --mode prompt")
    p.add_argument("--pretrain_ckpt", type=str, required=True,
                   help="PatchTST-FM backbone checkpoint directory (step_XXXXXX)")
    p.add_argument("--norm_mode", type=str, default="inhouse_asinh",
                   help="zero-shot only; in prompt mode this is read from the checkpoint config")
    p.add_argument("--min_forecast_span", type=int, default=0, help="zero-shot only")
    p.add_argument("--time_data", type=str, required=True)
    p.add_argument("--datasets", type=str, nargs="+", required=True,
                   help="dataset names, or all_datasets")
    p.add_argument("--terms", type=str, nargs="+", default=None,
                   choices=["short", "medium", "long"])
    p.add_argument("--context_length", type=int, default=4000,
                   help="visible TIME history cap; rows are silently right-truncated "
                        "to this length (release default: 4000)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = (args.precision == "bf16" and torch.cuda.is_available())

    config = load_dataset_config()
    all_names = list(config.get("datasets", {}).keys())
    names = all_names if args.datasets == ["all_datasets"] else args.datasets

    model = build_model(args, device)
    q_indices = make_qidx(model)

    os.makedirs(args.output_dir, exist_ok=True)
    tag = f"ptfm_{args.mode}{args.context_length}"

    for name in names:
        terms = args.terms or get_available_terms(name, config)
        for term in terms:
            settings = get_dataset_settings(name, term, config)
            pred_len = settings.get("prediction_length")
            ds = Dataset(name=name, term=term, to_univariate=False,
                         prediction_length=pred_len,
                         test_length=settings.get("test_length"),
                         val_length=settings.get("val_length"),
                         storage_path=args.time_data)
            season = get_seasonality(ds.freq)
            inputs = list(ds.test_data.input)
            print(f"[{tag}] {name}/{term}: {len(inputs)} instances, pred={pred_len}, freq={ds.freq}")

            rows, counts = [], []
            for d in inputs:
                t = np.asarray(d["target"], dtype=np.float32)
                if t.ndim == 1:
                    t = t[np.newaxis, :]
                counts.append(t.shape[0])
                for v in range(t.shape[0]):
                    rows.append(visible_ctx(t[v], args.context_length))

            preds = []
            bs = args.batch_size
            i = 0
            while i < len(rows):
                chunk = rows[i:i + bs]
                try:
                    preds.append(predict_rows(model, chunk, pred_len, q_indices, use_amp))
                    i += bs
                except torch.cuda.OutOfMemoryError:
                    if bs <= 1:
                        raise
                    bs = max(1, bs // 2)
                    torch.cuda.empty_cache()
                    print(f"  [OOM] batch_size -> {bs}")
            preds = np.concatenate(preds, axis=0)             # (N_rows, 9, pred)
            if preds.shape[0] != len(rows):
                raise RuntimeError(f"forecast row count mismatch: {preds.shape[0]} vs {len(rows)}")

            fc, idx = [], 0
            for v in counts:
                item = preds[idx:idx + v]                      # (V, 9, pred)
                fc.append(np.transpose(item, (1, 0, 2))[np.newaxis, ...])  # (1, 9, V, pred)
                idx += v
            fc_quantiles = np.concatenate(fc, axis=0)          # (B, 9, V, pred)

            meta = save_window_predictions(
                dataset=ds,
                fc_quantiles=fc_quantiles,
                ds_config=f"{name}/{term}",
                output_base_dir=args.output_dir,
                seasonality=season,
                model_hyperparams={"model": tag, "context_length": args.context_length,
                                   "mode": args.mode, "ckpt": args.ckpt,
                                   "quantile_levels": Q_LEVELS, "per_variate_rows": True},
                quantile_levels=Q_LEVELS,
            )
            print(f"  saved: {meta['num_series']} series × {meta['num_windows']} windows")

    print(f"[DONE] {tag} → {args.output_dir}")


if __name__ == "__main__":
    main()
