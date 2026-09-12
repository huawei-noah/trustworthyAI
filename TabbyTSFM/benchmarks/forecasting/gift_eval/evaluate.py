#!/usr/bin/env python3
"""
GIFT-Eval evaluation of a pretrained PatchTST-FM, zero-shot or prompt-tuned.

  --mode zs      bare backbone (prompt_len=0 through the same wrapper path)
  --mode prompt  load a train.py checkpoint (--ckpt)

Both modes share ONE predict path (PromptedPatchTSTFM.forward), so the
prompt-vs-zero-shot delta is never confounded by harness differences. Per-config
metrics are written to results/<out_csv>.csv, and the Seasonal-Naive normalised
geometric mean over the matched configs is printed at the end as the headline
number (see ``tabby.posttraining.sn_metrics``).

Example:
    python benchmarks/forecasting/gift_eval/evaluate.py \
        --mode prompt --ckpt runs/<run>/checkpoint_best.pt \
        --pretrain_ckpt /path/to/checkpoint/step_0165000 \
        --gift_eval_repo /path/to/gift-eval \
        --dataset_properties /path/to/dataset_properties.json \
        --context_length 8096 --datasets all --out_csv my_eval
"""
import sys, json, csv, argparse, logging
from pathlib import Path

# gluonts warns once per forecast that "mean prediction is not stored" - we emit
# QuantileForecast, which has no mean by construction. Over all 97 configs this
# repeats hundreds of thousands of times and bloats the log to tens of MB, so it
# is filtered out here (the reference harness does the same).
logging.getLogger("gluonts.model.forecast").addFilter(
    type("F", (logging.Filter,), {
        "filter": lambda self, r: "mean prediction is not stored" not in r.getMessage()
    })())

import numpy as np
import torch
from gluonts.itertools import batcher
from gluonts.model.forecast import QuantileForecast
from gluonts.model import evaluate_model
from gluonts.time_feature import get_seasonality
from gluonts.ev.metrics import (
    MSE, MAE, MASE, MAPE, SMAPE, MSIS, RMSE, NRMSE, ND, MeanWeightedSumQuantileLoss
)

from tabby.posttraining.model import (
    PatchTSTFMPromptCFG, PromptedPatchTSTFM, load_patchtstfm_backbone,
)

GIFT_EVAL_Q = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class PatchTSTFMGiftPredictor:
    def __init__(self, model, prediction_length, context_length, batch_size=256,
                 quantile_levels=GIFT_EVAL_Q, precision="bf16"):
        self.model = model
        self.prediction_length = prediction_length
        self.context_length = context_length
        # Long-history prompt: when set above context_length, the prompt generator sees
        # a wider right-aligned window while the backbone still consumes context_length.
        self.long_ctx_len = int(getattr(model.cfg, "long_ctx_len", 0) or 0)
        self.batch_size = batch_size
        self.quantile_levels = quantile_levels
        self.model_quantiles = np.asarray(model.quantile_levels, dtype=np.float64)
        self.use_amp = (precision == "bf16" and torch.cuda.is_available())

    def _qidx(self):
        idx = []
        for q in self.quantile_levels:
            i = int(np.argmin(np.abs(self.model_quantiles - q)))
            if abs(self.model_quantiles[i] - q) > 1e-6:
                raise ValueError(f"quantile {q} not on the model grid (nearest {self.model_quantiles[i]})")
            idx.append(i)
        return idx

    @torch.no_grad()
    def predict(self, test_data_input):
        device = self.model.device
        q_indices = self._qidx()
        bs = self.batch_size
        # Materialise the input as a list before any OOM/batch-size retry: if
        # test_data_input were a one-shot generator, the retry would read zero items
        # from an exhausted iterator and silently emit an empty forecast set.
        if not isinstance(test_data_input, (list, tuple)):
            test_data_input = list(test_data_input)
        n_expected = len(test_data_input)
        while True:
            try:
                all_forecasts = []
                for batch in batcher(test_data_input, batch_size=bs):
                    keep = max(self.context_length, self.long_ctx_len)
                    longs = []
                    for entry in batch:
                        t = np.asarray(entry["target"], dtype=np.float32)
                        if t.ndim > 1:
                            t = t.reshape(-1)
                        if len(t) > keep:
                            t = t[-keep:]
                        longs.append(t)
                    # The backbone window is the last context_length steps of the long
                    targets = [t[-self.context_length:] if len(t) > self.context_length else t
                               for t in longs]
                    max_len = max(len(t) for t in targets)
                    B = len(targets)
                    context = np.full((B, max_len), np.nan, dtype=np.float32)
                    # leading positions before each series' start are PADDING (as in
                    # predictor.py); interior NaNs stay "missing"
                    is_pad = np.zeros((B, max_len), dtype=np.bool_)
                    for i, t in enumerate(targets):
                        context[i, -len(t):] = t
                        is_pad[i, :max_len - len(t)] = True
                    context_t = torch.tensor(context, device=device)
                    context_mask = torch.isnan(context_t).logical_not().float()
                    pad_t = torch.tensor(is_pad, device=device)
                    lctx_t = lmask_t = None
                    if self.long_ctx_len > self.context_length:
                        lmax = max(len(t) for t in longs)
                        lctx = np.zeros((B, lmax), dtype=np.float32)
                        lmsk = np.zeros((B, lmax), dtype=np.float32)
                        for i, t in enumerate(longs):
                            lctx[i, lmax - len(t):] = np.nan_to_num(t, nan=0.0)
                            lmsk[i, lmax - len(t):] = np.isfinite(t).astype(np.float32)
                        lctx_t = torch.tensor(lctx, device=device)
                        lmask_t = torch.tensor(lmsk, device=device)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                        out = self.model(context=context_t, context_mask=context_mask,
                                         past_is_pad=pad_t,
                                         prediction_length=self.prediction_length,
                                         long_context=lctx_t, long_context_mask=lmask_t)
                    qp = out["quantile_preds"].float().cpu().numpy()   # [B, K, H]
                    qp = qp[:, q_indices, :self.prediction_length]
                    for i, entry in enumerate(batch):
                        all_forecasts.append(QuantileForecast(
                            item_id=entry.get("item_id", None),
                            forecast_arrays=qp[i],
                            start_date=entry["start"] + len(entry["target"]),
                            forecast_keys=list(map(str, self.quantile_levels)),
                        ))
                break
            except torch.cuda.OutOfMemoryError:
                if bs <= 1:
                    raise
                print(f"[OOM] batch_size {bs} -> {bs // 2}")
                bs = max(1, bs // 2); torch.cuda.empty_cache()
        # Count check: silently producing fewer forecasts would make the downstream
        if len(all_forecasts) != n_expected:
            raise RuntimeError(
                f"forecast count mismatch: got {len(all_forecasts)}, expected {n_expected} "
                "(likely an exhausted input after an OOM retry, or a dropped batch)")
        return all_forecasts


def build_model(args, device):
    backbone, model_cfg, step = load_patchtstfm_backbone(args.pretrain_ckpt, device)
    print(f"Backbone loaded: step={step}")

    if args.mode == "zs":
        cfg = PatchTSTFMPromptCFG(
            prompt_len=0, context_aware=False,
            prediction_length=96, context_length=args.context_length,
            norm_mode=args.norm_mode, min_forecast_span=args.min_forecast_span)
        model = PromptedPatchTSTFM(backbone, cfg).to(device)
        model.freeze_base(); model.eval()
        return model, cfg, "patchtstfm_zs"

    if not args.ckpt:
        raise SystemExit("--mode prompt requires --ckpt")
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
        long_ctx_len=int(c.get("long_ctx_len", 0) or 0),
        long_ctx_mode=c.get("long_ctx_mode", "add"),
        long_max_segments=int(c.get("long_max_segments", 0) or 0),
        # Normalisation mode travels with the checkpoint (written into its config at
        norm_mode=c.get("norm_mode", "inhouse_asinh"),
        min_forecast_span=int(c.get("min_forecast_span", 0)),
    )
    if int(cfg.context_length) != int(args.context_length):
        print(f"[WARN] --context_length={args.context_length} != training "
              f"context_length={cfg.context_length}; segments anchor to the training "
              f"value — pass --context_length {cfg.context_length} unless intentional.")
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
    return model, cfg, "patchtstfm_prompted"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="prompt", choices=["zs", "prompt"])
    p.add_argument("--ckpt", type=str, default=None, help="prompt checkpoint (mode=prompt)")
    p.add_argument("--pretrain_ckpt", type=str, required=True,
                   help="in-house step dir/.bin, or an HF snapshot dir (config.json+model.safetensors, r1)")
    # Zero-shot knobs only; in prompt mode these are read from the checkpoint config.
    p.add_argument("--norm_mode", type=str, default="inhouse_asinh",
                   choices=["inhouse_asinh", "revin_official"])
    p.add_argument("--min_forecast_span", type=int, default=0)
    p.add_argument("--gift_eval_repo", type=str, required=True)
    p.add_argument("--dataset_properties", type=str, required=True)
    p.add_argument("--context_length", type=int, default=4096,
                   help="history cap Tc (model window is fixed 8192; [pad|hist|future])")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--datasets", type=str, default="all")
    p.add_argument("--out_csv", type=str, default="patchtstfm_prompted")
    args = p.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else args.device if args.device != "auto" else "cpu")

    sys.path.append(args.gift_eval_repo)
    from src.gift_eval.data import Dataset
    dataset_properties_map = json.load(open(args.dataset_properties))

    model, cfg, model_name = build_model(args, device)

    # smoke: one dummy forward so API breakage dies here, not mid-eval
    with torch.no_grad():
        dummy = torch.randn(2, 128, device=device)
        out = model(context=dummy, prediction_length=8)
        assert out["quantile_preds"].shape == (2, model.num_quantiles, 8), \
            f"smoke shape {out['quantile_preds'].shape}"
    print("[smoke] dummy predict OK")

    # ---- dataset lists (verbatim from test_chronos2_prompt_d.py) ----
    electricity_datasets = "electricity/15T electricity/H electricity/D electricity/W"
    m4_datasets = "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly"
    solar_data = "solar/10T solar/H solar/D solar/W"
    kdd_data = "kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D"
    other_data = "bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_l2c/5T bizitobs_l2c/H bizitobs_service car_parts_with_missing covid_deaths ett1/15T ett1/D ett1/H ett1/W ett2/15T ett2/D ett2/H ett2/W hierarchical_sales/D hierarchical_sales/W hospital jena_weather/10T jena_weather/D jena_weather/H LOOP_SEATTLE/5T LOOP_SEATTLE/D LOOP_SEATTLE/H M_DENSE/D M_DENSE/H restaurant saugeenday/D saugeenday/M saugeenday/W SZ_TAXI/15T SZ_TAXI/H temperature_rain_with_missing us_births/D us_births/M us_births/W"
    long_ctx_data = "LOOP_SEATTLE/5T LOOP_SEATTLE/H bitbrains_rnd/5T bizitobs_l2c/5T bizitobs_application solar/10T solar/H SZ_TAXI/15T us_births/D saugeenday/D M_DENSE/H jena_weather/H jena_weather/10T ett2/15T bizitobs_service ett2/H ett1/15T ett1/H electricity/H electricity/15T kdd_cup_2018_with_missing/H bitbrains_fast_storage/5T"
    short_ctx_data = "electricity/D electricity/W m4_monthly m4_quarterly m4_yearly m4_daily m4_hourly m4_weekly solar/D solar/W kdd_cup_2018_with_missing/D bitbrains_fast_storage/H bitbrains_rnd/H bizitobs_l2c/H car_parts_with_missing covid_deaths ett1/D ett1/W ett2/D ett2/W hierarchical_sales/D hierarchical_sales/W hospital LOOP_SEATTLE/D jena_weather/D M_DENSE/D restaurant saugeenday/M saugeenday/W SZ_TAXI/H temperature_rain_with_missing us_births/M us_births/W"
    med_long_datasets = "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
    no_long_medium = ["bitbrains_fast_storage/H", "bitbrains_rnd/H", "car_parts_with_missing", "covid_deaths", "ett1/D", "ett1/W", "ett2/D", "ett2/W", "hierarchical_sales/D", "hierarchical_sales/W", "hospital", "jena_weather/D", "LOOP_SEATTLE/D", "M_DENSE/D", "restaurant", "saugeenday/D", "saugeenday/M", "saugeenday/W", "SZ_TAXI/H", "temperature_rain_with_missing", "us_births/D", "us_births/M", "us_births/W", "solar/D", "solar/W", "electricity/D", "electricity/W", "kdd_cup_2018_with_missing/D", "m4_weekly", "m4_monthly", "m4_daily", "m4_yearly", "m4_quarterly", "m4_hourly"]
    pretty_names = {"saugeenday": "saugeen", "temperature_rain_with_missing": "temperature_rain", "kdd_cup_2018_with_missing": "kdd_cup_2018", "car_parts_with_missing": "car_parts"}

    ds_map = {
        "all": list(set(electricity_datasets.split() + m4_datasets.split() + solar_data.split() + kdd_data.split() + other_data.split())),
        "long": list(set(long_ctx_data.split())),
        "short": list(set(short_ctx_data.split())),
    }
    all_datasets = ds_map.get(args.datasets, args.datasets.split())

    metrics = [MSE(forecast_type="mean"), MSE(forecast_type=0.5), MAE(), MASE(),
               MAPE(), SMAPE(), MSIS(), RMSE(), NRMSE(), ND(),
               MeanWeightedSumQuantileLoss(quantile_levels=list(GIFT_EVAL_Q))]

    out_csv = Path("results/", f"{args.out_csv}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerow([
            "dataset_config", "model", "context_length",
            "MSE[mean]", "MSE[0.5]", "MAE[0.5]", "MASE[0.5]", "MAPE[0.5]", "sMAPE[0.5]",
            "MSIS", "RMSE[mean]", "NRMSE[mean]", "ND[0.5]", "mean_weighted_sum_quantile_loss",
            "domain", "num_variates"])

    for ds_name in sorted(all_datasets):
        for term in ["short", "medium", "long"]:
            if term in ["long", "medium"] and ds_name in no_long_medium:
                continue
            if (term in ["medium", "long"]) and (ds_name not in med_long_datasets.split()) and (args.datasets not in ["long", "short"]):
                continue
            if "/" in ds_name:
                ds_key = ds_name.split("/")[0].lower(); ds_freq = ds_name.split("/")[1]
            else:
                ds_key = ds_name.lower()
                ds_freq = dataset_properties_map[pretty_names.get(ds_key, ds_key)]["frequency"]
            ds_key_pretty = pretty_names.get(ds_key, ds_key)
            ds_config = f"{ds_key_pretty}/{ds_freq}/{term}"
            try:
                td = Dataset(name=ds_name, term=term, to_univariate=False).target_dim
                dataset = Dataset(name=ds_name, term=term, to_univariate=(td != 1))
            except Exception as e:
                print(f"  [SKIP] {ds_name}/{term}: {e}")
                continue
            predictor = PatchTSTFMGiftPredictor(
                model=model, prediction_length=dataset.prediction_length,
                context_length=args.context_length, batch_size=args.batch_size,
                quantile_levels=GIFT_EVAL_Q, precision=args.precision)
            try:
                res = evaluate_model(predictor, test_data=dataset.test_data, metrics=metrics,
                                     batch_size=args.batch_size, axis=None,
                                     mask_invalid_label=True, allow_nan_forecast=False,
                                     seasonality=get_seasonality(dataset.freq))
            except Exception as e:
                print(f"  [ERROR] {ds_config}: {e}")
                import traceback; traceback.print_exc()
                continue
            with open(out_csv, "a", newline="") as f:
                csv.writer(f).writerow([
                    ds_config, model_name, args.context_length,
                    res["MSE[mean]"].iloc[0], res["MSE[0.5]"].iloc[0], res["MAE[0.5]"].iloc[0],
                    res["MASE[0.5]"].iloc[0], res["MAPE[0.5]"].iloc[0], res["sMAPE[0.5]"].iloc[0],
                    res["MSIS"].iloc[0], res["RMSE[mean]"].iloc[0], res["NRMSE[mean]"].iloc[0],
                    res["ND[0.5]"].iloc[0], res["mean_weighted_sum_quantile_loss"].iloc[0],
                    dataset_properties_map[ds_key_pretty]["domain"],
                    dataset_properties_map[ds_key_pretty]["num_variates"]])
            print(f"  [OK] {ds_config}: MASE={res['MASE[0.5]'].iloc[0]:.4f} "
                  f"CRPS={res['mean_weighted_sum_quantile_loss'].iloc[0]:.4f}")

    # headline: SN-normalized geometric mean over matched configs -> lands in .train.log
    try:
        from tabby.posttraining.sn_metrics import read_metrics
        m, c, n = read_metrics(str(out_csv))
        if m is not None:
            print(f"\n[SN-agg] geo-mean MASE={m:.4f} CRPS={(c if c else float('nan')):.4f} "
                  f"over {n} configs ({model_name}, ctx {args.context_length})")
    except Exception as e:
        print(f"[SN-agg] skipped: {e}")

    print(f"\nDone. Results: {out_csv}")


if __name__ == "__main__":
    main()
