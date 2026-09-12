# Tabby zero-shot anomaly detection on TSB-AD

This directory contains only the Tabby-specific anomaly detector, checkpoint
adapter, experiment runner, tuned configuration, and released result tables.
The upstream TSB-AD implementation is an external dependency; it is not copied
into the Tabby repository.

## What the evaluation does

The frozen Tabby-Pretrain encoder produces one latent vector per patch. Each
patch is scored by its distance from the latent distribution of the same time
series, and patch scores are broadcast back to timesteps. Multivariate channel
scores are combined by either mean or max. There is no training or per-series
fine-tuning.

`run_tsb_ad.py` obtains `find_length_rank` and `get_metrics` from the official
TSB-AD package, so its VUS-PR, AUC-PR, and VUS-ROC calculations use the benchmark
implementation directly.

## Layout

| Path | Purpose |
| --- | --- |
| `detector.py` | Latent-distance detector and channel aggregation |
| `predictor.py` | Loads Tabby-Pretrain once and exposes patch embeddings |
| `run_tsb_ad.py` | U-Tuning sweep and U/M final evaluation |
| `configs/tabby_tsb_u.json` | Tuned configuration used for the released U result |
| `results/ablation/` | Per-series tuning results and tuning summary |
| `results/final_U.csv` | Released 350-series TSB-AD-U evaluation |

The predictor imports the canonical model from `src/tabby`; this benchmark does
not vendor another model implementation.

## Dependencies and data

Install Tabby from the repository root, then install the official TSB-AD package:

```bash
python -m pip install -e .
git clone https://github.com/TheDatumOrg/TSB-AD.git /path/to/TSB-AD
python -m pip install -e /path/to/TSB-AD
```

Download TSB-AD-U (and TSB-AD-M if needed) by following the upstream dataset
instructions. Keep the datasets outside this repository. The commands below
expect the official file lists and data directories under `/path/to/TSB-AD/Datasets`.
TSB-AD code is Apache-2.0 licensed; the datasets have their own terms and are not
redistributed here.

## Checkpoint

Pass the Tabby-Pretrain checkpoint with `d_model=768`. `--checkpoint` is resolved
by [`tabby.checkpoint`](../../src/tabby/checkpoint.py), so it accepts a Hugging
Face snapshot directory (`config.json` + `model.safetensors`), a
`pytorch_model.bin` file, the step directory holding one, a parent directory
whose `latest` entry points to that step, or a Hub repository id. Architecture
values are read from the checkpoint rather than supplied on the command line.

## Tune on TSB-AD-U

From the Tabby repository root:

```bash
python benchmarks/anomaly/run_tsb_ad.py \
  --checkpoint /path/to/step_0165000 \
  --dev_u /path/to/TSB-AD/Datasets/File_List/TSB-AD-U-Tuning.csv \
  --data_u /path/to/TSB-AD/Datasets/TSB-AD-U \
  --out_dir artifacts/tsb_ad/ablation \
  --device cuda:0 --workers 8
```

The default sweep evaluates all four latent metrics with non-overlapping context
windows. It writes per-series CSV files, a ranked summary, and
`best_config.json`.

## Reproduce the released TSB-AD-U result

Use the committed tuned configuration explicitly. Without `--config_json`, the
runner searches `--out_dir/best_config.json` and then falls back to its original
centroid placeholder; that fallback is not the released Mahalanobis setup.

```bash
python benchmarks/anomaly/run_tsb_ad.py \
  --checkpoint /path/to/step_0165000 \
  --eval_u /path/to/TSB-AD/Datasets/File_List/TSB-AD-U-Eva.csv \
  --data_u /path/to/TSB-AD/Datasets/TSB-AD-U \
  --results_dir artifacts/tsb_ad/results \
  --config_json benchmarks/anomaly/configs/tabby_tsb_u.json \
  --final --skip_multivariate \
  --device cuda:0 --workers 8
```

The committed `results/final_U.csv` contains 350 successful series and gives:

| Metric | Mean | Median |
| --- | ---: | ---: |
| VUS-PR | 0.4282 | 0.4024 |
| AUC-PR | 0.3408 | 0.2643 |
| VUS-ROC | 0.8364 | 0.9334 |

The runner preserves the original experiment behavior: a failing series is
recorded as a row with NaN metrics and an error string, while pandas computes
the displayed mean and median over non-NaN values. Inspect the `error` column
and successful-row count before treating a new aggregate as comparable.

## Multivariate track

Omit `--skip_multivariate` and additionally provide `--eval_m` and `--data_m` to
run TSB-AD-M. The released files supplied with Tabby contain a final U result but
no final M result.
