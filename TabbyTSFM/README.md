# Tabby

Tabby is a time series foundation model with a frozen pretrained backbone,
prompt (post-training), and evaluation code for forecasting, classification, and
anomaly detection.

## Repository layout

```text
src/tabby/
  models/             canonical Tabby-Pretrain backbone
  data/               GIFT-Eval-Pretrain, BLAST, KernelSynth readers, and CauKer V2
  utils/              shared preprocessing
  predictor.py        GluonTS-compatible base predictor
  posttraining/       prompt model, data split, and metrics
recipes/
  pretrain/            distributed pretraining entry point
  posttrain/           prompt/post-training entry point and smoke test
benchmarks/
  forecasting/         GIFT-Eval and TIME prediction evaluation
  classification/      UCR frozen-representation evaluation
  anomaly/             TSB-AD zero-shot latent evaluation
tests/                 source, release-contract and checkpoint-loading checks
```

Classification and anomaly detection both import the same canonical
`tabby.models.PatchTSTFM` implementation; no second model copy is maintained in
either benchmark.

## Installation

```bash
python -m pip install -e .
```

Install benchmark-specific dependencies only when needed:

```bash
python -m pip install -e ".[train]"
python -m pip install -e ".[forecast]"
python -m pip install -e ".[classification]"
python -m pip install -e ".[anomaly]"
```

The repository does not bundle checkpoints or benchmark datasets. Pass local
checkpoint/data paths to each recipe.

Every checkpoint argument in the repository is resolved by
[`tabby.checkpoint`](src/tabby/checkpoint.py), so the recipes, the predictor and
all four benchmarks accept the same inputs:

| Input | Layout |
| --- | --- |
| Hugging Face snapshot directory | `config.json` + `model.safetensors` — the published Tabby-Pretrain weights, a `huggingface-cli download` target, or anything `save_pretrained` wrote |
| Training checkpoint | a `pytorch_model.bin` file, the `step_XXXXXXX` directory holding one, or a run directory with a `latest` symlink |
| Hub repository id | an `org/name` string with no local path of that name, downloaded via `huggingface_hub` |

A snapshot's tensor names already match the bundled architecture, so no key
translation happens on load; the HF config vocabulary (`d_patch`, `n_layer`,
`num_quantile`, ...) is mapped onto the trainer-facing dataclass fields, and any
missing or unexpected tensor is an error rather than a silent partial load. The
training step is read from the snapshot's safetensors metadata when present.

Copy `.env.example` to `.env` for the local dataset paths used by post-training
and TIME. The pretraining shell recipe reads its path and mixture variables from
the environment.

## Python forecasting API

Install the forecasting extra and pass the downloaded Tabby-Pretrain checkpoint —
either a Hugging Face snapshot directory or an in-house step directory:

```python
import numpy as np
import pandas as pd

from tabby.predictor import PatchTSTFMPredictor

predictor = PatchTSTFMPredictor(
    checkpoint="/path/to/checkpoint",
    prediction_length=24,
    device="cuda:0",
)
items = [{
    "start": pd.Period("2024-01-01", freq="h"),
    "target": np.asarray([1.0, 1.2, 1.1, 1.4], dtype=np.float32),
}]
forecasts = predictor.predict(items)
```

## Benchmark data

Neither benchmark is vendored here: each has its own repository, its own dataset
on the Hugging Face Hub, and its own licence. Clone the repository and download
the data once, then point the evaluation entry points at both.

> The benchmarks' own READMEs use `huggingface-cli download`. That command is
> deprecated and fails outright on `huggingface_hub >= 1.0`; use `hf download`.

### GIFT-Eval

```bash
git clone https://github.com/SalesforceAIResearch/gift-eval.git
hf download Salesforce/GiftEval --repo-type=dataset --local-dir /path/to/gifteval_data
```

The download is ~1.5 GB across 28 base datasets, each holding one directory per
frequency. `GIFT_EVAL` must point at that directory — GIFT-Eval's own `Dataset`
class reads it — and `--gift_eval_repo` at the clone.

### TIME

```bash
git clone https://github.com/zqiao11/TIME.git
hf download Real-TSF/TIME --repo-type=dataset --local-dir /path/to/timebench_data
```

The code and the data live in different places: the benchmark is the `zqiao11`
GitHub repository, the datasets are the `Real-TSF` Hub organisation. The download
is ~200 MB across 39 base datasets. The clone must contain
`src/timebench`; `--time_data` points at the data and `TIME_REPO` at the clone,
which is how `evaluate.py` puts `timebench` on the import path.

Both paths can also live in `.env` (see `.env.example`).

## Running the benchmarks

Both entry points take `--mode zs` for the bare backbone or `--mode prompt`
with a prompt/post-training checkpoint. The examples below are zero-shot.

### GIFT-Eval

```bash
GIFT_EVAL=/path/to/gifteval_data \
python benchmarks/forecasting/gift_eval/evaluate.py \
  --mode zs \
  --pretrain_ckpt /path/to/checkpoint \
  --gift_eval_repo /path/to/gift-eval \
  --dataset_properties benchmarks/forecasting/gift_eval/dataset_properties.json \
  --datasets all \
  --context_length 4000 --batch_size 128 \
  --device cuda --precision bf16 \
  --out_csv tabby_zeroshot
```

Writes one row per configuration to `results/tabby_zeroshot.csv` as it goes, and
prints the headline figure itself when it finishes:

```text
[SN-agg] geo-mean MASE=... CRPS=... over N configs
```

That aggregate is already normalised by Seasonal Naive, using the reference
values in [`tabby.posttraining.sn_metrics`](src/tabby/posttraining/sn_metrics.py).
Pass a space-separated list to `--datasets` (`"us_births/D hospital"`) to run a
subset. A run that exits cleanly with an empty CSV means every configuration was
skipped, not that everything passed.

### TIME

```bash
TIME_REPO=/path/to/TIME \
python benchmarks/forecasting/time/evaluate.py \
  --mode zs \
  --pretrain_ckpt /path/to/checkpoint \
  --time_data /path/to/timebench_data \
  --datasets all_datasets \
  --context_length 4000 --batch_size 64 \
  --device cuda --precision bf16 \
  --output_dir results/tabby_time
```

Writes `metrics.npz`, `predictions.npz` and `config.json` per configuration
under `results/tabby_time/<dataset>/<freq>/<term>/`. TIME dataset names carry a
frequency suffix (`Oil_Price/B`, not `Oil_Price/D`); add `--terms short` to
restrict the horizons.

Batch size is the knob to watch: one series costs roughly 21 MiB of activations
at the 8192-point window, on top of ~0.6 GB of weights, so a 128-series batch
needs about 3.2 GB. Both entry points halve the batch and retry on
out-of-memory, so an oversized `--batch_size` costs throughput rather than the
run.

### The TIME overall leaderboard

TIME scores a model by normalising it against Seasonal Naive per configuration.
Its `compute_local_leaderboard.py` reads every model under `output/results/` and
downloads the official Seasonal Naive results from the Hub, so the only step
needed is to place a run there under the name it should appear as:

```bash
cd /path/to/TIME
mkdir -p output/results
cp -r /path/to/TabbyTSFM/results/tabby_time output/results/tabby_pretrain
python scripts/compute_local_leaderboard.py
```
The layout under `output/results/` has to
be `<model>/<dataset>/<freq>/<term>/metrics.npz`, which is exactly what
`--output_dir` produces; only the model-name level is added by the copy.

## Tests

```bash
python -m pip install -e ".[dev,forecast]"
python -m pytest tests
```

| File | Covers |
| --- | --- |
| `test_release_structure.py` | release contracts; dependency-free |
| `test_hf_checkpoint_loading.py` | checkpoint loading across every accepted layout |
| `test_gift_eval_evaluation.py` | the GIFT-Eval script loading a `model.safetensors` snapshot |

The checkpoint tests run against a real Hugging Face snapshot that
[`tests/make_tiny_hf_snapshot.py`](tests/make_tiny_hf_snapshot.py) generates at
run time (~35k parameters, same layout as the release weights); run that script
standalone to materialize one for inspection.

Two tests are opt-in because they need artifacts this repository does not ship.
Point `TABBY_TEST_CHECKPOINT` at a snapshot directory to additionally load real
weights, and set the GIFT-Eval variables to drive the evaluation CLI end to end
over one real configuration:

```bash
TABBY_TEST_CHECKPOINT=/path/to/Tabby-Pretrain \
GIFT_EVAL=/path/to/gift-eval/data \
TABBY_TEST_GIFT_EVAL_REPO=/path/to/gift-eval \
  python -m pytest tests
```

The CLI test runs on the generated tiny snapshot, not release weights: it checks
that the script completes and writes finite metrics, not that the metrics are
good. Override the configuration it uses with `TABBY_TEST_GIFT_EVAL_DATASET`.

## Entry points

- Pretraining: [`recipes/pretrain/README.md`](recipes/pretrain/README.md)
- Prompt/post-training: [`recipes/posttrain/README.md`](recipes/posttrain/README.md)
- GIFT-Eval: [`benchmarks/forecasting/gift_eval/README.md`](benchmarks/forecasting/gift_eval/README.md)
- TIME: [`benchmarks/forecasting/time/README.md`](benchmarks/forecasting/time/README.md)
- UCR: [`benchmarks/classification/README.md`](benchmarks/classification/README.md)
- TSB-AD: [`benchmarks/anomaly/README.md`](benchmarks/anomaly/README.md)

