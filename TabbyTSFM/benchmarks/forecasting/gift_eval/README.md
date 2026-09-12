# GIFT-Eval forecasting

`evaluate.py` evaluates either the frozen Tabby-Pretrain backbone (`--mode zs`)
or a prompt/post-trained Tabby checkpoint (`--mode prompt`) through the same
prediction wrapper. The GIFT-Eval repository and datasets remain external.

Install this repository as a package, clone GIFT-Eval separately, and pass its
root through `--gift_eval_repo`. `dataset_properties.json` is the dataset
metadata used by the original evaluation script. Point `GIFT_EVAL` at the
downloaded dataset directory, which is what GIFT-Eval's own `Dataset` class
reads.

`--pretrain_ckpt` is resolved by [`tabby.checkpoint`](../../../src/tabby/checkpoint.py):
a Hugging Face snapshot directory (`config.json` + `model.safetensors`), a
`pytorch_model.bin`, its step directory, a `latest` pointer, or a Hub repository
id. `--ckpt` is separate and always a prompt/post-training `.pt` checkpoint.

Zero-shot, straight off a downloaded snapshot:

```bash
GIFT_EVAL=/path/to/gift-eval/data \
python benchmarks/forecasting/gift_eval/evaluate.py \
  --mode zs \
  --pretrain_ckpt /path/to/Tabby-Pretrain \
  --gift_eval_repo /path/to/gift-eval \
  --dataset_properties benchmarks/forecasting/gift_eval/dataset_properties.json \
  --context_length 4000 \
  --datasets all \
  --out_csv tabby_zeroshot
```

Prompt/post-trained, from an in-house step directory:

```bash
GIFT_EVAL=/path/to/gift-eval/data \
python benchmarks/forecasting/gift_eval/evaluate.py \
  --mode prompt \
  --ckpt /path/to/checkpoint_best.pt \
  --pretrain_ckpt /path/to/step_0165000 \
  --gift_eval_repo /path/to/gift-eval \
  --dataset_properties benchmarks/forecasting/gift_eval/dataset_properties.json \
  --context_length 8096 \
  --datasets all \
  --out_csv tabby_posttrained
```

Pass a space-separated list to `--datasets` (for example `"us_births/D
hospital"`) to evaluate a subset. The evaluator retains the original
per-configuration error handling and writes one CSV row per successful
configuration, so a run that exits cleanly with an empty CSV means every
configuration was skipped.

[`tests/test_gift_eval_evaluation.py`](../../../tests/test_gift_eval_evaluation.py)
covers `build_model` loading a `model.safetensors` snapshot with no benchmark
data required, and drives this CLI end to end over one real configuration when
`GIFT_EVAL` and `TABBY_TEST_GIFT_EVAL_REPO` are set.
