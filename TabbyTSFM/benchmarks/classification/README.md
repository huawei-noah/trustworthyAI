# UCR classification

`ucr_rf_layer_fusion.py` evaluates the frozen Tabby-Pretrain encoder on the
official UCR TRAIN/TEST splits. It preserves the original protocol: interpolate
each series, pool selected Transformer layers, train a random forest only on
TRAIN, and evaluate once on TEST.

Install the package from the repository root before running the script. The
paper configuration uses `--resize_length 1024`, `--layers all`,
`--layer_fusion concat`, and `--n_estimators 400`. The CLI default resize length
remains 512 for compatibility with the original script; pass 1024 explicitly to
reproduce the paper setting.

`--checkpoint` is resolved by [`tabby.checkpoint`](../../src/tabby/checkpoint.py)
and accepts a Hugging Face snapshot directory (`config.json` +
`model.safetensors`) as well as the in-house `pytorch_model.bin` layouts.

```bash
python benchmarks/classification/ucr_rf_layer_fusion.py \
  --checkpoint /path/to/step_0165000 \
  --ucr_root /path/to/UCRArchive_2018 \
  --output_dir results/ucr \
  --resize_length 1024 \
  --layers all \
  --layer_fusion concat \
  --n_estimators 400
```

The UCR archive is an external dataset and is not bundled with this repository.
