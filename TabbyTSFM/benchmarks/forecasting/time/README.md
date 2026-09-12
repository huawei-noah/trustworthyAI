# TIME forecasting

`evaluate.py` runs Tabby-Pretrain (`--mode zs`) or prompt/post-trained Tabby
(`--mode prompt`) through the TIME benchmark's own data and result writer.
The TIME repository and datasets are external and are not vendored here.

The release protocol intentionally caps every visible input row at the last
4000 observations. This cap is independent of the prompt checkpoint's training
context and is applied silently in both modes.

`--pretrain_ckpt` is resolved by [`tabby.checkpoint`](../../../src/tabby/checkpoint.py)
and accepts a Hugging Face snapshot directory (`config.json` +
`model.safetensors`) as well as the in-house `pytorch_model.bin` layouts.
`--ckpt` is separate and always a prompt/post-training `.pt` checkpoint.

```bash
TIME_REPO=/path/to/TIME \
python benchmarks/forecasting/time/evaluate.py \
  --mode prompt \
  --ckpt /path/to/checkpoint_best.pt \
  --pretrain_ckpt /path/to/step_0165000 \
  --context_length 4000 \
  --time_data /path/to/TIME/data \
  --datasets all_datasets \
  --output_dir results/time/prompt
```
