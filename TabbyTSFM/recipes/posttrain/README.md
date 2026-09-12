# Tabby prompt/post-training

`train.py` freezes Tabby-Pretrain and optimizes only the prompt module. The data
loader always removes each known official GIFT-Eval test window before creating
the time-based train/validation split. This train-only constraint is a code
invariant and is saved in the run config as `strict_test_cut=true`.

Install Tabby and the forecast dependencies, set `GIFT_EVAL` to the local
dataset root, then run:

```bash
export GIFT_EVAL=/path/to/GIFT-Eval/data
GPU=0 CKPT=/path/to/step_0165000 SEED=4 \
  bash recipes/posttrain/train_tabby.sh
```

`CKPT` is resolved by [`tabby.checkpoint`](../../src/tabby/checkpoint.py), so a
Hugging Face snapshot directory (`config.json` + `model.safetensors`) works
wherever a step directory does.

The prompt model, loss, segmentation, checkpoint format, and optimization loop
are otherwise preserved from the research code. Use `smoke_test.py` for a
checkpoint-backed forward/backward check without benchmark data. Use
`smoke_pipeline.sh` for a tiny train-save-load-GIFT-Eval integration run.
