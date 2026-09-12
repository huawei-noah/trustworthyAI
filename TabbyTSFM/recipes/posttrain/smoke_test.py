#!/usr/bin/env python3
"""End-to-end smoke test: no data, no GIFT-Eval, one GPU (or CPU), ~30 seconds.

It checks the things that actually break when this code is moved to a new
machine, in the order they would break:

  1. every installed Tabby module imports
  2. the backbone checkpoint loads
  3. the prompt module builds and reports a sane parameter count
  4. a forward pass produces the expected quantile shape
  5. a loss backward pass reaches the prompt parameters AND NOTHING ELSE
     (i.e. the backbone really is frozen)
  6. the prompt-only state dict round-trips through save/load

None of these checks needs a GPU. If the machine's GPUs are busy, run it on the
CPU (a minute or two) rather than queueing for a card.

Usage:
    python recipes/posttrain/smoke_test.py --pretrain_ckpt /path/to/step_0165000
    python smoke_test.py --pretrain_ckpt ... --device cpu       # no GPU at all
    python smoke_test.py --pretrain_ckpt ... --device cuda:5    # a specific card
    CUDA_VISIBLE_DEVICES=5 python smoke_test.py ...             # equivalent
"""
import argparse
import tempfile
from pathlib import Path

import torch

from tabby.posttraining.model import (
    PatchTSTFMPromptCFG, PromptedPatchTSTFM, load_patchtstfm_backbone,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain_ckpt", required=True)
    p.add_argument("--device", default="auto",
                   help="auto | cpu | cuda | cuda:<n>. The checks do not need a GPU.")
    p.add_argument("--context_length", type=int, default=8096)
    p.add_argument("--prediction_length", type=int, default=96)
    p.add_argument("--prompt_len", type=int, default=160)
    p.add_argument("--batch", type=int, default=2)
    args = p.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else args.device if args.device != "auto" else "cpu")
    print(f"[1/6] imports OK; device={device}")
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        print(f"      GPU {device}: {free / 2**30:.1f} GiB free of {total / 2**30:.1f} GiB")
        if free < 6 * 2**30:
            print("      NOTE: little free memory. Use --device cpu (these checks do not "
                  "need a GPU) or point --device at another card.")

    backbone, model_cfg, step = load_patchtstfm_backbone(args.pretrain_ckpt, device)
    print(f"[2/6] backbone loaded: step={step}, d_model={getattr(model_cfg, 'd_model', '?')}, "
          f"layers={getattr(model_cfg, 'num_layers', '?')}")

    cfg = PatchTSTFMPromptCFG(
        prompt_len=args.prompt_len, prompt_init_mode="anchor_delta",
        prompt_gate_init=5.0, ctx_gate_init=5.0,
        context_aware=True, segmented=True, max_segments=16,
        seg_mode="adaptive_horizon", min_seg_len=48, season_period=48,
        ctx_rank=4, ctx_hidden=32,
        prediction_length=args.prediction_length,
        context_length=args.context_length,
    )
    model = PromptedPatchTSTFM(backbone, cfg).to(device)
    model.freeze_base()
    summary = model.param_summary()
    trainable = summary.get("total_trainable")
    frozen = summary.get("backbone_frozen")
    if not trainable or trainable <= 0:
        raise SystemExit(f"[FAIL] no trainable parameters: {summary}")
    pct = 100.0 * trainable / (trainable + frozen) if frozen else float("nan")
    print(f"[3/6] prompt built: trainable={trainable:,} "
          f"({pct:.2f}% of {trainable + frozen:,}), frozen={frozen:,}")

    B, T, H = args.batch, min(1024, args.context_length), args.prediction_length
    ctx = torch.randn(B, T, device=device)
    out = model(context=ctx, prediction_length=H)
    want = (B, model.num_quantiles, H)
    if tuple(out["quantile_preds"].shape) != want:
        raise SystemExit(f"[FAIL] quantile shape {tuple(out['quantile_preds'].shape)} != {want}")
    print(f"[4/6] forward OK: quantile_preds{tuple(out['quantile_preds'].shape)}")

    future = torch.randn(B, H, device=device)
    fmask = torch.ones(B, H, device=device)
    out = model(context=ctx, future_target=future, future_target_mask=fmask)
    loss = out["loss"]
    if loss is None:
        raise SystemExit("[FAIL] loss is None with a future target supplied")
    loss.backward()

    got_grad = [n for n, q in model.named_parameters() if q.grad is not None]
    backbone_grad = [n for n in got_grad if n.startswith("backbone.")]
    if backbone_grad:
        raise SystemExit(f"[FAIL] backbone received gradients (not frozen): {backbone_grad[:5]}")
    if not got_grad:
        raise SystemExit("[FAIL] no parameter received a gradient")
    print(f"[5/6] backward OK: loss={loss.item():.5f}, "
          f"{len(got_grad)} prompt tensors got gradients, backbone got none")

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ckpt.pt"
        torch.save({"model_state_dict": model.prompt_state_dict()}, path)
        state = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        real_missing = [k for k in missing if not k.startswith("backbone.")]
        if real_missing or unexpected:
            raise SystemExit(f"[FAIL] state dict round-trip: missing={real_missing} unexpected={unexpected}")
        size_mb = path.stat().st_size / 2**20
    print(f"[6/6] checkpoint round-trip OK ({size_mb:.1f} MB, prompt only)")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
