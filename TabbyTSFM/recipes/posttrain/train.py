#!/usr/bin/env python3
"""
Prompt tuning on a pretrained PatchTST-FM time-series foundation model.

The backbone is frozen; only the prompt module is trained. Training data comes
from the *training* portion of the GIFT-Eval datasets (see
``tabby.posttraining.data`` for the time-based train/validation split), and the
objective is the backbone's own pinball loss. Official GIFT-Eval test windows
are always excluded by the release data loader.

Notes:
  - the loss is natively a per-sample masked mean over the horizon, so
    --horizon_mask_tasks works on its own (the mask lands in future_observed);
  - --precision bf16 wraps the forward pass in autocast, matching how the
    backbone was pretrained, while the prompt parameters stay fp32 masters.

Example:
    python recipes/posttrain/train.py \
        --pretrain_ckpt /path/to/checkpoint/step_0165000 \
        --datasets m4_monthly electricity/H ... \
        --run_name my_run --context_length 8096 --prompt_len 160 \
        --context_aware --segmented --seg_mode adaptive_horizon
See recipes/posttrain/train_tabby.sh for the exact released configuration.
"""
import json, math, argparse, random
from pathlib import Path
import torch
import numpy as np

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from tabby.posttraining.data import (
    STRICT_TEST_CUT,
    TaskSpec,
    make_posttraining_loader_dataset,
)
from tabby.posttraining.model import (
    PatchTSTFMPromptCFG, PromptedPatchTSTFM, load_patchtstfm_backbone,
)


def batch_to_ptfm(batch, device):
    past = batch["past_target"].squeeze(-1).to(device)
    obs = batch["past_observed_target"].squeeze(-1).float().to(device)
    pad = batch["past_is_pad"].to(device)
    context_mask = obs * (~pad).float()
    context = torch.where(context_mask > 0, past, torch.tensor(float('nan'), device=device))
    future = batch["future_target"].squeeze(-1).to(device)          # NaN-padded kept
    future_obs = batch["future_observed_target"].squeeze(-1).float().to(device)
    # Long-history prompt: absent unless enabled, in which case forward() takes the wide-window path.
    lctx = lmask = lpad_out = None
    if "long_past_target" in batch:
        lp = batch["long_past_target"].squeeze(-1).to(device)
        lo = batch["long_past_observed_target"].squeeze(-1).float().to(device)
        lpad = batch["long_past_is_pad"].to(device)
        lmask = lo * (~lpad).float()
        lctx = torch.where(lmask > 0, lp, torch.zeros_like(lp))
        lpad_out = lpad
    else:
        lpad_out = None
    return context, context_mask, pad, future, future_obs, lctx, lmask, lpad_out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain_ckpt", type=str, required=True,
                   help="Tabby-Pretrain checkpoint: step dir / pytorch_model.bin / parent with `latest`")
    p.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp32"])
    # Normalisation convention. inhouse_asinh matches the Tabby checkpoints; revin_official
    # reproduces the released PatchTST-FM r1 inference path.
    p.add_argument("--norm_mode", type=str, default="inhouse_asinh",
                   choices=["inhouse_asinh", "revin_official"],
                   help="inhouse_asinh: sqrt(var+eps) (Tabby checkpoint); "
                        "revin_official: RevIN std_min floor (r1 compatibility)")
    p.add_argument("--min_forecast_span", type=int, default=0,
                   help="0 = mask exactly H; 128 mirrors official r1 inference (mask max(H,128), keep first H)")

    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--context_length", type=int, default=4096,
                   help="HISTORY width Tc; Tc + prediction_length must fit the model window (8192)")
    p.add_argument("--prediction_length", type=int, default=96)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--val_batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_instances_per_series", type=int, default=8)
    # Windows drawn per series on the validation side (training uses --num_instances_per_series).
    # Raising it reduces validation noise; every number reported for this model used the default 4.
    p.add_argument("--val_instances", type=int, default=4)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--temperature_alpha", type=float, default=0.5)
    p.add_argument("--epoch_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_series_per_task", type=int, default=50000)

    p.add_argument("--prompt_len", type=int, default=10)
    p.add_argument("--prompt_init_mode", type=str, default="anchor_delta", choices=["anchor_delta", "direct"])
    p.add_argument("--loss_quantile_subset", action="store_true",
                   help="restrict the pinball gradient to the 9 GIFT-Eval quantiles (0.1..0.9); "
                        "the backbone predicts 99, of which 90 never enter the score. "
                        "The loss is rescaled by K_full/K_sub to keep its magnitude comparable.")
    p.add_argument("--prompt_gate_init", type=float, default=-3.0)
    p.add_argument("--data_init", action="store_true")

    p.add_argument("--context_aware", action="store_true")
    p.add_argument("--n_stats", type=int, default=10)
    p.add_argument("--ctx_rank", type=int, default=4)
    p.add_argument("--ctx_hidden", type=int, default=32)
    p.add_argument("--ctx_mlp_layers", type=int, default=2)
    p.add_argument("--ctx_gate_init", type=float, default=-3.0)

    p.add_argument("--segmented", action="store_true")
    p.add_argument("--max_segments", type=int, default=16)
    p.add_argument("--seg_attn_dim", type=int, default=64)
    p.add_argument("--seg_attn_heads", type=int, default=4)
    p.add_argument("--seg_mode", type=str, default="fixed_len",
                   choices=["fixed_len", "adaptive_horizon"])
    p.add_argument("--seg_fixed_len", type=int, default=256)
    p.add_argument("--min_seg_len", type=int, default=48)
    p.add_argument("--season_period", type=int, default=1)
    p.add_argument("--long_ctx_mode", type=str, default="add", choices=["add","replace","all"],
                   help="add = sum the short-window and long-window branches; replace = use the long branch only")
    p.add_argument("--long_ctx_len", type=int, default=0,
                   help="width of the extra right-aligned history window shown to the prompt generator "
                        "(0 = disabled; must exceed --context_length to take effect). "
                        "The backbone input length is unchanged.")
    p.add_argument("--long_max_segments", type=int, default=0,
                   help="segment count for the long window (0 = same as --max_segments)")
    p.add_argument("--no_seg_relative_pos", action="store_true")
    p.add_argument("--no_query_from_base", action="store_true")

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--cosine_tmax", type=int, default=None,
                   help="cosine period (default = --epochs). Set it equal to --stop_epoch for a full anneal.")
    p.add_argument("--cosine_eta_ratio", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--grad_accum_steps", type=int, default=1)

    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="runs_patchtstfm")
    p.add_argument("--wandb_project", type=str, default="none")
    p.add_argument("--min_future", type=int, default=None)
    p.add_argument("--val_min_future", type=int, default=None)
    p.add_argument("--horizon_mask_tasks", nargs="*", default=None,
                   help="per-task GIFT-Eval short-horizon mask (data-level; loss "
                        "here is natively masked-mean so no extra flag needed)")
    p.add_argument("--stop_epoch", type=int, default=50)
    p.add_argument("--save_epochs", type=str, default="",
                   help="comma-separated epochs to snapshot (prompt-only, a few MB); 'all' = every epoch")

    args = p.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else args.device if args.device != "auto" else "cpu")

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    task_specs = [TaskSpec(name=n, series_fraction=None, max_series=args.max_series_per_task,
                           storage_env_var="GIFT_EVAL") for n in args.datasets]

    print("Building data loaders...")
    train_ds, train_loader = make_posttraining_loader_dataset(
        tasks=task_specs, context_length=args.context_length,
        prediction_length=args.prediction_length, split="train",
        batch_size=args.batch_size, num_workers=args.num_workers,
        temperature_alpha=args.temperature_alpha, epoch_size=args.epoch_size,
        use_sampler=True, allow_padding=True,
        num_instances_per_series=args.num_instances_per_series,
        val_fraction=args.val_fraction, to_univariate=True,
        seed=args.seed, storage_env_var="GIFT_EVAL",
        min_future=args.min_future,
        horizon_mask_tasks=args.horizon_mask_tasks,
        long_context_length=args.long_ctx_len,
    )
    _, val_loader = make_posttraining_loader_dataset(
        tasks=task_specs, context_length=args.context_length,
        prediction_length=args.prediction_length, split="val",
        batch_size=args.val_batch_size or args.batch_size, num_workers=args.num_workers,
        temperature_alpha=args.temperature_alpha, epoch_size=None,
        use_sampler=False, allow_padding=True, num_instances_per_series=args.val_instances,
        val_fraction=args.val_fraction, to_univariate=True,
        seed=args.seed, storage_env_var="GIFT_EVAL",
        min_future=(args.val_min_future if args.val_min_future is not None
                    else args.min_future),
        horizon_mask_tasks=args.horizon_mask_tasks,
        long_context_length=args.long_ctx_len,
    )

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print(f"Loading PatchTST-FM backbone from {args.pretrain_ckpt} ...")
    backbone, model_cfg, step = load_patchtstfm_backbone(args.pretrain_ckpt, device)
    print(f"  loaded step={step}  model_cfg={model_cfg}")

    cfg = PatchTSTFMPromptCFG(
        prompt_len=args.prompt_len, prompt_init_mode=args.prompt_init_mode,
        prompt_gate_init=args.prompt_gate_init,
        loss_quantile_subset=args.loss_quantile_subset, context_aware=args.context_aware,
        n_stats=args.n_stats, ctx_rank=args.ctx_rank, ctx_hidden=args.ctx_hidden,
        ctx_mlp_layers=args.ctx_mlp_layers, ctx_gate_init=args.ctx_gate_init,
        segmented=args.segmented, max_segments=args.max_segments,
        seg_attn_dim=args.seg_attn_dim, seg_attn_heads=args.seg_attn_heads,
        query_from_base=(not args.no_query_from_base),
        seg_mode=args.seg_mode, seg_fixed_len=args.seg_fixed_len,
        min_seg_len=args.min_seg_len,
        seg_relative_pos=(not args.no_seg_relative_pos),
        prediction_length=args.prediction_length,
        season_period=args.season_period,
        long_ctx_len=args.long_ctx_len,
        long_ctx_mode=args.long_ctx_mode,
        long_max_segments=args.long_max_segments,
        context_length=args.context_length,
        norm_mode=args.norm_mode, min_forecast_span=args.min_forecast_span,
    )

    model = PromptedPatchTSTFM(backbone, cfg).to(device)
    model.freeze_base()

    if args.data_init and args.prompt_init_mode == "anchor_delta":
        model.initialize_anchor_from_data(train_loader, device, num_batches=10)

    summary = model.param_summary()
    print(f"\n{'='*60}\nPatchTST-FM Prompt Tuning:")
    print(f"  pretrain_ckpt={args.pretrain_ckpt} (step {step})")
    print(f"  segmented={cfg.segmented}, S={cfg.max_segments}, seg_mode={cfg.seg_mode}")
    print(f"  loss = native per-sample masked-mean pinball (K={model.num_quantiles})")
    for k, v in summary.items():
        print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")
    print(f"{'='*60}\n")

    prompt_params = model.trainable_parameters()
    optimizer = torch.optim.AdamW(prompt_params, lr=args.lr, weight_decay=args.weight_decay)
    # With T_max = --epochs and an early stop well before it, the LR stays nearly constant;
    # setting --cosine_tmax = --stop_epoch gives a real anneal down to eta_min.
    _tmax = getattr(args, "cosine_tmax", None) or args.epochs
    _eta_min = args.lr * getattr(args, "cosine_eta_ratio", 0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_tmax, eta_min=_eta_min)
    _stop = getattr(args, "stop_epoch", None) or args.epochs
    print(f"[lr-plan] AdamW lr={args.lr:g}  cosine T_max={_tmax}  eta_min={_eta_min:g}  "
          f"{'full anneal' if _tmax <= _stop * 1.5 else 'WARNING: near-constant lr (T_max >> stop_epoch)'}")

    config_dict = {**vars(args), **vars(cfg)}
    config_dict["pretrain_step"] = step
    config_dict["strict_test_cut"] = STRICT_TEST_CUT
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2, default=str)

    use_wandb = HAS_WANDB and args.wandb_project.lower() != "none"
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.run_name, config=config_dict)

    use_amp = (args.precision == "bf16" and device.type == "cuda")

    best = best50 = float("inf")
    patience = gstep = 0

    def _save(path, ep, vl):
        torch.save({"model_state_dict": model.prompt_state_dict(),
                    "epoch": ep, "val_loss": vl, "config": config_dict}, path)

    for epoch in range(1, args.epochs + 1):
        model.backbone.eval()
        tot, nb = 0.0, 0
        optimizer.zero_grad()
        for batch in train_loader:
            if not batch:
                continue
            ctx, cmask, pad, ft, fo, lctx, lmask, lpad = batch_to_ptfm(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = model(context=ctx, context_mask=cmask, past_is_pad=pad,
                            future_target=ft, future_target_mask=fo,
                            long_context=lctx, long_context_mask=lmask, long_past_is_pad=lpad)
            loss = out["loss"]
            if loss is None:
                continue
            loss = loss / args.grad_accum_steps
            loss.backward()
            tot += loss.item() * args.grad_accum_steps; nb += 1
            if nb % args.grad_accum_steps == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(prompt_params, args.grad_clip)
                optimizer.step(); optimizer.zero_grad()
            gstep += 1
            if gstep == 1:
                mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0
                print(f"[mem] first batch: max_allocated={mem:.1f} GiB "
                      f"(windows/step = {args.batch_size}x{args.num_instances_per_series})")
            if use_wandb:
                wandb.log({"train/loss_step": loss.item() * args.grad_accum_steps, "global_step": gstep})
        avg_train = tot / max(nb, 1)

        vl, vb = 0.0, 0
        model.eval()   # Prompt modules go to eval too. Harmless today (no dropout/BN), but it
                       # would silently add validation noise if either were introduced later.
        with torch.no_grad():
            for batch in val_loader:
                if not batch:
                    continue
                ctx, cmask, pad, ft, fo, lctx, lmask, lpad = batch_to_ptfm(batch, device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    out = model(context=ctx, context_mask=cmask, past_is_pad=pad,
                                future_target=ft, future_target_mask=fo,
                                long_context=lctx, long_context_mask=lmask, long_past_is_pad=lpad)
                if out["loss"] is not None:
                    vl += out["loss"].item(); vb += 1
        val_empty = (vb == 0)
        model.train()
        model.backbone.eval()   # the backbone stays frozen and in eval mode at all times
        avg_val = vl / vb if vb > 0 else float("nan")

        info = out.get("info", {})
        gate_str = f"pg={info.get('prompt_gate', 0):.4f}"
        if cfg.context_aware:
            gate_str += f" cg={info.get('ctx_gate', 0):.4f}"
        # Log the long-window validity ratio when the long branch is active.
        if info.get("long_seg_valid") is not None:
            gate_str += f" lsv={info['long_seg_valid']:.2f}"
        lr_now = scheduler.get_last_lr()[0]
        print(f"[Epoch {epoch:3d}/{args.epochs}] train={avg_train:.5f} val={avg_val:.5f} lr={lr_now:.6f} {gate_str}")
        if val_empty:
            print("  [WARN] validation produced 0 batches this epoch; skipping best/patience update")
        if use_wandb:
            wandb.log({"epoch": epoch, "train/loss": avg_train, "val/loss": avg_val, "lr": lr_now})
        scheduler.step()

        is_best = (not val_empty) and (avg_val < best)


        if args.save_epochs:
            _se = args.save_epochs.strip().lower()
            if _se == "all" or (epoch in {int(x) for x in _se.split(",") if x.strip()}):
                _save(out_dir / f"checkpoint_e{epoch}.pt", epoch, avg_val)
                print(f"  → Snapshot checkpoint_e{epoch}.pt")

        if val_empty:
            if epoch >= args.stop_epoch:
                break
            continue

        if is_best:
            best = avg_val; patience = 0
            _save(out_dir / "checkpoint_best.pt", epoch, avg_val)
            print(f"  → Saved best (val={avg_val:.5f})")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break
        if epoch <= args.stop_epoch and avg_val < best50:
            best50 = avg_val
            _save(out_dir / "checkpoint_best_e50.pt", epoch, avg_val)
            print(f"  → Saved best_e50 (epoch={epoch}, val={avg_val:.5f})")
        if epoch >= args.stop_epoch:
            break

    if "epoch" not in dir() or args.epochs < 1:   # guard the epochs=0 corner case
        print("[warn] training loop never ran (epochs < 1); skipping the final checkpoint")
    else:
        _save(out_dir / "checkpoint_final.pt", epoch, avg_val)
    print(f"\nDone. best={best:.5f}, ≤stop_epoch={best50:.5f}")


    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
