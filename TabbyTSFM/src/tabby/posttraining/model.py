"""
Prompted PatchTST-FM: prompt tuning on a pretrained PatchTST-FM time-series
foundation model.

Injection point: M prompt tokens are PREPENDED to
the patch-token sequence between (in_layer + pos_embed) and the transformer
blocks, so prompts carry no positional embedding (the analogue of position_id=0
in the Chronos-2 version). After the blocks the first M tokens are dropped
before out_layer (the output head is per-patch, so this is forced by shape).

The prompt-GENERATION pathway (global stats -> low-rank prompt, segmented
refinement) lives in prompt_blocks.py and is backbone-agnostic. What is specific
to this backbone is only the plumbing around it:
  - normalization is the pretraining pipeline's mask-aware asinh
    (``tabby.utils.input_preprocessing``), applied OUTSIDE the model — we
    mirror the ``tabby.models.PatchTSTFM`` adapter forward (key-only Boolean
    attention mask), which is the exact function the checkpoint was trained through, not
    PatchTSTFMModel.forward (internal RevIN, query+key masking).
  - stats/segments are computed on the HISTORY SLICE of the padded window
    (layout [left-pad | history | future]); the slice restores the
    right-aligned-valid-region assumption SegmentedStatsExtractor requires.
  - loss is the pretraining pinball loss verbatim (per-sample masked mean over
    time, sum over quantiles / sqrt(K), mean over batch).

Import contract: the canonical Tabby backbone is imported from the installed
``tabby.models`` package; no source-directory injection is required.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from tabby.posttraining.prompt_blocks import (
    Chronos2StatsExtractor,
    SegmentedStatsExtractor,
    SegmentRefiner,
    Chronos2ContextPromptGenerator,
)

# Checkpoint loading lives in ``tabby.checkpoint`` so the recipes, the GluonTS
# predictor and the benchmarks all accept the same inputs: a Hugging Face
# snapshot (config.json + model.safetensors -- the published Tabby-Pretrain and
# patchtst-fm-r1 layout) or an in-house pytorch_model.bin.
from tabby.checkpoint import (
    clean_state_dict as _clean_state_dict,
    is_hf_snapshot as _is_hf_snapshot,
    load_backbone,
)


def load_patchtstfm_backbone(ckpt: str, device):
    """Load the adapter from a Tabby-Pretrain checkpoint or an HF snapshot dir.

    Returns (model, trainer_cfg, step); model on `device`, eval mode,
    frozen-ready.  See :mod:`tabby.checkpoint` for the accepted layouts.
    """
    return load_backbone(ckpt, device)


# =============================================================================
# Configuration (subset of Chronos2PromptCFG: no neighbor line, no masked_mean
# switch — see module docstring)
# =============================================================================
@dataclass
class PatchTSTFMPromptCFG:
    prompt_len: int = 10
    prompt_init_mode: str = "anchor_delta"     # anchor_delta | direct
    prompt_gate_init: float = -3.0

    # True = restrict the pinball gradient to the 9 GIFT-Eval quantiles. The backbone
    # predicts 99, of which 90 never enter the score. The sum is rescaled by K_full/K_sub so
    # the loss magnitude stays comparable; otherwise changing the quantile set would also
    # change the effective learning rate.
    loss_quantile_subset: bool = False

    context_aware: bool = False
    n_stats: int = 10
    ctx_rank: int = 4
    ctx_hidden: int = 32
    ctx_mlp_layers: int = 2
    ctx_gate_init: float = -3.0

    segmented: bool = False
    max_segments: int = 16
    seg_attn_dim: int = 64
    seg_attn_heads: int = 4
    query_from_base: bool = True
    seg_mode: str = "fixed_len"                # fixed_len (A) | adaptive_horizon (C)
    seg_fixed_len: int = 256
    seg_len_k: int = 2
    min_seg_len: int = 48
    seg_relative_pos: bool = True
    prediction_length: int = 96
    season_period: int = 1
    context_length: int = 4096                 # HISTORY width Tc; the CONSTANT the
                                               # segment split anchors to (train==test)

    # --- Long-history prompt (decoupled summariser context) ---
    # long_ctx_len = 0 disables it and forward() is bit-for-bit the old path: no extra module
    # is built and no new branch is taken.
    # When set above context_length, the backbone input is unchanged (still Tc). Only the
    # prompt generator additionally sees a wider right-aligned history window, summarised by
    # a second (parameter-free) segment extractor into S segment tokens, which are refined by
    # the *same* seg_refiner weights and added as a second ctx_prompt term.
    #   * No new learnable parameters (the extractor has none and the refiner is shared), so
    #     the difference from the baseline is information, not capacity.
    #   * Segmentation reuses the valid_len logic of SegmentedStatsExtractor (right-aligned,
    #     most recent segment last), so a segment never falls on padding; asserted below.
    long_ctx_len: int = 0
    # The segmenter is already adaptive (window length sets segment length, the count stays
    # S), so the long window can be fed to the existing extractor/refiner unchanged. seg_pos
    # would only need widening if the short and long segments were concatenated instead of
    # added, which would confound "more history" with "one more refinement path".
    #   long_ctx_mode:
    #     "add"     = sum the short-window and long-window branches (default).
    #     "replace" = use the long-window branch only, skipping the short-window refiner.
    #                 The only change is a wider summariser window, at the cost of coarser
    #                 recent segments.
    #     "all"     = as "replace", and the global-statistics branch also uses the long
    #                 window, so the whole conditioning path sees long history while the
    #                 backbone still sees only the short window.
    long_ctx_mode: str = "add"
    # Applies to long_seg_extractor only; 0 means "same as max_segments".
    long_max_segments: int = 0
    # Compatibility normalization and forecast-span knobs. Defaults reproduce
    # the Tabby checkpoint path bit-for-bit.
    norm_mode: str = "inhouse_asinh"           # inhouse_asinh: sqrt(var+eps), n<2 -> 1
                                               # revin_official: sqrt(var), std<=1e-5 -> 1
                                               #   (= tabby.models.normalization.RevIN,
                                               #   what the official r1 was trained through)
    min_forecast_span: int = 0                 # 0 = mask exactly H; official inference
                                               # masks max(H, 128) then keeps the first H


# =============================================================================
# Main module
# =============================================================================
class PromptedPatchTSTFM(nn.Module):
    def __init__(self, backbone, cfg: PatchTSTFMPromptCFG):
        super().__init__()
        self.backbone = backbone               # tabby.models.PatchTSTFM adapter
        self.cfg = cfg
        self.model_cfg = backbone.config       # trainer dataclass: context_length=8192, ...
        d = self.model_cfg.d_model

        Tm = self.model_cfg.context_length
        if cfg.context_length + cfg.prediction_length > Tm:
            raise ValueError(
                f"context_length({cfg.context_length}) + prediction_length({cfg.prediction_length}) "
                f"must fit the model window {Tm} ([pad|history|future] layout)")

        K = self.model_cfg.num_quantiles
        # same grid as configuration_patchtst_fm / predictor.model_quantile_levels
        if K == 99:
            levels = [i / 100.0 for i in range(1, 100)]
        else:
            levels = [i / (K + 1) for i in range(1, K + 1)]
        self.register_buffer("quantile_levels_t", torch.tensor(levels, dtype=torch.float32))

        # Quantile-subset indices; see quantile_subset.py. Fails fast rather than approximating.
        from tabby.posttraining.quantile_subset import gifteval_subset_idx
        _idx = gifteval_subset_idx(levels)
        self.register_buffer("_qsub_idx", torch.tensor(_idx, dtype=torch.long), persistent=False)
        if cfg.loss_quantile_subset:
            print(f"  [loss] quantile subset ON: K={K}, gradient restricted to the "
                  f"{len(_idx)} GIFT-Eval quantiles (indices {_idx}), "
                  f"rescaled by K_full/K_sub={K/len(_idx):.3f}")

        if cfg.prompt_init_mode == "anchor_delta":
            self.register_buffer("prompt_anchor", torch.zeros(cfg.prompt_len, d))
            self.prompt_delta = nn.Parameter(torch.zeros(cfg.prompt_len, d))
            if cfg.prompt_len > 0:
                nn.init.normal_(self.prompt_delta, std=0.02)
            self.prompt_gate = nn.Parameter(torch.tensor(cfg.prompt_gate_init))
        elif cfg.prompt_init_mode == "direct":
            self.prompt_embedding = nn.Parameter(torch.zeros(cfg.prompt_len, d))
            if cfg.prompt_len > 0:
                nn.init.normal_(self.prompt_embedding, std=0.02)
        else:
            raise ValueError(f"Unknown prompt_init_mode: {cfg.prompt_init_mode}")

        if cfg.context_aware:
            self.ctx_gate = nn.Parameter(torch.tensor(cfg.ctx_gate_init))
            self.stats_extractor = Chronos2StatsExtractor(n_stats=cfg.n_stats)
            self.ctx_generator = Chronos2ContextPromptGenerator(
                feat_dim=cfg.n_stats, prompt_len=cfg.prompt_len, d_model=d,
                rank=cfg.ctx_rank, hidden_dim=cfg.ctx_hidden, n_layers=cfg.ctx_mlp_layers,
            )
            if cfg.segmented:
                self.seg_extractor = SegmentedStatsExtractor(
                    n_stats=cfg.n_stats, max_segments=cfg.max_segments,
                    seg_mode=cfg.seg_mode, seg_fixed_len=cfg.seg_fixed_len,
                    seg_len_k=cfg.seg_len_k, min_seg_len=cfg.min_seg_len,
                    prediction_length=cfg.prediction_length,
                    season_period=cfg.season_period,
                    context_length=cfg.context_length)
                self.seg_refiner = SegmentRefiner(
                    n_stats=cfg.n_stats, d_model=d, prompt_len=cfg.prompt_len,
                    n_segments=cfg.max_segments, d_attn=cfg.seg_attn_dim,
                    n_heads=cfg.seg_attn_heads, query_from_base=cfg.query_from_base,
                    relative_pos=cfg.seg_relative_pos,
                )
                # Long-history path: only one extra parameter-free segment extractor, with
                # the context_length anchor replaced by the long-window width so its segment
                # length is self-consistent. The refiner weights are shared, not duplicated:
                # it maps relative oldest-to-newest position, which means the same thing in
                # both windows.
                if int(getattr(cfg, "long_ctx_len", 0) or 0) > cfg.context_length:
                    self.long_seg_extractor = SegmentedStatsExtractor(
                        n_stats=cfg.n_stats,
                        max_segments=int(getattr(cfg, "long_max_segments", 0) or cfg.max_segments),
                        seg_mode=cfg.seg_mode, seg_fixed_len=cfg.seg_fixed_len,
                        seg_len_k=cfg.seg_len_k, min_seg_len=cfg.min_seg_len,
                        prediction_length=cfg.prediction_length,
                        season_period=cfg.season_period,
                        context_length=int(cfg.long_ctx_len))

    # ------------------------------------------------------------------ props
    @property
    def device(self):
        return next(self.backbone.parameters()).device

    @property
    def num_quantiles(self):
        return self.model_cfg.num_quantiles

    @property
    def quantile_levels(self):
        return [float(q) for q in self.quantile_levels_t.tolist()]

    def freeze_base(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def get_prompt_embedding(self):
        if self.cfg.prompt_init_mode == "anchor_delta":
            return self.prompt_anchor + torch.sigmoid(self.prompt_gate) * self.prompt_delta
        return self.prompt_embedding

    # ------------------------------------------------------------- window build
    def _build_window(self, context, context_mask, past_is_pad, future_target,
                      future_observed, H):
        """[left-pad | history | future] layout over the fixed model width Tm.

        Returns x (values, NaN-free), observed, padding, pred — all [B, Tm].
        `observed` on the future span reflects future_observed (False when no
        target / NaN-padded future), which is exactly what makes those
        positions drop out of both stats and loss downstream.
        """
        device = context.device
        B, Tc = context.shape
        Tm = self.model_cfg.context_length
        # forecast span actually masked: >= H (official r1 inference floors it at
        # d_patch*max(pretrain_mask_cont,2)=128 and keeps the first H steps)
        Hs = max(H, int(self.cfg.min_forecast_span or 0))

        cap = Tm - Hs
        if Tc > cap:  # keep the most recent history
            context = context[:, -cap:]
            context_mask = context_mask[:, -cap:]
            if past_is_pad is not None:
                past_is_pad = past_is_pad[:, -cap:]
            Tc = cap
        left = Tm - Tc - Hs

        x = torch.zeros(B, Tm, device=device, dtype=torch.float32)
        observed = torch.zeros(B, Tm, dtype=torch.bool, device=device)
        padding = torch.zeros(B, Tm, dtype=torch.bool, device=device)
        pred = torch.zeros(B, Tm, dtype=torch.bool, device=device)

        padding[:, :left] = True
        hist_sl = slice(left, left + Tc)
        fut_sl = slice(Tm - Hs, Tm)                 # whole masked span (Hs)
        tgt_sl = slice(Tm - Hs, Tm - Hs + H)        # the H steps we predict/score

        ctx_finite = torch.isfinite(context)
        obs_hist = ctx_finite & (context_mask > 0)
        if past_is_pad is not None:
            pad_hist = past_is_pad.bool()
            obs_hist = obs_hist & ~pad_hist
            padding[:, hist_sl] = pad_hist
        x[:, hist_sl] = torch.nan_to_num(context.float(), nan=0.0)
        observed[:, hist_sl] = obs_hist

        pred[:, fut_sl] = True
        if future_target is not None:
            fut_finite = torch.isfinite(future_target)
            obs_fut = fut_finite
            if future_observed is not None:
                obs_fut = obs_fut & (future_observed > 0)
            x[:, tgt_sl] = torch.nan_to_num(future_target.float(), nan=0.0)
            observed[:, tgt_sl] = obs_fut          # span beyond H stays unobserved

        return x, observed, padding, pred, hist_sl, tgt_sl

    # ------------------------------------------------------------ normalization
    def _normalize(self, x, observed, pred, padding):
        """Dispatch on cfg.norm_mode. Both modes: visible-history-only mean, asinh
        transform, zeros at masked positions, patch_pad = all-pad patches.
        - inhouse_asinh : tabby.utils.input_preprocessing
          (std=sqrt(var+eps); n<2 -> 1)
        - revin_official: tabby.models.normalization.RevIN(std_min=1e-5,
          use_sinh=True) statistics (std=sqrt(var); std<=1e-5 -> 1) — the path
          the official r1 checkpoint was trained/inferred through.
        Near-constant series is where the two differ (~300x in std)."""
        from tabby.utils.input_preprocessing import mask_aware_normalize_for_inference
        x_norm, mean, std, union_mask, patch_pad = mask_aware_normalize_for_inference(
            x=x, observed_mask=observed, pred_mask=pred, padding_mask=padding,
            cfg=self.model_cfg)
        if self.cfg.norm_mode == "inhouse_asinh":
            return x_norm, mean, std, union_mask, patch_pad
        if self.cfg.norm_mode != "revin_official":
            raise ValueError(f"unknown norm_mode {self.cfg.norm_mode}")
        # RevIN._get_statistics(x, mask=union_mask) verbatim (fp32, no eps)
        with torch.autocast(device_type="cuda", enabled=False):
            xf = x.float()
            unmask = (~union_mask).float()
            count = unmask.sum(dim=1, keepdim=True).clamp(min=1)
            r_mean = (xf * unmask).sum(dim=1, keepdim=True) / count
            r_std = ((((xf - r_mean) * unmask) ** 2).sum(dim=1, keepdim=True) / count).sqrt()
            r_std = torch.where(r_std > 1e-5, r_std, torch.ones_like(r_std))
            xn = torch.asinh((xf - r_mean) / r_std)
            xn = torch.where(union_mask, torch.zeros_like(xn), xn)
        return xn, r_mean, r_std, union_mask, patch_pad

    def _prep_long(self, long_context, long_context_mask, long_past_is_pad, mean, std):
        """Long window -> (normalised values, mask), using the short window's mean/std so
        both live in the same coordinate frame and no future information leaks in."""
        lv = long_context.squeeze(-1) if long_context.dim() == 3 else long_context
        if long_context_mask is None:
            lm = torch.isfinite(lv).float()
        else:
            lm = (long_context_mask.squeeze(-1) if long_context_mask.dim() == 3
                  else long_context_mask).float()
        lv = torch.nan_to_num(lv, nan=0.0, posinf=0.0, neginf=0.0)
        ln = torch.asinh((lv - mean.view(-1, 1)) / std.view(-1, 1).clamp(min=1e-6)) * lm
        if self.training and long_past_is_pad is not None:
            pd = long_past_is_pad
            pd = (pd.squeeze(-1) if pd.dim() == 3 else pd).bool()
            n_pad = pd.sum(1)
            idx = torch.arange(pd.shape[1], device=pd.device).unsqueeze(0)
            if not bool((pd == (idx < n_pad.unsqueeze(1))).all()):
                raise AssertionError("long_past_is_pad is not a contiguous left prefix; "
                                     "segments would fall on padding.")
        return ln, lm

    # ------------------------------------------------------------------ forward
    def forward(self, context, context_mask=None, past_is_pad=None,
                future_target=None, future_target_mask=None, prediction_length=None,
                long_context=None, long_context_mask=None, long_past_is_pad=None):
        """
        context:            [B, Tc] raw values (NaN allowed at unobserved)
        context_mask:       [B, Tc] 1 = valid observed history (default: isfinite)
        past_is_pad:        [B, Tc] bool, loader left-padding (optional)
        future_target:      [B, H] raw future (NaN-padded); None at inference
        future_target_mask: [B, H] 1 = valid target position
        prediction_length:  overrides cfg.prediction_length (inference with a
                            dataset-specific horizon); training leaves it None
        Returns {"quantile_preds": [B, K, H] denormalized, "loss", "info"}.
        """
        cfg = self.cfg
        B = context.shape[0]
        M = cfg.prompt_len
        H = int(prediction_length) if prediction_length is not None else cfg.prediction_length
        Tm = self.model_cfg.context_length
        L = self.model_cfg.patch_size
        N = self.model_cfg.num_patches
        K = self.model_cfg.num_quantiles
        Hs = max(H, int(cfg.min_forecast_span or 0))
        if not (1 <= Hs <= Tm - L):
            raise ValueError(f"forecast span={Hs} (H={H}) out of range for model window {Tm}")

        if context_mask is None:
            context_mask = torch.isfinite(context).float()

        x, observed, padding, pred, hist_sl, tgt_sl = self._build_window(
            context, context_mask, past_is_pad, future_target, future_target_mask, H)

        # normalization (mean/std from visible history only; mode = cfg.norm_mode)
        x_norm, mean, std, union_mask, patch_pad = self._normalize(x, observed, pred, padding)

        # ---- prompt pathway (on the right-aligned history slice) ----
        prompt = self.get_prompt_embedding().unsqueeze(0).expand(B, M, -1)
        info = {}
        if cfg.prompt_init_mode == "anchor_delta":
            info["prompt_gate"] = torch.sigmoid(self.prompt_gate).item()

        if cfg.context_aware and M > 0:
            hist_vals = x_norm[:, hist_sl]                    # zeros at invisible
            hist_mask = (~union_mask[:, hist_sl]).float()     # 1 = visible history
            _long_ready = None
            if (getattr(self, "long_seg_extractor", None) is not None
                    and long_context is not None):
                _long_ready = self._prep_long(long_context, long_context_mask,
                                              long_past_is_pad, mean, std)
            # mode="all": the global-statistics branch also uses the long window
            if _long_ready is not None and getattr(cfg, "long_ctx_mode", "add") == "all":
                global_stats = self.stats_extractor(_long_ready[0], _long_ready[1])
            else:
                global_stats = self.stats_extractor(hist_vals, hist_mask)   # [B, n_stats]
            base_prompt = self.ctx_generator(global_stats)              # [B, M, d]
            if cfg.segmented:
                seg_matrix, seg_valid = self.seg_extractor(hist_vals, hist_mask)
                ctx_prompt = base_prompt + self.seg_refiner(seg_matrix, seg_valid, base_prompt)
                # --- Long-history path: the backbone has already consumed its own window; ---
        # --- only the prompt side gains information here.                          ---
                if _long_ready is not None:
                    ln, lm = _long_ready
                    seg_m2, seg_v2 = self.long_seg_extractor(ln, lm)
                    long_refine = self.seg_refiner(seg_m2, seg_v2, base_prompt)
                    if getattr(cfg, "long_ctx_mode", "add") in ("replace", "all"):
                        # Long-window branch only: the short-window refinement is not added
                        ctx_prompt = base_prompt + long_refine
                    else:
                        ctx_prompt = ctx_prompt + long_refine
                    info["long_seg_valid"] = float(seg_v2.sum(1).float().mean().item())
            else:
                ctx_prompt = base_prompt
            prompt = prompt + torch.sigmoid(self.ctx_gate) * ctx_prompt
            info["ctx_gate"] = torch.sigmoid(self.ctx_gate).item()

        # ---- backbone drive: mirror models/PatchTSTFM.py forward + M prompts ----
        bb = self.backbone.backbone            # PatchTSTFMModel submodules
        value_patch = x_norm.reshape(B, N, L)
        mask_patch = union_mask.reshape(B, N, L).to(x_norm.dtype)

        h = bb.in_layer(torch.cat([value_patch, 1.0 - mask_patch], dim=-1))
        h = bb.pos_embed(h)                    # prompts get NO positional embedding
        if M > 0:
            h = torch.cat([prompt.to(h.dtype), h], dim=1)

        key_ok = ~patch_pad.bool()             # [B, N]; True = attendable (key-only)
        if M > 0:
            prompt_ok = torch.ones(B, M, dtype=torch.bool, device=h.device)
            key_ok = torch.cat([prompt_ok, key_ok], dim=1)
        attn_mask = key_ok[:, None, None, :]
        for block in bb.blocks:
            h = block(h, attn_mask)
        if M > 0:
            h = h[:, M:]
        h = bb.out_layer(h)

        # monotone quantile head, identical to the adapter
        q_raw = h.reshape(B, N, K + 1, L).permute(0, 2, 1, 3)
        q = q_raw[:, 0, :, :].unsqueeze(1) + torch.cumsum(
            nn.functional.softplus(q_raw[:, 1:, :, :]) / K, dim=1)
        q_hat = q.permute(0, 2, 3, 1).reshape(B, Tm, K)

        q_f = q_hat[:, tgt_sl, :]                              # [B, H, K] normalized
                                                               # (first H of the masked span)

        # ---- loss: pretraining pinball, per-sample masked mean (native) ----
        loss = None
        if future_target is not None:
            qf32 = q_f.float()                 # pinball in fp32 even under bf16 autocast
            target_f = torch.asinh((x[:, tgt_sl] - mean) / std)          # [B, H]
            lm = (pred & observed)[:, tgt_sl].float()                    # valid targets
            t = target_f.unsqueeze(-1)
            quantiles = self.quantile_levels_t.view(1, 1, -1)
            ql = 2 * torch.abs((t - qf32) * ((t <= qf32).float() - quantiles))
            ql = ql * lm.unsqueeze(-1)
            per = ql.sum(dim=1) / lm.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, K]
            if cfg.loss_quantile_subset:
                _i = self._qsub_idx.to(per.device)
                from tabby.posttraining.quantile_subset import subset_rescale
                loss = (per.index_select(-1, _i).sum(dim=-1).mean() / math.sqrt(K)
                        * subset_rescale(K, _i.numel()))
            else:
                loss = per.sum(dim=-1).mean() / math.sqrt(K)

        # ---- denormalize predictions ----
        q_out = torch.sinh(q_f.float()) * std.unsqueeze(-1) + mean.unsqueeze(-1)
        q_out = q_out.permute(0, 2, 1)                         # [B, K, H]

        return {"quantile_preds": q_out, "loss": loss, "info": info}

    # ------------------------------------------------------------- anchor init
    @torch.no_grad()
    def initialize_anchor_from_data(self, dataloader, device, num_batches=10):
        """Anchor = mean pre-pos-embed patch token over visible patches
        (the analogue of Chronos-2's mean input_patch_embedding)."""
        if self.cfg.prompt_init_mode != "anchor_delta" or self.cfg.prompt_len == 0:
            return
        bb = self.backbone.backbone
        L, N = self.model_cfg.patch_size, self.model_cfg.num_patches
        self.backbone.eval()
        embeds, count = [], 0
        for batch in dataloader:
            if not batch:
                continue
            past = batch["past_target"].squeeze(-1).to(device)
            obs = batch["past_observed_target"].squeeze(-1).float().to(device)
            pad = batch["past_is_pad"].to(device)
            cmask = obs * (~pad).float()
            ctx = torch.where(cmask > 0, past, torch.tensor(float("nan"), device=device))
            x, observed, padding, pred, _, _ = self._build_window(
                ctx, cmask, pad, None, None, self.cfg.prediction_length)
            x_norm, _, _, union_mask, _ = self._normalize(x, observed, pred, padding)
            B = x_norm.shape[0]
            value_patch = x_norm.reshape(B, N, L)
            mask_patch = union_mask.reshape(B, N, L).to(x_norm.dtype)
            tok = bb.in_layer(torch.cat([value_patch, 1.0 - mask_patch], dim=-1))
            pv = (~union_mask).reshape(B, N, L).any(-1).to(tok.dtype)    # visible patches
            avg = (tok * pv.unsqueeze(-1)).sum(dim=(0, 1)) / pv.sum().clamp(min=1)
            embeds.append(avg); count += 1
            if count >= num_batches:
                break
        if embeds:
            anchor = torch.stack(embeds).mean(0)
            self.prompt_anchor.copy_(anchor.unsqueeze(0).expand(self.cfg.prompt_len, -1))
            print(f"[init] Anchor from {count} batches, norm={anchor.norm():.4f}")

    # --------------------------------------------------------------- summaries
    def trainable_parameters(self):
        params = []
        if self.cfg.prompt_init_mode == "anchor_delta":
            params.extend([self.prompt_delta, self.prompt_gate])
        else:
            params.append(self.prompt_embedding)
        if self.cfg.context_aware:
            params.extend(list(self.ctx_generator.parameters()))
            if self.cfg.segmented:
                params.extend(list(self.seg_refiner.parameters()))
            params.append(self.ctx_gate)
        return [p for p in params if p.requires_grad]

    def param_summary(self) -> dict:
        s = {}
        if self.cfg.prompt_init_mode == "anchor_delta":
            s["phase1_prompt"] = self.prompt_delta.numel() + 1
        else:
            s["phase1_prompt"] = self.prompt_embedding.numel()
        if self.cfg.context_aware:
            s["ctx_generator(base)"] = sum(p.numel() for p in self.ctx_generator.parameters())
            if self.cfg.segmented:
                s["seg_refiner"] = sum(p.numel() for p in self.seg_refiner.parameters())
                s["mode"] = (f"segmented(S={self.cfg.max_segments}, seg_mode={self.cfg.seg_mode}, "
                             f"rel_pos={self.cfg.seg_relative_pos}, q_from_base={self.cfg.query_from_base})")
            s["ctx_gate"] = 1
        s["backbone_frozen"] = sum(p.numel() for p in self.backbone.parameters())
        s["total_trainable"] = sum(p.numel() for p in self.trainable_parameters())
        return s

    def prompt_state_dict(self):
        """Prompt-only state dict (backbone excluded) for small checkpoints."""
        return {k: v.cpu() for k, v in self.state_dict().items()
                if not k.startswith("backbone.")}
