#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-./outputs/tabby-pretrain-165k}"
: "${BLAST_DATA_ROOT:?Set BLAST_DATA_ROOT to the BLAST training directory}"
: "${SYNTHETIC_ARROW_ROOT:?Set SYNTHETIC_ARROW_ROOT to pre-generated KernelSynth Arrow shards}"
: "${BLAST_RATIO:?Set BLAST_RATIO to the actual BLAST fraction used by the 165K run}"
: "${CAUKER_V2_RATIO:?Set CAUKER_V2_RATIO to the actual CauKer V2 fraction used by the 165K run}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-6}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

RESUME_ARGS=()
if [[ "${RESUME:-0}" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi

# Run from the repository root after installing the package. Tabby-Pretrain does
# not use GIFT data. The KernelSynth fraction is the remainder after BLAST and
# CauKer V2 (1 - BLAST_RATIO - CAUKER_V2_RATIO). Ratios are required instead of
# guessed here because the released source material does not identify the final
# no-GIFT mixture. Six processes, a per-device batch of 90, and four accumulation
# steps give an effective batch size of 2160.
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" recipes/pretrain/train.py \
  --output_dir "${OUT_DIR}" \
  --blast_data_root "${BLAST_DATA_ROOT}" \
  --blast_ratio "${BLAST_RATIO}" \
  --min_real_length 64 \
  --min_blast_sample_length 96 \
  --max_blast_sample_length 4096 \
  --synthetic_mode kernel_mixed \
  --synthetic_arrow_root "${SYNTHETIC_ARROW_ROOT}" \
  --synthetic_arrow_balance_files \
  --min_synth_sample_length 96 \
  --max_synth_sample_length 8192 \
  --mixup_ratio 0 \
  --gift_ratio 0.3 \
  --use_cauker_V2 \
  --cauker_V2_ratio "${CAUKER_V2_RATIO}" \
  --min_series_length 96 \
  --max_series_length 2048 \
  --length_sampling uniform \
  --cauker_features 9 \
  --cauker_num_nodes 9 \
  --cauker_max_parents 2 \
  --context_length 8192 \
  --num_blast_workers 2 \
  --num_synth_workers 2 \
  --num_mixup_workers 0 \
  --prefetch_factor 4 \
  --patch_size 16 \
  --num_layers 20 \
  --d_model 768 \
  --head_dim 64 \
  --num_quantiles 99 \
  --dropout 0.10 \
  --mask_ratio 0.40 \
  --length_weight_alpha 0.5 \
  --cpm_blocks 8 \
  --terminal_mask_min 0 \
  --terminal_mask_max 2 \
  --total_steps 165000 \
  --warmup_steps 10000 \
  --decay_steps 10000 \
  --per_device_batch_size 90 \
  --grad_accum_steps 4 \
  --scheduler wsd \
  --num_cycles 3 \
  --peak_lr 2e-4 \
  --min_lr 1e-5 \
  --weight_decay 0.1 \
  --beta1 0.9 \
  --beta2 0.95 \
  --max_grad_norm 1.0 \
  --precision bf16 \
  --save_every 2000 \
  --log_file train_log.jsonl \
  --log_every 100 \
  --flow_supervision \
  --num_flow_exits 5 \
  --lambda_ds 0.5 \
  --lambda_fm 0.1 \
  --ds_gamma 1.0 \
  --flow_exit_layers 0 5 10 15 20 \
  "${RESUME_ARGS[@]}"
