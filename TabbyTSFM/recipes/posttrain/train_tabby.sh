#!/usr/bin/env bash
# Train the prompt module on top of a frozen PatchTST-FM backbone.
#
# This reproduces the released configuration: prompt length 160, segmented
# context-aware prompts, gate initialised at sigma(+5), cosine annealing over 20
# epochs, 8096-step context window.
#
# Usage:
#   GPU=0 CKPT=/path/to/step_0165000 SEED=0 \
#   bash recipes/posttrain/train_tabby.sh
#
# Required environment (read by tabby.posttraining.data through python-dotenv or
# the shell): GIFT_EVAL must point at the GIFT-Eval data root. The data loader
# always excludes the official test windows; this recipe cannot disable that cut.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

GPU=${GPU:-0}
CKPT=${CKPT:?"set CKPT to the PatchTST-FM checkpoint (a step directory, a .bin, or an HF snapshot dir)"}
SEED=${SEED:-0}
CTX=${CTX:-8096}                  # history width; CTX + PRED must fit the 8192 model window
PRED=${PRED:-96}
PLEN=${PLEN:-160}                 # prompt length; 160 is the released setting
GATE=${GATE:-5.0}                 # logit of the initial gate value, sigma(5.0) ~ 0.993
STOP=${STOP:-20}
LR=${LR:-1e-3}
MICRO_BS=${MICRO_BS:-24}
ACCUM=${ACCUM:-2}                 # effective batch = MICRO_BS * ACCUM
OUT=${OUT:-runs}
RUN=${RUN:-tabby_plen${PLEN}_ctx${CTX}_seed${SEED}}

# Long-context GIFT-Eval training tasks (evaluated at short/medium/long terms).
long_tasks="LOOP_SEATTLE/5T LOOP_SEATTLE/H bitbrains_rnd/5T bizitobs_l2c/5T bizitobs_application \
solar/10T solar/H SZ_TAXI/15T us_births/D saugeenday/D M_DENSE/H jena_weather/H jena_weather/10T \
ett2/15T bizitobs_service ett2/H ett1/15T ett1/H electricity/H electricity/15T \
kdd_cup_2018_with_missing/H bitbrains_fast_storage/5T"

# Short-horizon tasks. These are also passed to --horizon_mask_tasks so that the
# training target is truncated to the horizon each task is actually evaluated at.
short_tasks="electricity/D electricity/W m4_monthly m4_quarterly m4_yearly m4_daily m4_hourly \
m4_weekly solar/D kdd_cup_2018_with_missing/D bitbrains_fast_storage/H bitbrains_rnd/H \
bizitobs_l2c/H car_parts_with_missing covid_deaths ett1/D hierarchical_sales/D hierarchical_sales/W \
hospital LOOP_SEATTLE/D M_DENSE/D restaurant saugeenday/M SZ_TAXI/H temperature_rain_with_missing \
us_births/W"

echo "[cfg] run=${RUN} gpu=${GPU} ctx=${CTX} plen=${PLEN} seed=${SEED}"
echo "[cfg] backbone=${CKPT}"

CUDA_VISIBLE_DEVICES=${GPU} python recipes/posttrain/train.py \
    --pretrain_ckpt "${CKPT}" \
    --datasets ${long_tasks} ${short_tasks} \
    --horizon_mask_tasks ${short_tasks} \
    --context_length ${CTX} --prediction_length ${PRED} \
    --min_future 8 --val_min_future 2 \
    --prompt_len ${PLEN} --prompt_init_mode anchor_delta --data_init \
    --prompt_gate_init ${GATE} --ctx_gate_init ${GATE} \
    --context_aware --segmented --max_segments 16 \
    --seg_mode adaptive_horizon --min_seg_len 48 --season_period 48 \
    --ctx_rank 4 --ctx_hidden 32 \
    --num_instances_per_series 8 --max_series_per_task 2000 --temperature_alpha 0.2 \
    --batch_size ${MICRO_BS} --grad_accum_steps ${ACCUM} \
    --lr ${LR} --epochs 100 --stop_epoch ${STOP} --patience 25 \
    --cosine_tmax ${STOP} --cosine_eta_ratio 0.1 \
    --save_epochs 5,8,10,12,15,20 \
    --seed ${SEED} --output_dir "${OUT}" --run_name "${RUN}"

echo "[done] checkpoints in ${OUT}/${RUN}/"
