#!/usr/bin/env bash
# Tiny end-to-end run: one short training epoch, then one evaluated config.
#
# smoke_test.py covers the model path only. This covers the two parts it cannot:
# the GIFT-Eval training loader (tabby.posttraining.data) and the evaluation
# harness. Those are where a fresh machine
# usually breaks: GIFT_EVAL not set, gift-eval repo missing, dataset cache absent.
#
# It deliberately uses a tiny configuration (short context, 2 datasets, capped
# steps) so it finishes in minutes. The numbers it produces are meaningless; the
# point is that every stage runs and writes what the next stage expects.
#
# Usage:
#   GPU=5 CKPT=/path/to/step_0165000 GIFT=/path/to/gift-eval \
#   bash recipes/posttrain/smoke_pipeline.sh
#
# Requires GIFT_EVAL to point at the GIFT-Eval data root (or a .env providing it).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

GPU=${GPU:-0}
CKPT=${CKPT:?"set CKPT to the PatchTST-FM backbone checkpoint"}
GIFT=${GIFT:?"set GIFT to the gift-eval repository root"}
PROPS=${PROPS:-${REPO_ROOT}/benchmarks/forecasting/gift_eval/dataset_properties.json}
RUN=${RUN:-smoke_pipeline}
OUT=${OUT:-runs_smoke}

[ -f "${PROPS}" ] || { echo "[ABORT] ${PROPS} not found (ships with this release)"; exit 1; }
[ -d "${GIFT}" ] || { echo "[ABORT] gift-eval repo not found at ${GIFT}"; exit 1; }
# Resolve GIFT_EVAL through the package itself, so the check and the loader use
# the same search path (see tabby.posttraining.check_env for why that matters).
python -m tabby.posttraining.check_env || exit 1

echo "==== [1/2] one short training epoch (loader + train loop) ===="
CUDA_VISIBLE_DEVICES=${GPU} python recipes/posttrain/train.py \
    --pretrain_ckpt "${CKPT}" \
    --datasets m4_monthly hospital \
    --context_length 512 --prediction_length 48 --min_future 8 --val_min_future 2 \
    --prompt_len 16 --prompt_init_mode anchor_delta \
    --prompt_gate_init 5.0 --ctx_gate_init 5.0 \
    --context_aware --segmented --max_segments 8 \
    --seg_mode adaptive_horizon --min_seg_len 16 --season_period 12 \
    --ctx_rank 4 --ctx_hidden 32 \
    --batch_size 4 --num_instances_per_series 2 --num_workers 2 \
    --max_series_per_task 50 --epoch_size 20 \
    --epochs 1 --stop_epoch 1 --patience 5 \
    --seed 0 --output_dir "${OUT}" --run_name "${RUN}"

BEST="${OUT}/${RUN}/checkpoint_best.pt"
[ -f "${BEST}" ] || { echo "[ABORT] training produced no ${BEST}"; exit 1; }
echo "[1/2] OK -> ${BEST}"

echo
echo "==== [2/2] evaluate one config (harness + SN aggregate) ===="
CUDA_VISIBLE_DEVICES=${GPU} python benchmarks/forecasting/gift_eval/evaluate.py \
    --mode prompt --ckpt "${BEST}" \
    --pretrain_ckpt "${CKPT}" \
    --gift_eval_repo "${GIFT}" --dataset_properties "${PROPS}" \
    --context_length 512 --batch_size 64 \
    --datasets hospital --out_csv "${RUN}_eval"

CSV="results/${RUN}_eval.csv"
rows=$(( $(grep -c '' "${CSV}") - 1 ))
[ "${rows}" -ge 1 ] || { echo "[ABORT] ${CSV} has no result rows"; exit 1; }
echo
echo "==== pipeline OK: trained, saved, loaded, evaluated (${rows} config row(s) in ${CSV}) ===="
echo "The metric values here are meaningless (tiny context, 20 steps); only the plumbing was tested."
