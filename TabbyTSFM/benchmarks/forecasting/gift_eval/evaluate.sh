#!/usr/bin/env bash
# Evaluate on GIFT-Eval, either zero-shot or with a trained prompt.
#
# Usage:
#   # prompt-tuned
#   GPU=0 CKPT=/path/to/step_0165000 GIFT=/path/to/gift-eval \
#   PROMPT=runs/<run>/checkpoint_best.pt \
#   bash benchmarks/forecasting/gift_eval/evaluate.sh
#
#   # zero-shot baseline (same predict path, no prompt)
#   GPU=0 CKPT=... GIFT=... MODE=zs \
#   bash benchmarks/forecasting/gift_eval/evaluate.sh
#
# Results: results/<OUT>.csv (one row per dataset/term config). The Seasonal-Naive
# normalised geometric mean over the matched configs is printed at the end; that
# aggregate is the headline number.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

GPU=${GPU:-0}
MODE=${MODE:-prompt}
CKPT=${CKPT:?"set CKPT to the PatchTST-FM backbone checkpoint"}
GIFT=${GIFT:?"set GIFT to the gift-eval repository root"}
PROPS=${PROPS:-${REPO_ROOT}/benchmarks/forecasting/gift_eval/dataset_properties.json}
CTX=${CTX:-8096}
BS=${BS:-256}
DATASETS=${DATASETS:-all}
OUT=${OUT:-tabby_${MODE}_ctx${CTX}}

EXTRA=()
if [ "${MODE}" = "prompt" ]; then
    PROMPT=${PROMPT:?"MODE=prompt requires PROMPT=<path to checkpoint_best.pt>"}
    EXTRA+=(--ckpt "${PROMPT}")
fi

echo "[cfg] mode=${MODE} ctx=${CTX} datasets=${DATASETS} out=${OUT}"

CUDA_VISIBLE_DEVICES=${GPU} python benchmarks/forecasting/gift_eval/evaluate.py \
    --mode "${MODE}" "${EXTRA[@]}" \
    --pretrain_ckpt "${CKPT}" \
    --gift_eval_repo "${GIFT}" --dataset_properties "${PROPS}" \
    --context_length ${CTX} --batch_size ${BS} \
    --datasets "${DATASETS}" --out_csv "${OUT}"
