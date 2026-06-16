#!/bin/bash
set -euo pipefail

export device="${device:-${DEVICE:-0}}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"
cd "${repo_root}"

python_bin="${PYTHON_BIN:-python}"
eval_split="${EVAL_SPLIT:-valid}"
importance_mode="${IMPORTANCE_MODE:-attnXgrad}"
target_source="${TARGET_SOURCE:-pred}"
aopc_mode="${AOPC_MODE:-mean}"
mask_fill_strategy="${MASK_FILL_STRATEGY:-mean}"
ks="${KS:-5,10,15,20,25}"
random_repeats="${RANDOM_REPEATS:-10}"
ckpt_dir="${CKPT_DIR:-saves/vanilla/mimic}"
ckpt_pattern="${CKPT_PATTERN:-*.pth}"
suffix="${SUFFIX:-${importance_mode}_vanilla}"
out_dir="${OUT_DIR:-}"

extra_args=()
if [[ -n "$out_dir" ]]; then
    extra_args+=(--out_dir "$out_dir")
fi

if [[ ! -d "$ckpt_dir" ]]; then
    echo "[ERROR] checkpoint directory not found: $ckpt_dir"
    exit 1
fi

mapfile -t ckpts < <(find "$ckpt_dir" -maxdepth 1 -type f -name "$ckpt_pattern" ! -name "*rep_cka*" | sort)
if (( ${#ckpts[@]} == 0 )); then
    echo "[ERROR] no vanilla checkpoints found in $ckpt_dir with pattern $ckpt_pattern"
    exit 1
fi

echo "[RUN] dataset=mimic method=vanilla ckpts=${#ckpts[@]} split=${eval_split} suffix=${suffix}"
CUDA_VISIBLE_DEVICES=$device "$python_bin" -m src.common.evaluation.masking_eval_gshardgate \
    --dataset mimic \
    --ckpt "${ckpts[@]}" \
    --eval_split "$eval_split" \
    --importance_mode "$importance_mode" \
    --target_source "$target_source" \
    --aopc_mode "$aopc_mode" \
    --mask_fill_strategy "$mask_fill_strategy" \
    --Ks "$ks" \
    --random_repeats "$random_repeats" \
    --batch_size 32 \
    --hidden_dim 128 \
    --num_patches 4 \
    --num_experts 4 \
    --top_k 2 \
    --suffix "$suffix" \
    "${extra_args[@]}"
