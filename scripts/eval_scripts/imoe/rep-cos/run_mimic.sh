#!/bin/bash
set -euo pipefail

export device="${device:-${DEVICE:-0}}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"
cd "${repo_root}"

python_bin="${PYTHON_BIN:-python}"
lambda_grid="${LAMBDA_GRID:-1e-05 3e-05 1e-04 3e-04 1e-03 3e-03 1e-02 5e-02 1e-01 0.2 0.5 1 2 3 4 6 8 10}"
weight_grid="${WEIGHT_GRID:-$lambda_grid}"
eval_split="${EVAL_SPLIT:-valid}"
importance_mode="${IMPORTANCE_MODE:-attnXgrad}"
target_source="${TARGET_SOURCE:-pred}"
aopc_mode="${AOPC_MODE:-mean}"
mask_fill_strategy="${MASK_FILL_STRATEGY:-mean}"
ks="${KS:-5,10,15,20,25}"
random_repeats="${RANDOM_REPEATS:-10}"
ckpt_dir="${CKPT_DIR:-saves/imoe/transformer/mimic/rep_cos}"
ckpt_tag="${CKPT_TAG:-rep_cos}"
suffix_prefix="${SUFFIX_PREFIX:-${importance_mode}_rep_cos}"
out_dir="${OUT_DIR:-}"

extra_args=()
if [[ -n "$out_dir" ]]; then
    extra_args+=(--out_dir "$out_dir")
fi

if [[ ! -d "$ckpt_dir" ]]; then
    echo "[ERROR] checkpoint directory not found: $ckpt_dir"
    exit 1
fi

evaluated=0
for regularizer_weight in $weight_grid
do
    mapfile -t ckpts < <(find "$ckpt_dir" -maxdepth 1 -type f -name "*_${ckpt_tag}_${regularizer_weight}_run*.pth" | sort)
    if (( ${#ckpts[@]} == 0 )); then
        echo "[SKIP] dataset=mimic method=rep-cos lambda=${regularizer_weight} no checkpoints in $ckpt_dir"
        continue
    fi

    suffix="${suffix_prefix}_${regularizer_weight}"
    echo "[RUN] dataset=mimic method=rep-cos lambda=${regularizer_weight} ckpts=${#ckpts[@]} split=${eval_split} suffix=${suffix}"
    CUDA_VISIBLE_DEVICES=$device "$python_bin" -m src.common.evaluation.masking_eval_imoe \
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
    evaluated=$((evaluated + 1))
done

if (( evaluated == 0 )); then
    echo "[ERROR] no evaluations were run for dataset=mimic method=rep-cos"
    exit 1
fi
