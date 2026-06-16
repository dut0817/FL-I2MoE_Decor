#!/bin/bash
set -euo pipefail

export device="${device:-${DEVICE:-0}}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"
cd "${repo_root}"

lr_grid="${LR_GRID:-0.0001}"
run_seed="${RUN_SEED:-0}"
n_runs_override="${N_RUNS_OVERRIDE:-3}"
seed_end=$((run_seed + n_runs_override - 1))
run_suffix_prefix="${RUN_SUFFIX:-}"

for lr in $lr_grid
do
if [[ -n "$run_suffix_prefix" ]]; then
    run_suffix="${run_suffix_prefix}_seed${run_seed}-${seed_end}"
else
    run_suffix="seed${run_seed}-${seed_end}"
fi
CUDA_VISIBLE_DEVICES=$device python src/gshardgate/train_gshardgate.py \
    --lr $lr \
    --data mmimdb \
    --gate GShardGate \
    --train_epochs 40 \
    --modality LI \
    --fusion_sparse True \
    --batch_size 32 \
    --hidden_dim 256 \
    --num_layers_fus 2 \
    --num_layers_enc 2 \
    --num_layers_pred 2 \
    --num_patches 16 \
    --num_experts 4 \
    --num_routers 1 \
    --top_k 2 \
    --num_heads 4 \
    --dropout 0.5 \
    --n_runs $n_runs_override \
    --seed $run_seed \
    --run_suffix "$run_suffix" \
    --gate_loss_weight 0.01 \
    --save True \
    --use_common_ids True
done
