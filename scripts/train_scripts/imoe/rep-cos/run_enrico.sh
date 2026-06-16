#!/bin/bash
set -euo pipefail

export device="${device:-${DEVICE:-0}}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"
cd "${repo_root}"

lambda_grid="${LAMBDA_GRID:-1e-05 3e-05 1e-04 3e-04 1e-03 3e-03 1e-02 5e-02 1e-01 0.2 0.5 1 2 3 4 6 8 10}"
rep_cos_loss_weight_grid="${REP_COS_LOSS_WEIGHT:-$lambda_grid}"
run_seed="${RUN_SEED:-0}"
n_runs_override="${N_RUNS_OVERRIDE:-3}"
save_subdir="${SAVE_SUBDIR:-rep_cos}"

for lr in 0.0001
do
for temperature_rw in 2.0
do
for hidden_dim_rw in 256
do
for num_layer_rw in 3
do
for interaction_loss_weight in 0.0
do
for num_heads in 4
do
for rep_cos_loss_weight in $rep_cos_loss_weight_grid
do
CUDA_VISIBLE_DEVICES=$device python src/imoe/train_transformer.py \
    --temperature_rw $temperature_rw \
    --hidden_dim_rw $hidden_dim_rw \
    --num_layer_rw $num_layer_rw \
    --interaction_loss_weight $interaction_loss_weight \
    --lr $lr \
    --data enrico \
    --gate None \
    --train_epochs 50 \
    --modality SW \
    --fusion_sparse False \
    --batch_size 32 \
    --hidden_dim 256 \
    --num_layers_fus 2 \
    --num_layers_enc 2 \
    --num_layers_pred 2 \
    --num_patches 8 \
    --num_experts 2 \
    --num_routers 1 \
    --top_k 2 \
    --num_heads $num_heads \
    --dropout 0.5 \
    --n_runs $n_runs_override \
    --seed $run_seed \
    --gate_loss_weight 0.01 \
    --save True \
    --use_common_ids True \
    --regularizer rep-cos \
    --rep_cos_loss_weight $rep_cos_loss_weight \
    --save_subdir "$save_subdir"
done
done
done
done
done
done
done
