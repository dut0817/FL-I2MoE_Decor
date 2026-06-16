#!/bin/bash
set -euo pipefail

export device="${device:-${DEVICE:-0}}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"
cd "${repo_root}"

lambda_grid="${LAMBDA_GRID:-1e-05 3e-05 1e-04 3e-04 1e-03 3e-03 1e-02 5e-02 1e-01 0.2 0.5 1 2 3 4 6 8 10}"
rep_cka_loss_weight_grid="${REP_CKA_LOSS_WEIGHT:-${CKA_GRID:-${CKA_LOSS_WEIGHT:-$lambda_grid}}}"
setting="${SYNTHETIC_SETTING:-redundancy}"
synthetic_root="${SYNTHETIC_ROOT:-${repo_root}/data/synthetic}"
synthetic_pickle="${SYNTHETIC_PICKLE:-${synthetic_root}/VEC2XOR_DATA_${setting}.pickle}"

for rep_cka_loss_weight in $rep_cka_loss_weight_grid
do
CUDA_VISIBLE_DEVICES=$device python src/imoe/train_transformer.py \
    --data synthetic \
    --synthetic_pickle $synthetic_pickle \
    --train_epochs 10 \
    --modality 01 \
    --fusion_sparse False \
    --batch_size 128 \
    --hidden_dim 128 \
    --num_layers_fus 2 \
    --num_layers_enc 1 \
    --num_layers_pred 2 \
    --num_patches 4 \
    --num_experts 4 \
    --num_routers 1 \
    --top_k 2 \
    --num_heads 4 \
    --dropout 0.5 \
    --lr 1e-4 \
    --temperature_rw 1.0 \
    --hidden_dim_rw 256 \
    --num_layer_rw 3 \
    --interaction_loss_weight 0.5 \
    --regularizer rep-cka \
    --rep_cka_loss_weight $rep_cka_loss_weight \
    --save_subdir rep_cka \
    --n_runs 3 \
    --seed 0 \
    --gate None \
    --gate_loss_weight 0.01 \
    --save True \
    --use_common_ids True
done
