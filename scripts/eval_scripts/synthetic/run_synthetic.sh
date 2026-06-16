#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
data_dir="${SYNTHETIC_DATA_DIR:-${repo_dir}/data/synthetic}"

cd "$repo_dir"
mkdir -p outputs

setting="${SETTING:-redundancy}"
modality="${MODALITY:-01}"

# Matches train_scripts/imoe/rep-cka/run_synthetic_vec2.sh defaults.
ckpt_glob="${CKPT_GLOB:-${repo_dir}/saves/imoe/transformer/synthetic/synthetic_dataset_dim200200/${setting}/seed_*_modality_${modality}_train_epochs_10_val_acc_*_cka_*_run*.pth}"

PYTHONPATH="$repo_dir" \
    "$python_bin" \
    -m src.common.evaluation.synthetic_eval_checkpoints \
  --ckpt_glob "$ckpt_glob" \
  --synthetic_pickle "${data_dir}/VEC2XOR_DATA_${setting}.pickle" \
  --modality "$modality" \
  --setting "$setting" \
  --temperature_rw "${TEMPERATURE_RW:-1.0}" \
  --out_csv "${OUT_CSV:-${repo_dir}/outputs/eval_per_ckpt_vec2_${setting}_dim200200_allcka.csv}" \
  --out_summary_csv "${OUT_SUMMARY_CSV:-${repo_dir}/outputs/eval_summary_vec2_${setting}_dim200200_allcka.csv}"
