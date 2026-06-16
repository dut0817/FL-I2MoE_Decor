# Evaluation scripts

Masking-faithfulness evaluation scripts are organized like `train_scripts`:

- `imoe/rep-cos`, `imoe/rep-cka`, `imoe/rep-barlow` call `src.common.evaluation.masking_eval_imoe`.
- `gshardgate/vanilla`, `gshardgate/rep-cka` call `src.common.evaluation.masking_eval_gshardgate`.
- `synthetic/run_synthetic.sh` calls `src.common.evaluation.synthetic_eval_checkpoints`.

Each method folder has one `run_<dataset>.sh` script for `mimic`, `mmimdb`, and `enrico`.
The default split is `valid`; use `EVAL_SPLIT=test` for test evaluation.
Evaluation implementations live in `src/common/evaluation`; these shell scripts are wrappers only.
Useful overrides: `LAMBDA_GRID`, `CKPT_DIR`, `IMPORTANCE_MODE`, `OUT_DIR`, `PYTHON_BIN`, `DEVICE`.
