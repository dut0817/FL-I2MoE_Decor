# Does Role Specialization Matter for Explanation Faithfulness in Mixture-of-Experts?

Official implementation for "Does Role Specialization Matter for Explanation Faithfulness in Mixture-of-Experts?" accepted by ECML PKDD 2026.

- Authors: Yeji Kim, Housam Babiker, Mi-Young Kim, and Randy Goebel

The paper studies whether role-specialized experts improve explanation faithfulness in multimodal mixture-of-experts models by comparing interaction-aware I2MoE against standard sparse MoE baselines with representation regularization. The repository includes I2MoE training with representation-level regularizers, GShardGate baselines, and masking-based faithfulness evaluation for multimodal datasets.

## Relation to I2MoE

This repository builds on the official I2MoE implementation:

Jiayi Xin et al., "I2MoE: Interpretable Multimodal Interaction-aware Mixture-of-Experts"  
Code: https://github.com/Raina-Xin/I2MoE

The original I2MoE code is released under the MIT License. This repository contains modifications for FL-I2MoE Decor, including representation regularizers and evaluation scripts for MIMIC-IV, MMIMDb, ENRICO, and synthetic experiments.

## Environment Setup

Install FastMoE first if it is not already available in your environment:

```bash
git clone https://github.com/laekov/fastmoe.git
cd fastmoe
pip install -e .
cd ..
```

Then create the environment for this repository:

```bash
conda create -n fli2moe-decor python=3.10 -y
conda activate fli2moe-decor
pip install -r requirements.txt
```

## Prerequisites

Before running training or evaluation, prepare the pretrained encoder checkpoints locally. By default, the code expects Hugging Face-style model folders under `models/`:

```text
models/roberta-base/
models/clip-vit-base-patch16/
```


## Datasets

Please download each dataset from its original source and place the processed files under the `data/` directory.

### Dataset Sources

MIMIC-IV  
https://physionet.org/content/mimiciv/

MMIMDb  
https://github.com/johnarevalo/gmu-mmimdb

ENRICO  
https://github.com/pliang279/MultiBench

### Synthetic Dataset

The synthetic experiments use HD-XOR / VEC2XOR datasets generated with the InterSHAP data generator. In InterSHAP, run the generator and save the output under this repository's `data/synthetic/` directory:

```bash
bash generate_data/generate_VEC2.sh ../I2MoE_decor/data/synthetic
```

The default training and evaluation scripts expect files such as:

```text
data/synthetic/VEC2XOR_DATA_redundancy.pickle
```

Supported synthetic settings include `uniqueness0`, `uniqueness1`, `synergy`, and `redundancy`. Each generated pickle contains `train`, `valid`, and `test` splits with modality keys such as `0`, `1`, and a `label` field.


## Training

### Train FL-I2MoE Decor

FL-I2MoE Decor uses interaction-aware experts and representation-level regularization.

Available regularizers:

- `rep-cos`: penalizes squared cosine similarity between normalized expert latent representations.
- `rep-cka`: penalizes representation similarity using CKA-style dependence.
- `rep-barlow`: penalizes off-diagonal cross-correlation between expert representations.

Training scripts follow this pattern:

```bash
DEVICE=<gpu_id> bash scripts/train_scripts/imoe/<regularizer>/run_<dataset>.sh
```

where `<regularizer>` is one of `rep-cos`, `rep-cka`, or `rep-barlow`, and `<dataset>` is one of `mimic`, `mmimdb`, or `enrico`.


### Train GShardGate Baselines

The GShardGate baseline is a standard sparse MoE without I2MoE role assignment.

Training scripts follow this pattern:

```bash
DEVICE=<gpu_id> bash scripts/train_scripts/gshardgate/<method>/run_<dataset>.sh
```

where `<method>` is `vanilla` or `rep-cka`, and `<dataset>` is one of `mimic`, `mmimdb`, or `enrico`.


## Evaluation Overview

The repository includes two evaluation families:

Masking Evaluation

- feature/token-level attribution
- top-K percent masking
- comprehensiveness and sufficiency-style faithfulness scores
- drop-curve/AOPC metrics

Synthetic Checkpoint Evaluation

- checkpoint-level performance summaries
- expert routing summaries for synthetic settings

## Masking Evaluation

### Run FL-I2MoE Decor Masking

Evaluation scripts follow this pattern:

```bash
DEVICE=<gpu_id> bash scripts/eval_scripts/imoe/<regularizer>/run_<dataset>.sh
```

where `<regularizer>` is one of `rep-cos`, `rep-cka`, or `rep-barlow`, and `<dataset>` is one of `mimic`, `mmimdb`, or `enrico`.

### Run GShardGate Masking

Evaluation scripts follow this pattern:

```bash
DEVICE=<gpu_id> bash scripts/eval_scripts/gshardgate/<method>/run_<dataset>.sh
```

where `<method>` is `vanilla` or `rep-cka`, and `<dataset>` is one of `mimic`, `mmimdb`, or `enrico`.

## Synthetic Evaluation

```bash
DEVICE=0 bash scripts/eval_scripts/synthetic/run_synthetic.sh
```

