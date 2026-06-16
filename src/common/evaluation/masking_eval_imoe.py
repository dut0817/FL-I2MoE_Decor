#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified I2MoE masking evaluation for 3 datasets:
  - MIMIC   (binary classification)
  - MMIMDb  (multi-label classification)
  - ENRICO  (multiclass classification)

Key behavior:
  - Per-modality top-K% masking (not global top-K across modalities)
  - Comprehensiveness masks top-K important tokens; sufficiency keeps only top-K important tokens
  - Importance mode: attn | attnXgrad | ig
  - Computes comprehensiveness, sufficiency, and AOPC (ERASER probability-difference)
  - AOPC aggregation: mean over K bins (default) or normalized trapz AUC
  - Supports multiple checkpoints and saves mean/std CSV
"""

import os
import sys
import gc
import argparse
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score,
    accuracy_score,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(os.path.dirname(cwd))

from src.common.datasets.mimic import load_and_preprocess_data_mimic
from src.common.datasets.mmimdb import load_and_preprocess_data_mmimdb
from src.common.datasets.enrico import load_and_preprocess_data_enrico
from src.common.datasets.MultiModalDataset import create_loaders
from src.common.fusion_models.transformer import Transformer
from src.imoe.InteractionMoE import InteractionMoE
from src.imoe.imoe_train import _encode_batch as imoe_encode_batch
from src.imoe.imoe_train import _to_device_tree as imoe_to_device_tree


TEXT_LIKE_NAMES = {"language", "text", "txt", "note", "notes"}
FINAL_OUTPUT_COLS = [
    "K_percent",
    "orig_acc",
    "orig_micro_f1",
    "orig_macro_f1",
    "comprehensiveness_prob_mean_identified",
    "sufficiency_prob_mean_identified",
    "comprehensiveness_prob_mean_random",
    "sufficiency_prob_mean_random",
    "aopc_comprehensiveness_prob_identified",
    "aopc_sufficiency_prob_identified",
    "aopc_comprehensiveness_prob_random",
    "aopc_sufficiency_prob_random",
]
GAP_OUTPUT_COLS = [
    "comprehensiveness_prob_gap",
    "sufficiency_prob_gap",
    "aopc_comprehensiveness_prob_gap",
    "aopc_sufficiency_prob_gap",
]
OUTPUT_COLS_WITH_GAPS = FINAL_OUTPUT_COLS + GAP_OUTPUT_COLS


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y", "t"}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True, choices=["mimic", "mmimdb", "enrico"])
    p.add_argument("--ckpt", type=str, nargs="+", required=True, help="iMoE checkpoint path(s)")
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", type=str2bool, default=True)
    p.add_argument("--modality", type=str, default=None)

    p.add_argument("--hidden_dim", type=int, default=None)
    p.add_argument("--num_patches", type=int, default=None)
    p.add_argument("--num_layers_fus", type=int, default=2)
    p.add_argument("--num_layers_pred", type=int, default=2)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--fusion_sparse", type=str2bool, default=False)
    p.add_argument("--gate", type=str, default="None")
    p.add_argument("--num_experts", type=int, default=4)
    p.add_argument("--num_routers", type=int, default=1)
    p.add_argument("--top_k", type=int, default=2)

    p.add_argument("--temperature_rw", type=float, default=None)
    p.add_argument("--hidden_dim_rw", type=int, default=256)
    p.add_argument("--num_layer_rw", type=int, default=None)

    p.add_argument("--Ks", type=str, default="5,10,15,20,25")
    p.add_argument("--random_repeats", type=int, default=10)
    p.add_argument("--importance_mode", type=str, default="attnXgrad", choices=["attn", "attnXgrad", "ig"])
    p.add_argument(
        "--target_source",
        type=str,
        default="pred",
        choices=["true", "pred"],
        help="Faithfulness target source: true labels or model prediction on original input.",
    )
    p.add_argument(
        "--aopc_mode",
        type=str,
        default="mean",
        choices=["mean", "auc"],
        help="AOPC aggregation: mean over K bins (ERASER-like) or normalized trapz AUC.",
    )
    p.add_argument("--ig_steps", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--mask_fill_strategy",
        type=str,
        default="mean",
        choices=["zero", "mean"],
        help="Mask fill strategy for selected tokens: zero or dataset-level modality mean.",
    )

    p.add_argument("--as_percent", type=str2bool, default=True)
    p.add_argument("--suffix", type=str, default="")
    p.add_argument("--out_dir", type=str, default="")
    p.add_argument(
        "--eval_split",
        type=str,
        default="valid",
        choices=["valid", "test"],
        help="Dataset split used for masking-faithfulness evaluation.",
    )
    p.add_argument(
        "--auto_split_suffix",
        type=str2bool,
        default=True,
        help="Append '_val' or '_test' to output filename suffix automatically.",
    )

    p.add_argument("--strict_enc", type=str2bool, default=True)
    p.add_argument("--strict_model", type=str2bool, default=False)
    return p.parse_args()


def normalize_device_arg(s):
    s = str(s).strip()
    if s == "cpu":
        return 0, "cpu"
    if s.startswith("cuda:"):
        idx = int(s.split(":")[1])
        if torch.cuda.is_available():
            return idx, s
        return idx, "cpu"
    if s.isdigit():
        idx = int(s)
        if torch.cuda.is_available():
            return idx, f"cuda:{idx}"
        return idx, "cpu"
    return 0, s


def apply_dataset_defaults(args):
    defaults = {
        "mimic": {
            "modality": "LNC",
            "batch_size": 32,
            "hidden_dim": 128,
            "num_patches": 4,
            "num_layer_rw": 3,
            "temperature_rw": 1.0,
        },
        "mmimdb": {
            "modality": "LI",
            "batch_size": 8,
            "hidden_dim": 256,
            "num_patches": 16,
            "num_layer_rw": 2,
            "temperature_rw": 2.0,
        },
        "enrico": {
            "modality": "SW",
            "batch_size": 32,
            "hidden_dim": 256,
            "num_patches": 8,
            "num_layer_rw": 3,
            "temperature_rw": 2.0,
        },
    }[args.dataset]

    for k, v in defaults.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    return args


@torch.no_grad()
def metrics_binary(logits, labels):
    logits_cpu = logits.detach().cpu()
    labs = labels.long().detach().cpu().numpy()

    preds = logits_cpu.argmax(dim=1).numpy()
    acc = accuracy_score(labs, preds)
    micro = f1_score(labs, preds, average="micro", zero_division=0)
    macro = f1_score(labs, preds, average="macro", zero_division=0)
    return {"acc": float(acc), "micro_f1": float(micro), "macro_f1": float(macro)}


@torch.no_grad()
def metrics_multiclass(logits, labels):
    preds = logits.argmax(dim=1).detach().cpu().numpy()
    labs = labels.long().detach().cpu().numpy()
    acc = accuracy_score(labs, preds)
    micro = f1_score(labs, preds, average="micro", zero_division=0)
    macro = f1_score(labs, preds, average="macro", zero_division=0)
    return {"acc": float(acc), "micro_f1": float(micro), "macro_f1": float(macro)}


@torch.no_grad()
def metrics_multilabel(logits, labels):
    preds = (torch.sigmoid(logits) > 0.5).long().detach().cpu().numpy()
    labs = labels.long().detach().cpu().numpy()
    acc = accuracy_score(labs, preds)
    micro = f1_score(labs, preds, average="micro", zero_division=0)
    macro = f1_score(labs, preds, average="macro", zero_division=0)
    return {"acc": float(acc), "micro_f1": float(micro), "macro_f1": float(macro)}


def compute_metrics(logits, labels, task_type):
    if task_type == "binary":
        return metrics_binary(logits, labels)
    elif task_type == "multiclass":
        return metrics_multiclass(logits, labels)
    elif task_type == "multilabel":
        return metrics_multilabel(logits, labels)
    else:
        raise ValueError(f"Unknown task_type={task_type}")


def build_true_target(logits, labels, task_type):
    """
    Build true-label-based targets to align attribution and
    faithfulness metrics on the same criterion.
    """
    bsz, n_cls = logits.shape
    if labels is None:
        raise ValueError("labels must be provided for true-label target construction.")

    if task_type in {"binary", "multiclass"}:
        if labels.dim() == 2 and labels.size(1) == n_cls:
            return labels.argmax(dim=1).long()
        return labels.long().view(-1)
    if task_type == "multilabel":
        target = labels.float()
        if target.dim() == 1:
            target = F.one_hot(target.long(), num_classes=n_cls).float()
        target = target.view(bsz, n_cls)
        pos_cnt = target.sum(dim=1, keepdim=True)
        no_pos = pos_cnt <= 0
        if no_pos.any():
            top1 = logits.argmax(dim=1, keepdim=True)
            top1_oh = torch.zeros_like(target)
            top1_oh.scatter_(1, top1, 1.0)
            no_pos_expand = no_pos.expand_as(target)
            target = torch.where(no_pos_expand, top1_oh, target)
        return target

    raise ValueError(f"Unknown task_type={task_type}")


def build_pred_target(logits, task_type):
    bsz, n_cls = logits.shape

    if task_type in {"binary", "multiclass"}:
        return logits.argmax(dim=1).long().view(-1)

    if task_type == "multilabel":
        probs = torch.sigmoid(logits)
        target = (probs >= 0.5).float().view(bsz, n_cls)
        pos_cnt = target.sum(dim=1, keepdim=True)
        no_pos = pos_cnt <= 0
        if no_pos.any():
            top1 = logits.argmax(dim=1, keepdim=True)
            top1_oh = torch.zeros_like(target)
            top1_oh.scatter_(1, top1, 1.0)
            target = torch.where(no_pos.expand_as(target), top1_oh, target)
        return target

    raise ValueError(f"Unknown task_type={task_type}")


def build_eval_target(logits, labels, task_type, target_source):
    if target_source == "true":
        return build_true_target(logits, labels, task_type)
    if target_source == "pred":
        return build_pred_target(logits, task_type)
    raise ValueError(f"Unknown target_source={target_source}")


def target_probability_per_sample(logits, labels, task_type):
    bsz, n_cls = logits.shape

    if task_type in {"binary", "multiclass"}:
        probs = torch.softmax(logits, dim=1)
        if labels is None:
            idx = logits.argmax(dim=1)
        else:
            if labels.dim() == 2 and labels.size(1) == n_cls:
                idx = labels.argmax(dim=1)
            else:
                idx = labels.long().view(-1)
        return probs.gather(1, idx.long().view(-1, 1)).squeeze(1)

    if task_type == "multilabel":
        probs = torch.sigmoid(logits)
        if labels is None:
            target = (probs >= 0.5).float()
        else:
            target = labels.float()
            if target.dim() == 1:
                target = F.one_hot(target.long(), num_classes=n_cls).float()
            target = target.view(bsz, n_cls)

        pos_cnt = target.sum(dim=1, keepdim=True)
        no_pos = pos_cnt <= 0
        if no_pos.any():
            top1 = logits.argmax(dim=1, keepdim=True)
            top1_oh = torch.zeros_like(target)
            top1_oh.scatter_(1, top1, 1.0)
            no_pos_expand = no_pos.expand_as(target)
            target = torch.where(no_pos_expand, top1_oh, target)
            pos_cnt = target.sum(dim=1, keepdim=True)

        return (probs * target).sum(dim=1) / pos_cnt.squeeze(1).clamp_min(1.0)

    raise ValueError(f"Unknown task_type={task_type}")


def logprob_target_per_sample(logits, labels, task_type):
    bsz, n_cls = logits.shape

    if task_type in {"binary", "multiclass"}:
        if labels is None:
            idx = logits.argmax(dim=1)
        else:
            if labels.dim() == 2 and labels.size(1) == n_cls:
                idx = labels.argmax(dim=1)
            else:
                idx = labels.long().view(-1)
        idx = idx.long()
        return F.log_softmax(logits, dim=1).gather(1, idx.view(-1, 1)).squeeze(1)

    if task_type == "multilabel":
        if labels is None:
            target = (torch.sigmoid(logits) >= 0.5).float()
        else:
            target = labels.float()
            if target.dim() == 1:
                target = F.one_hot(target.long(), num_classes=n_cls).float()
            target = target.view(bsz, n_cls)

        pos_cnt = target.sum(dim=1, keepdim=True)
        no_pos = pos_cnt <= 0
        if no_pos.any():
            top1 = logits.argmax(dim=1, keepdim=True)
            top1_oh = torch.zeros_like(target)
            top1_oh.scatter_(1, top1, 1.0)
            no_pos_expand = no_pos.expand_as(target)
            target = torch.where(no_pos_expand, top1_oh, target)
            pos_cnt = target.sum(dim=1, keepdim=True)

        eps = 1e-12
        lp_pos = torch.log(torch.sigmoid(logits).clamp_min(eps))
        return (lp_pos * target).sum(dim=1) / pos_cnt.squeeze(1).clamp_min(1.0)

    raise ValueError(f"Unknown task_type={task_type}")


def target_logit_per_sample(logits, labels, task_type):
    """
    Per-sample target logit used for attribution backprop.
    """
    bsz, n_cls = logits.shape

    if task_type in {"binary", "multiclass"}:
        if labels is None:
            idx = logits.argmax(dim=1)
        else:
            if labels.dim() == 2 and labels.size(1) == n_cls:
                idx = labels.argmax(dim=1)
            else:
                idx = labels.long().view(-1)
        idx = idx.long()
        return logits.gather(1, idx.view(-1, 1)).squeeze(1)

    if task_type == "multilabel":
        if labels is None:
            target = (torch.sigmoid(logits) >= 0.5).float()
        else:
            target = labels.float()
            if target.dim() == 1:
                target = F.one_hot(target.long(), num_classes=n_cls).float()
            target = target.view(bsz, n_cls)

        pos_cnt = target.sum(dim=1, keepdim=True)
        no_pos = pos_cnt <= 0
        if no_pos.any():
            top1 = logits.argmax(dim=1, keepdim=True)
            top1_oh = torch.zeros_like(target)
            top1_oh.scatter_(1, top1, 1.0)
            no_pos_expand = no_pos.expand_as(target)
            target = torch.where(no_pos_expand, top1_oh, target)
            pos_cnt = target.sum(dim=1, keepdim=True)

        return (logits * target).sum(dim=1) / pos_cnt.squeeze(1).clamp_min(1.0)

    raise ValueError(f"Unknown task_type={task_type}")


def build_logit_score(logits, labels, task_type):
    return target_logit_per_sample(logits, labels, task_type).sum()


def build_logprob_score(logits, labels, task_type):
    return logprob_target_per_sample(logits, labels, task_type).sum()


def normalized_auc(k_values: np.ndarray, y_values: np.ndarray) -> float:
    if k_values.size == 0 or y_values.size == 0:
        return float("nan")
    order = np.argsort(k_values)
    x = k_values[order]
    y = y_values[order]
    if x.size == 1:
        return float(y[0])
    area = float(np.trapz(y, x))
    width = float(x[-1] - x[0])
    if width <= 0.0:
        return area
    return area / width


def aggregate_aopc(k_values: np.ndarray, y_values: np.ndarray, mode: str) -> float:
    if y_values.size == 0:
        return float("nan")
    if mode == "mean":
        return float(np.nanmean(y_values))
    if mode == "auc":
        return normalized_auc(k_values, y_values)
    raise ValueError(f"Unknown aopc_mode={mode}")


def aggregate_eraser_aopc(k_values: np.ndarray, per_k_instance_values, mode: str) -> float:
    """
    ERASER-style aggregation over (instance, K):
      - mean: flatten all per-instance deltas across K then average.
      - auc: first compute per-K mean over instances, then normalized trapz over K.
    """
    xs = []
    vals = []
    for k, v in zip(k_values, per_k_instance_values):
        arr = np.asarray(v, dtype=float).reshape(-1)
        if arr.size == 0:
            continue
        xs.append(float(k))
        vals.append(arr)

    if len(vals) == 0:
        return float("nan")

    if mode == "mean":
        return float(np.nanmean(np.concatenate(vals, axis=0)))

    if mode == "auc":
        y = np.array([np.nanmean(v) for v in vals], dtype=float)
        x = np.array(xs, dtype=float)
        return normalized_auc(x, y)

    raise ValueError(f"Unknown aopc_mode={mode}")


def _set_attn_capture_flags(model, capture=True, require_grad=False):
    if model is None:
        return
    for m in model.modules():
        if hasattr(m, "capture_attn") and hasattr(m, "require_attn_grad"):
            m.capture_attn = capture
            m.require_attn_grad = require_grad


def _clear_last_attn(model):
    if model is None:
        return
    for m in model.modules():
        if hasattr(m, "last_attn"):
            m.last_attn = None


def collect_attns_from_transformer(transformer_model):
    attn_list = []
    for m in transformer_model.modules():
        if hasattr(m, "last_attn"):
            if m.last_attn is not None:
                attn_list.append(m.last_attn)
    if not attn_list:
        raise RuntimeError("No attention maps found (last_attn is empty).")
    return attn_list


def attention_rollout(attn_list, head_fusion="mean", add_residual=True):
    fused = []
    for A in attn_list:
        if head_fusion == "mean":
            A = A.mean(dim=1)
        elif head_fusion == "max":
            A = A.max(dim=1).values
        else:
            raise ValueError("head_fusion must be 'mean' or 'max'")

        if add_residual:
            I = torch.eye(A.size(-1), device=A.device).unsqueeze(0)
            A = A + I

        A = A / A.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        fused.append(A)

    R = fused[0]
    for A in fused[1:]:
        R = A @ R
    return R.mean(dim=1)


def _minmax_norm_masked(x, m):
    if x.numel() == 0:
        return x
    m_bool = m.to(torch.bool)
    big_pos = torch.finfo(x.dtype).max
    big_neg = torch.finfo(x.dtype).min
    x_min = torch.where(m_bool, x, torch.full_like(x, big_pos))
    x_max = torch.where(m_bool, x, torch.full_like(x, big_neg))
    min_v = x_min.min(dim=1, keepdim=True).values
    max_v = x_max.max(dim=1, keepdim=True).values
    denom = (max_v - min_v).clamp_min(1e-12)
    x_norm = (x - min_v) / denom
    return torch.where(m_bool, x_norm, x)


def normalize_importances_per_modality(imps, masks, modal_names):
    out = []
    for imp, m, name in zip(imps, masks, modal_names):
        if m is None:
            m = torch.ones_like(imp, device=imp.device)
        m = m.float()
        if name.lower() in TEXT_LIKE_NAMES and imp.size(1) > 1:
            cls = imp[:, :1]
            body = imp[:, 1:]
            m_body = m[:, 1:]
            if body.numel() > 0:
                body = _minmax_norm_masked(body, m_body)
            out.append(torch.cat([cls, body], dim=1))
        else:
            out.append(_minmax_norm_masked(imp, m))
    return out


def get_expert_fusions(ensemble_model):
    if not hasattr(ensemble_model, "interaction_experts"):
        return []
    fusions = []
    for ex in ensemble_model.interaction_experts:
        fus = getattr(ex, "fusion_model", None)
        if fus is not None:
            fusions.append(fus)
    return fusions


def split_importance_by_modalities(imp_global, lengths):
    out = []
    start = 0
    for L in lengths:
        out.append(imp_global[:, start:start + L])
        start += L
    return out


def _extract_outputs_and_routing(forward_out, num_expected_experts=None):
    outputs = None
    routing = None

    if isinstance(forward_out, (tuple, list)) and len(forward_out) >= 3:
        if torch.is_tensor(forward_out[2]):
            outputs = forward_out[2]
        if torch.is_tensor(forward_out[1]) and forward_out[1].dim() == 2:
            routing = forward_out[1]
        if outputs is not None:
            return outputs, routing

    tensors = []
    if torch.is_tensor(forward_out):
        tensors.append(forward_out)
    elif isinstance(forward_out, (tuple, list)):
        for x in forward_out:
            if torch.is_tensor(x):
                tensors.append(x)

    two_d = [t for t in tensors if t.dim() == 2 and t.size(0) > 0]
    if num_expected_experts is not None:
        for t in two_d:
            if t.size(1) == num_expected_experts:
                routing = t
                break
    if outputs is None:
        non_routing = [t for t in two_d if routing is None or t is not routing]
        if non_routing:
            outputs = non_routing[-1]
        elif two_d:
            outputs = two_d[-1]

    return outputs, routing


def forward_logits_and_routing(ensemble_model, feats, masks):
    expected_e = None
    if hasattr(ensemble_model, "interaction_experts"):
        expected_e = len(ensemble_model.interaction_experts)

    if hasattr(ensemble_model, "inference"):
        out = ensemble_model.inference(feats, masks)
    else:
        out = ensemble_model(feats, masks)

    logits, routing = _extract_outputs_and_routing(out, num_expected_experts=expected_e)
    if logits is None:
        raise RuntimeError("Failed to extract logits from model output.")
    return logits, routing


def ensure_mask_list(feats, masks, device):
    out = []
    for f, m in zip(feats, masks):
        if m is None:
            out.append(torch.ones((f.size(0), f.size(1)), device=device))
        else:
            out.append(m.to(device).float())
    return out


def get_valid_row(mask_row, modal_name):
    valid = mask_row.to(torch.bool)
    if modal_name.lower() in TEXT_LIKE_NAMES and valid.numel() > 1:
        valid = valid.clone()
        valid[0] = False
    return valid


def extract_batch_samples_labels(batch):
    if not isinstance(batch, (tuple, list)):
        raise TypeError("Expected loader batch as tuple/list.")
    # Validation loaders use 4 fields: (samples, labels, mcs, observeds)
    if len(batch) == 4:
        return batch[0], batch[1]
    # Test loaders use 5 fields: (samples, sample_ids, labels, mcs, observeds)
    if len(batch) >= 5:
        return batch[0], batch[2]
    raise TypeError(
        f"Unexpected loader batch length={len(batch)}. Expected 4 or 5 fields."
    )


@torch.no_grad()
def compute_dataset_modality_mean_vectors(data_loader, encoder_dict, device, ns):
    sum_vecs = None
    valid_counts = None

    for batch in data_loader:
        if not isinstance(batch, (tuple, list)) or len(batch) == 0:
            raise TypeError("Expected loader batch to be a non-empty tuple/list.")
        batch_samples = batch[0]
        batch_samples = imoe_to_device_tree(batch_samples, device)

        feats, masks, modal_names = imoe_encode_batch(batch_samples, encoder_dict, ns)
        masks = ensure_mask_list(feats, masks, device)

        if sum_vecs is None:
            sum_vecs = [torch.zeros(f.size(-1), device=f.device, dtype=f.dtype) for f in feats]
            valid_counts = [0 for _ in feats]

        for mi, (feat_m, mask_m, name_m) in enumerate(zip(feats, masks, modal_names)):
            bsz = feat_m.size(0)
            valid = torch.stack([get_valid_row(mask_m[b], name_m) for b in range(bsz)], dim=0)

            if valid.any():
                valid_feat = feat_m[valid]
                sum_vecs[mi] += valid_feat.sum(dim=0)
                valid_counts[mi] += valid_feat.size(0)

    if sum_vecs is None:
        raise RuntimeError("Failed to compute dataset-level modality means: loader is empty.")

    means = []
    for mi in range(len(sum_vecs)):
        if valid_counts[mi] > 0:
            means.append(sum_vecs[mi] / float(valid_counts[mi]))
        else:
            means.append(torch.zeros_like(sum_vecs[mi]))
    return means


def topk_mask_indices_per_sample(imp_row, K_percent, valid_mask_row, largest=True):
    cand_idx = torch.nonzero(valid_mask_row, as_tuple=False).squeeze(-1)
    if cand_idx.numel() == 0:
        return imp_row.new_empty(0, dtype=torch.long)
    m = max(1, int(np.ceil(cand_idx.numel() * (K_percent / 100.0))))
    cand_imp = imp_row[cand_idx]
    _, top_local = torch.topk(cand_imp, k=min(m, cand_imp.numel()), largest=largest, sorted=False)
    return cand_idx[top_local]


def complement_mask_indices(valid_mask_row, keep_idx):
    cand_idx = torch.nonzero(valid_mask_row, as_tuple=False).squeeze(-1)
    if cand_idx.numel() == 0:
        return cand_idx
    if keep_idx.numel() == 0:
        return cand_idx
    keep_flags = torch.zeros_like(valid_mask_row, dtype=torch.bool)
    keep_flags[keep_idx] = True
    return cand_idx[~keep_flags[cand_idx]]


def apply_modality_masking(
    feats,
    masks,
    modal_names,
    idx_groups,
    modal_means=None,
    fill_strategy="zero",
):
    # Keep signature for compatibility with existing call sites.
    del modal_names
    feats_masked = [t.clone() for t in feats]
    masks_masked = [None if m is None else m.clone() for m in masks]

    for mi in range(len(feats_masked)):
        mean_vec = None
        if fill_strategy == "mean" and modal_means is not None and mi < len(modal_means):
            mean_vec = modal_means[mi].to(device=feats_masked[mi].device, dtype=feats_masked[mi].dtype)
        for b, idx in enumerate(idx_groups[mi]):
            if idx.numel() == 0:
                continue
            if fill_strategy == "mean" and mean_vec is not None:
                feats_masked[mi][b, idx, :] = mean_vec.view(1, -1)
            else:
                feats_masked[mi][b, idx, :] = 0.0

    return feats_masked, masks_masked


def attn_importance_per_modality(ensemble_model, feats, masks, modal_names):
    fusions = get_expert_fusions(ensemble_model)
    if len(fusions) == 0:
        raise RuntimeError("No expert fusion models found for attention rollout.")

    was_training = ensemble_model.training
    ensemble_model.eval()

    with torch.no_grad():
        routing = ensemble_model.reweight(feats, masks).detach()

    lengths = [f.size(1) for f in feats]
    bsz = feats[0].size(0)
    device = feats[0].device
    acc_imps = [torch.zeros((bsz, L), device=device) for L in lengths]

    for e, fus in enumerate(fusions):
        _clear_last_attn(fus)
        _set_attn_capture_flags(fus, capture=True, require_grad=False)

        with torch.no_grad():
            _ = fus(feats, masks, return_latent=False)

        attn_list = collect_attns_from_transformer(fus)
        imp_global = attention_rollout(attn_list, head_fusion="mean", add_residual=True)
        imps_e = split_importance_by_modalities(imp_global, lengths)
        # Keep routing weighting as a heuristic in plain-attn mode,
        # but normalize once only after additive expert aggregation.
        w = routing[:, e].unsqueeze(1)
        for i in range(len(acc_imps)):
            acc_imps[i] += w * imps_e[i]

        _set_attn_capture_flags(fus, capture=False, require_grad=False)

    if was_training:
        ensemble_model.train(True)
    return normalize_importances_per_modality(acc_imps, masks, modal_names)


def grad_attn_importance_per_modality(ensemble_model, feats, masks, modal_names, task_type, target):
    fusions = get_expert_fusions(ensemble_model)
    if len(fusions) == 0:
        raise RuntimeError("No expert fusion models found for attnXgrad.")

    was_training = ensemble_model.training
    ensemble_model.eval()
    ensemble_model.zero_grad(set_to_none=True)

    # Routing does not depend on expert-internal attention tensors.
    # We use it only to build the final mixture target score.
    with torch.no_grad():
        routing = ensemble_model.reweight(feats, masks)

    for fus in fusions:
        _clear_last_attn(fus)
        _set_attn_capture_flags(fus, capture=True, require_grad=True)

    logits_list = []
    for fus in fusions:
        logits_list.append(fus(feats, masks, return_latent=False))
    all_logits = torch.stack(logits_list, dim=1)
    weighted_logits = (all_logits * routing.unsqueeze(-1)).sum(dim=1)

    # Backprop from final mixture target logit (not log-probability).
    score = build_logit_score(weighted_logits, target, task_type)
    score.backward()

    lengths = [f.size(1) for f in feats]
    bsz = feats[0].size(0)
    acc_imps = [torch.zeros((bsz, L), device=weighted_logits.device) for L in lengths]

    for fus in fusions:
        attn_list = collect_attns_from_transformer(fus)
        mul_layers = []
        for A in attn_list:
            G = A.grad
            if G is None:
                raise RuntimeError("Attention gradient is None in attnXgrad mode.")

            AG = torch.relu(A) * torch.relu(G)
            M = AG.mean(dim=1)
            I = torch.eye(M.size(-1), device=M.device).unsqueeze(0)
            M = M + I
            M = M / M.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            mul_layers.append(M)

        R = mul_layers[0]
        for M in mul_layers[1:]:
            R = M @ R
        imp_global = R.mean(dim=1)

        # Do not multiply routing again here:
        # gradients w.r.t. expert attentions already come from final mixture score.
        imps_e = split_importance_by_modalities(imp_global, lengths)
        for i in range(len(acc_imps)):
            acc_imps[i] += imps_e[i]

    for fus in fusions:
        _set_attn_capture_flags(fus, capture=False, require_grad=False)

    if was_training:
        ensemble_model.train(True)
    # Normalize once after additive expert aggregation, per modality.
    return normalize_importances_per_modality(acc_imps, masks, modal_names)


def ig_importance_per_modality(ensemble_model, feats, masks, modal_names, task_type, target, steps=16):
    was_training = ensemble_model.training
    ensemble_model.eval()

    base_feats = [torch.zeros_like(f) for f in feats]

    grads_sum = [torch.zeros_like(f) for f in feats]

    with torch.no_grad():
        logits_orig, _ = forward_logits_and_routing(ensemble_model, feats, masks)

    for s in range(1, steps + 1):
        alpha = float(s) / float(steps)
        x_alpha_list = []
        for x0, x in zip(base_feats, feats):
            xa = (x0 + alpha * (x - x0)).detach()
            xa.requires_grad_(True)
            x_alpha_list.append(xa)

        ensemble_model.zero_grad(set_to_none=True)
        logits_alpha, _ = forward_logits_and_routing(ensemble_model, x_alpha_list, masks)
        # IG also uses target logit objective for gradient accumulation.
        score = build_logit_score(logits_alpha, target, task_type)
        score.backward()

        for i, xa in enumerate(x_alpha_list):
            if xa.grad is None:
                raise RuntimeError("IG gradient is None.")
            grads_sum[i] += xa.grad.detach()

        del x_alpha_list, logits_alpha, score
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    imps = []
    for x0, x, gsum in zip(base_feats, feats, grads_sum):
        ig = (x - x0) * (gsum / float(steps))
        imps.append(ig.abs().mean(dim=-1))

    imps = normalize_importances_per_modality(imps, masks, modal_names)

    if was_training:
        ensemble_model.train(True)
    return imps


@torch.no_grad()
def compute_base_metrics(data_loader, encoder_dict, ensemble_model, device, ns, task_type, return_logits=False):
    all_logits, all_labels = [], []
    for batch in data_loader:
        batch_samples, batch_labels = extract_batch_samples_labels(batch)
        batch_samples = imoe_to_device_tree(batch_samples, device)
        batch_labels = batch_labels.to(device, non_blocking=True)

        feats, masks, _modal_names = imoe_encode_batch(batch_samples, encoder_dict, ns)
        masks = ensure_mask_list(feats, masks, device)
        logits, _ = forward_logits_and_routing(ensemble_model, feats, masks)
        all_logits.append(logits)
        all_labels.append(batch_labels)

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(logits, labels, task_type)
    if return_logits:
        return metrics, logits, labels
    return metrics


def run_full_test_drop_curve(eval_loader, encoder_dict, ensemble_model, device, ns, args, task_type, Ks, modal_means):
    csa_largest = True
    mask_fill_strategy = args.mask_fill_strategy
    print(f"[INFO] mask_fill_strategy={mask_fill_strategy}")
    print(
        f"[INFO] CSA metrics: comprehensiveness/sufficiency use ERASER-style "
        f"{args.target_source}-target probability difference."
    )
    print(f"[INFO] aopc_mode={args.aopc_mode}")
    print(f"[INFO] target_source={args.target_source}")

    base = compute_base_metrics(eval_loader, encoder_dict, ensemble_model, device, ns, task_type)
    base_acc = float(base.get("acc", float("nan")))
    base_micro_f1 = float(base.get("micro_f1", base_acc))
    base_macro_f1 = float(base.get("macro_f1", float("nan")))
    print(f"[BASE] acc={base_acc:.4f} | micro_f1={base_micro_f1:.4f} | macro_f1={base_macro_f1:.4f}")

    out = {
        K: {
            "comp_vals_idf": [],
            "suff_vals_idf": [],
        } for K in Ks
    }
    if args.random_repeats > 0:
        for K in Ks:
            out[K]["comp_vals_rnd"] = []
            out[K]["suff_vals_rnd"] = []

    if device.type == "cuda":
        gen = torch.Generator(device="cuda")
    else:
        gen = torch.Generator()
    gen.manual_seed(int(args.seed))

    for it, batch in enumerate(eval_loader):
        batch_samples, batch_labels = extract_batch_samples_labels(batch)
        batch_samples = imoe_to_device_tree(batch_samples, device)
        batch_labels = batch_labels.to(device, non_blocking=True)

        feats, masks, modal_names = imoe_encode_batch(batch_samples, encoder_dict, ns)
        masks = ensure_mask_list(feats, masks, device)
        bsz = feats[0].size(0)
        with torch.no_grad():
            logits_orig, _ = forward_logits_and_routing(ensemble_model, feats, masks)
        eval_target = build_eval_target(logits_orig.detach(), batch_labels, task_type, args.target_source)
        prob_orig_target = target_probability_per_sample(logits_orig, eval_target, task_type)

        if args.importance_mode == "attn":
            imps = attn_importance_per_modality(ensemble_model, feats, masks, modal_names)
        elif args.importance_mode == "attnXgrad":
            imps = grad_attn_importance_per_modality(
                ensemble_model, feats, masks, modal_names, task_type, eval_target
            )
        elif args.importance_mode == "ig":
            imps = ig_importance_per_modality(
                ensemble_model, feats, masks, modal_names, task_type, eval_target, steps=args.ig_steps
            )
        else:
            raise ValueError(f"Unknown importance_mode={args.importance_mode}")

        for K in Ks:
            csa_idx_groups = [[] for _ in imps]
            csa_keep_complement_idx_groups = [[] for _ in imps]

            for mi, (imp_m, mask_m, name_m) in enumerate(zip(imps, masks, modal_names)):
                for b in range(bsz):
                    valid_row = get_valid_row(mask_m[b], name_m)
                    csa_idx = topk_mask_indices_per_sample(
                        imp_m[b], K, valid_row, largest=csa_largest
                    )
                    csa_idx_groups[mi].append(csa_idx)
                    csa_keep_complement_idx_groups[mi].append(complement_mask_indices(valid_row, csa_idx))

            feats_comp, masks_comp = apply_modality_masking(
                feats,
                masks,
                modal_names,
                csa_idx_groups,
                modal_means=modal_means,
                fill_strategy=mask_fill_strategy,
            )
            feats_suff, masks_suff = apply_modality_masking(
                feats,
                masks,
                modal_names,
                csa_keep_complement_idx_groups,
                modal_means=modal_means,
                fill_strategy=mask_fill_strategy,
            )

            with torch.no_grad():
                logits_comp, _ = forward_logits_and_routing(ensemble_model, feats_comp, masks_comp)
                logits_suff, _ = forward_logits_and_routing(ensemble_model, feats_suff, masks_suff)
            prob_comp_target = target_probability_per_sample(logits_comp, eval_target, task_type)
            prob_suff_target = target_probability_per_sample(logits_suff, eval_target, task_type)
            comp_idf = prob_orig_target - prob_comp_target
            suff_idf = prob_orig_target - prob_suff_target
            out[K]["comp_vals_idf"].append(comp_idf.detach().cpu())
            out[K]["suff_vals_idf"].append(suff_idf.detach().cpu())

            if args.random_repeats > 0:
                comp_rnd_repeats = []
                suff_rnd_repeats = []
                for _r in range(args.random_repeats):
                    rnd_idx_groups = [[] for _ in imps]
                    rnd_keep_complement_idx_groups = [[] for _ in imps]
                    for mi, (mask_m, name_m) in enumerate(zip(masks, modal_names)):
                        for b in range(bsz):
                            m_cnt = csa_idx_groups[mi][b].numel()
                            valid_row = get_valid_row(mask_m[b], name_m)
                            if m_cnt == 0:
                                empty_idx = torch.empty(0, dtype=torch.long, device=feats[mi].device)
                                rnd_idx_groups[mi].append(empty_idx)
                                rnd_keep_complement_idx_groups[mi].append(complement_mask_indices(valid_row, empty_idx))
                                continue
                            cand = torch.nonzero(valid_row, as_tuple=False).squeeze(-1)
                            if cand.numel() == 0:
                                empty_idx = torch.empty(0, dtype=torch.long, device=feats[mi].device)
                                rnd_idx_groups[mi].append(empty_idx)
                                rnd_keep_complement_idx_groups[mi].append(empty_idx)
                                continue
                            perm = cand[torch.randperm(cand.numel(), generator=gen, device=cand.device)[:m_cnt]]
                            rnd_idx_groups[mi].append(perm)
                            rnd_keep_complement_idx_groups[mi].append(complement_mask_indices(valid_row, perm))

                    feats_rnd, masks_rnd = apply_modality_masking(
                        feats,
                        masks,
                        modal_names,
                        rnd_idx_groups,
                        modal_means=modal_means,
                        fill_strategy=mask_fill_strategy,
                    )
                    feats_rnd_keep, masks_rnd_keep = apply_modality_masking(
                        feats,
                        masks,
                        modal_names,
                        rnd_keep_complement_idx_groups,
                        modal_means=modal_means,
                        fill_strategy=mask_fill_strategy,
                    )

                    with torch.no_grad():
                        logits_rnd, _ = forward_logits_and_routing(ensemble_model, feats_rnd, masks_rnd)
                        logits_rnd_keep, _ = forward_logits_and_routing(ensemble_model, feats_rnd_keep, masks_rnd_keep)
                    prob_rnd_target = target_probability_per_sample(logits_rnd, eval_target, task_type)
                    prob_rnd_keep_target = target_probability_per_sample(logits_rnd_keep, eval_target, task_type)
                    comp_rnd = prob_orig_target - prob_rnd_target
                    suff_rnd = prob_orig_target - prob_rnd_keep_target
                    comp_rnd_repeats.append(comp_rnd.detach())
                    suff_rnd_repeats.append(suff_rnd.detach())

                if comp_rnd_repeats:
                    # Random baseline should average repeats per sample before dataset aggregation.
                    comp_rnd_mean_batch = torch.stack(comp_rnd_repeats, dim=0).mean(dim=0)
                    suff_rnd_mean_batch = torch.stack(suff_rnd_repeats, dim=0).mean(dim=0)
                    out[K]["comp_vals_rnd"].append(comp_rnd_mean_batch.cpu())
                    out[K]["suff_vals_rnd"].append(suff_rnd_mean_batch.cpu())

        if torch.cuda.is_available() and (it % 5 == 0):
            torch.cuda.empty_cache()
        gc.collect()

    rows = []
    for K in Ks:
        row = {
            "K_percent": K,
            "orig_acc": base_acc,
            "orig_micro_f1": base_micro_f1,
            "orig_macro_f1": base_macro_f1,
        }
        if out[K]["comp_vals_idf"]:
            comp_vals_k = torch.cat(out[K]["comp_vals_idf"], dim=0).numpy()
            row["comprehensiveness_prob_mean_identified"] = float(np.nanmean(comp_vals_k))
        else:
            row["comprehensiveness_prob_mean_identified"] = float("nan")
        if out[K]["suff_vals_idf"]:
            suff_vals_k = torch.cat(out[K]["suff_vals_idf"], dim=0).numpy()
            row["sufficiency_prob_mean_identified"] = float(np.nanmean(suff_vals_k))
        else:
            row["sufficiency_prob_mean_identified"] = float("nan")

        if args.random_repeats > 0 and out[K].get("comp_vals_rnd", []):
            comp_vals_rnd_k = torch.cat(out[K]["comp_vals_rnd"], dim=0).numpy()
            row["comprehensiveness_prob_mean_random"] = float(np.nanmean(comp_vals_rnd_k))
        else:
            row["comprehensiveness_prob_mean_random"] = float("nan")
        if args.random_repeats > 0 and out[K].get("suff_vals_rnd", []):
            suff_vals_rnd_k = torch.cat(out[K]["suff_vals_rnd"], dim=0).numpy()
            row["sufficiency_prob_mean_random"] = float(np.nanmean(suff_vals_rnd_k))
        else:
            row["sufficiency_prob_mean_random"] = float("nan")

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("K_percent")
    comp_per_k_idf = [
        torch.cat(out[K]["comp_vals_idf"], dim=0).numpy() if out[K]["comp_vals_idf"] else np.array([], dtype=float)
        for K in Ks
    ]
    suff_per_k_idf = [
        torch.cat(out[K]["suff_vals_idf"], dim=0).numpy() if out[K]["suff_vals_idf"] else np.array([], dtype=float)
        for K in Ks
    ]
    df["aopc_comprehensiveness_prob_identified"] = aggregate_eraser_aopc(
        np.array(Ks, dtype=float),
        comp_per_k_idf,
        args.aopc_mode,
    )
    df["aopc_sufficiency_prob_identified"] = aggregate_eraser_aopc(
        np.array(Ks, dtype=float),
        suff_per_k_idf,
        args.aopc_mode,
    )

    if args.random_repeats > 0:
        comp_per_k_rnd = [
            torch.cat(out[K]["comp_vals_rnd"], dim=0).numpy() if out[K].get("comp_vals_rnd", []) else np.array([], dtype=float)
            for K in Ks
        ]
        suff_per_k_rnd = [
            torch.cat(out[K]["suff_vals_rnd"], dim=0).numpy() if out[K].get("suff_vals_rnd", []) else np.array([], dtype=float)
            for K in Ks
        ]
        df["aopc_comprehensiveness_prob_random"] = aggregate_eraser_aopc(
            np.array(Ks, dtype=float),
            comp_per_k_rnd,
            args.aopc_mode,
        )
        df["aopc_sufficiency_prob_random"] = aggregate_eraser_aopc(
            np.array(Ks, dtype=float),
            suff_per_k_rnd,
            args.aopc_mode,
        )
    else:
        df["aopc_comprehensiveness_prob_random"] = float("nan")
        df["aopc_sufficiency_prob_random"] = float("nan")

    # Store seed-level gaps explicitly, then aggregate mean/std over gap itself.
    # (Table-2 style: std of gap is not inferred from identified/random std.)
    df["comprehensiveness_prob_gap"] = (
        df["comprehensiveness_prob_mean_identified"] - df["comprehensiveness_prob_mean_random"]
    )
    df["sufficiency_prob_gap"] = (
        df["sufficiency_prob_mean_random"] - df["sufficiency_prob_mean_identified"]
    )
    df["aopc_comprehensiveness_prob_gap"] = (
        df["aopc_comprehensiveness_prob_identified"] - df["aopc_comprehensiveness_prob_random"]
    )
    df["aopc_sufficiency_prob_gap"] = (
        df["aopc_sufficiency_prob_random"] - df["aopc_sufficiency_prob_identified"]
    )

    for col in OUTPUT_COLS_WITH_GAPS:
        if col not in df.columns:
            df[col] = float("nan")
    df = df[OUTPUT_COLS_WITH_GAPS]

    print("\n=== Drop curve (per-modality K% masking) ===")
    print(df.to_string(index=False))
    return df


def load_ckpt_imoe(ckpt_path, encoder_dict, ensemble_model, device, strict_enc=True, strict_model=False):
    sd = torch.load(ckpt_path, map_location="cpu")
    if "encoder_dict" not in sd or "ensemble_model" not in sd:
        raise KeyError(f"[{ckpt_path}] checkpoint must contain 'encoder_dict' and 'ensemble_model'")

    for key, enc in encoder_dict.items():
        if key not in sd["encoder_dict"]:
            raise KeyError(f"[{ckpt_path}] encoder_dict missing key='{key}'")
        enc.load_state_dict(sd["encoder_dict"][key], strict=strict_enc)
        encoder_dict[key] = enc.to(device).eval()

    missing, unexpected = ensemble_model.load_state_dict(sd["ensemble_model"], strict=strict_model)
    print(f"[CKPT:{os.path.basename(ckpt_path)}] missing={len(missing)} | unexpected={len(unexpected)}")
    if len(missing) > 0:
        print("  - missing (sample):", list(missing)[:10], "..." if len(missing) > 10 else "")
    if len(unexpected) > 0:
        print("  - unexpected (sample):", list(unexpected)[:10], "..." if len(unexpected) > 10 else "")
    ensemble_model.eval()


def build_dataset_context(args, device, device_id):
    num_modalities = len(str(args.modality))

    ns = argparse.Namespace(
        data=args.dataset,
        modality=args.modality,
        use_common_ids=True,
        initial_filling="mean",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        patch=(args.dataset == "mimic"),
        device=(str(device_id) if args.dataset == "enrico" else str(device)),
        num_modalities=num_modalities,
        num_patches=args.num_patches,
        hidden_dim=args.hidden_dim,
        num_layers_fus=args.num_layers_fus,
        num_layers_enc=(1 if args.dataset == "mimic" else 2),
        num_layers_pred=args.num_layers_pred,
        num_experts=args.num_experts,
        num_routers=args.num_routers,
        top_k=args.top_k,
        num_heads=args.num_heads,
        dropout=args.dropout,
        fusion_sparse=args.fusion_sparse,
        gate=args.gate,
        hidden_dim_rw=args.hidden_dim_rw,
        num_layer_rw=args.num_layer_rw,
        temperature_rw=args.temperature_rw,
        debug=False,
    )

    if args.dataset == "mimic":
        (
            data_dict,
            encoder_dict,
            labels,
            train_ids,
            valid_ids,
            test_ids,
            n_labels,
            input_dims,
            transforms,
            masks,
            observed_idx_arr,
            *_rest,
        ) = load_and_preprocess_data_mimic(ns)
        dataset_name = "mimic"
        n_out = 2 if n_labels is None else int(n_labels)
        task_type = "binary"

    elif args.dataset == "mmimdb":
        (
            data_dict,
            encoder_dict,
            labels,
            train_ids,
            valid_ids,
            test_ids,
            n_labels,
            input_dims,
            transforms,
            masks,
            observed_idx_arr,
            *_rest,
        ) = load_and_preprocess_data_mmimdb(ns)
        dataset_name = "mmimdb"
        n_out = 23 if n_labels is None else int(n_labels)
        task_type = "multilabel"

    elif args.dataset == "enrico":
        (
            data_dict,
            encoder_dict,
            labels,
            train_ids,
            valid_ids,
            test_ids,
            n_labels,
            input_dims,
            transforms,
            masks,
            observed_idx_arr,
            *_rest,
        ) = load_and_preprocess_data_enrico(ns)
        dataset_name = "enrico"
        n_out = 20 if n_labels is None else int(n_labels)
        task_type = "multiclass"
    else:
        raise ValueError(f"Unknown dataset={args.dataset}")

    train_loader, valid_loader, test_loader = create_loaders(
        data_dict,
        observed_idx_arr,
        labels,
        train_ids,
        valid_ids,
        test_ids,
        batch_size=ns.batch_size,
        num_workers=ns.num_workers,
        pin_memory=ns.pin_memory,
        input_dims=input_dims,
        transforms=transforms,
        masks=masks,
        use_common_ids=ns.use_common_ids,
        dataset=dataset_name,
    )

    fusion_model = Transformer(
        num_modalities=num_modalities,
        num_patches=ns.num_patches,
        hidden_dim=ns.hidden_dim,
        output_dim=n_out,
        num_layers=ns.num_layers_fus,
        num_layers_pred=ns.num_layers_pred,
        num_experts=ns.num_experts,
        num_routers=ns.num_routers,
        top_k=ns.top_k,
        num_heads=ns.num_heads,
        dropout=ns.dropout,
        mlp_sparse=ns.fusion_sparse,
        gate=ns.gate,
    ).to(device).eval()

    ensemble_model = InteractionMoE(
        num_modalities=num_modalities,
        fusion_model=deepcopy(fusion_model),
        fusion_sparse=ns.fusion_sparse,
        hidden_dim=ns.hidden_dim,
        hidden_dim_rw=ns.hidden_dim_rw,
        num_layer_rw=ns.num_layer_rw,
        temperature_rw=ns.temperature_rw,
    ).to(device).eval()

    return ns, encoder_dict, train_loader, valid_loader, test_loader, ensemble_model, task_type


def format_mean_std(mean_val, std_val, as_percent=True, decimals=2):
    if as_percent:
        mean_val *= 100.0
        std_val *= 100.0
    return f"{mean_val:.{decimals}f}±{std_val:.{decimals}f}"


def main():
    args = get_args()
    args = apply_dataset_defaults(args)

    device_id, torch_dev = normalize_device_arg(args.device)
    device = torch.device(torch_dev)
    print(f"[INFO] dataset={args.dataset} | modality={args.modality} | device={device}")

    Ks = tuple(int(x.strip()) for x in args.Ks.split(",") if x.strip() != "")
    if len(Ks) == 0:
        raise ValueError("Ks is empty. Example: --Ks 5,10,15")

    ns, encoder_dict, train_loader, valid_loader, test_loader, ensemble_model, task_type = build_dataset_context(
        args, device, device_id
    )
    if args.eval_split == "valid":
        eval_loader = valid_loader
    else:
        eval_loader = test_loader
    if eval_loader is None:
        raise RuntimeError(f"Requested eval_split='{args.eval_split}' but that loader is not available.")
    print(f"[INFO] Faithfulness evaluation split: {args.eval_split}")

    df_list = []

    for ckpt_path in args.ckpt:
        print(f"\n=== EVAL CKPT: {ckpt_path} ===")
        load_ckpt_imoe(
            ckpt_path,
            encoder_dict,
            ensemble_model,
            device,
            strict_enc=args.strict_enc,
            strict_model=args.strict_model,
        )
        modal_means = None
        if args.mask_fill_strategy == "mean":
            mean_loader = train_loader if train_loader is not None else eval_loader
            mean_split = "train" if train_loader is not None else args.eval_split
            modal_means = compute_dataset_modality_mean_vectors(
                mean_loader, encoder_dict, device, ns
            )
            print(f"[INFO] Computed dataset-level modality means from {mean_split} split for this ckpt.")
        else:
            print("[INFO] mask_fill_strategy=zero, skip modality-mean computation.")

        df = run_full_test_drop_curve(
            eval_loader=eval_loader,
            encoder_dict=encoder_dict,
            ensemble_model=ensemble_model,
            device=device,
            ns=ns,
            args=args,
            task_type=task_type,
            Ks=Ks,
            modal_means=modal_means,
        )
        df["_ckpt"] = ckpt_path
        df_list.append(df)

    big = pd.concat(df_list, ignore_index=True)
    group_cols = ["K_percent"]
    selected_numeric_cols = [c for c in OUTPUT_COLS_WITH_GAPS if c != "K_percent"]
    for col in selected_numeric_cols:
        if col not in big.columns:
            big[col] = float("nan")

    mean_df = big.groupby(group_cols)[selected_numeric_cols].mean().reset_index()
    std_df = big.groupby(group_cols)[selected_numeric_cols].std(ddof=1).fillna(0.0).reset_index()
    merged = mean_df.merge(std_df, on="K_percent", suffixes=("_m", "_s"))

    pretty_df = mean_df[["K_percent"]].copy()
    for col in selected_numeric_cols:
        use_percent = args.as_percent
        pretty_df[col] = [
            format_mean_std(row[f"{col}_m"], row[f"{col}_s"], as_percent=use_percent, decimals=2)
            for _, row in merged.iterrows()
        ]
    pretty_df = pretty_df[OUTPUT_COLS_WITH_GAPS]

    suffix = args.suffix
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix
    if args.auto_split_suffix:
        split_tag = "_val" if args.eval_split == "valid" else "_test"
        if split_tag not in suffix:
            suffix = f"{suffix}{split_tag}"
    out_dir = args.out_dir.strip() if args.out_dir.strip() else os.path.dirname(args.ckpt[0])
    if out_dir == "":
        out_dir = "."
    os.makedirs(out_dir, exist_ok=True)

    prefix = f"{args.dataset}_imoe_masking_per_modality"
    raw_path = os.path.join(out_dir, f"{prefix}_mean_std_raw{suffix}.csv")
    pretty_path = os.path.join(out_dir, f"{prefix}_mean_std_pretty{suffix}.csv")

    wide = mean_df.merge(std_df, on="K_percent", suffixes=("_mean", "_std"))
    wide_cols = ["K_percent"] + [f"{c}_mean" for c in selected_numeric_cols] + [f"{c}_std" for c in selected_numeric_cols]
    wide = wide[wide_cols]
    wide.to_csv(raw_path, index=False)
    pretty_df.to_csv(pretty_path, index=False)

    print("\n=== Mean ± Std ===")
    print(pretty_df.to_string(index=False))
    print(f"\n[RESULT] Saved CSVs:\n - {raw_path}\n - {pretty_path}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("[DONE]")


if __name__ == "__main__":
    main()
