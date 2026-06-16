import argparse
import csv
import glob
import os
import re
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.common.datasets.synthetic import load_and_preprocess_data_synthetic
from src.common.datasets.MultiModalDataset import create_loaders
from src.common.fusion_models.transformer import Transformer
from src.imoe.InteractionMoE import InteractionMoE
from src.imoe.imoe_train import _encode_batch, _to_device_tree


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate saved synthetic checkpoints.")
    p.add_argument("--ckpt_glob", type=str, required=True, help="Glob for .pth files")
    p.add_argument(
        "--synthetic_pickle",
        type=str,
        default="",
        help="Single synthetic pickle (backward compatible)",
    )
    p.add_argument(
        "--synthetic_pickles",
        nargs="+",
        default=None,
        help="Multiple synthetic pickles to merge (train/valid/test concatenated by split)",
    )
    p.add_argument("--modality", type=str, default="01", help="e.g., 01 / 012 / 01234")
    p.add_argument(
        "--setting",
        type=str,
        required=True,
        choices=["redundancy", "synergy", "uniqueness0", "uniqueness1", "uniqueness2", "uniqueness3", "uniqueness4"],
        help="Dataset setting used for target-expert metric",
    )
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pin_memory", action="store_true")

    # Model hyperparameters (must match training)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers_fus", type=int, default=2)
    p.add_argument("--num_layers_pred", type=int, default=2)
    p.add_argument("--num_experts", type=int, default=4)
    p.add_argument("--num_routers", type=int, default=1)
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--fusion_sparse", action="store_true")
    p.add_argument("--gate", type=str, default="None")
    p.add_argument("--hidden_dim_rw", type=int, default=256)
    p.add_argument("--num_layer_rw", type=int, default=3)
    p.add_argument("--temperature_rw", type=float, default=0.3)
    p.add_argument("--num_patches", type=int, default=4)

    p.add_argument("--out_csv", type=str, default="outputs/synthetic_eval_per_ckpt.csv")
    p.add_argument("--out_summary_csv", type=str, default="outputs/synthetic_eval_summary_by_decor.csv")
    return p.parse_args()

def resolve_synthetic_paths(args):
    paths = []
    if args.synthetic_pickles:
        for item in args.synthetic_pickles:
            for p in str(item).split(","):
                p = p.strip()
                if p:
                    paths.append(p)
    if args.synthetic_pickle:
        for p in str(args.synthetic_pickle).split(","):
            p = p.strip()
            if p:
                paths.append(p)

    unique_paths = []
    seen = set()
    for p in paths:
        if p not in seen:
            unique_paths.append(p)
            seen.add(p)

    if len(unique_paths) == 0:
        raise ValueError("Provide --synthetic_pickle or --synthetic_pickles")
    return unique_paths


def parse_meta_from_ckpt(path):
    name = os.path.basename(path)
    seed = None
    decor = None
    val_acc = None
    m = re.search(r"seed_(\d+)", name)
    if m:
        seed = int(m.group(1))
    m = re.search(r"(?:decor|cka)_([0-9.]+)", name)
    if m:
        decor = float(m.group(1))
    m = re.search(r"val_acc_([0-9.]+)", name)
    if m:
        val_acc = float(m.group(1))
    return seed, decor, val_acc


def target_expert_index(setting, num_modalities):
    if setting == "synergy":
        return num_modalities
    if setting == "redundancy":
        return num_modalities + 1
    if setting.startswith("uniqueness"):
        i = int(setting.replace("uniqueness", ""))
        if i >= num_modalities:
            raise ValueError(f"{setting} invalid for {num_modalities} modalities")
        return i
    raise ValueError(f"Unsupported setting: {setting}")


def expert_name(idx, num_modalities):
    if idx < num_modalities:
        return f"uni_{idx+1}"
    if idx == num_modalities:
        return "syn"
    return "red"


def expert_mean_key(idx, num_modalities):
    return f"expert_mean_weight_{expert_name(idx, num_modalities)}"

def normalize_routing_weights(routing_weights, expected_num_experts):
    """
    Convert routing weights to shape (B, E).

    Supported input shapes:
      - (B, E)
      - (B, R, E): reduce router axis by mean -> (B, E)
    """
    if routing_weights.ndim == 2:
        rw = routing_weights
    elif routing_weights.ndim == 3:
        rw = routing_weights.mean(dim=1)
    else:
        raise ValueError(
            f"Unsupported routing_weights shape {tuple(routing_weights.shape)}. "
            "Expected (B, E) or (B, R, E)."
        )

    if rw.shape[-1] != expected_num_experts:
        raise ValueError(
            f"Routing expert dim mismatch: got E={rw.shape[-1]}, "
            f"expected {expected_num_experts}."
        )
    return rw


def evaluate_one_ckpt(args, ckpt_path):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    ds_args = SimpleNamespace(
        device=args.device,
        hidden_dim=args.hidden_dim,
        modality=args.modality,
        synthetic_pickle=args.synthetic_pickle,
        synthetic_pickles=args.synthetic_pickles,
    )

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
        _,
        _,
    ) = load_and_preprocess_data_synthetic(ds_args)

    _, _, test_loader = create_loaders(
        data_dict=data_dict,
        observed_idx=observed_idx_arr,
        labels=labels,
        train_ids=train_ids,
        valid_ids=valid_ids,
        test_ids=test_ids,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        input_dims=input_dims,
        transforms=transforms,
        masks=masks,
        use_common_ids=True,
        dataset="synthetic",
    )

    num_modalities = int(observed_idx_arr.shape[1])
    num_branches = num_modalities + 2  # uni + syn + red
    fusion_model = Transformer(
        num_modalities=num_modalities,
        num_patches=args.num_patches,
        hidden_dim=args.hidden_dim,
        output_dim=n_labels,
        num_layers=args.num_layers_fus,
        num_layers_pred=args.num_layers_pred,
        num_experts=args.num_experts,
        num_routers=args.num_routers,
        top_k=args.top_k,
        num_heads=args.num_heads,
        dropout=args.dropout,
        mlp_sparse=args.fusion_sparse,
        gate=args.gate,
    ).to(device)

    model = InteractionMoE(
        num_modalities=num_modalities,
        fusion_model=fusion_model,
        fusion_sparse=args.fusion_sparse,
        hidden_dim=args.hidden_dim,
        hidden_dim_rw=args.hidden_dim_rw,
        num_layer_rw=args.num_layer_rw,
        temperature_rw=args.temperature_rw,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["ensemble_model"], strict=True)
    for k, enc in encoder_dict.items():
        enc.load_state_dict(ckpt["encoder_dict"][k], strict=True)
        enc.eval()
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    all_routing = []
    routing_shape_raw = None
    routing_shape_used = None

    with torch.no_grad():
        for batch_samples, batch_ids, batch_labels, batch_mcs, batch_observed in test_loader:
            batch_samples = _to_device_tree(batch_samples, device)
            batch_labels = batch_labels.to(device, non_blocking=True)
            feats, masks_, _ = _encode_batch(batch_samples, encoder_dict, ds_args)
            _, routing_weights, outputs = model.inference(feats, masks_)
            if routing_shape_raw is None:
                routing_shape_raw = tuple(routing_weights.shape)
            routing_weights = normalize_routing_weights(
                routing_weights, expected_num_experts=num_branches
            )
            if routing_shape_used is None:
                routing_shape_used = tuple(routing_weights.shape)

            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_routing.append(routing_weights.cpu().numpy())

    all_routing = np.concatenate(all_routing, axis=0)
    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_preds)
    y_prob1 = np.asarray(all_probs)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_micro = f1_score(y_true, y_pred, average="micro")
    auc = roc_auc_score(y_true, y_prob1)

    # --- Target expert metrics ---
    t_idx = target_expert_index(args.setting, num_modalities)
    if t_idx >= all_routing.shape[1]:
        raise ValueError(
            f"Target expert idx {t_idx} out of range for routing shape {all_routing.shape}."
        )

    # (Old) top-1 selection rate (argmax)
    top1 = all_routing.argmax(axis=1)
    target_top1_rate = float(np.mean(top1 == t_idx))

    # (A option) usage percentage of target expert
    target_mean_weight = float(np.mean(all_routing[:, t_idx]))
    target_usage_pct = float(100.0 * target_mean_weight)

    # target margin vs best other expert
    others_max = np.max(np.delete(all_routing, t_idx, axis=1), axis=1)
    target_margin = float(np.mean(all_routing[:, t_idx] - others_max))
    mean_weights = all_routing.mean(axis=0)

    seed, decor, val_acc = parse_meta_from_ckpt(ckpt_path)

    row = {
        "ckpt": ckpt_path,
        "seed": seed,
        "decor_w": decor,
        "val_acc_in_name": val_acc,
        "test_acc": acc,
        "test_f1_macro": f1_macro,
        "test_f1_micro": f1_micro,
        "test_auc": auc,
        "target_setting": args.setting,
        "target_expert_idx": t_idx,
        "target_expert_name": expert_name(t_idx, num_modalities),
        "routing_shape_raw": str(routing_shape_raw),
        "routing_shape_used": str(routing_shape_used),
        "target_top1_rate": target_top1_rate,
        "target_mean_weight": target_mean_weight,
        "target_usage_pct": target_usage_pct,
        "target_margin": target_margin,
    }
    for i in range(all_routing.shape[1]):
        row[expert_mean_key(i, num_modalities)] = float(mean_weights[i])
    return row


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def summarize_by_decor(rows):
    groups = {}
    for r in rows:
        k = r["decor_w"]
        groups.setdefault(k, []).append(r)

    def mstd(vals):
        vals = np.asarray(vals, dtype=np.float64)
        return float(vals.mean()), float(vals.std())

    expert_cols = sorted(
        [k for k in rows[0].keys() if k.startswith("expert_mean_weight_")]
    )

    out = []
    for decor, rs in sorted(groups.items(), key=lambda x: (x[0] is None, x[0])):
        acc_m, acc_s = mstd([x["test_acc"] for x in rs])
        f1_m, f1_s = mstd([x["test_f1_macro"] for x in rs])
        auc_m, auc_s = mstd([x["test_auc"] for x in rs])

        t1_m, t1_s = mstd([x["target_top1_rate"] for x in rs])
        wt_m, wt_s = mstd([x["target_mean_weight"] for x in rs])

        # (A option) usage percentage summary
        usage_m, usage_s = mstd([100.0 * x["target_mean_weight"] for x in rs])

        mg_m, mg_s = mstd([x["target_margin"] for x in rs])

        summary = {
            "decor_w": decor,
            "n_models": len(rs),
            "test_acc_mean": acc_m,
            "test_acc_std": acc_s,
            "test_f1_macro_mean": f1_m,
            "test_f1_macro_std": f1_s,
            "test_auc_mean": auc_m,
            "test_auc_std": auc_s,
            "target_top1_rate_mean": t1_m,
            "target_top1_rate_std": t1_s,
            "target_mean_weight_mean": wt_m,
            "target_mean_weight_std": wt_s,
            "target_usage_pct_mean": usage_m,
            "target_usage_pct_std": usage_s,
            "target_margin_mean": mg_m,
            "target_margin_std": mg_s,
        }
        for c in expert_cols:
            c_m, c_s = mstd([x[c] for x in rs])
            summary[f"{c}_mean"] = c_m
            summary[f"{c}_std"] = c_s
        out.append(summary)
    return out


def main():
    args = parse_args()
    synthetic_paths = resolve_synthetic_paths(args)
    # Keep compatibility with dataset loader path resolution.
    args.synthetic_pickles = synthetic_paths
    if not args.synthetic_pickle:
        args.synthetic_pickle = synthetic_paths[0]

    ckpts = sorted(glob.glob(args.ckpt_glob))
    if len(ckpts) == 0:
        raise FileNotFoundError(f"No checkpoint matched: {args.ckpt_glob}")

    rows = []
    for i, ckpt in enumerate(ckpts):
        print(f"[{i+1}/{len(ckpts)}] Evaluating: {ckpt}")
        rows.append(evaluate_one_ckpt(args, ckpt))

    fieldnames = list(rows[0].keys())
    write_csv(args.out_csv, rows, fieldnames)
    print(f"Saved per-checkpoint results: {args.out_csv}")

    summary_rows = summarize_by_decor(rows)
    write_csv(args.out_summary_csv, summary_rows, list(summary_rows[0].keys()))
    print(f"Saved decor summary: {args.out_summary_csv}")

    print("\n=== Summary by decor_w ===")
    for r in summary_rows:
        print(
            f"decor={r['decor_w']} n={r['n_models']} "
            f"acc={r['test_acc_mean']:.4f}±{r['test_acc_std']:.4f} "
            f"target_top1={r['target_top1_rate_mean']:.4f}±{r['target_top1_rate_std']:.4f} "
            f"target_w={r['target_mean_weight_mean']:.4f}±{r['target_mean_weight_std']:.4f} "
            f"target_usage%={r['target_usage_pct_mean']:.2f}±{r['target_usage_pct_std']:.2f}"
        )


if __name__ == "__main__":
    main()
