import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import torch
import numpy as np
import argparse
from pathlib import Path
from copy import deepcopy
from tqdm import trange
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, message="os.fork()")

from src.common.datasets.mimic import load_and_preprocess_data_mimic
from src.common.datasets.enrico import load_and_preprocess_data_enrico
from src.common.datasets.mmimdb import load_and_preprocess_data_mmimdb
from src.common.datasets.MultiModalDataset import create_loaders
from src.common.fusion_models.transformer import Transformer
from src.common.utils import setup_logger, str2bool, seed_everything, plot_total_loss_curves


SUPPORTED_DATASETS = {"mimic", "enrico", "mmimdb"}
DEFAULT_MODALITY = {
    "mimic": "LNC",
    "mmimdb": "LI",
    "enrico": "SW",
}
VALID_MODALITY_CHARS = {
    "mimic": set("LNC"),
    "mmimdb": set("LI"),
    "enrico": set("SW"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="GShard-Transformer")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--data", type=str, default="mimic")
    parser.add_argument("--gate", type=str, default="GShardGate")
    parser.add_argument(
        "--modality", type=str, default="IGCB"
    )  # I G C B for ADNI, L N C for MIMIC
    parser.add_argument("--initial_filling", type=str, default="mean")
    parser.add_argument("--train_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers_enc", type=int, default=1)
    parser.add_argument("--num_layers_fus", type=int, default=1)
    parser.add_argument("--num_layers_pred", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=str2bool, default=True)
    parser.add_argument("--use_common_ids", type=str2bool, default=True)
    parser.add_argument("--patch", type=str2bool, default=True)
    parser.add_argument("--num_patches", type=int, default=16)
    parser.add_argument("--num_experts", type=int, default=16)
    parser.add_argument("--num_routers", type=int, default=1)
    parser.add_argument("--fusion_sparse", type=str2bool, default=True)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--gate_loss_weight", type=float, default=1e-2)
    # CKA-only training path for GShardGate.
    parser.add_argument("--regularizer_weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument(
        "--run_suffix",
        type=str,
        default="",
        help="Optional checkpoint filename suffix, e.g. run1/run2/run3.",
    )
    parser.add_argument("--save", type=str2bool, default=True)
    parser.add_argument("--debug", type=str2bool, default=False)

    return parser.parse_known_args()


def _validate_args(args):
    if args.data not in SUPPORTED_DATASETS:
        supported = ", ".join(sorted(SUPPORTED_DATASETS))
        raise ValueError(f"Unsupported dataset: {args.data}. Supported: {supported}")

    args.modality = args.modality.upper()
    if args.modality == "IGCB":
        args.modality = DEFAULT_MODALITY[args.data]

    invalid = [ch for ch in args.modality if ch not in VALID_MODALITY_CHARS[args.data]]
    if invalid:
        raise ValueError(
            f"Invalid modality '{args.modality}' for dataset '{args.data}'. "
            f"Allowed chars: {''.join(sorted(VALID_MODALITY_CHARS[args.data]))}"
        )

    if args.gate != "GShardGate":
        raise ValueError("train_gshardgate.py expects --gate GShardGate")
    if args.top_k != 2:
        raise ValueError("GShardGate requires --top_k 2")
    if args.n_runs < 1:
        raise ValueError("--n_runs must be >= 1")
    if args.run_suffix and any(ch in args.run_suffix for ch in ("/", "\\")):
        raise ValueError("--run_suffix must not contain path separators")


def _to_device_tree(batch_samples, device):
    out = {}
    for k, v in batch_samples.items():
        if isinstance(v, dict):
            out[k] = {}
            for kk, vv in v.items():
                if torch.is_tensor(vv):
                    out[k][kk] = vv.to(device, non_blocking=True)
                else:
                    out[k][kk] = vv
        elif torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _encode_batch(batch_samples, encoder_dict, args):
    key_order = []
    for ch in args.modality:
        if ch == "L":
            if "lab" in encoder_dict:
                key_order.append("lab")
            elif "language" in encoder_dict:
                key_order.append("language")
            else:
                raise KeyError("L modality requested, but no lab/language encoder found.")
        elif ch == "N":
            key_order.append("note")
        elif ch == "C":
            key_order.append("code")
        elif ch == "I":
            key_order.append("img")
        elif ch == "S":
            key_order.append("screenshot")
        elif ch == "W":
            key_order.append("wireframe")
        else:
            raise ValueError(f"Unsupported modality char: {ch}")

    features = []
    masks = []
    for key in key_order:
        if key not in encoder_dict or key not in batch_samples:
            raise KeyError(f"Missing key '{key}' in encoder_dict or batch_samples.")

        x = batch_samples[key]
        encoder = encoder_dict[key]

        if isinstance(x, dict):
            if "texts" in x:
                out = encoder(x["texts"])
            elif "paths" in x:
                out = encoder(x["paths"])
            else:
                raise KeyError(f"Unsupported payload for key '{key}'.")
        else:
            out = encoder(x)

        if isinstance(out, (tuple, list)) and len(out) == 2:
            feat, mask = out
        else:
            feat, mask = out, None
        features.append(feat)
        masks.append(mask)

    return features, masks


def _cpu_state_dict(state_dict):
    return {k: v.cpu() for k, v in state_dict.items()}


def _reset_moe_route_cache(fusion_model):
    for module in fusion_model.modules():
        if hasattr(module, "all_gates"):
            if hasattr(module, "last_topk_indices"):
                module.last_topk_indices = None
            if hasattr(module, "last_topk_scores"):
                module.last_topk_scores = None
            gate = getattr(module, "gate", None)
            if gate is not None and hasattr(gate, "reset_topk_logit"):
                gate.reset_topk_logit()


def _register_moe_capture_hooks(fusion_model, capture_state, capture_pre_mix=False):
    handles = []

    def _hook(module, _inputs, output):
        # Capture only during train() to avoid val/test memory growth.
        if not module.training:
            return
        if not torch.is_tensor(output):
            return

        topk_idx = getattr(module, "last_topk_indices", None)
        topk_scores = getattr(module, "last_topk_scores", None)
        tot_expert = int(
            getattr(
                module,
                "last_tot_expert",
                getattr(getattr(module, "gate", None), "tot_expert", 0),
            )
        )

        if topk_idx is None or tot_expert <= 0:
            return
        if topk_idx.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise RuntimeError(
                f"Gate top-k tensor must be integer indices, got dtype={topk_idx.dtype}"
            )
        if topk_idx.dim() == 1:
            topk_idx = topk_idx.unsqueeze(1)
        if topk_scores is not None and topk_scores.dim() == 1:
            topk_scores = topk_scores.unsqueeze(1)
        topk_pre_mix = getattr(module, "last_topk_pre_mix", None) if capture_pre_mix else None

        # We keep token_hidden in graph; routing tensors are detached.
        capture_state["records"].append(
            {
                "token_hidden": output,
                "topk_idx": topk_idx.detach(),
                # Keep topk_scores attached only for strict expert-CKA mode.
                "topk_scores": (
                    None
                    if topk_scores is None
                    else (topk_scores if capture_pre_mix else topk_scores.detach())
                ),
                "token_hidden_pre_mix": topk_pre_mix,
                "tot_expert": tot_expert,
            }
        )

    for module in fusion_model.modules():
        # FMoETransformerMLP blocks expose routing attrs and output token-level hidden states.
        if hasattr(module, "all_gates") and hasattr(module, "gate"):
            if hasattr(module, "capture_topk_pre_mix"):
                module.capture_topk_pre_mix = bool(capture_pre_mix)
            handles.append(module.register_forward_hook(_hook))

    return handles


def _build_expert_latents_from_record(rec, eps=1e-6, use_topk_scores=True):
    """
    Convert one MoE capture record into per-expert latents and presence masks.

    Returns:
      latents:  (E, B, D)
      presence: (E, B) bool
    """
    token_hidden = rec["token_hidden"]
    topk_idx = rec["topk_idx"]
    topk_scores = rec["topk_scores"]
    token_hidden_pre_mix = rec.get("token_hidden_pre_mix", None)
    tot_expert = int(rec["tot_expert"])

    if token_hidden.dim() == 2:
        token_hidden = token_hidden.unsqueeze(1)  # (B, 1, H)
    if token_hidden.dim() != 3:
        raise RuntimeError(
            f"Expected token_hidden with dim 2/3, got {tuple(token_hidden.shape)}"
        )
    if tot_expert <= 0:
        raise RuntimeError(f"Expected positive tot_expert, got {tot_expert}")

    B, T, H = token_hidden.shape
    n_tokens = B * T

    if topk_idx.dim() == 1:
        topk_idx = topk_idx.unsqueeze(1)
    if topk_idx.shape[0] != n_tokens:
        raise RuntimeError(
            f"Routing/hidden size mismatch: topk_idx={tuple(topk_idx.shape)}, "
            f"token_hidden={tuple(token_hidden.shape)}"
        )

    K = topk_idx.shape[1]
    if not use_topk_scores:
        topk_scores = None

    if topk_scores is not None:
        if topk_scores.dim() == 1:
            topk_scores = topk_scores.unsqueeze(1)
        if topk_scores.shape != topk_idx.shape:
            raise RuntimeError(
                f"topk_scores shape mismatch: scores={tuple(topk_scores.shape)}, "
                f"indices={tuple(topk_idx.shape)}"
            )
    else:
        topk_scores = token_hidden.new_ones((n_tokens, K))

    sample_ids = (
        torch.arange(B, device=token_hidden.device).unsqueeze(1).expand(B, T).reshape(-1)
    )  # (B*T,)

    if token_hidden_pre_mix is not None:
        # Expected from FixedFMoE.forward() pre-bmm cache: (N, K, H)
        if token_hidden_pre_mix.dim() == 4:
            token_hidden_pre_mix = token_hidden_pre_mix.reshape(
                n_tokens, token_hidden_pre_mix.shape[2], token_hidden_pre_mix.shape[3]
            )
        if token_hidden_pre_mix.dim() != 3:
            raise RuntimeError(
                f"Expected token_hidden_pre_mix with dim 3/4, got {tuple(token_hidden_pre_mix.shape)}"
            )
        if token_hidden_pre_mix.shape[0] != n_tokens or token_hidden_pre_mix.shape[1] != K:
            raise RuntimeError(
                f"Pre-mix/route size mismatch: pre_mix={tuple(token_hidden_pre_mix.shape)}, "
                f"topk_idx={tuple(topk_idx.shape)}, token_hidden={tuple(token_hidden.shape)}"
            )
        h_dim = token_hidden_pre_mix.shape[-1]
        sums_flat = token_hidden.new_zeros((tot_expert * B, h_dim))
    else:
        h = token_hidden.reshape(n_tokens, H)
        sums_flat = token_hidden.new_zeros((tot_expert * B, H))
    counts_flat = token_hidden.new_zeros((tot_expert * B,))

    for k in range(K):
        idx_k = topk_idx[:, k]
        valid = idx_k >= 0
        if not valid.any():
            continue

        expert_ids = idx_k[valid].long()
        sample_k = sample_ids[valid].long()
        if expert_ids.numel() > 0 and expert_ids.max().item() >= tot_expert:
            raise RuntimeError(
                f"Routing index out of range: max={expert_ids.max().item()}, tot_expert={tot_expert}"
            )
        if token_hidden_pre_mix is not None:
            h_k = token_hidden_pre_mix[valid, k, :]
            if topk_scores is not None:
                w_k = topk_scores[valid, k].to(h_k.dtype)
                contrib = h_k * w_k.unsqueeze(1)
            else:
                w_k = h_k.new_ones((h_k.shape[0],))
                contrib = h_k
        else:
            w_k = topk_scores[valid, k].to(h.dtype)
            contrib = h[valid] * w_k.unsqueeze(1)
        target = expert_ids * B + sample_k

        sums_flat.index_add_(0, target, contrib)
        counts_flat.index_add_(0, target, w_k)

    # Use the actually accumulated hidden dim (works for both mix/pre-mix paths).
    sums = sums_flat.view(tot_expert, B, -1)
    counts = counts_flat.view(tot_expert, B)
    latents = sums / counts.clamp_min(eps).unsqueeze(-1)  # (E, B, D)
    presence = counts > 0                                  # (E, B)
    return latents, presence


def compute_rep_cka_loss_from_moe_all_latents(records, reference_tensor=None):
    """
    Build per-expert latents from MoE routing records and compute pairwise
    linear CKA across experts (sample axis is batch dimension B).
    """
    if not records:
        if reference_tensor is not None:
            return reference_tensor.new_zeros(())
        return torch.tensor(0.0)

    layer_terms = []
    for rec in records:
        if rec.get("token_hidden_pre_mix", None) is None:
            raise RuntimeError(
                "rep_cka requires pre-mix expert outputs, but token_hidden_pre_mix is missing."
            )
        latents, presence = _build_expert_latents_from_record(
            rec, use_topk_scores=False
        )  # (E,B,H), (E,B)
        active = presence.sum(dim=1) > 0
        idxs = torch.where(active)[0]
        if idxs.numel() <= 1:
            continue

        terms = []
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                e1 = idxs[i].item()
                e2 = idxs[j].item()
                valid = presence[e1] & presence[e2]  # (B,)
                if valid.sum() < 2:
                    continue
                x = latents[e1][valid]  # (Bv, H)
                y = latents[e2][valid]  # (Bv, H)
                terms.append(_linear_cka(x, y))

        if terms:
            layer_terms.append(torch.stack(terms).mean())

    if layer_terms:
        return torch.stack(layer_terms).mean()

    if reference_tensor is not None:
        return reference_tensor.new_zeros(())
    return records[0]["token_hidden"].new_zeros(())


def _center_features(z):
    return z - z.mean(dim=0, keepdim=True)


def _linear_cka(x, y, eps=1e-8):
    x = _center_features(x)
    y = _center_features(y)
    xty = x.T @ y
    xtx = x.T @ x
    yty = y.T @ y
    hsic = (xty * xty).sum()
    norm_x = torch.linalg.norm(xtx, ord="fro")
    norm_y = torch.linalg.norm(yty, ord="fro")
    return hsic / (norm_x * norm_y + eps)


def _safe_auc(data, all_labels, all_probs, n_labels):
    try:
        if data == "enrico":
            return roc_auc_score(
                np.array(all_labels),
                np.array(all_probs),
                multi_class="ovo",
                labels=list(range(n_labels)),
            )
        if data == "mimic":
            return roc_auc_score(all_labels, all_probs)
    except ValueError:
        return float("nan")
    return 0.0


def train_and_evaluate(args, seed):
    seed_everything(seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    num_modalities = len(args.modality)

    if args.data == "mimic":
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
        ) = load_and_preprocess_data_mimic(args)
    elif args.data == "enrico":
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
        ) = load_and_preprocess_data_enrico(args)
        n_labels = 20
    elif args.data == "mmimdb":
        mmimdb_args = deepcopy(args)
        mmimdb_args.device = str(device)
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
        ) = load_and_preprocess_data_mmimdb(mmimdb_args)
    else:
        raise ValueError(f"Unsupported dataset for this repo setup: {args.data}")

    train_loader, val_loader, test_loader = create_loaders(
        data_dict,
        observed_idx_arr,
        labels,
        train_ids,
        valid_ids,
        test_ids,
        args.batch_size,
        args.num_workers,
        args.pin_memory,
        input_dims,
        transforms,
        masks,
        args.use_common_ids,
        dataset=args.data,
    )

    fusion_model = Transformer(
        num_modalities,
        args.num_patches,
        args.hidden_dim,
        n_labels,
        args.num_layers_fus,
        args.num_layers_pred,
        args.num_experts,
        args.num_routers,
        args.top_k,
        args.num_heads,
        args.dropout,
        args.fusion_sparse,
        args.gate,
    ).to(device)

    params = list(fusion_model.parameters()) + [
        param for encoder in encoder_dict.values() for param in encoder.parameters()
    ]
    optimizer = torch.optim.Adam(params, lr=args.lr)

    if args.data == "mimic":
        criterion = torch.nn.CrossEntropyLoss(torch.tensor([0.25, 0.75]).to(device))
    elif args.data == "enrico":
        criterion = torch.nn.CrossEntropyLoss()
    elif args.data == "mmimdb":
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unsupported dataset for criterion: {args.data}")

    if args.data == "mmimdb":
        best_metric = -np.inf
    else:
        best_metric = -np.inf
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_val_auc = 0.0

    best_model_fus = deepcopy(fusion_model.state_dict())
    best_model_enc = {
        modality: deepcopy(encoder.state_dict())
        for modality, encoder in encoder_dict.items()
    }
    if args.save:
        best_model_fus_cpu = _cpu_state_dict(best_model_fus)
        best_model_enc_cpu = {
            modality: _cpu_state_dict(enc_state)
            for modality, enc_state in best_model_enc.items()
        }

    plotting_total_losses = {"task": [], "gate": []}
    cka_w = float(args.regularizer_weight)
    if cka_w > 0.0:
        plotting_total_losses["regularizer"] = []
    capture_state = {"records": []}
    capture_handles = []
    if cka_w > 0.0:
        if not args.fusion_sparse:
            raise ValueError("rep_cka requires sparse fusion (set --fusion_sparse True)")
        capture_handles = _register_moe_capture_hooks(
            fusion_model,
            capture_state,
            capture_pre_mix=True,
        )

    try:
        for epoch in trange(args.train_epochs):
            fusion_model.train()
            for encoder in encoder_dict.values():
                encoder.train()

            batch_task_losses = []
            batch_gate_losses = []
            batch_reg_losses = []

            for batch_samples, batch_labels, batch_mcs, batch_observed in train_loader:
                batch_samples = _to_device_tree(batch_samples, device)
                batch_labels = batch_labels.to(device, non_blocking=True)
                batch_mcs = batch_mcs.to(device, non_blocking=True)
                batch_observed = batch_observed.to(device, non_blocking=True)
                optimizer.zero_grad()

                fusion_input, fusion_masks = _encode_batch(batch_samples, encoder_dict, args)
                use_reg = cka_w > 0.0
                if use_reg:
                    capture_state["records"].clear()
                    _reset_moe_route_cache(fusion_model)

                outputs = fusion_model(fusion_input, masks=fusion_masks)

                if args.data == "mmimdb":
                    task_loss = criterion(outputs, batch_labels.float())
                else:
                    task_loss = criterion(outputs, batch_labels)

                if args.fusion_sparse:
                    gate_loss = fusion_model.gate_loss()
                    if not torch.is_tensor(gate_loss):
                        gate_loss = outputs.new_tensor(float(gate_loss))
                else:
                    gate_loss = outputs.new_zeros(())

                if use_reg:
                    reg_loss = compute_rep_cka_loss_from_moe_all_latents(
                        capture_state["records"], reference_tensor=outputs
                    )
                else:
                    reg_loss = outputs.new_zeros(())

                loss = (
                    task_loss
                    + args.gate_loss_weight * gate_loss
                    + cka_w * reg_loss
                )
                loss.backward()
                if args.data == "enrico":
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

                batch_task_losses.append(task_loss.item())
                batch_gate_losses.append(float(gate_loss.detach().item()))
                batch_reg_losses.append(float(reg_loss.detach().item()))

            plotting_total_losses["task"].append(np.mean(batch_task_losses))
            plotting_total_losses["gate"].append(np.mean(batch_gate_losses))
            if cka_w > 0.0:
                plotting_total_losses["regularizer"].append(np.mean(batch_reg_losses))

            fusion_model.eval()
            for encoder in encoder_dict.values():
                encoder.eval()

            all_preds = []
            all_labels = []
            all_probs = []
            val_losses = []

            with torch.no_grad():
                for batch_samples, batch_labels, batch_mcs, batch_observed in val_loader:
                    batch_samples = _to_device_tree(batch_samples, device)
                    batch_labels = batch_labels.to(device, non_blocking=True)
                    batch_mcs = batch_mcs.to(device, non_blocking=True)
                    batch_observed = batch_observed.to(device, non_blocking=True)

                    fusion_input, fusion_masks = _encode_batch(
                        batch_samples, encoder_dict, args
                    )
                    outputs = fusion_model(fusion_input, masks=fusion_masks)

                    if args.data == "mmimdb":
                        val_loss = criterion(outputs, batch_labels.float())
                        preds = torch.sigmoid(outputs).round()
                    else:
                        val_loss = criterion(outputs, batch_labels)
                        _, preds = torch.max(outputs, 1)
                    val_losses.append(val_loss.item())
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch_labels.cpu().numpy())

                    if args.data == "mimic":
                        all_probs.extend(
                            torch.nn.functional.softmax(outputs, dim=1)[:, 1]
                            .cpu()
                            .numpy()
                        )
                    elif args.data == "enrico":
                        probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                        if probs.shape[1] != n_labels:
                            raise ValueError("Incorrect output shape from the model")
                        all_probs.extend(probs)

            val_loss = float(np.mean(val_losses))
            val_acc = accuracy_score(all_labels, all_preds)
            val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
            val_auc = _safe_auc(args.data, all_labels, all_probs, n_labels)
            gate_loss_mean = float(np.mean(batch_gate_losses)) if batch_gate_losses else 0.0

            print(
                f"[Seed {seed}] [Epoch {epoch+1}/{args.train_epochs}] "
                f"Val Loss: {val_loss:.2f}, Val Acc: {val_acc*100:.2f}, "
                f"Val F1: {val_f1*100:.2f}, Val AUC: {val_auc*100:.2f}, "
                f"Gate Loss: {gate_loss_mean:.2f}"
            )
            if cka_w > 0.0:
                print(
                    f"  Regularizer(rep_cka(all_latents)): {float(np.mean(batch_reg_losses)):.4f} "
                    f"(lambda={cka_w:.4f})"
                )

            if args.data == "mmimdb":
                is_better = val_f1 > best_metric
            else:
                is_better = val_acc > best_metric

            if is_better:
                print(
                    f"[(**Best**) Epoch {epoch+1}/{args.train_epochs}] "
                    f"Val Acc: {val_acc*100:.2f}, Val F1: {val_f1*100:.2f}, Val AUC: {val_auc*100:.2f}"
                )
                best_metric = val_f1 if args.data == "mmimdb" else val_acc
                best_val_acc = val_acc
                best_val_f1 = val_f1
                best_val_auc = val_auc

                best_model_fus = deepcopy(fusion_model.state_dict())
                best_model_enc = {
                    modality: deepcopy(encoder.state_dict())
                    for modality, encoder in encoder_dict.items()
                }
                if args.save:
                    best_model_fus_cpu = _cpu_state_dict(best_model_fus)
                    best_model_enc_cpu = {
                        modality: _cpu_state_dict(enc_state)
                        for modality, enc_state in best_model_enc.items()
                    }
    finally:
        for handle in capture_handles:
            handle.remove()
        for module in fusion_model.modules():
            if hasattr(module, "capture_topk_pre_mix"):
                module.capture_topk_pre_mix = False

    plot_total_loss_curves(
        args,
        plotting_total_losses=plotting_total_losses,
        framework="baseline",
        fusion="gshardgate",
    )

    if args.save:
        Path("./saves").mkdir(exist_ok=True, parents=True)
        Path(f"./saves/vanilla/{args.data}").mkdir(exist_ok=True, parents=True)

        if args.data == "mmimdb":
            save_path = (
                f"./saves/vanilla/{args.data}/"
                f"seed_{seed}_modality_{args.modality}_Sparse_{args.fusion_sparse}_"
                f"gate_{args.gate}_train_epochs_{args.train_epochs}_val_f1_{best_val_f1:.2f}.pth"
            )
        else:
            save_path = (
                f"./saves/vanilla/{args.data}/"
                f"seed_{seed}_modality_{args.modality}_Sparse_{args.fusion_sparse}_"
                f"gate_{args.gate}_train_epochs_{args.train_epochs}_val_acc_{best_val_acc:.2f}.pth"
            )
        if cka_w > 0.0:
            save_path = save_path.replace(".pth", f"_rep_cka_{cka_w}.pth")
        run_suffix = str(args.run_suffix).strip()
        if run_suffix:
            run_suffix = run_suffix.lstrip("_")
            save_path = save_path.replace(".pth", f"_{run_suffix}.pth")

        torch.save(
            {"fusion_model": best_model_fus_cpu, "encoder_dict": best_model_enc_cpu},
            save_path,
        )
        print(f"Best model saved to {save_path}")

    for modality, encoder in encoder_dict.items():
        encoder.load_state_dict(best_model_enc[modality])
        encoder.eval()
    fusion_model.load_state_dict(best_model_fus)
    fusion_model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for (
            batch_samples,
            batch_ids,
            batch_labels,
            batch_mcs,
            batch_observed,
        ) in test_loader:
            batch_samples = _to_device_tree(batch_samples, device)
            batch_labels = batch_labels.to(device, non_blocking=True)
            batch_ids = batch_ids.to(device, non_blocking=True)
            batch_mcs = batch_mcs.to(device, non_blocking=True)
            batch_observed = batch_observed.to(device, non_blocking=True)

            fusion_input, fusion_masks = _encode_batch(batch_samples, encoder_dict, args)
            outputs = fusion_model(fusion_input, masks=fusion_masks)

            if args.data == "mmimdb":
                preds = torch.sigmoid(outputs).round()
            else:
                _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

            if args.data == "mimic":
                all_probs.extend(
                    torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                )
            elif args.data == "enrico":
                all_probs.extend(
                    torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                )

    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    test_f1_micro = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    test_auc = _safe_auc(args.data, all_labels, all_probs, n_labels)

    return (
        best_val_acc,
        best_val_f1,
        best_val_auc,
        test_acc,
        test_f1,
        test_f1_micro,
        test_auc,
    )


def main():
    args, _ = parse_args()
    _validate_args(args)

    logger = setup_logger(
        f"./logs/gshardgate/{args.data}",
        f"gshardgate_{args.data}",
        f"{args.modality}_SP_{args.fusion_sparse}_GT_{args.gate}.txt",
    )
    seeds = np.arange(args.seed, args.seed + args.n_runs)

    log_summary = "======================================================================================\n"
    model_kwargs = {
        "model": "Baseline_Transformer",
        "modality": args.modality,
        "gate": args.gate,
        "fusion_sparse": args.fusion_sparse,
        "initial_filling": args.initial_filling,
        "use_common_ids": args.use_common_ids,
        "train_epochs": args.train_epochs,
        "num_experts": args.num_experts,
        "num_routers": args.num_routers,
        "top_k": args.top_k,
        "num_layers_enc": args.num_layers_enc,
        "num_layers_fus": args.num_layers_fus,
        "num_layers_pred": args.num_layers_pred,
        "num_heads": args.num_heads,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "num_patches": args.num_patches,
        "gate_loss_weight": args.gate_loss_weight,
        "regularizer": "rep_cka",
        "regularizer_weight": args.regularizer_weight,
        "run_suffix": args.run_suffix,
    }
    log_summary += f"Model configuration: {model_kwargs}\n"

    print("Modality:", args.modality)

    val_accs = []
    val_f1s = []
    val_aucs = []
    test_accs = []
    test_f1s = []
    test_f1_micros = []
    test_aucs = []

    for seed in seeds:
        val_acc, val_f1, val_auc, test_acc, test_f1, test_f1_micro, test_auc = (
            train_and_evaluate(args, seed)
        )
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        val_aucs.append(val_auc)
        test_accs.append(test_acc)
        test_f1s.append(test_f1)
        test_f1_micros.append(test_f1_micro)
        test_aucs.append(test_auc)

    val_avg_acc = np.mean(val_accs) * 100
    val_std_acc = np.std(val_accs) * 100
    val_avg_f1 = np.mean(val_f1s) * 100
    val_std_f1 = np.std(val_f1s) * 100
    val_avg_auc = np.mean(val_aucs) * 100
    val_std_auc = np.std(val_aucs) * 100

    test_avg_acc = np.mean(test_accs) * 100
    test_std_acc = np.std(test_accs) * 100
    test_avg_f1 = np.mean(test_f1s) * 100
    test_std_f1 = np.std(test_f1s) * 100
    test_avg_f1_micro = np.mean(test_f1_micros) * 100
    test_std_f1_micro = np.std(test_f1_micros) * 100
    test_avg_auc = np.mean(test_aucs) * 100
    test_std_auc = np.std(test_aucs) * 100

    log_summary += f"[Val] Average Accuracy: {val_avg_acc:.2f} ± {val_std_acc:.2f} "
    log_summary += f"[Val] Average F1 Score: {val_avg_f1:.2f} ± {val_std_f1:.2f} "
    log_summary += f"[Val] Average AUC: {val_avg_auc:.2f} ± {val_std_auc:.2f} / "
    log_summary += (
        f"[Test] Average Accuracy: {test_avg_acc:.2f} ± {test_std_acc:.2f} "
    )
    log_summary += (
        f"[Test] Average F1 (Macro) Score: {test_avg_f1:.2f} ± {test_std_f1:.2f} "
    )
    log_summary += f"[Test] Average F1 (Micro) Score: {test_avg_f1_micro:.2f} ± {test_std_f1_micro:.2f} "
    log_summary += f"[Test] Average AUC: {test_avg_auc:.2f} ± {test_std_auc:.2f} "

    print(model_kwargs)
    print(
        f"[Val] Average Accuracy: {val_avg_acc:.2f} ± {val_std_acc:.2f} / "
        f"Average F1 Score: {val_avg_f1:.2f} ± {val_std_f1:.2f} / "
        f"Average AUC: {val_avg_auc:.2f} ± {val_std_auc:.2f}"
    )
    print(
        f"[Test] Average Accuracy: {test_avg_acc:.2f} ± {test_std_acc:.2f} / "
        f"Average F1 Score: {test_avg_f1:.2f} ± {test_std_f1:.2f} / "
        f"Average AUC: {test_avg_auc:.2f} ± {test_std_auc:.2f}"
    )
    logger.info(log_summary)


if __name__ == "__main__":
    main()
