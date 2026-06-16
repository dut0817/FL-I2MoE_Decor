import torch
from tqdm import trange
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error
from copy import deepcopy
from datetime import datetime
from fvcore.nn import FlopCountAnalysis, parameter_count
import time

from src.common.datasets.mimic import load_and_preprocess_data_mimic
from src.common.datasets.enrico import load_and_preprocess_data_enrico
from src.common.datasets.mmimdb import load_and_preprocess_data_mmimdb
from src.common.datasets.synthetic import load_and_preprocess_data_synthetic
from src.common.datasets.MultiModalDataset import create_loaders

from src.common.utils import (
    seed_everything,
    plot_total_loss_curves,
    plot_interaction_loss_curves,
    visualize_sample_weights,
    visualize_expert_logits,
    visualize_expert_logits_distribution,
    set_style,
)

from src.imoe.InteractionMoE import InteractionMoE
from src.imoe.regularizers import (
    canonicalize_regularizer,
    compute_regularizer_loss,
    format_regularizer_details,
    lambda_warmup_scale,
    regularizer_log_token,
    regularizer_save_subdir,
    regularizer_slug,
    regularizer_warmup_epochs_from_args,
    regularizer_weight_from_args,
)

set_style()

def _to_device_tree(batch_samples, device):
    out = {}
    for k, v in batch_samples.items():
        if isinstance(v, dict):
            sub = {}
            for kk, vv in v.items():
                if torch.is_tensor(vv):
                    sub[kk] = vv.to(device, non_blocking=True)
                else:
                    # Keep lists (texts/paths) and other Python objects as-is.
                    sub[kk] = vv
            out[k] = sub
        elif torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out

def _encode_batch_previous(batch_samples, encoder_dict):
    """
    Returns:
      feats: [ (B,T_txt,H), (B,T_img,H), ... ]
      masks: [ (B,T_txt),   (B,T_img),   ... ]  # 1(valid)/0(pad)
    """
    feats, masks = [], []

    # language (List[str])
    if "language" in batch_samples and "texts" in batch_samples["language"]:
        texts = batch_samples["language"]["texts"]              # List[str]
        x_lang, m_lang = encoder_dict["language"](texts)        # (B,T,H), (B,T)
        feats.append(x_lang)
        masks.append(m_lang)

    # image (List[str] of file paths OR List[PIL])
    if "img" in batch_samples and "paths" in batch_samples["img"]:
        paths = batch_samples["img"]["paths"]                   # List[str]
        x_img, m_img = encoder_dict["img"](paths)               # (B,T,H), (B,T)
        feats.append(x_img)
        masks.append(m_img)

    return feats, masks

def _encode_batch(batch_samples, encoder_dict, args):
    """
    Build and return feature/mask lists following args.modality order.
    Example: args.modality="LNC" -> fixed order ["lab", "note", "code"].

    Returns:
      feats: List[Tensor] (encoder output per modality)
      masks: List[Tensor or None]
      modal_names: List[str]
    """
    assert args is not None and hasattr(args, "modality"), "args.modality required"

    feats, masks, modal_names = [], [], []

    desired_keys = []
    for ch in str(args.modality).upper():
        if ch == "L":
            # L means lab for MIMIC and language for MMIMDb.
            if ("lab" in encoder_dict) or ("lab" in batch_samples):
                desired_keys.append("lab")
            elif ("language" in encoder_dict) or ("language" in batch_samples):
                desired_keys.append("language")
        elif ch == "N":
            desired_keys.append("note")
        elif ch == "C":
            desired_keys.append("code")
        elif ch == "I":
            desired_keys.append("img")
        elif ch == "T":
            desired_keys.append("language")
        elif ch == "S":
            desired_keys.append("screenshot")
        elif ch == "W":
            desired_keys.append("wireframe")
        elif ch.isdigit():
            desired_keys.append(f"m{ch}")

    def encode_one(key: str):
        if key not in encoder_dict:
            raise KeyError(f"Missing '{key}' encoder in encoder_dict")
        if key not in batch_samples:
            raise KeyError(f"Missing '{key}' in batch_samples")

        v = batch_samples[key]
        enc = encoder_dict[key]

        if isinstance(v, dict):
            if "texts" in v:
                x, m = enc(v["texts"])
                return x, m
            if "paths" in v:
                x, m = enc(v["paths"])
                return x, m
            raise KeyError(f"Unsupported dict payload for '{key}': expected 'texts' or 'paths'")

        if not torch.is_tensor(v):
            raise TypeError(f"batch_samples['{key}'] must be a Tensor/dict, got {type(v)}")

        out = enc(v)
        if isinstance(out, (tuple, list)) and len(out) == 2:
            x, m = out
        else:
            x, m = out, None
        return x, m

    # 1) Add modalities first in args.modality order (critical behavior).
    used = set()
    for k in desired_keys:
        x, m = encode_one(k)
        feats.append(x); masks.append(m); modal_names.append(k)
        used.add(k)

    # 2) Append extra encoders not listed in args.modality if present in batch (safe fallback).
    #    (Remove this block entirely if you do not want this behavior.)
    extra_keys = []
    for k in encoder_dict.keys():
        if k in used:
            continue
        if k in ["language", "note", "img"]:
            # Check whether this key is present as a dict in batch_samples.
            if k in batch_samples:
                extra_keys.append(k)
        else:
            if k in batch_samples and torch.is_tensor(batch_samples[k]):
                extra_keys.append(k)

    for k in sorted(extra_keys):
        x, m = encode_one(k)
        feats.append(x); masks.append(m); modal_names.append(k)

    return feats, masks, modal_names

def train_and_evaluate_imoe(args, seed, fusion_model, fusion):
    """Train and evaluate interaction MoE.

    Args:
        args (argparser.args): argument
        seed (int): random seed
        ensemble_model (nn.Module): ensemble model
        fusion (str): name of fusion method

    Raises:
        ValueError

    Returns:
        tuple: (best_val_acc, best_val_f1, best_val_auc, test_acc, test_f1, test_auc)
    """
    seed_everything(seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(device)
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
    elif args.data == "mmimdb":
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
        ) = load_and_preprocess_data_mmimdb(args)
    elif args.data == "synthetic":
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
        ) = load_and_preprocess_data_synthetic(args)
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

    ensemble_model = InteractionMoE(
        num_modalities=num_modalities,
        fusion_model=deepcopy(fusion_model),
        fusion_sparse=args.fusion_sparse,
        hidden_dim=args.hidden_dim,
        hidden_dim_rw=args.hidden_dim_rw,
        num_layer_rw=args.num_layer_rw,
        temperature_rw=args.temperature_rw,
    ).to(device)

    params = list(ensemble_model.parameters()) + [
        param for encoder in encoder_dict.values() for param in encoder.parameters()
    ]

    optimizer = torch.optim.Adam(params, lr=args.lr)
    if args.data == "mimic":
        criterion = torch.nn.CrossEntropyLoss(torch.tensor([0.25, 0.75]).to(device))
    elif args.data == "enrico":
        criterion = torch.nn.CrossEntropyLoss()
    elif args.data == "mmimdb":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.data == "synthetic":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported dataset for criterion: {args.data}")

    if args.data == "mmimdb":
        best_val_f1 = 0
    else:
        best_val_acc = 0.0

    regularizer = canonicalize_regularizer(getattr(args, "regularizer", "rep-cos"))
    regularizer_w = regularizer_weight_from_args(args, regularizer)
    warmup_epochs = regularizer_warmup_epochs_from_args(args, regularizer)
    print(
        f"[CFG] regularizer={regularizer} "
        f"regularizer_weight={regularizer_w} warmup_epochs={warmup_epochs}"
    )

    if args.fusion_sparse:
        plotting_total_losses = {"task": [], "interaction": [], "gate": []}
    else:
        plotting_total_losses = {"task": [], "interaction": []}
    plotting_total_losses["regularizer"] = []

    plotting_interaction_losses = {}
    for i in range(len(args.modality)):
        plotting_interaction_losses[f"uni_{i+1}"] = []
    plotting_interaction_losses[f"syn"] = []
    plotting_interaction_losses[f"red"] = []

    ############ efficiency
    train_time = 0
    ############ efficiency

    for epoch in trange(args.train_epochs):
        ############ efficiency
        epoch_start_time = time.time()
        ############ efficiency

        regularizer_w_eff = regularizer_w * lambda_warmup_scale(epoch, warmup_epochs)
        ensemble_model.train()

        for encoder in encoder_dict.values():
            encoder.train()

        batch_task_losses = []
        if args.fusion_sparse:
            batch_gate_losses = []
        batch_interaction_losses = []
        batch_regularizer_losses = []

        num_interaction_experts = len(args.modality) + 2
        interaction_loss_sums = [0] * (num_interaction_experts)
        minibatch_count = len(train_loader)

        for batch_samples, batch_labels, batch_mcs, batch_observed in train_loader:
            batch_samples = _to_device_tree(batch_samples, device)
            batch_labels  = batch_labels.to(device, non_blocking=True)
            batch_mcs     = batch_mcs.to(device, non_blocking=True)
            batch_observed= batch_observed.to(device, non_blocking=True)
            optimizer.zero_grad()

            features, masks, modal_names = _encode_batch(batch_samples, encoder_dict, args)

            # ==================== Quick sanity check (first batch only) ====================
            if epoch == 0 and len(batch_task_losses) == 0:
                print("\n[DBG] ====== Input Sanity Check ======")

                def masked_mean(x, m=None):
                    if m is None:
                        return x.mean(dim=1)
                    m = m.float().unsqueeze(-1)
                    return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)

                for name, x, m in zip(modal_names, features, masks):
                    # x can vary by encoder, e.g., (B,T,H) or (B,H).
                    x_shape = tuple(x.shape) if torch.is_tensor(x) else None
                    m_shape = tuple(m.shape) if torch.is_tensor(m) else None
                    print(f"[DBG] {name:>8} feat={x_shape} mask={m_shape}")

                    if torch.is_tensor(m):
                        print(f"      mask_sum={float(m.sum().item()):.1f} mask_uniq={m.unique().detach().cpu().tolist()}")
                    else:
                        print("      mask=None")

                    # Print norm (masked mean for sequences, direct norm for vectors).
                    if torch.is_tensor(x):
                        if x.dim() >= 3:   # (B,T,H)
                            v = masked_mean(x, m)
                            n = v.norm(dim=1).mean().item()
                        elif x.dim() == 2: # (B,H)
                            n = x.norm(dim=1).mean().item()
                        else:
                            n = x.float().abs().mean().item()
                        print(f"      ||v|| mean={float(n):.4f}")

                print("[DBG] =================================\n")

            # ================================================================
            if args.fusion_sparse:
                expert_outputs, interaction_weights, outputs, interaction_losses, gate_losses, all_latents = ensemble_model(
                    features, masks
                )
            else:
                expert_outputs, interaction_weights, outputs, interaction_losses, all_latents = ensemble_model(features, masks)

            if args.data == "mmimdb":
                task_loss = criterion(outputs, batch_labels.float())
            else:
                task_loss = criterion(outputs, batch_labels)

            interaction_loss = sum(interaction_losses) / (len(args.modality) + 2)
            if regularizer_w_eff > 0.0:
                regularizer_loss, regularizer_details = compute_regularizer_loss(
                    regularizer,
                    all_latents=all_latents,
                    reference_tensor=outputs,
                )
            else:
                regularizer_loss = outputs.new_zeros(())
                regularizer_details = {}
            
            if args.fusion_sparse:
                if len(gate_losses) == 0:
                    gate_loss = outputs.new_zeros(())
                elif torch.is_tensor(gate_losses[0]):
                    gate_loss = torch.stack(gate_losses).mean()
                else:
                    gate_loss = outputs.new_tensor(gate_losses).mean()
                loss = (
                    task_loss
                    + args.interaction_loss_weight * interaction_loss
                    + args.gate_loss_weight * gate_loss
                    + regularizer_w_eff * regularizer_loss
                )
            else:
                loss = (
                    task_loss
                    + args.interaction_loss_weight * interaction_loss
                    + regularizer_w_eff * regularizer_loss
                )

            loss.backward()
            optimizer.step()
            if (len(batch_task_losses) % 100) == 0:
                mw = interaction_weights.mean(dim=0).detach().cpu().tolist()
                detail_str = format_regularizer_details(regularizer_details)
                detail_str = f", {detail_str}" if detail_str else ""
                print(
                    f"[Epoch {epoch+1}] step {len(batch_task_losses)} "
                    f"T:{task_loss.item():.4f} I:{interaction_loss.item():.4f} "
                    f"{regularizer_log_token(regularizer)}:{regularizer_loss.item():.4f} "
                    f"(λeff={regularizer_w_eff:.4f}{detail_str}) | "
                    f"W:{[round(x,3) for x in mw]}"
                )
            batch_task_losses.append(task_loss.item())
            batch_interaction_losses.append(interaction_loss.item())
            batch_regularizer_losses.append(float(regularizer_loss.detach().item()))
            if args.fusion_sparse:
                batch_gate_losses.append(gate_loss.item())

            for idx, loss in enumerate(interaction_losses):
                interaction_loss_sums[idx] += loss.item()

            if args.data == "enrico":
                torch.nn.utils.clip_grad_norm_(params, 1.0)

        ############ efficiency
        epoch_end_time = time.time()
        train_epoch_time = epoch_end_time - epoch_start_time
        train_time += train_epoch_time
        ############ efficiency

        plotting_total_losses["task"].append(np.mean(batch_task_losses))
        plotting_total_losses["interaction"].append(np.mean(batch_interaction_losses))
        plotting_total_losses["regularizer"].append(np.mean(batch_regularizer_losses))
        if args.fusion_sparse:
            plotting_total_losses["gate"].append(np.mean(batch_gate_losses))

        for i in range(len(args.modality)):
            avg_loss = interaction_loss_sums[i] / minibatch_count
            plotting_interaction_losses[f"uni_{i+1}"].append(avg_loss)

        # For syn and red interaction losses
        plotting_interaction_losses["syn"].append(
            interaction_loss_sums[-2] / minibatch_count
        )
        plotting_interaction_losses["red"].append(
            interaction_loss_sums[-1] / minibatch_count
        )

        ensemble_model.eval()
        for encoder in encoder_dict.values():
            encoder.eval()

        all_preds = []
        all_labels = []
        all_probs = []
        val_losses = []

        with torch.no_grad():
            for batch_samples, batch_labels, batch_mcs, batch_observed in val_loader:
                batch_samples  = _to_device_tree(batch_samples, device)
                batch_labels   = batch_labels.to(device, non_blocking=True)
                batch_mcs      = batch_mcs.to(device, non_blocking=True)
                batch_observed = batch_observed.to(device, non_blocking=True)

                features, masks, _ = _encode_batch(batch_samples, encoder_dict, args)

                _, _, outputs = ensemble_model.inference(features, masks)

                if args.data == "mmimdb":
                    val_loss = criterion(outputs, batch_labels.float())
                else:
                    val_loss = criterion(outputs, batch_labels)
                val_losses.append(val_loss.item())
                if args.data == "mmimdb":
                    preds = torch.sigmoid(outputs).round()
                else:
                    _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                if args.data in ["mimic", "synthetic"]:
                    all_probs.extend(
                        torch.nn.functional.softmax(outputs, dim=1)[:, 1]
                        .cpu()
                        .numpy()
                    )
                else:
                    probs = (
                        torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                    )
                    all_probs.extend(probs)
                    if probs.shape[1] != n_labels:
                        raise ValueError("Incorrect output shape from the model")

            val_acc = accuracy_score(all_labels, all_preds)
            val_f1 = f1_score(all_labels, all_preds, average="macro")
            if args.data == "enrico":
                val_auc = roc_auc_score(
                    np.array(all_labels),
                    np.array(all_probs),
                    multi_class="ovo",
                    labels=list(range(n_labels)),
                )
            elif args.data == "mimic":
                val_auc = roc_auc_score(all_labels, all_probs)
            elif args.data == "synthetic":
                val_auc = roc_auc_score(all_labels, all_probs)
            elif args.data == "mmimdb":
                val_auc = 0

            print(
                f"[Seed {seed}/{args.n_runs-1}] [Epoch {epoch+1}/{args.train_epochs}]  Val Loss: {val_loss:.2f}, Val Acc: {val_acc*100:.2f}, Val F1: {val_f1*100:.2f}, Val AUC: {val_auc*100:.2f}"
            )

            if args.data == "mmimdb":
                # if False:
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_val_acc = val_acc
                    best_val_auc = val_auc
                    print(
                        f" [(**Best**) Epoch {epoch+1}/{args.train_epochs}] Val Acc: {val_acc*100:.2f}, Val F1: {val_f1*100:.2f}, Val AUC: {val_auc*100:.2f}"
                    )

                    best_model_fus = deepcopy(ensemble_model.state_dict())
                    best_model_enc = {
                        modality: deepcopy(encoder.state_dict())
                        for modality, encoder in encoder_dict.items()
                    }

                    if args.save:
                        best_model_fus_cpu = {
                            k: v.cpu() for k, v in best_model_fus.items()
                        }
                        best_model_enc_cpu = {
                            modality: {k: v.cpu() for k, v in enc_state.items()}
                            for modality, enc_state in best_model_enc.items()
                        }
            else:
                if val_acc > best_val_acc:
                    print(
                        f" [(**Best**) Epoch {epoch+1}/{args.train_epochs}] Val Acc: {val_acc*100:.2f}, Val F1: {val_f1*100:.2f}, Val AUC: {val_auc*100:.2f}"
                    )
                    best_val_acc = val_acc
                    best_val_f1 = val_f1
                    best_val_auc = val_auc
                    best_model_fus = deepcopy(ensemble_model.state_dict())
                    best_model_enc = {
                        modality: deepcopy(encoder.state_dict())
                        for modality, encoder in encoder_dict.items()
                    }
                    # Move the models to CPU for saving (only state_dict)
                    if args.save:
                        best_model_fus_cpu = {
                            k: v.cpu() for k, v in best_model_fus.items()
                        }
                        best_model_enc_cpu = {
                            modality: {k: v.cpu() for k, v in enc_state.items()}
                            for modality, enc_state in best_model_enc.items()
                        }
    ############ efficiency
    total_param = parameter_count(ensemble_model)[""]
    # flop = FlopCountAnalysis(ensemble_model, fusion_input)
    total_flop = 0
    ############ efficiency

    plot_total_loss_curves(
        args,
        plotting_total_losses=plotting_total_losses,
        framework="imoe",
        fusion=fusion,
    )

    plot_interaction_loss_curves(
        args,
        plotting_interaction_losses=plotting_interaction_losses,
        framework="imoe",
        fusion=fusion,
    )
    # Save the best model
    if args.save:
        save_subdir = regularizer_save_subdir(args, regularizer)
        run_tag = int(seed) + 1
        regularizer_name = regularizer_slug(regularizer)
        ckpt_dir = Path(f"./saves/imoe/{fusion}/{args.data}/{save_subdir}")

        Path("./saves").mkdir(exist_ok=True, parents=True)
        ckpt_dir.mkdir(exist_ok=True, parents=True)

        if args.data == "mmimdb":
            save_path = ckpt_dir / (
                f"seed_{seed}_modality_{args.modality}_train_epochs_{args.train_epochs}"
                f"_val_f1_{best_val_f1:.2f}_{regularizer_name}_{regularizer_w}_run{run_tag}.pth"
            )
        else:
            save_path = ckpt_dir / (
                f"seed_{seed}_modality_{args.modality}_train_epochs_{args.train_epochs}"
                f"_val_acc_{best_val_acc:.2f}_{regularizer_name}_{regularizer_w}_run{run_tag}.pth"
            )
        torch.save(
            {"ensemble_model": best_model_fus_cpu, "encoder_dict": best_model_enc_cpu},
            str(save_path),
        )

        print(f"Best model saved to {save_path}")

    # Load best model for test evaluation
    for modality, encoder in encoder_dict.items():
        encoder.load_state_dict(best_model_enc[modality])
        encoder.eval()

    ensemble_model.load_state_dict(best_model_fus)
    ensemble_model.eval()

    all_preds = []
    all_labels = []
    all_ids = []
    all_probs = []
    test_losses = []
    all_routing_weights = []
    num_experts = len(args.modality) + 2
    all_expert_outputs = [[] for _ in range(num_experts)]

    ############ efficiency
    infer_time = 0
    ############ efficiency

    with torch.no_grad():
        ############ efficiency
        epoch_start_time = time.time()
        ############ efficiency

        for (
            batch_samples,
            batch_ids,
            batch_labels,
            batch_mcs,
            batch_observed,
        ) in test_loader:
            batch_samples  = _to_device_tree(batch_samples, device)
            batch_labels   = batch_labels.to(device, non_blocking=True)
            batch_mcs      = batch_mcs.to(device, non_blocking=True)
            batch_observed = batch_observed.to(device, non_blocking=True)

            features, masks, _ = _encode_batch(batch_samples, encoder_dict, args)
            expert_outputs, routing_weights, outputs = ensemble_model.inference(
                features, masks
            )

            for expert_idx in range(num_experts):
                all_expert_outputs[expert_idx].extend(
                    expert_outputs[expert_idx].cpu().numpy()
                )

            all_routing_weights.extend(routing_weights.cpu().numpy())

            if args.data == "mmimdb":
                preds = torch.sigmoid(outputs).round()
            else:
                _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            all_ids.extend(batch_ids.cpu().numpy())

            if args.data in ["mimic", "synthetic"]:
                all_probs.extend(
                    torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                )
            else:
                all_probs.extend(
                    torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                )

    ############ efficiency
    epoch_end_time = time.time()
    infer_epoch_time = epoch_end_time - epoch_start_time
    infer_time += infer_epoch_time
    ############ efficiency

    visualize_expert_logits(
        expert_outputs, routing_weights, outputs, args, framework="imoe", fusion=fusion
    )

    visualize_expert_logits_distribution(
        all_expert_outputs, args, framework="imoe", fusion=fusion
    )

    visualize_sample_weights(all_routing_weights, args, framework="imoe", fusion=fusion)

    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average="macro")
    test_f1_micro = f1_score(all_labels, all_preds, average="micro")
    if args.data == "enrico":
        test_auc = roc_auc_score(
            np.array(all_labels),
            np.array(all_probs),
            multi_class="ovo",
            labels=list(range(n_labels)),
        )
    elif args.data == "mimic":
        test_auc = roc_auc_score(all_labels, all_probs)
    elif args.data == "synthetic":
        test_auc = roc_auc_score(all_labels, all_probs)
    elif args.data == "mmimdb":
        test_auc = 0
    else:
        raise ValueError(f"Unsupported dataset for AUC: {args.data}")

    now = datetime.now()
    save_dir = Path(
        f"./outputs/imoe/{fusion}/{args.data}_{now.strftime('%Y-%m-%d_%H:%M:%S')}"
    )
    save_dir.mkdir(exist_ok=True, parents=True)
    np.save(save_dir / "all_expert_outputs.npy", np.array(all_expert_outputs))
    np.save(save_dir / "all_routing_weights.npy", np.array(all_routing_weights))
    np.save(save_dir / "all_preds.npy", np.array(all_preds))
    np.save(save_dir / "all_labels.npy", np.array(all_labels))
    np.save(save_dir / "all_ids.npy", np.array(all_ids))

    return (
        best_val_acc,
        best_val_f1,
        best_val_auc,
        test_acc,
        test_f1,
        test_f1_micro,
        test_auc,
        train_time / args.train_epochs,
        infer_time,
        total_flop,
        total_param,
    )
