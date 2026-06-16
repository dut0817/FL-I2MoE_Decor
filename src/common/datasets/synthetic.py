import os
import pickle
import re

import numpy as np
import torch

from src.common.modules.common import TabularTokenEncoder
from src.common.utils import get_modality_combinations


def _stack_modal(split_dict, key):
    arr = np.asarray(split_dict[key])
    if arr.dtype == object:
        arr = np.stack(split_dict[key], axis=0)
    return arr.astype(np.float32)


def _infer_num_modalities_from_path(path):
    m = re.search(r"VEC(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


def _resolve_synthetic_pickles(args):
    paths = []

    multi = getattr(args, "synthetic_pickles", None)
    if multi:
        if isinstance(multi, str):
            multi = [multi]
        for item in multi:
            for p in str(item).split(","):
                p = p.strip()
                if p:
                    paths.append(p)

    single = getattr(args, "synthetic_pickle", None)
    if single:
        for p in str(single).split(","):
            p = p.strip()
            if p:
                paths.append(p)

    # Keep order and drop duplicates.
    unique_paths = []
    seen = set()
    for p in paths:
        if p not in seen:
            unique_paths.append(p)
            seen.add(p)

    if len(unique_paths) == 0:
        raise ValueError(
            "Provide --synthetic_pickle <path> or --synthetic_pickles <p1 p2 ...> when --data synthetic"
        )

    for p in unique_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Synthetic pickle not found: {p}")

    return unique_paths


def load_and_preprocess_data_synthetic(args):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    pickle_paths = _resolve_synthetic_pickles(args)
    raws = []

    for path in pickle_paths:
        with open(path, "rb") as f:
            raw = pickle.load(f)
        for split in ["train", "valid", "test"]:
            if split not in raw:
                raise ValueError(f"Missing split '{split}' in synthetic pickle: {path}")
            if "label" not in raw[split]:
                raise ValueError(f"Missing 'label' in split '{split}' in: {path}")
        raws.append(raw)

    modality_keys = [k for k in raws[0]["train"].keys() if k != "label"]
    modality_keys = sorted(
        modality_keys, key=lambda x: int(x) if str(x).isdigit() else str(x)
    )
    if len(modality_keys) == 0:
        raise ValueError("No modality keys found in synthetic pickle")

    for idx, raw in enumerate(raws[1:], start=1):
        cur_keys = [k for k in raw["train"].keys() if k != "label"]
        cur_keys = sorted(
            cur_keys, key=lambda x: int(x) if str(x).isdigit() else str(x)
        )
        if cur_keys != modality_keys:
            raise ValueError(
                f"Modality keys mismatch across pickles. "
                f"Ref={modality_keys}, current={cur_keys}, path={pickle_paths[idx]}"
            )

    if not hasattr(args, "modality") or not args.modality:
        n_mod = _infer_num_modalities_from_path(pickle_paths[0]) or len(modality_keys)
        args.modality = "".join(str(i) for i in range(n_mod))

    if len(args.modality) != len(modality_keys):
        raise ValueError(
            f"len(args.modality)={len(args.modality)} does not match "
            f"pickle modalities={len(modality_keys)} ({modality_keys}). "
            f"Use modality like '01', '012', '01234' to match the dataset."
        )

    train_n = sum(len(raw["train"]["label"]) for raw in raws)
    valid_n = sum(len(raw["valid"]["label"]) for raw in raws)
    test_n = sum(len(raw["test"]["label"]) for raw in raws)
    total_n = train_n + valid_n + test_n

    data_dict = {}
    encoder_dict = {}
    input_dims = {}
    transforms = {}
    masks = {}

    for i, src_key in enumerate(modality_keys):
        dst_key = f"m{i}"
        x_train_parts = []
        x_valid_parts = []
        x_test_parts = []

        feat_dim_ref = None
        for pidx, raw in enumerate(raws):
            x_train_cur = _stack_modal(raw["train"], src_key)
            x_valid_cur = _stack_modal(raw["valid"], src_key)
            x_test_cur = _stack_modal(raw["test"], src_key)

            if x_train_cur.ndim != 2:
                raise ValueError(
                    f"Synthetic modality '{src_key}' must be 2D tabular (N,F), "
                    f"got {x_train_cur.shape} in {pickle_paths[pidx]}"
                )

            feat_dim_cur = x_train_cur.shape[1]
            if feat_dim_ref is None:
                feat_dim_ref = feat_dim_cur
            elif feat_dim_cur != feat_dim_ref:
                raise ValueError(
                    f"Feature dim mismatch for modality '{src_key}': "
                    f"ref={feat_dim_ref}, current={feat_dim_cur}, path={pickle_paths[pidx]}"
                )

            x_train_parts.append(x_train_cur)
            x_valid_parts.append(x_valid_cur)
            x_test_parts.append(x_test_cur)

        x_train = np.concatenate(x_train_parts, axis=0).astype(np.float32)
        x_valid = np.concatenate(x_valid_parts, axis=0).astype(np.float32)
        x_test = np.concatenate(x_test_parts, axis=0).astype(np.float32)

        x_all = np.concatenate([x_train, x_valid, x_test], axis=0).astype(np.float32)
        data_dict[dst_key] = x_all

        feat_dim = x_all.shape[1]
        encoder_dict[dst_key] = TabularTokenEncoder(
            num_features=feat_dim,
            out_dim=args.hidden_dim,
        ).to(device)
        input_dims[dst_key] = feat_dim

    y_train = np.concatenate(
        [np.asarray(raw["train"]["label"]).astype(np.int64) for raw in raws], axis=0
    )
    y_valid = np.concatenate(
        [np.asarray(raw["valid"]["label"]).astype(np.int64) for raw in raws], axis=0
    )
    y_test = np.concatenate(
        [np.asarray(raw["test"]["label"]).astype(np.int64) for raw in raws], axis=0
    )
    labels = np.concatenate([y_train, y_valid, y_test], axis=0)
    n_labels = int(np.max(labels) + 1)

    observed_idx_arr = np.ones((total_n, len(modality_keys)), dtype=bool)

    train_idxs = np.arange(0, train_n).tolist()
    valid_idxs = np.arange(train_n, train_n + valid_n).tolist()
    test_idxs = np.arange(train_n + valid_n, total_n).tolist()

    combination_to_index = get_modality_combinations(args.modality)
    full_comb = "".join(sorted(set(args.modality)))
    full_idx = combination_to_index[full_comb]
    data_dict["modality_comb"] = np.full((total_n,), full_idx, dtype=np.int64).tolist()

    mc_num_to_mc = {v: k for k, v in combination_to_index.items()}
    mc_idx_dict = {
        mc_num_to_mc[full_idx]: list(range(total_n))
    }

    return (
        data_dict,
        encoder_dict,
        labels,
        train_idxs,
        valid_idxs,
        test_idxs,
        n_labels,
        input_dims,
        transforms,
        masks,
        observed_idx_arr,
        mc_idx_dict,
        mc_num_to_mc,
    )
