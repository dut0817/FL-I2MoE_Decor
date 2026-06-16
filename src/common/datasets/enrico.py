import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import csv
import random
import numpy as np
import torch

from src.common.encoders.vit_encoder import ViTImageEncoder
from src.common.utils import get_modality_combinations


def load_and_preprocess_data_enrico(args):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    data_dir = "data/enrico"
    csv_file = os.path.join(data_dir, "design_topics.csv")
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        example_list = list(reader)

    random_seed = 42
    train_split = 0.65
    val_split = 0.15

    img_dir = os.path.join(data_dir, "screenshots")
    wireframe_dir = os.path.join(data_dir, "wireframes")

    # Wireframe files are corrupted for these IDs.
    ignores = {"50105", "50109"}
    example_list = [e for e in example_list if e["screen_id"] not in ignores]

    keys = list(range(len(example_list)))
    random.Random(random_seed).shuffle(keys)

    train_keys = keys[: int(len(example_list) * train_split)]
    val_keys = keys[int(len(example_list) * train_split) : int(len(example_list) * (train_split + val_split))]
    test_keys = keys[int(len(example_list) * (train_split + val_split)) :]

    topics = sorted({e["topic"] for e in example_list})
    topic2idx = {topic: i for i, topic in enumerate(topics)}

    data_dict = {}
    encoder_dict = {}
    input_dims = {}
    transforms = {}
    masks = {}

    id_to_idx = {id_: idx for idx, id_ in enumerate(keys)}
    observed_idx_arr = np.zeros((len(keys), 2), dtype=bool)

    modality_combinations = [""] * len(id_to_idx)

    def update_modality_combinations(idx, modality):
        if modality_combinations[idx] == "":
            modality_combinations[idx] = modality
        else:
            modality_combinations[idx] += modality

    if ("S" in args.modality) and ("W" in args.modality):
        screenshot_paths = []
        wireframe_paths = []
        label_list = []

        for idx in range(len(keys)):
            example = example_list[keys[idx]]
            screen_id = example["screen_id"]

            screenshot_paths.append(os.path.join(img_dir, screen_id + ".jpg"))
            wireframe_paths.append(os.path.join(wireframe_dir, screen_id + ".png"))

            screen_label = topic2idx[example["topic"]]
            label_list.append(screen_label)

            update_modality_combinations(idx, "S")
            update_modality_combinations(idx, "W")

        observed_idx_arr[:, 0] = True
        observed_idx_arr[:, 1] = True

        # Keep image paths and encode on-the-fly in ViT encoders.
        data_dict["screenshot"] = {"paths": screenshot_paths}
        data_dict["wireframe"] = {"paths": wireframe_paths}

        image_encoder_path = os.environ.get("IMAGE_ENCODER_NAME_OR_PATH", "models/clip-vit-base-patch16")
        local_only = os.environ.get("TRANSFORMERS_LOCAL_ONLY", "0") == "1"
        encoder_dict["screenshot"] = ViTImageEncoder(
            name_or_path=image_encoder_path,
            out_dim=args.hidden_dim,
            freeze=True,
            use_cls=False,
            local_only=local_only,
        ).to(device)
        encoder_dict["wireframe"] = ViTImageEncoder(
            name_or_path=image_encoder_path,
            out_dim=args.hidden_dim,
            freeze=True,
            use_cls=False,
            local_only=local_only,
        ).to(device)

        input_dims["screenshot"] = args.hidden_dim
        input_dims["wireframe"] = args.hidden_dim
    else:
        raise ValueError("ENRICO setup requires modality to include both 'S' and 'W'.")

    combination_to_index = get_modality_combinations(args.modality)
    modality_combinations = ["".join(sorted(set(comb))) for comb in modality_combinations]

    combo_keys = combination_to_index.keys()
    data_dict["modality_comb"] = [
        combination_to_index[comb] if comb in combo_keys else -1
        for comb in modality_combinations
    ]

    train_idxs = [id_to_idx[id_] for id_ in train_keys if id_ in id_to_idx]
    valid_idxs = [id_to_idx[id_] for id_ in val_keys if id_ in id_to_idx]
    test_idxs = [id_to_idx[id_] for id_ in test_keys if id_ in id_to_idx]

    def all_modalities_missing(idx):
        return data_dict["modality_comb"][idx] == -1

    train_idxs = [idx for idx in train_idxs if not all_modalities_missing(idx)]
    valid_idxs = [idx for idx in valid_idxs if not all_modalities_missing(idx)]
    test_idxs = [idx for idx in test_idxs if not all_modalities_missing(idx)]

    mc_num_to_mc = {v: k for k, v in combination_to_index.items()}
    mc_idx_dict = {
        mc_num_to_mc[mc_num]: list(np.where(np.array(data_dict["modality_comb"]) == mc_num)[0])
        for mc_num in set(data_dict["modality_comb"])
        if mc_num != -1
    }

    labels = np.array(label_list)
    n_labels = 20

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
