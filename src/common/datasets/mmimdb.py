# src/common/datasets/mmimdb.py

import os, json, glob
import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer

# Encoders added for this project setup.
from src.common.encoders.transformer_text_encoder import BERTTextEncoder
from src.common.encoders.vit_encoder import ViTImageEncoder

GENRES = [
    "drama","comedy","romance","thriller","crime","action","adventure","horror",
    "documentary","mystery","sci-fi","music","fantasy","family","biography","war",
    "history","animation","musical","western","sport","short","film-noir"
]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
_MM_ROOT = os.environ.get("MMIMDB_ROOT", os.path.join(_REPO_ROOT, "data", "mm-imdb"))
_DS_DIR = os.environ.get("MMIMDB_DATASET_DIR", os.path.join(_MM_ROOT, "dataset"))

def _list_ids_from_folder(ds_dir: str):
    """Collect imdb_id list (without extension) from dataset/*.json filenames."""
    paths = sorted(glob.glob(os.path.join(ds_dir, "*.json")))
    ids = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    return ids

def _gather_samples(ds_dir: str, ids):
    """Build a list of tuples: (mid, caption, genres, img_path)."""
    out = []
    for mid in ids:
        j = os.path.join(ds_dir, f"{mid}.json")
        with open(j, "r") as f:
            meta = json.load(f)
        plots  = meta.get("plot", [])
        cap    = plots[0] if len(plots) > 0 else ""
        genres = [g.lower() for g in meta.get("genres", [])]
        imgp   = os.path.join(ds_dir, f"{mid}.jpeg")
        out.append((mid, cap, genres, imgp))
    return out

def _contiguous_splits(N: int):
    """
    Prefer legacy contiguous split ranges used by the old HDF5 setup.
    - If total size is 25959: (0:15552)=train, (15552:18160)=val, (18160:25959)=test
    - Otherwise use approximate ratio split (60/10/30).
    """
    if N == 25959:
        t1, t2 = 15552, 18160
        return (0, t1), (t1, t2), (t2, N)
    # Ratio split (rounded).
    tr = int(round(N * 0.599))  # 15552/25959 ≈ 0.599
    va = int(round(N * 0.100))  # 2608/25959  ≈ 0.100
    te = N - tr - va
    return (0, tr), (tr, tr + va), (tr + va, N)

def load_and_preprocess_data_mmimdb(args):
    """
    Build train/val/test splits from contiguous indices directly from
    MMIMDb raw folder, without split.json,
    and use CLIP-Text / CLIP-ViT encoders for on-the-fly encoding.
    """
    device = torch.device(args.device)
    H      = args.hidden_dim
    use_modal = args.modality  # one of: "LI", "L", "I"

    # 1) ID list (sorted by filename).
    ids_all = _list_ids_from_folder(_DS_DIR)
    if len(ids_all) == 0:
        raise RuntimeError(f"No JSON files under: {_DS_DIR}")

    # 2) Contiguous index split.
    (a1, a2), (b1, b2), (c1, c2) = _contiguous_splits(len(ids_all))
    ids_train = ids_all[a1:a2]
    ids_val   = ids_all[b1:b2]
    ids_test  = ids_all[c1:c2]

    # 3) Load metadata.
    samples_all = _gather_samples(_DS_DIR, ids_all)
    id2idx = {mid: i for i, (mid, *_ ) in enumerate(samples_all)}
    idx2id = {v: k for k, v in id2idx.items()}

    train_idxs = [id2idx[m] for m in ids_train]
    valid_idxs = [id2idx[m] for m in ids_val]
    test_idxs  = [id2idx[m] for m in ids_test]

    # 4) Labels (one-hot).
    mlb = MultiLabelBinarizer(classes=GENRES)
    mlb.fit([GENRES])
    labels = np.zeros((len(samples_all), len(GENRES)), dtype=np.float32)
    for i, (_mid, _cap, gs, _imgp) in enumerate(samples_all):
        labels[i] = mlb.transform([gs])[0].astype(np.float32)
    n_labels = len(GENRES)

    # 5) data_dict: keep only raw assets.
    texts = [cap for (_mid, cap, _gs, _ip) in samples_all]
    paths = [ip  for (_mid, _cap, _gs, ip) in samples_all]
    data_dict = {}
    if "L" in use_modal: data_dict["language"] = {"texts": texts}
    if "I" in use_modal: data_dict["img"]      = {"paths": paths}

    # 6) Observation mask / modality combination.
    obs = np.zeros((len(samples_all), 2), dtype=bool)  # (L,I)
    if "L" in use_modal: obs[:, 0] = True
    if "I" in use_modal: obs[:, 1] = True

    # Single/multi-modal index mapping (keep legacy format).
    mc_num_to_mc = {0:"L", 1:"I", 2:"LI"}
    if use_modal == "LI":
        mc_idx_dict = {"LI": list(range(len(samples_all)))}
        data_dict["modality_comb"] = [2] * len(samples_all)
    elif use_modal == "L":
        mc_idx_dict = {"L": list(range(len(samples_all)))}
        data_dict["modality_comb"] = [0] * len(samples_all)
    else:  # "I"
        mc_idx_dict = {"I": list(range(len(samples_all)))}
        data_dict["modality_comb"] = [1] * len(samples_all)

    # 7) Encoders (freezing recommended).
    encoder_dict = {}
    if "L" in use_modal:
        encoder_dict["language"] = BERTTextEncoder(
            name_or_path=os.environ.get("TEXT_ENCODER_NAME_OR_PATH", "models/roberta-base"),
            out_dim=H, max_len=getattr(args, "max_txt_len", 320),
            freeze=True,
            local_only=os.environ.get("TRANSFORMERS_LOCAL_ONLY", "0") == "1",
        ).to(device).eval()
    if "I" in use_modal:
        encoder_dict["img"] = ViTImageEncoder(
            name_or_path=os.environ.get("IMAGE_ENCODER_NAME_OR_PATH", "models/clip-vit-base-patch16"),
            out_dim=H,
            freeze=True,
            use_cls=False,
            local_only=os.environ.get("TRANSFORMERS_LOCAL_ONLY", "0") == "1",
        ).to(device).eval()

    # 8) Input dimension dict.
    input_dims = {}
    if "L" in use_modal: input_dims["language"] = H
    if "I" in use_modal: input_dims["img"]      = H

    # Placeholders for fields unused in this project.
    transforms, masks = {}, {}

    # 9) (Optional) save mapping from test index to imdb_id.
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/test_idx_to_mmimdb_ids.json", "w") as f:
        json.dump({idx: idx2id[idx] for idx in test_idxs}, f)

    return (
        data_dict,
        encoder_dict,
        labels,
        train_idxs, valid_idxs, test_idxs,
        n_labels,
        input_dims,
        transforms, masks,
        obs,
        mc_idx_dict, mc_num_to_mc
    )
