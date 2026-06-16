import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MultiModalDataset(Dataset):
    def __init__(
        self,
        data_dict,
        observed_idx,
        ids,
        labels,
        input_dims,
        transforms,
        masks,
        use_common_ids=True,
    ):
        self.data_dict = data_dict
        self.mc = np.array(data_dict["modality_comb"])
        self.observed = observed_idx
        self.ids = np.array(ids)

        self.text_modalities = {}  # key -> List[str]
        self.path_modalities = {}  # key -> List[str]

        for k, v in data_dict.items():
            if isinstance(v, dict) and "texts" in v:
                self.text_modalities[k] = v["texts"]
            if isinstance(v, dict) and "paths" in v:
                self.path_modalities[k] = v["paths"]

        self.raw_tabular_keys = []
        self.note_texts = None

        for k, v in data_dict.items():
            if k == "modality_comb":
                continue
            if isinstance(v, dict) and ("paths" in v or "texts" in v):
                continue
            if k == "note":
                self.note_texts = v
            else:
                self.raw_tabular_keys.append(k)

        self.labels = np.array(labels)
        self.input_dims = input_dims
        self.transforms = transforms
        self.masks = masks
        self.use_common_ids = use_common_ids
        
        self.label = self.labels[ids]
        self.mc = self.mc[ids]
        self.observed = self.observed[ids]

    def process_2d_to_3d(self, data, idx, masks, transforms):
        subj1 = data[idx]
        subj_gm_3d = np.zeros(masks.shape, dtype=np.float32)
        subj_gm_3d.ravel()[masks] = subj1
        subj_gm_3d = subj_gm_3d.reshape((91, 109, 91))
        if transforms:
            subj_gm_3d = transforms(subj_gm_3d)
        sample = subj_gm_3d[None, :, :, :]  # Add channel dimension
        output = np.array(sample)

        return output

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        sample_data = {}

        for k, texts in self.text_modalities.items():
            sample_data.setdefault(k, {})
            sample_data[k]["text"] = texts[idx]

        for k, paths in self.path_modalities.items():
            sample_data.setdefault(k, {})
            sample_data[k]["path"] = paths[idx]

        for k in self.raw_tabular_keys:
            arr = self.data_dict[k][idx]  # (F,)
            if isinstance(arr, np.ndarray):
                sample_data[k] = arr.astype(np.float32)
            else:
                sample_data[k] = np.array(arr, dtype=np.float32)

        if self.note_texts is not None:
            sample_data.setdefault("note", {})
            sample_data["note"]["text"] = self.note_texts[idx]

        lab = self.labels[idx]
        if np.ndim(lab) == 0:
            label = int(lab)
        else:
            label = lab.astype(np.float32)
        mc = self.mc[i]
        observed = self.observed[i]

        return int(idx), sample_data, label, mc, observed


def collate_fn(batch):
    _, data, labels, mcs, observeds = zip(*batch)
    modalities = data[0].keys()
    collated_data = {
        modality: torch.tensor(
            np.stack([d[modality] for d in data]), dtype=torch.float32
        )
        for modality in modalities
    }

    labels = torch.tensor(labels, dtype=torch.long)
    mcs = torch.tensor(mcs, dtype=torch.long)
    observeds = torch.tensor(np.vstack(observeds))
    return collated_data, labels, mcs, observeds


def collate_fn_test(batch):
    sampele_ids, data, labels, mcs, observeds = zip(*batch)
    modalities = data[0].keys()
    collated_data = {
        modality: torch.tensor(
            np.stack([d[modality] for d in data]), dtype=torch.float32
        )
        for modality in modalities
    }
    sampele_ids = torch.tensor(sampele_ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    mcs = torch.tensor(mcs, dtype=torch.long)
    observeds = torch.tensor(np.vstack(observeds))
    return collated_data, sampele_ids, labels, mcs, observeds



def collate_fn_mmimdb(batch):
    # batch: [(sample_id, data_dict, label, mc, observed), ...]
    _, data, labels, mcs, observeds = zip(*batch)

    out = {}
    # language
    if "language" in data[0]:
        out.setdefault("language", {})
        out["language"]["texts"] = [d["language"]["text"] for d in data]  # List[str]

    # image
    if "img" in data[0]:
        out.setdefault("img", {})
        out["img"]["paths"] = [d["img"]["path"] for d in data]            # List[str]

    labels = torch.from_numpy(np.stack(labels)).float()   # (B,23)
    mcs = torch.tensor(mcs, dtype=torch.long)             # (B,)
    observeds = torch.from_numpy(np.vstack(observeds))    # e.g., (B,2)

    return out, labels, mcs, observeds


def collate_fn_mmimdb_test(batch):
    sample_ids, data, labels, mcs, observeds = zip(*batch)

    out = {}
    if "language" in data[0]:
        out.setdefault("language", {})
        out["language"]["texts"] = [d["language"]["text"] for d in data]  # List[str]

    if "img" in data[0]:
        out.setdefault("img", {})
        out["img"]["paths"] = [d["img"]["path"] for d in data]            # List[str]

    sample_ids = torch.tensor(sample_ids, dtype=torch.long)
    labels = torch.from_numpy(np.stack(labels)).float()
    mcs = torch.tensor(mcs, dtype=torch.long)
    observeds = torch.from_numpy(np.vstack(observeds))

    return out, sample_ids, labels, mcs, observeds


def collate_fn_enrico(batch):
    _, data, labels, mcs, observeds = zip(*batch)

    out = {}
    for k in data[0].keys():
        if isinstance(data[0][k], dict) and "path" in data[0][k]:
            out.setdefault(k, {})
            out[k]["paths"] = [d[k]["path"] for d in data]
        else:
            out[k] = torch.tensor(np.stack([d[k] for d in data]), dtype=torch.float32)

    labels = torch.tensor([int(l) for l in labels], dtype=torch.long)
    mcs = torch.tensor(mcs, dtype=torch.long)
    observeds = torch.tensor(np.vstack(observeds))
    return out, labels, mcs, observeds


def collate_fn_enrico_test(batch):
    sample_ids, data, labels, mcs, observeds = zip(*batch)

    out = {}
    for k in data[0].keys():
        if isinstance(data[0][k], dict) and "path" in data[0][k]:
            out.setdefault(k, {})
            out[k]["paths"] = [d[k]["path"] for d in data]
        else:
            out[k] = torch.tensor(np.stack([d[k] for d in data]), dtype=torch.float32)

    sample_ids = torch.tensor(sample_ids, dtype=torch.long)
    labels = torch.tensor([int(l) for l in labels], dtype=torch.long)
    mcs = torch.tensor(mcs, dtype=torch.long)
    observeds = torch.tensor(np.vstack(observeds))
    return out, sample_ids, labels, mcs, observeds

def collate_fn_mimic(batch):
    # batch: [(sample_id, data_dict, label, mc, observed), ...]
    _, data, labels, mcs, observeds = zip(*batch)

    out = {}

    # tabular modalities: lab, code, ... (excluding language/note/img)
    tabular_keys = [
        k for k in data[0].keys()
        if k not in ["language", "img", "note"]
    ]
    for k in tabular_keys:
        out[k] = torch.tensor(
            np.stack([d[k] for d in data]), dtype=torch.float32
        )  # (B, F)

    # note text -> input for BERTTextEncoder
    if "note" in data[0]:
        out.setdefault("note", {})
        out["note"]["texts"] = [d["note"]["text"] for d in data]  # List[str]

    # Cast labels to integer class ids.
    # labels: tuple of np.float32 (ex: (np.float32(0.), np.float32(1.), ...))
    labels = torch.tensor([int(l) for l in labels], dtype=torch.long)
    mcs = torch.tensor(mcs, dtype=torch.long)
    observeds = torch.tensor(np.vstack(observeds))

    return out, labels, mcs, observeds


def collate_fn_mimic_test(batch):
    sample_ids, data, labels, mcs, observeds = zip(*batch)

    out = {}

    tabular_keys = [
        k for k in data[0].keys()
        if k not in ["language", "img", "note"]
    ]
    for k in tabular_keys:
        out[k] = torch.tensor(
            np.stack([d[k] for d in data]), dtype=torch.float32
        )

    if "note" in data[0]:
        out.setdefault("note", {})
        out["note"]["texts"] = [d["note"]["text"] for d in data]

    sample_ids = torch.tensor(sample_ids, dtype=torch.long)
    # Same label casting for test collate.
    labels = torch.tensor([int(l) for l in labels], dtype=torch.long)
    mcs = torch.tensor(mcs, dtype=torch.long)
    observeds = torch.tensor(np.vstack(observeds))

    return out, sample_ids, labels, mcs, observeds

def create_loaders(
    data_dict,
    observed_idx,
    labels,
    train_ids,
    valid_ids,
    test_ids,
    batch_size,
    num_workers,
    pin_memory,
    input_dims,
    transforms,
    masks,
    use_common_ids=True,
    dataset="mimic",
):
    if "image" in list(data_dict.keys()) and dataset == "adni":
        train_transfrom = val_transform = test_transform = transforms["image"]
        # val_transform = test_transform = False
        mask = masks["image"]
    else:
        train_transfrom = val_transform = test_transform = False
        mask = None

    train_dataset = MultiModalDataset(
        data_dict,
        observed_idx,
        train_ids,
        labels,
        input_dims,
        train_transfrom,
        mask,
        use_common_ids,
    )
    valid_dataset = MultiModalDataset(
        data_dict,
        observed_idx,
        valid_ids,
        labels,
        input_dims,
        val_transform,
        mask,
        use_common_ids,
    )
    test_dataset = MultiModalDataset(
        data_dict,
        observed_idx,
        test_ids,
        labels,
        input_dims,
        test_transform,
        mask,
        use_common_ids,
    )

    if dataset == "mmimdb":
        # if False:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn_mmimdb,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn_mmimdb,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn_mmimdb_test,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    elif dataset == "enrico":
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn_enrico,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn_enrico,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn_enrico_test,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    elif dataset == "mimic":
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn_mimic,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn_mimic,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn_mimic_test,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn_test,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return train_loader, val_loader, test_loader
