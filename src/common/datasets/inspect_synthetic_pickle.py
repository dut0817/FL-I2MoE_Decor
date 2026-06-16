import argparse
import pickle
import numpy as np


def to_array(v):
    arr = np.asarray(v)
    if arr.dtype == object:
        arr = np.stack(v, axis=0)
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle", required=True, help="Path to synthetic pickle")
    args = parser.parse_args()

    with open(args.pickle, "rb") as f:
        data = pickle.load(f)

    print(f"Loaded: {args.pickle}")
    print(f"Top-level keys: {list(data.keys())}")

    for split in ["train", "valid", "test"]:
        if split not in data:
            print(f"[{split}] missing")
            continue
        print(f"\n[{split}]")
        keys = list(data[split].keys())
        print(f"keys: {keys}")
        for k in keys:
            arr = to_array(data[split][k])
            print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")
        y = np.asarray(data[split]["label"])
        uniq, cnt = np.unique(y, return_counts=True)
        dist = ", ".join([f"{int(u)}:{int(c)}" for u, c in zip(uniq, cnt)])
        print(f"  label_dist: {dist}")


if __name__ == "__main__":
    main()

