"""
Step 5a - Prepare compact arrays for the deep model (run once per feature folder).

What it does:
  * Loads all feature files (step 3 output) into one table.
  * Transforms features: signed log1p (heavy tails) -> z-score using TRAIN rows
    only -> NaN set to 0 -> clipped to [-10, 10]. Statistics are saved.
  * Builds two orderings used to assemble the model inputs on the GPU:
      link order : (file, receiver, pseudonym, time)  -> message history of a link
      rx order   : (file, receiver, time)             -> what else the receiver heard
  * Writes everything as .npy files that the training script loads directly.

Usage:
    python step5a_prepare.py --features F:\\veremi_features_v3 --out F:\\dl_cache
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

FEATURE_PREFIXES = ("k_", "r_", "c_", "n_", "m_")
SPLIT_CODE = {"train": 0, "val": 1, "test": 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.features, "*.parquet")))
    first = pd.read_parquet(files[0])
    feats = [c for c in first.columns if c.startswith(FEATURE_PREFIXES)]
    if "aux_pos_x" not in first.columns:
        raise SystemExit("aux_pos_x missing: re-run the latest step3_features_splits.py first.")
    cols = feats + ["aux_pos_x", "aux_pos_y", "label_msg", "split", "sender",
                    "receiver", "senderPseudo", "rcvTime", "attack_folder", "period"]

    parts = []
    for i, f in enumerate(files):
        d = pd.read_parquet(f, columns=cols)
        d["file_id"] = np.int32(i)
        parts.append(d)
        print(f"read {i + 1}/{len(files)}  {os.path.basename(f)}  rows={len(d):,}")
    df = pd.concat(parts, ignore_index=True)
    del parts
    df = df[df["split"].notna()].reset_index(drop=True)
    n = len(df)
    print(f"Total rows: {n:,}")

    # ---- feature transform (train statistics only) -----------------------
    X = df[feats].to_numpy(np.float32)
    X = np.sign(X) * np.log1p(np.abs(X))
    split = df["split"].map(SPLIT_CODE).to_numpy(np.int8)
    tr = split == 0
    mean = np.nanmean(X[tr], axis=0)
    std = np.nanstd(X[tr], axis=0)
    std[std < 1e-6] = 1.0
    X = (X - mean) / std
    X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
    X = np.clip(X, -10, 10).astype(np.float16)

    # ---- orderings --------------------------------------------------------
    link_id = pd.factorize(pd.MultiIndex.from_arrays(
        [df["file_id"], df["receiver"], df["senderPseudo"]]))[0].astype(np.int64)
    rx_id = pd.factorize(pd.MultiIndex.from_arrays(
        [df["file_id"], df["receiver"]]))[0].astype(np.int64)
    t = df["rcvTime"].to_numpy(np.float64)
    order_link = np.lexsort((t, link_id)).astype(np.int64)
    order_rx = np.lexsort((t, rx_id)).astype(np.int64)
    rank_link = np.empty(n, np.int64); rank_link[order_link] = np.arange(n)
    rank_rx = np.empty(n, np.int64); rank_rx[order_rx] = np.arange(n)

    labels_raw = df["label_msg"].to_numpy()
    classes = np.unique(labels_raw)
    y = np.searchsorted(classes, labels_raw).astype(np.int16)

    save = {
        "X": X, "y": y, "split": split,
        "sender": df["sender"].to_numpy(np.int64),
        "link_id": link_id, "rx_id": rx_id,
        "pseudo": df["senderPseudo"].to_numpy(np.int64),
        "time": t,
        "pos": df[["aux_pos_x", "aux_pos_y"]].to_numpy(np.float32),
        "order_link": order_link, "rank_link": rank_link,
        "order_rx": order_rx, "rank_rx": rank_rx,
    }
    for k, v in save.items():
        np.save(os.path.join(args.out, f"{k}.npy"), v)
    df[["attack_folder", "period"]].to_parquet(os.path.join(args.out, "meta.parquet"), index=False)
    with open(os.path.join(args.out, "info.json"), "w") as f:
        json.dump({"features": feats, "classes": [int(c) for c in classes],
                   "mean": mean.tolist(), "std": std.tolist(), "n_rows": int(n),
                   "source": os.path.abspath(args.features)}, f, indent=1)
    print("Classes:", list(classes))
    print("Rows per split:", {k: int((split == v).sum()) for k, v in SPLIT_CODE.items()})
    print(f"Saved arrays to {args.out}")


if __name__ == "__main__":
    main()
