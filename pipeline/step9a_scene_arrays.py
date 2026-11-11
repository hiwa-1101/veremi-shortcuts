"""
Step 9a - Extra arrays for mixed-scene training (run once, about 5 minutes).

Adds to the deep-model cache, in exactly the same row order as step5a:
  receiver.npy  - receiving vehicle id
  file_id.npy   - which simulation file (attack folder x period x hour) the row comes from
  group_id.npy  - (period, sim_hour, receiver): the SAME receiver at the SAME time in every
                  attack folder (finding A4: all folders replay the same traffic)
  gs_id.npy     - (period, sim_hour, receiver, sender): one sender as heard by one receiver;
                  its rows in the different folders are the versions of the same reception
The row order is verified against the existing cache (sender and time must match).

Usage:
    python step9a_scene_arrays.py --features F:\\hiwa\\veremi_features_v3 --cache F:\\hiwa\\dl_cache
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--cache", required=True)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.features, "*.parquet")))
    parts = []
    for i, f in enumerate(files):
        d = pd.read_parquet(f, columns=["receiver", "sim_hour", "period", "sender", "rcvTime", "split"])
        d = d[d["split"].notna()]
        d["file_id"] = np.int32(i)
        parts.append(d.drop(columns=["split"]))
    df = pd.concat(parts, ignore_index=True)
    sender = np.load(os.path.join(args.cache, "sender.npy"), mmap_mode="r")
    t = np.load(os.path.join(args.cache, "time.npy"), mmap_mode="r")
    if len(df) != len(sender) or not np.array_equal(df["sender"].to_numpy(np.int64), np.asarray(sender)) \
            or not np.allclose(df["rcvTime"].to_numpy(np.float64), np.asarray(t)):
        raise SystemExit("Row order differs from the cache - stop and report this.")
    period = df["period"].astype(str).str.zfill(4)
    group = pd.factorize(pd.MultiIndex.from_arrays([period, df["sim_hour"], df["receiver"]]))[0]
    np.save(os.path.join(args.cache, "receiver.npy"), df["receiver"].to_numpy(np.int64))
    np.save(os.path.join(args.cache, "file_id.npy"), df["file_id"].to_numpy(np.int32))
    np.save(os.path.join(args.cache, "group_id.npy"), group.astype(np.int64))
    gs = pd.factorize(pd.MultiIndex.from_arrays([period, df["sim_hour"], df["receiver"], df["sender"]]))[0]
    np.save(os.path.join(args.cache, "gs_id.npy"), gs.astype(np.int64))
    print(f"rows={len(df):,} files={len(files)} receiver-time groups={group.max() + 1:,}  (order verified)")


if __name__ == "__main__":
    main()
