"""
Step 11c - Make NextGen fit in memory: keep a fixed random share of RECEIVING vehicles.

NextGen has 196.5 million received messages (2.6 x VeReMi Extension), more than fits in
64 GB RAM / 16 GB GPU memory with our pipeline. We therefore keep a fixed random fraction of
receivers (default 25 %) in every scenario and split, with ALL their messages:
  * a kept receiver keeps its complete stream, so link histories and the 1-s neighbourhood
    are exactly what that vehicle really heard (no message inside a stream is dropped);
  * the SAME receivers are kept in all 15 attack folders (they replay the same traffic), so
    mixed-scene training still finds every reception in the other folders;
  * the choice uses only receiver ids and a fixed seed (no labels), and is applied identically
    to train, validation and test - every model is trained and tested on the same rows.

Usage:
    python step11c_nextgen_subsample.py --features F:\\hiwa\\nextgen\\features --out F:\\hiwa\\nextgen\\features_sub --fraction 0.25
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fraction", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.features, "*.parquet")))

    sel_path = os.path.join(args.out, "kept_receivers.csv")
    if os.path.exists(sel_path):
        keep = pd.read_csv(sel_path, dtype={"period": str})
    else:
        rx = []
        for f in files:
            d = pd.read_parquet(f, columns=["period", "sim_hour", "receiver"]).drop_duplicates()
            rx.append(d)
        rx = pd.concat(rx).drop_duplicates().sort_values(["period", "sim_hour", "receiver"]).reset_index(drop=True)
        rng = np.random.default_rng(args.seed)
        parts = []
        for _, g in rx.groupby(["period", "sim_hour"], sort=True):
            n = max(1, int(round(args.fraction * len(g))))
            parts.append(g.iloc[np.sort(rng.choice(len(g), n, replace=False))])
        keep = pd.concat(parts, ignore_index=True)
        keep.to_csv(sel_path, index=False)
        summ = rx.groupby(["period", "sim_hour"]).size().rename("receivers").to_frame()
        summ["kept"] = keep.groupby(["period", "sim_hour"]).size()
        print(summ.to_string())
    keep["period"] = keep["period"].astype(str)

    total_in = total_out = 0
    for f in files:
        out = os.path.join(args.out, os.path.basename(f))
        if os.path.exists(out):
            continue
        d = pd.read_parquet(f)
        d["period"] = d["period"].astype(str)
        n0 = len(d)
        d = d.merge(keep, on=["period", "sim_hour", "receiver"], how="inner")
        tmp = out + ".tmp"
        d.to_parquet(tmp, index=False)
        os.replace(tmp, out)
        total_in += n0; total_out += len(d)
        print(f"{os.path.basename(f)}: {n0:,} -> {len(d):,}", flush=True)
    json.dump({"fraction": args.fraction, "seed": args.seed}, open(os.path.join(args.out, "subsample.json"), "w"))
    if total_in:
        print(f"rows {total_in:,} -> {total_out:,} ({total_out / total_in:.1%})")


if __name__ == "__main__":
    main()
