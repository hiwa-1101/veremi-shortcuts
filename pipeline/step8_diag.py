"""
Step 8 diagnostic - why does MixAll contain only Genuine and DoS labels?

Message labels need the ground-truth file: a message of an attacker counts as an
attack only if its content differs from the ground truth (oracle_d* > 1e-6), except
plain DoS. If the ground truth cannot be matched (different messageIDs), every
attack message silently becomes "genuine". This script checks that per attack type.

Usage:
    python step8_diag.py --raw F:\\mixall\\raw --zip F:\\mixall_zip\\MixAll_0024.zip --main F:\\veremi_parquet
"""

import argparse
import glob
import io
import json
import os
import zipfile

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--main", required=True, help="raw parquet folder of the main dataset (for comparison)")
    args = ap.parse_args()
    L = []
    p = lambda *a: (print(*a, flush=True), L.append(" ".join(map(str, a))))

    cols = ["label_vehicle", "oracle_gt_found", "oracle_dpos", "oracle_dspd", "oracle_dacl", "oracle_dhed",
            "messageID", "sender", "senderPseudo"]
    for name, pattern in [("MixAll", os.path.join(args.raw, "MixAll_0024_6*.parquet")),
                          ("Main (ConstPos 0709)", os.path.join(args.main, "ConstPos_0709_*.parquet"))]:
        files = sorted(glob.glob(pattern))[:2]
        if not files:
            p(f"{name}: no files for {pattern}")
            continue
        df = pd.concat([pd.read_parquet(f, columns=cols) for f in files])
        p(f"\n=== {name}: {len(df):,} rows from {[os.path.basename(f) for f in files]}")
        g = df.groupby("label_vehicle")
        t = pd.DataFrame({
            "rows": g.size(),
            "gt_found": g["oracle_gt_found"].mean().round(3),
            "dpos>1e-6": g["oracle_dpos"].apply(lambda s: (s > 1e-6).mean()).round(3),
            "dspd>1e-6": g["oracle_dspd"].apply(lambda s: (s > 1e-6).mean()).round(3),
            "median_dpos": g["oracle_dpos"].median().round(3),
        })
        p(t.to_string())
        p("messageID range:", int(df["messageID"].min()), int(df["messageID"].max()))

    # Raw look at the MixAll files of one hour: type-3 record keys and GT keys
    outer = zipfile.ZipFile(args.zip)
    name = [n for n in outer.namelist() if "VeReMi_61200_" in n][0]
    inner = zipfile.ZipFile(io.BytesIO(outer.read(name)))
    logs = [n for n in inner.namelist() if "traceJSON-" in n and "-A0-" not in n][:1]
    gts = [n for n in inner.namelist() if "GroundTruth" in n]
    p("\nattacker log file:", logs, "| ground-truth files:", gts)
    for n in logs:
        with inner.open(n) as f:
            for i, line in enumerate(f):
                r = json.loads(line)
                if r.get("type") == 3:
                    p("first type-3 record:", json.dumps(r)[:600])
                    break
    for n in gts[:1]:
        with inner.open(n) as f:
            p("first GT record:", f.readline().decode()[:600])
    # Which attack ids appear in MixAll file names?
    ids = pd.Series([n.split("-A")[1].split("-")[0] for n in inner.namelist() if "traceJSON-" in n]).value_counts()
    p("attack ids in file names (hour 61200):", ids.to_dict())
    open(os.path.join(os.path.dirname(args.raw.rstrip("\\/")), "mixall_diag.txt"), "w", encoding="utf-8").write("\n".join(L))
    print("\nSaved", os.path.join(os.path.dirname(args.raw.rstrip("\\/")), "mixall_diag.txt"))


if __name__ == "__main__":
    main()
