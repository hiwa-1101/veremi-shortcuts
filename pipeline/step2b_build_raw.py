"""
Step 2b - Build the raw, labeled message table for VeReMi Extension.

One Parquet file is written per simulation (attack folder x hour). Each row is
one received BSM (type 3) with:
  * identifiers: attack_folder, period, sim_hour, receiver, sender, senderPseudo, messageID
  * reported kinematics: pos/spd/acl/hed (x, y) and their *_noise fields
  * receiver's own latest GPS state (type 2) at reception time (rx_*)
  * labels:
      label_vehicle  = attack type of the SENDER, taken from its log-file name (A<k>)
  * ORACLE columns (prefix 'oracle_'): deviation of the message from ground truth.
      These are for labeling / sanity checks ONLY and must NEVER be used as
      model features (that would be label leakage).

A summary of noise magnitudes and oracle deviations per simulation and class is
written to step2_summary.csv so that we can decide how to model sensor noise.

Usage:
    python step2b_build_raw.py --root F:\\veremi --out F:\\veremi_parquet --workers 8
Re-runnable: simulations whose Parquet file already exists are skipped.
"""

import argparse
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

try:
    import orjson as _json
    loads = _json.loads
except ImportError:  # fall back to the standard library
    import json as _json
    loads = _json.loads

NAME_RE = re.compile(r"^traceJSON-(\d+)-(\d+)-A(\d+)-(\d+)-(\d+)\.json$")
VEC_FIELDS = ["pos", "spd", "acl", "hed"]


def read_lines(path):
    out = []
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(loads(line))
                except Exception:
                    pass
    return out


def xy(rec, key):
    v = rec.get(key)
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return float(v[0]), float(v[1])
    return math.nan, math.nan


def sim_meta(sim_dir, root):
    """Derive attack folder, period and simulation hour from the path."""
    rel = os.path.relpath(sim_dir, root).replace("\\", "/").split("/")
    attack_folder, period = rel[0], ""
    m = re.match(r"^(.*)_(\d{4})$", rel[0])
    if m:
        attack_folder, period = m.group(1), m.group(2)
    m2 = re.search(r"VeReMi_(\d+)_(\d+)", rel[-1])
    sim_hour = int(m2.group(1)) if m2 else -1
    return attack_folder, period, sim_hour


def process_sim(sim_dir, root, out_dir):
    attack_folder, period, sim_hour = sim_meta(sim_dir, root)
    out_path = os.path.join(out_dir, f"{attack_folder}_{period}_{sim_hour}.parquet")
    if os.path.exists(out_path):
        return out_path, None, "exists"

    files = os.listdir(sim_dir)
    logs = [f for f in files if NAME_RE.match(f)]
    gts = [f for f in files if f.startswith("traceGroundTruthJSON")]

    # Vehicle id -> attack type, from file names
    vehicle_attack = {}
    for f in logs:
        m = NAME_RE.match(f)
        vehicle_attack[int(m.group(1))] = int(m.group(3))

    # Ground truth indexed by messageID
    gt_rows = []
    for g in gts:
        for r in read_lines(os.path.join(sim_dir, g)):
            if "messageID" not in r:
                continue
            row = {"messageID": r["messageID"], "gt_sendTime": r.get("sendTime", math.nan)}
            for k in VEC_FIELDS:
                row[f"gt_{k}_x"], row[f"gt_{k}_y"] = xy(r, k)
            gt_rows.append(row)
    gt = pd.DataFrame(gt_rows).drop_duplicates("messageID").set_index("messageID")

    frames = []
    for f in logs:
        m = NAME_RE.match(f)
        receiver = int(m.group(1))
        recs = read_lines(os.path.join(sim_dir, f))
        own, msgs = [], []
        for r in recs:
            t = r.get("type")
            if t == 2:
                o = {"rx_time": r.get("rcvTime", math.nan)}
                for k in ["pos", "spd", "hed"]:
                    o[f"rx_{k}_x"], o[f"rx_{k}_y"] = xy(r, k)
                own.append(o)
            elif t == 3:
                row = {
                    "receiver": receiver,
                    "rcvTime": r.get("rcvTime", math.nan),
                    "sendTime": r.get("sendTime", math.nan),
                    "sender": r.get("sender", -1),
                    "senderPseudo": r.get("senderPseudo", -1),
                    "messageID": r.get("messageID", -1),
                }
                for k in VEC_FIELDS:
                    row[f"{k}_x"], row[f"{k}_y"] = xy(r, k)
                    row[f"{k}_noise_x"], row[f"{k}_noise_y"] = xy(r, f"{k}_noise")
                msgs.append(row)
        if not msgs:
            continue
        mdf = pd.DataFrame(msgs).sort_values("rcvTime")
        if own:
            odf = pd.DataFrame(own).sort_values("rx_time")
            mdf = pd.merge_asof(mdf, odf, left_on="rcvTime", right_on="rx_time",
                                direction="backward")
        frames.append(mdf)

    if not frames:
        return out_path, None, "no messages"
    df = pd.concat(frames, ignore_index=True)

    # Labels
    df["label_vehicle"] = df["sender"].map(vehicle_attack)
    n_unlabeled = int(df["label_vehicle"].isna().sum())
    df = df[df["label_vehicle"].notna()].copy()
    df["label_vehicle"] = df["label_vehicle"].astype("int16")
    df["receiver_attack"] = df["receiver"].map(vehicle_attack).fillna(-1).astype("int16")

    # Oracle deviations from ground truth (NOT features)
    df = df.join(gt, on="messageID")
    df["oracle_gt_found"] = df["gt_pos_x"].notna()
    for k in VEC_FIELDS:
        df[f"oracle_d{k}"] = np.hypot(df[f"{k}_x"] - df[f"gt_{k}_x"],
                                      df[f"{k}_y"] - df[f"gt_{k}_y"])
    df["oracle_dtime"] = (df["sendTime"] - df["gt_sendTime"]).abs()
    df = df.drop(columns=[c for c in df.columns if c.startswith("gt_")])

    df.insert(0, "attack_folder", attack_folder)
    df.insert(1, "period", period)
    df.insert(2, "sim_hour", sim_hour)
    df.to_parquet(out_path, index=False)

    # Per-class summary for the noise / label sanity report
    summ = []
    for lab, g in df.groupby("label_vehicle"):
        row = {"attack_folder": attack_folder, "period": period, "sim_hour": sim_hour,
               "label_vehicle": int(lab), "n_msgs": len(g),
               "n_senders": g["sender"].nunique(),
               "pseudonyms_per_sender_max": int(g.groupby("sender")["senderPseudo"].nunique().max()),
               "unlabeled_msgs_dropped": n_unlabeled,
               "gt_match_rate": round(float(g["oracle_gt_found"].mean()), 4)}
        for k in VEC_FIELDS:
            row[f"median_|{k}_noise|"] = float(np.nanmedian(np.hypot(g[f"{k}_noise_x"], g[f"{k}_noise_y"])))
            d = g[f"oracle_d{k}"]
            row[f"share_d{k}>1e-6"] = round(float((d > 1e-6).mean()), 4)
            row[f"median_d{k}"] = float(np.nanmedian(d))
        row["share_dtime>1e-6"] = round(float((g["oracle_dtime"] > 1e-6).mean()), 4)
        summ.append(row)
    return out_path, summ, "ok"


def find_sims(root):
    sims = []
    for dirpath, _, files in os.walk(root):
        if any(f.startswith("traceGroundTruthJSON") for f in files):
            sims.append(dirpath)
    return sorted(sims)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sims = find_sims(args.root)
    print(f"Simulations found: {len(sims)}")
    summary_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_sim, s, args.root, args.out): s for s in sims}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                path, summ, status = fut.result()
                print(f"[{status}] {os.path.basename(path)}")
                if summ:
                    summary_rows.extend(summ)
            except Exception as e:
                print(f"[ERROR] {s}: {type(e).__name__}: {e}")

    if summary_rows:
        new = pd.DataFrame(summary_rows)
        path = "step2_summary.csv"
        if os.path.exists(path):
            # read 'period' as text so "0709" is not turned into 709 (caused duplicate rows)
            old = pd.read_csv(path, dtype={"period": str})
            new = pd.concat([old, new]).drop_duplicates(
                ["attack_folder", "period", "sim_hour", "label_vehicle"], keep="last")
        new.sort_values(["attack_folder", "period", "sim_hour", "label_vehicle"]).to_csv(path, index=False)
        pd.set_option("display.width", 250)
        pd.set_option("display.max_columns", 40)
        print("\n", new.to_string(index=False))
        print("\nSaved: step2_summary.csv")


if __name__ == "__main__":
    main()
