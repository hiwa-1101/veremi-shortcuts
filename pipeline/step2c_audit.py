"""
Step 2c - Data audit: find dataset artifacts that would inflate results.

Findings from step2_summary.csv that this script verifies on the full Parquet data:
  A) Genuine traffic seems IDENTICAL across attack folders of the same period/hour
     (same message counts). If true, pooling all folders duplicates genuine data
     and a split per folder leaks the same trajectories into train and test.
  B) The same vehicles may be attackers in every attack folder.
  C) DelayedMessages attackers have *_noise == 0 (a trivial shortcut feature).
  D) Sybil attackers use up to 100 pseudonyms; the real 'sender' ID is not
     observable by a real receiver (only 'senderPseudo' is).
  E) Plain DoS messages are content-identical to ground truth; only the
     message RATE differs.

Usage:
    python step2c_audit.py --parquet F:\\veremi_parquet
Output:
    step2c_audit.json and a printed report
"""

import argparse
import glob
import itertools
import json
import os

import numpy as np
import pandas as pd

COLS = ["attack_folder", "period", "sim_hour", "receiver", "sender", "senderPseudo",
        "messageID", "rcvTime", "pos_x", "pos_y", "pos_noise_x", "pos_noise_y",
        "spd_noise_x", "spd_noise_y", "label_vehicle"]


def load(path):
    df = pd.read_parquet(path, columns=COLS)
    df["period"] = df["period"].astype(str).str.zfill(4)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--max-pairs", type=int, default=6,
                    help="Folder pairs compared per simulation hour (keeps it fast)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.parquet, "*.parquet")))
    print(f"Parquet files: {len(files)}")
    groups = {}
    for f in files:
        base = os.path.basename(f)[:-8]
        folder, period, hour = base.rsplit("_", 2)
        groups.setdefault((period.zfill(4), int(hour)), []).append((folder, f))

    report = {"A_genuine_overlap": [], "B_attacker_overlap": [],
              "C_zero_noise_share": {}, "D_pseudonyms": {}, "E_rate": {}}
    zero_noise, pseudo_stats, rate_stats = {}, {}, {}

    for (period, hour), items in sorted(groups.items()):
        cache = {}
        for folder, f in items:
            df = load(f)
            gen = df[df["label_vehicle"] == 0]
            att = df[df["label_vehicle"] != 0]
            cache[folder] = {
                "gen_senders": set(gen["sender"].unique()),
                "att_senders": set(att["sender"].unique()),
                # key that identifies the same physical reception, independent of messageID
                "gen_keys": set(zip(gen["receiver"], gen["sender"],
                                    gen["rcvTime"].round(3), gen["pos_x"].round(2))),
                "gen_ids": set(gen["messageID"].unique()),
            }
            for lab, g in df.groupby("label_vehicle"):
                key = f"{folder}|{lab}"
                zn = ((g["pos_noise_x"] == 0) & (g["pos_noise_y"] == 0)).mean()
                zero_noise[key] = round(float(zn), 4)
                pps = g.groupby("sender")["senderPseudo"].nunique()
                pseudo_stats[key] = {"median": float(pps.median()), "max": int(pps.max())}
                # messages per second per (receiver, pseudonym) link
                link = g.groupby(["receiver", "senderPseudo"])["rcvTime"].agg(["count", "min", "max"])
                dur = (link["max"] - link["min"]).replace(0, np.nan)
                rate = (link["count"] / dur).dropna()
                rate_stats[key] = {"median_msgs_per_s": round(float(rate.median()), 3)} if len(rate) else None
            del df

        folders = sorted(cache)
        for a, b in list(itertools.combinations(folders, 2))[: args.max_pairs]:
            A, B = cache[a], cache[b]
            jac_s = len(A["gen_senders"] & B["gen_senders"]) / max(1, len(A["gen_senders"] | B["gen_senders"]))
            key_ov = len(A["gen_keys"] & B["gen_keys"]) / max(1, len(A["gen_keys"]))
            id_ov = len(A["gen_ids"] & B["gen_ids"]) / max(1, len(A["gen_ids"]))
            jac_a = len(A["att_senders"] & B["att_senders"]) / max(1, len(A["att_senders"] | B["att_senders"]))
            report["A_genuine_overlap"].append({
                "period": period, "hour": hour, "pair": f"{a} vs {b}",
                "genuine_sender_jaccard": round(jac_s, 4),
                "genuine_reception_overlap": round(key_ov, 4),
                "genuine_messageID_overlap": round(id_ov, 4)})
            report["B_attacker_overlap"].append({
                "period": period, "hour": hour, "pair": f"{a} vs {b}",
                "attacker_sender_jaccard": round(jac_a, 4)})
        print(f"done: {period} {hour} ({len(folders)} folders)")

    report["C_zero_noise_share"] = zero_noise
    report["D_pseudonyms"] = pseudo_stats
    report["E_rate"] = rate_stats
    with open("step2c_audit.json", "w") as f:
        json.dump(report, f, indent=2)

    pd.set_option("display.width", 220)
    print("\n=== A) Genuine data overlap between attack folders ===")
    print(pd.DataFrame(report["A_genuine_overlap"]).to_string(index=False))
    print("\n=== B) Attacker vehicle overlap between attack folders ===")
    print(pd.DataFrame(report["B_attacker_overlap"]).to_string(index=False))
    print("\n=== C) Share of messages with pos_noise == 0 (folder|label) ===")
    for k, v in sorted(zero_noise.items()):
        if v > 0:
            print(f"  {k}: {v}")
    print("  (all other folder|label combinations: 0.0)")
    print("\n=== D) Pseudonyms per real sender (folder|label) ===")
    for k, v in sorted(pseudo_stats.items()):
        if v["max"] > 1:
            print(f"  {k}: {v}")
    print("\n=== E) Message rate per link (folder|label) ===")
    for k, v in sorted(rate_stats.items()):
        print(f"  {k}: {v}")
    print("\nSaved: step2c_audit.json")


if __name__ == "__main__":
    main()
