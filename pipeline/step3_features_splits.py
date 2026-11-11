"""
Step 3 - Leakage-free splits + receiver-observable feature engineering.

Design decisions (all derived from the step 2c audit):
  * The SAME genuine vehicles and the SAME attacker vehicles appear in every
    attack folder of a given period/hour. Therefore the split unit is the
    physical vehicle: (period, sim_hour, sender). Each vehicle is assigned to
    exactly one of train/val/test, and that assignment is applied to ALL
    attack folders. No trajectory can appear on both sides of the split.
  * Only information a real receiver can observe is used as a feature:
      - NOT used: real 'sender' ID, messageID, *_noise fields (DelayedMessages
        has noise == 0 -> shortcut), absolute map positions (could identify
        vehicles), any 'oracle_' column.
      - Sequences/links are keyed by (receiver, senderPseudo), i.e. what the
        receiver actually sees.
  * Two labels:
      label_vehicle : attack type of the sender (from file name)
      label_msg     : label_vehicle if that message is actually manipulated
                      (deviates from ground truth), else 0 (genuine).
                      Plain DoS (13) is content-genuine by design; its excess
                      messages ARE the attack, so DoS messages keep label 13.

Feature groups (prefix -> group used later in the ablation study):
  k_  : single-message kinematics (speed, acceleration, heading/velocity agreement)
  r_  : receiver-relative (distance to receiver, relative speed)
  c_  : link consistency between consecutive messages of the same pseudonym
        (physics-informed residuals, timing, replay indicators)
  n_  : neighbourhood view of the receiver in 1-second buckets
        (message load, pseudonym count, co-located pseudonyms, new pseudonyms)
        + replay indicator (same position already reported by another pseudonym)
  m_  : map plausibility (density of genuine train-split traffic at the
        reported position; stand-in for a road map an on-board unit would have)

Usage:
    python step3_features_splits.py --raw F:\\veremi_parquet --out F:\\veremi_features --workers 6
Outputs:
    <out>/<file>.parquet  : features + labels + split per message
    splits_vehicles.csv   : the vehicle -> split assignment (for reproducibility)
"""

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

SEED = 42
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
CONTENT_EPS = 1e-6
DOS_CONTENT_GENUINE = {13}          # plain DoS: content not manipulated
CELL_M = 5.0                        # spatial cell for co-location (metres)
MAP_CELL_M = 5.0                    # cell size of the data-driven road map (metres)
REPLAY_WINDOW_S = 10.0              # look-back window for the replay indicator (seconds)


# --------------------------------------------------------------------------
# 1. Vehicle-level split shared by all attack folders
# --------------------------------------------------------------------------
def build_vehicle_splits(raw_files):
    parts = []
    for f in raw_files:
        d = pd.read_parquet(f, columns=["period", "sim_hour", "sender", "label_vehicle"])
        d["period"] = d["period"].astype(str).str.zfill(4)
        d["is_attacker"] = (d["label_vehicle"] != 0).astype(int)
        parts.append(d[["period", "sim_hour", "sender", "is_attacker"]].drop_duplicates())
    veh = pd.concat(parts).groupby(["period", "sim_hour", "sender"], as_index=False)["is_attacker"].max()

    rng = np.random.default_rng(SEED)
    veh["split"] = ""
    for (_, _, _), idx in veh.groupby(["period", "sim_hour", "is_attacker"]).groups.items():
        idx = np.array(list(idx))
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(SPLIT_FRACTIONS[0] * n))
        n_va = int(round(SPLIT_FRACTIONS[1] * n))
        veh.loc[idx[:n_tr], "split"] = "train"
        veh.loc[idx[n_tr:n_tr + n_va], "split"] = "val"
        veh.loc[idx[n_tr + n_va:], "split"] = "test"
    return veh


# --------------------------------------------------------------------------
# 2. Features for one simulation file
# --------------------------------------------------------------------------
def wrap_angle(a):
    return np.abs((a + np.pi) % (2 * np.pi) - np.pi)


def make_features(df, road_map):
    df = df.sort_values(["receiver", "senderPseudo", "rcvTime"]).reset_index(drop=True)

    # ---- k_: single-message kinematics -----------------------------------
    spd = np.hypot(df["spd_x"], df["spd_y"])
    acl = np.hypot(df["acl_x"], df["acl_y"])
    vel_ang = np.arctan2(df["spd_y"], df["spd_x"])
    hed_ang = np.arctan2(df["hed_y"], df["hed_x"])
    out = pd.DataFrame(index=df.index)
    out["k_spd"] = spd
    out["k_acl"] = acl
    out["k_hed_vel_diff"] = np.where(spd > 0.5, wrap_angle(vel_ang - hed_ang), 0.0)

    # ---- r_: receiver-relative -------------------------------------------
    out["r_range"] = np.hypot(df["pos_x"] - df["rx_pos_x"], df["pos_y"] - df["rx_pos_y"])
    out["r_rel_spd"] = np.hypot(df["spd_x"] - df["rx_spd_x"], df["spd_y"] - df["rx_spd_y"])

    # ---- c_: link consistency (same receiver, same pseudonym) ------------
    link = df.groupby(["receiver", "senderPseudo"], sort=False)
    prev = link[["rcvTime", "sendTime", "pos_x", "pos_y", "spd_x", "spd_y",
                 "acl_x", "acl_y", "hed_x", "hed_y"]].shift(1)
    first = prev["rcvTime"].isna()
    dt_r = df["rcvTime"] - prev["rcvTime"]
    dt_s = df["sendTime"] - prev["sendTime"]
    dt_pos = dt_s.where(dt_s > 1e-3)
    disp = np.hypot(df["pos_x"] - prev["pos_x"], df["pos_y"] - prev["pos_y"])
    prev_spd = np.hypot(prev["spd_x"], prev["spd_y"])
    pred_x = prev["pos_x"] + prev["spd_x"] * dt_pos + 0.5 * prev["acl_x"] * dt_pos ** 2
    pred_y = prev["pos_y"] + prev["spd_y"] * dt_pos + 0.5 * prev["acl_y"] * dt_pos ** 2

    out["c_first_on_link"] = first.astype(np.int8)
    out["c_link_msg_count_log"] = np.log1p(link.cumcount())
    out["c_dt_rcv"] = dt_r
    out["c_dt_send"] = dt_s
    out["c_send_not_increasing"] = (dt_s <= 0).astype(np.int8).where(~first, 0)
    out["c_age"] = df["rcvTime"] - df["sendTime"]
    out["c_disp"] = disp
    out["c_implied_spd"] = disp / dt_pos
    out["c_spd_resid"] = (out["c_implied_spd"] - 0.5 * (spd + prev_spd)).abs()
    out["c_kin_resid"] = np.hypot(df["pos_x"] - pred_x, df["pos_y"] - pred_y)
    out["c_dspd"] = (spd - prev_spd).abs()
    out["c_acl_resid"] = ((spd - prev_spd) / dt_pos - np.hypot(prev["acl_x"], prev["acl_y"])).abs()
    out["c_dhed"] = wrap_angle(hed_ang - np.arctan2(prev["hed_y"], prev["hed_x"]))
    out["c_same_content"] = ((disp < 1e-6) & ((spd - prev_spd).abs() < 1e-6)).astype(np.int8).where(~first, 0)

    # ---- n_: receiver neighbourhood in 1-second buckets ------------------
    bucket = np.floor(df["rcvTime"]).astype(np.int64)
    key = [df["receiver"], bucket]
    out["n_msgs_1s"] = df.groupby(key)["rcvTime"].transform("size")
    out["n_pseudos_1s"] = df.groupby(key)["senderPseudo"].transform("nunique")
    out["n_link_msgs_1s"] = df.groupby(key + [df["senderPseudo"]])["rcvTime"].transform("size")
    out["n_new_pseudos_1s"] = out["c_first_on_link"].groupby(key).transform("sum")
    cx = np.floor(df["pos_x"] / CELL_M).astype(np.int64)
    cy = np.floor(df["pos_y"] / CELL_M).astype(np.int64)
    out["n_colocated_pseudos_1s"] = df.groupby(key + [cx, cy])["senderPseudo"].transform("nunique") - 1

    # Replay indicator: this exact position (0.1 m) was already reported to this
    # receiver by a DIFFERENT pseudonym within the last REPLAY_WINDOW_S seconds.
    tmp = pd.DataFrame({"r": df["receiver"].to_numpy(),
                        "kx": np.round(df["pos_x"].to_numpy() * 10).astype(np.int64),
                        "ky": np.round(df["pos_y"].to_numpy() * 10).astype(np.int64),
                        "t": df["rcvTime"].to_numpy(),
                        "p": df["senderPseudo"].to_numpy()})
    tmp = tmp.sort_values(["r", "kx", "ky", "t"])
    g = tmp.groupby(["r", "kx", "ky"], sort=False)
    prev_t, prev_p = g["t"].shift(1), g["p"].shift(1)
    match = prev_t.notna() & (prev_p != tmp["p"]) & ((tmp["t"] - prev_t) <= REPLAY_WINDOW_S)
    out["n_replay_match_10s"] = match.reindex(df.index).astype(np.int8)

    # ---- m_: map plausibility -------------------------------------------
    # Density of genuine TRAIN-split traffic around the reported position
    # (a data-driven stand-in for a road map; built in build_road_map()).
    mx = np.floor(df["pos_x"] / MAP_CELL_M).astype(np.int64)
    my = np.floor(df["pos_y"] / MAP_CELL_M).astype(np.int64)
    lookup = pd.DataFrame({"mx": mx, "my": my}).merge(road_map, on=["mx", "my"], how="left")
    out["m_road_log_density"] = np.log1p(lookup["count3x3"].fillna(0).to_numpy())

    # ---- labels ----------------------------------------------------------
    manipulated = np.zeros(len(df), dtype=bool)
    for k in ["pos", "spd", "acl", "hed"]:
        col = f"oracle_d{k}"
        if col in df:
            manipulated |= (df[col].fillna(0).to_numpy() > CONTENT_EPS)
    lv = df["label_vehicle"].to_numpy()
    label_msg = np.where((lv != 0) & (manipulated | np.isin(lv, list(DOS_CONTENT_GENUINE))), lv, 0)

    ids = df[["attack_folder", "period", "sim_hour", "receiver", "senderPseudo", "sender", "rcvTime"]].copy()
    ids["period"] = ids["period"].astype(str).str.zfill(4)
    res = pd.concat([ids, out], axis=1)
    # aux_ columns: NOT model features. Only used by the deep model to compute
    # positions of neighbours RELATIVE to the target message.
    res["aux_pos_x"] = df["pos_x"].to_numpy(np.float32)
    res["aux_pos_y"] = df["pos_y"].to_numpy(np.float32)
    res["label_vehicle"] = lv.astype(np.int16)
    res["label_msg"] = label_msg.astype(np.int16)
    num = res.select_dtypes(include=["float64"]).columns
    res[num] = res[num].astype(np.float32).replace([np.inf, -np.inf], np.nan)
    return res


def build_road_map(raw_files, splits):
    """Count genuine messages from TRAIN-split vehicles per MAP_CELL_M cell and
    sum each cell with its 8 neighbours. Uses no validation/test vehicle, so the
    map cannot leak information about held-out trajectories."""
    train = splits[splits["split"] == "train"][["period", "sim_hour", "sender"]]
    counts = []
    for f in raw_files:
        d = pd.read_parquet(f, columns=["period", "sim_hour", "sender", "label_vehicle", "pos_x", "pos_y"])
        d["period"] = d["period"].astype(str).str.zfill(4)
        d = d[d["label_vehicle"] == 0].merge(train, on=["period", "sim_hour", "sender"])
        c = pd.DataFrame({"mx": np.floor(d["pos_x"] / MAP_CELL_M).astype(np.int64),
                          "my": np.floor(d["pos_y"] / MAP_CELL_M).astype(np.int64)})
        counts.append(c.value_counts().rename("n").reset_index())
    base = pd.concat(counts).groupby(["mx", "my"], as_index=False)["n"].sum()
    shifted = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            s = base.copy()
            s["mx"] += dx
            s["my"] += dy
            shifted.append(s)
    return pd.concat(shifted).groupby(["mx", "my"], as_index=False)["n"].sum().rename(columns={"n": "count3x3"})


def process_file(path, out_dir, splits, road_map):
    out_path = os.path.join(out_dir, os.path.basename(path))
    if os.path.exists(out_path):
        return out_path, "exists", None
    df = pd.read_parquet(path)
    df["period"] = df["period"].astype(str).str.zfill(4)
    feats = make_features(df, road_map)
    feats = feats.merge(splits, on=["period", "sim_hour", "sender"], how="left")
    feats = feats.drop(columns=["is_attacker"])
    feats.to_parquet(out_path, index=False)
    counts = feats.groupby(["split", "label_msg"]).size().rename("n").reset_index()
    counts.insert(0, "file", os.path.basename(path))
    return out_path, "ok", counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    raw_files = sorted(glob.glob(os.path.join(args.raw, "*.parquet")))
    print(f"Raw files: {len(raw_files)}")

    if os.path.exists("splits_vehicles.csv"):
        splits = pd.read_csv("splits_vehicles.csv", dtype={"period": str})
        print("Using existing splits_vehicles.csv (kept fixed for reproducibility)")
    else:
        splits = build_vehicle_splits(raw_files)
        splits.to_csv("splits_vehicles.csv", index=False)
        print("Created splits_vehicles.csv")
    print(splits.groupby(["split", "is_attacker"]).size().unstack(fill_value=0))

    if os.path.exists("road_map.parquet"):
        road_map = pd.read_parquet("road_map.parquet")
        print("Using existing road_map.parquet")
    else:
        road_map = build_road_map(raw_files, splits)
        road_map.to_parquet("road_map.parquet", index=False)
        print(f"Created road_map.parquet ({len(road_map):,} cells, train-split genuine traffic only)")

    all_counts = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_file, f, args.out, splits, road_map): f for f in raw_files}
        for fut in as_completed(futs):
            try:
                path, status, counts = fut.result()
                print(f"[{status}] {os.path.basename(path)}")
                if counts is not None:
                    all_counts.append(counts)
            except Exception as e:
                print(f"[ERROR] {futs[fut]}: {type(e).__name__}: {e}")

    if all_counts:
        c = pd.concat(all_counts)
        table = c.groupby(["label_msg", "split"])["n"].sum().unstack(fill_value=0)
        table.to_csv("step3_class_split_counts.csv")
        print("\nMessages per class (label_msg) and split:\n", table)
        print("\nSaved: step3_class_split_counts.csv")


if __name__ == "__main__":
    main()
