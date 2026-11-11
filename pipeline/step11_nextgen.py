"""
Step 11 - Second dataset: VeReMi NextGen (InTAS map, 15 attacks, predefined geographic splits).

Why: every other result comes from one simulator setup and one 1.61 km2 LuST map. NextGen
uses a different city (InTAS, Ingolstadt), a different attack set, and train / validation /
test splits that the dataset authors made in separate geographic areas. We re-run the SAME
pipeline on it (same 26 feature definitions, same model, same mixed-scene training, same
XGBoost) and compare - nothing is tuned for NextGen.

Data facts used (VeReMi NextGen documentation + our inspection, inspect_report_B):
  * one outer zip per scenario x attack, e.g. InTAS_urban_2_dataReplay.zip, with inner zips
    Train/, Validation/, Test/; each inner zip holds one JSON array per RECEIVING vehicle
    (veh_<id>.json);
  * every message has rcvTime/sendTime (ns), sender_id (true id - NOT a feature),
    sender_alias (pseudonym - what the receiver sees), messageID, attacker (0/1, message
    level: 1 = significant deviation), a 'receiver' block (own state) and a 'sender' block;
  * positions are strings "x,y,z"; speed, acceleration and heading are scalars
    (heading in degrees, convention not documented -> calibrated from genuine data below);
  * values are numbers in attack zips and strings in the benign zip -> float() handles both.
  * attack-free zips (InTAS_urban_2.zip) replay the same traffic as the attack zips and add
    nothing, so they are skipped; ground-truth JSON files are not needed (labels are given).

Labels: class 0 = genuine (attacker == 0); class k = attack folder k (attacker == 1), k = 1..15
in alphabetical order of the folder names (written to nextgen_classes.json).
Map plausibility: the test area is geographically separate from the training area, so a map
built from training traffic is empty there. m_road_log_density is therefore set to 0 for
EVERY model on NextGen (documented as a limitation; a real road map would restore it).

Stages (each skips finished work, safe to re-run after a power cut):
  raw      : zips -> one parquet per scenario x attack x split            (--stage raw)
  calib    : heading convention + time unit from genuine Train messages   (--stage calib)
  features : step3.make_features on every raw file                        (--stage features)
Usage:
    python step11_nextgen.py --nextgen F:\\hiwa\\nextgen_zips --work F:\\hiwa\\nextgen --stage all --workers 4
"""

import argparse
import glob
import io
import json
import math
import os
import re
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import step3_features_splits as s3

try:
    import orjson
    _loads = orjson.loads
except ImportError:
    _loads = json.loads

ZIP_RE = re.compile(r"^InTAS_(urban|highway)_(\d+)_(.+)\.zip$", re.I)
SPLITS = {"train": 0, "validation": 1, "test": 2}
SPLIT_NAME = {0: "train", 1: "val", 2: "test"}


def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def xy(s):
    if isinstance(s, str):
        a = s.split(",")
        return (f(a[0]), f(a[1])) if len(a) >= 2 else (math.nan, math.nan)
    if isinstance(s, (list, tuple)) and len(s) >= 2:
        return f(s[0]), f(s[1])
    return math.nan, math.nan


def vid(s):
    m = re.search(r"(\d+)", str(s))
    return int(m.group(1)) if m else -1


COLS = ["receiver", "rcvTime", "sendTime", "sender", "senderPseudo", "messageID", "attacker",
        "pos_x", "pos_y", "spd_s", "acl_s", "hed_deg",
        "rx_pos_x", "rx_pos_y", "rx_spd_s", "rx_acl_s", "rx_hed_deg"]


def parse_receiver_file(raw, receiver):
    rows = {c: [] for c in COLS}
    try:
        msgs = _loads(raw)
    except Exception:
        return rows, 1
    if isinstance(msgs, dict):
        msgs = [msgs]
    for m in msgs:
        s, r = m.get("sender") or {}, m.get("receiver") or {}
        px, py = xy(s.get("pos"))
        rx, ry = xy(r.get("pos"))
        rows["receiver"].append(receiver)
        rows["rcvTime"].append(f(m.get("rcvTime")))
        rows["sendTime"].append(f(m.get("sendTime")))
        rows["sender"].append(vid(m.get("sender_id")))
        rows["senderPseudo"].append(int(f(m.get("sender_alias", -1))) if m.get("sender_alias") is not None else -1)
        rows["messageID"].append(int(f(m.get("messageID", -1))))
        rows["attacker"].append(int(f(m.get("attacker", 0)) or 0))
        rows["pos_x"].append(px); rows["pos_y"].append(py)
        rows["spd_s"].append(f(s.get("spd"))); rows["acl_s"].append(f(s.get("acl"))); rows["hed_deg"].append(f(s.get("hed")))
        rows["rx_pos_x"].append(rx); rows["rx_pos_y"].append(ry)
        rows["rx_spd_s"].append(f(r.get("spd"))); rows["rx_acl_s"].append(f(r.get("acl"))); rows["rx_hed_deg"].append(f(r.get("hed")))
    return rows, 0


def raw_one(zip_path, raw_dir):
    """One outer zip (scenario x attack) -> up to three parquet files (one per split)."""
    m = ZIP_RE.match(os.path.basename(zip_path))
    scen = f"{m.group(1).lower()}_{m.group(2)}"
    attack = m.group(3)
    done = []
    outer = zipfile.ZipFile(zip_path)
    for inner in outer.namelist():
        if not inner.lower().endswith(".zip"):
            continue
        parts = [p.lower() for p in inner.replace("\\", "/").split("/")]
        sp = next((SPLITS[p] for p in parts if p in SPLITS), None)
        if sp is None:
            continue
        out = os.path.join(raw_dir, f"{scen}__{attack}__{SPLIT_NAME[sp]}.parquet")
        if os.path.exists(out):
            done.append((out, "exists", 0)); continue
        t0 = time.time()
        izf = zipfile.ZipFile(io.BytesIO(outer.read(inner)))
        frames, bad = [], 0
        for name in izf.namelist():
            base = os.path.basename(name)
            if not base.lower().endswith(".json"):
                continue
            rows, b = parse_receiver_file(izf.read(name), vid(base))
            bad += b
            if rows["receiver"]:
                frames.append(pd.DataFrame(rows))
        if not frames:
            done.append((out, "empty", bad)); continue
        df = pd.concat(frames, ignore_index=True)
        df.insert(0, "attack_folder", attack)
        df.insert(1, "period", scen)
        df.insert(2, "sim_hour", np.int32(sp))
        tmp = out + ".tmp"
        df.to_parquet(tmp, index=False)
        os.replace(tmp, out)
        done.append((out, f"ok rows={len(df):,} attackers={int(df['attacker'].sum()):,} [{time.time() - t0:.0f}s]", bad))
    return done


# ------------------------------------------------------------------ calibration
def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def calibrate(raw_dir):
    files = sorted(glob.glob(os.path.join(raw_dir, "*__train.parquet")))[:6]
    d = pd.concat([pd.read_parquet(x, columns=["receiver", "senderPseudo", "rcvTime", "sendTime", "pos_x", "pos_y",
                                                 "spd_s", "hed_deg", "attacker"]) for x in files], ignore_index=True)
    d = d[d["attacker"] == 0].sort_values(["receiver", "senderPseudo", "rcvTime"])
    g = d.groupby(["receiver", "senderPseudo"], sort=False)
    dt_raw = (d["sendTime"] - g["sendTime"].shift(1)).to_numpy()
    med = np.nanmedian(dt_raw[dt_raw > 0])
    scale = 1e9 if med > 1e6 else (1e3 if med > 1e2 else 1.0)       # beacon interval ~0.1-1 s
    dx = (d["pos_x"] - g["pos_x"].shift(1)).to_numpy()
    dy = (d["pos_y"] - g["pos_y"].shift(1)).to_numpy()
    dt = dt_raw / scale
    ok = (np.hypot(dx, dy) > 2.0) & (dt > 0) & (dt < 1.5) & (d["spd_s"].to_numpy() > 2.0)
    phi = np.arctan2(dy[ok], dx[ok])
    h = np.radians(d["hed_deg"].to_numpy()[ok])
    best = None
    for sign in (1, -1):
        for k in range(4):
            err = np.median(np.abs(wrap(phi - (sign * h + k * np.pi / 2))))
            if best is None or err < best[0]:
                best = (float(err), sign, k)
    # speed unit check: implied speed vs reported speed (genuine messages)
    implied = np.hypot(dx[ok], dy[ok]) / dt[ok]
    ratio = float(np.median(implied / d["spd_s"].to_numpy()[ok]))
    cal = {"time_scale": scale, "heading_sign": best[1], "heading_offset_deg": 90 * best[2],
           "median_heading_error_deg": math.degrees(best[0]), "median_implied_over_reported_speed": ratio,
           "n_pairs": int(ok.sum())}
    return cal


# ------------------------------------------------------------------ features
EMPTY_MAP = pd.DataFrame({"mx": pd.Series([], dtype="int64"), "my": pd.Series([], dtype="int64"),
                          "count3x3": pd.Series([], dtype="float64")})


def to_vectors(df, cal):
    df = df.copy()
    ts = cal["time_scale"]
    df["rcvTime"] = df["rcvTime"] / ts
    df["sendTime"] = df["sendTime"] / ts
    for pre, h, sp, ac in (("", "hed_deg", "spd_s", "acl_s"), ("rx_", "rx_hed_deg", "rx_spd_s", "rx_acl_s")):
        th = cal["heading_sign"] * np.radians(df[h].to_numpy()) + np.radians(cal["heading_offset_deg"])
        c, s = np.cos(th), np.sin(th)
        df[f"{pre}hed_x"], df[f"{pre}hed_y"] = c, s
        df[f"{pre}spd_x"], df[f"{pre}spd_y"] = df[sp].to_numpy() * c, df[sp].to_numpy() * s
        df[f"{pre}acl_x"], df[f"{pre}acl_y"] = df[ac].to_numpy() * c, df[ac].to_numpy() * s
    return df


def feat_one(raw_path, feat_dir, cal, class_of):
    out = os.path.join(feat_dir, os.path.basename(raw_path))
    if os.path.exists(out):
        return out, "exists"
    t0 = time.time()
    df = pd.read_parquet(raw_path)
    df = to_vectors(df, cal)
    k = class_of[df["attack_folder"].iloc[0]]
    df["label_vehicle"] = np.where(df["attacker"].to_numpy() == 1, k, 0).astype(np.int16)
    feats = s3.make_features(df, EMPTY_MAP)
    feats["label_msg"] = feats["label_vehicle"]          # NextGen labels are message-level already
    feats["split"] = feats["sim_hour"].map(SPLIT_NAME)
    tmp = out + ".tmp"
    feats.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    return out, f"ok rows={len(feats):,} [{time.time() - t0:.0f}s]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nextgen", required=True, help="folder with the downloaded NextGen zips")
    ap.add_argument("--work", required=True)
    ap.add_argument("--stage", choices=["raw", "calib", "features", "all"], default="all")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    raw_dir, feat_dir = os.path.join(args.work, "raw"), os.path.join(args.work, "features")
    os.makedirs(raw_dir, exist_ok=True); os.makedirs(feat_dir, exist_ok=True)
    log = open(os.path.join(args.work, "nextgen_log.txt"), "a", encoding="utf-8")

    def p(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()

    zips = sorted(z for z in glob.glob(os.path.join(args.nextgen, "*.zip")) if ZIP_RE.match(os.path.basename(z)))
    attacks = sorted({ZIP_RE.match(os.path.basename(z)).group(3) for z in zips}, key=str.lower)
    class_of = {a: i + 1 for i, a in enumerate(attacks)}
    json.dump({"0": "Genuine", **{str(v): k for k, v in class_of.items()}},
              open(os.path.join(args.work, "nextgen_classes.json"), "w"), indent=1)
    p(f"NextGen attack zips: {len(zips)}  attacks ({len(attacks)}): {attacks}")
    if not zips:
        raise SystemExit("No InTAS_<scenario>_<attack>.zip files found in " + args.nextgen)

    if args.stage in ("raw", "all"):
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(raw_one, z, raw_dir): z for z in zips}
            for fu in as_completed(futs):
                try:
                    for out, st, bad in fu.result():
                        p(f"[raw] {os.path.basename(out)}: {st}" + (f" unreadable_files={bad}" if bad else ""))
                except Exception as e:
                    p(f"[raw ERROR] {os.path.basename(futs[fu])}: {type(e).__name__}: {e}")

    cal_path = os.path.join(args.work, "nextgen_calibration.json")
    if args.stage in ("calib", "features", "all"):
        if not os.path.exists(cal_path):
            cal = calibrate(raw_dir)
            json.dump(cal, open(cal_path, "w"), indent=1)
        cal = json.load(open(cal_path))
        p("calibration:", cal)
        if cal["median_heading_error_deg"] > 15:
            p("STOP: heading convention unclear (median error > 15 deg) - send nextgen_log.txt before training")
            raise SystemExit(2)

    if args.stage in ("features", "all"):
        raws = sorted(glob.glob(os.path.join(raw_dir, "*.parquet")))
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(feat_one, r, feat_dir, cal, class_of): r for r in raws}
            for fu in as_completed(futs):
                try:
                    out, st = fu.result()
                    p(f"[features] {os.path.basename(out)}: {st}")
                except Exception as e:
                    p(f"[features ERROR] {os.path.basename(futs[fu])}: {type(e).__name__}: {e}")
        # class x split counts
        cnt = []
        for x in sorted(glob.glob(os.path.join(feat_dir, "*.parquet"))):
            d = pd.read_parquet(x, columns=["period", "split", "label_msg"])
            cnt.append(d.value_counts().rename("n").reset_index())
        c = pd.concat(cnt).groupby(["label_msg", "split"])["n"].sum().unstack(fill_value=0)
        p("\nMessages per class and split:\n" + c.to_string())
        sc = pd.concat(cnt).groupby(["period", "split"])["n"].sum().unstack(fill_value=0)
        p("\nMessages per scenario and split:\n" + sc.to_string())


if __name__ == "__main__":
    main()
