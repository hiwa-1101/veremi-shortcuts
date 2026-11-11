"""
Step 8 - External test on MixAll_0024 (VeReMi Extension, 24 h, all attacks mixed).

The trained model is applied UNCHANGED (no retraining, no tuning) to a separate
24-hour simulation of the same map in which all 19 attacks occur together.

Leakage control: LuST traffic demand is the same every day, so the MixAll hours that
coincide with the training periods (07-09 h and 14-16 h) may contain the same trips.
They are reported separately; the main external result uses the other hours only.

Stages (each one skips work that is already done):
  1. extract each hourly inner zip (':' in names replaced for Windows), salvage what
     can be read from a corrupt member, build the raw table with step2b.process_sim,
     then delete the extracted JSON to save disk space;
  2. features with step3.make_features, using the TRAINING road map (no MixAll labels);
  3. normalise with TRAINING statistics, run the 5 final seeds, report metrics.

Usage:
    python step8_mixall.py --zip F:\\mixall_zip\\MixAll_0024.zip --work F:\\mixall ^
        --cache F:\\dl_cache --results F:\\results_final --roadmap F:\\vanet_project\\road_map.parquet ^
        --out F:\\results_extra
"""

import argparse
import glob
import io
import json
import os
import re
import shutil
import time
import zipfile
import zlib

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support

import step2b_build_raw as s2
import step3_features_splits as s3
from step5b_train_model import GpuData, LinkNeighbourNet

TRAIN_WINDOWS = [(25200, 32400), (50400, 57600)]      # 07-09 h and 14-16 h (training periods)
LOG = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def read_member_salvage(zf, name):
    """Read a zip member. If its CRC is wrong, read it again with the CRC check
    disabled: the inner archive's own per-file CRCs then show which trace files
    are damaged, and only those are dropped."""
    try:
        return zf.read(name), True
    except (zipfile.BadZipFile, zlib.error) as e:
        p(f"   ! {name}: {e!r} - re-reading without the outer CRC check")
        chunks = []
        try:
            with zf.open(name) as f:
                f._expected_crc = None      # CPython ZipExtFile: skip the final CRC comparison
                while True:
                    b = f.read(1 << 22)
                    if not b:
                        break
                    chunks.append(b)
        except Exception as e2:
            p(f"   ! {name}: stream damaged ({e2!r}); keeping {sum(map(len, chunks)) / 1e6:.0f} MB")
        return b"".join(chunks), False


# ------------------------------------------------------------------ stage 1
def stage_raw(zip_path, work, raw_dir):
    os.makedirs(raw_dir, exist_ok=True)
    outer = zipfile.ZipFile(zip_path)
    inner_names = sorted([n for n in outer.namelist() if n.endswith(".zip")],
                         key=lambda n: int(re.search(r"VeReMi_(\d+)_", n).group(1)))
    report = []
    for name in inner_names:
        start = int(re.search(r"VeReMi_(\d+)_(\d+)", name).group(1))
        end = int(re.search(r"VeReMi_(\d+)_(\d+)", name).group(2))
        out_path = os.path.join(raw_dir, f"MixAll_0024_{start}.parquet")
        if os.path.exists(out_path):
            p(f"[skip] hour {start}: raw table exists")
            report.append({"start": start, "status": "exists"})
            continue
        t0 = time.time()
        data, intact = read_member_salvage(outer, name)
        sim_dir = os.path.join(work, "MixAll_0024", f"VeReMi_{start}_{end}")
        os.makedirs(sim_dir, exist_ok=True)
        n_ok = n_bad = 0
        gt_ok = False
        try:
            izf = zipfile.ZipFile(io.BytesIO(data))
            members = izf.infolist()
        except zipfile.BadZipFile:
            members, izf = [], None
            p(f"   ! hour {start}: inner archive unreadable")
        for m in members:
            if m.is_dir():
                continue
            fname = os.path.basename(m.filename.replace(":", "-"))
            try:
                with open(os.path.join(sim_dir, fname), "wb") as f:
                    f.write(izf.read(m))
                n_ok += 1
                gt_ok |= fname.startswith("traceGroundTruthJSON")
            except Exception:
                n_bad += 1
                try:
                    os.remove(os.path.join(sim_dir, fname))
                except OSError:
                    pass
        del data
        if not gt_ok:
            p(f"   ! hour {start}: ground truth missing or corrupt -> hour skipped (labels would be wrong)")
            shutil.rmtree(sim_dir, ignore_errors=True)
            report.append({"start": start, "status": "skipped_no_gt", "files_ok": n_ok, "files_bad": n_bad})
            continue
        path, summ, status = s2.process_sim(sim_dir, work, raw_dir)
        produced = os.path.join(raw_dir, os.path.basename(path))
        if os.path.exists(produced) and produced != out_path:
            os.replace(produced, out_path)
        shutil.rmtree(sim_dir, ignore_errors=True)
        p(f"hour {start}: {status}, trace files ok={n_ok} corrupt={n_bad} archive_intact={intact} "
          f"[{time.time() - t0:.0f}s]")
        report.append({"start": start, "status": status, "files_ok": n_ok, "files_bad": n_bad,
                       "archive_intact": intact})
    return report


# ------------------------------------------------------------------ stage 2
def stage_features(raw_dir, feat_dir, roadmap):
    os.makedirs(feat_dir, exist_ok=True)
    road_map = pd.read_parquet(roadmap)
    for f in sorted(glob.glob(os.path.join(raw_dir, "*.parquet"))):
        out = os.path.join(feat_dir, os.path.basename(f))
        if os.path.exists(out):
            continue
        t0 = time.time()
        df = pd.read_parquet(f)
        df["period"] = df["period"].astype(str).str.zfill(4)
        feats = s3.make_features(df, road_map)
        feats["split"] = "test"
        feats.to_parquet(out, index=False)
        p(f"features {os.path.basename(f)} rows={len(feats):,} [{time.time() - t0:.0f}s]")


# ------------------------------------------------------------------ stage 3
def make_data(df, info, device):
    feats, mean, std = info["features"], np.array(info["mean"]), np.array(info["std"])
    X = df[feats].to_numpy(np.float32)
    X = np.sign(X) * np.log1p(np.abs(X))
    X = (X - mean) / std
    X = np.clip(np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0), -10, 10).astype(np.float16)
    link_id = pd.factorize(pd.MultiIndex.from_arrays([df["receiver"], df["senderPseudo"]]))[0].astype(np.int64)
    rx_id = df["receiver"].to_numpy(np.int64)
    t = df["rcvTime"].to_numpy(np.float64)
    n = len(df)
    order_link = np.lexsort((t, link_id)).astype(np.int64)
    order_rx = np.lexsort((t, rx_id)).astype(np.int64)
    rank_link = np.empty(n, np.int64); rank_link[order_link] = np.arange(n)
    rank_rx = np.empty(n, np.int64); rank_rx[order_rx] = np.arange(n)
    classes = info["classes"]
    y = np.searchsorted(classes, df["label_msg"].to_numpy()).astype(np.int64)
    d = object.__new__(GpuData)
    T = lambda a, dt=None: (torch.from_numpy(a).to(dt) if dt is not None else torch.from_numpy(a)).to(device)
    d.X, d.y = T(X), T(y)
    d.link_id, d.rx_id, d.pseudo = T(link_id), T(rx_id), T(df["senderPseudo"].to_numpy(np.int64))
    d.time = T(t)
    d.pos = T(df[["aux_pos_x", "aux_pos_y"]].to_numpy(np.float32))
    d.order_link, d.rank_link, d.order_rx, d.rank_rx = T(order_link), T(rank_link), T(order_rx), T(rank_rx)
    d.n, d.f = X.shape
    d.device = device
    return d, y


@torch.no_grad()
def predict(model, data, cfg, bs=16384):
    out = []
    for i in range(0, data.n, bs):
        rows = torch.arange(i, min(i + bs, data.n), device=data.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=data.device.type == "cuda"):
            logits = model(*data.batch(rows, cfg["history"], cfg["neighbors"], cfg["window"]))
        out.append(logits.float().argmax(-1).cpu())
    return torch.cat(out).numpy().astype(np.int8)


def metrics(y, pred, classes):
    present = np.unique(y)
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=list(range(len(classes))), zero_division=0)
    att, patt = y != 0, pred != 0
    tp = int((att & patt).sum()); fp = int((~att & patt).sum()); fn = int((att & ~patt).sum())
    return {"n": int(len(y)), "classes_present": [int(classes[c]) for c in present],
            "macro_f1": float(np.mean(f1[present])),
            "accuracy": float((y == pred).mean()),
            "binary_f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "genuine_fpr": float(patt[~att].mean()) if (~att).any() else None,
            "attack_miss_rate": float((~patt[att]).mean()) if att.any() else None,
            "per_class": {str(classes[c]): {"precision": float(pr[c]), "recall": float(rc[c]),
                                            "f1": float(f1[c]), "support": int(sup[c])} for c in present}}


def main_reference_vehicle_labels(main_features, results, classes, seeds):
    """Proposed model on the MAIN test set, scored with vehicle-level labels
    (the only labels MixAll supports), so the external result has a fair reference."""
    files = sorted(glob.glob(os.path.join(main_features, "*.parquet")))
    lv = []
    for f in files:
        d = pd.read_parquet(f, columns=["label_vehicle", "split"])
        lv.append(d.loc[d["split"] == "test", "label_vehicle"].to_numpy())
    y = np.searchsorted(classes, np.concatenate(lv)).astype(np.int64)
    ref = {}
    for s in seeds:
        pr_ = np.load(os.path.join(results, f"tmodel_multi_vehicle_seed{s}_pred.npy")).astype(np.int64)
        if len(pr_) != len(y):
            p("main reference: row count mismatch, skipped")
            return None
        ref[str(s)] = metrics(y, pr_, classes)
    return ref


def stage_eval(feat_dir, cache, results, out, seeds, label="msg", main_features=None):
    info = json.load(open(os.path.join(cache, "info.json")))
    classes = info["classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cols = info["features"] + ["aux_pos_x", "aux_pos_y", "label_msg", "label_vehicle", "receiver",
                               "senderPseudo", "rcvTime", "sim_hour"]
    models = {}
    for s in seeds:
        tag = f"tmodel_multi_vehicle_seed{s}"
        cfg = json.load(open(os.path.join(results, tag + ".json")))["config"]
        m = LinkNeighbourNet(len(info["features"]), len(classes), cfg["history"], cfg["neighbors"],
                             d=cfg["d"], layers=cfg["layers"])
        m.load_state_dict(torch.load(os.path.join(results, tag + "_weights.pt"), map_location="cpu"))
        models[s] = (m.to(device).eval(), cfg)
    ys, hours, preds = [], [], {s: [] for s in seeds}
    for f in sorted(glob.glob(os.path.join(feat_dir, "*.parquet"))):
        t0 = time.time()
        df = pd.read_parquet(f, columns=cols)
        if label == "vehicle":
            df["label_msg"] = df["label_vehicle"]
        if len(df) == 0:
            p(f"skip {os.path.basename(f)}: no labelled messages")
            continue
        data, y = make_data(df, info, device)
        ys.append(y.astype(np.int8)); hours.append(df["sim_hour"].to_numpy(np.int32))
        for s, (m, cfg) in models.items():
            preds[s].append(predict(m, data, cfg))
        del data
        if device.type == "cuda":
            torch.cuda.empty_cache()
        p(f"evaluated {os.path.basename(f)} rows={len(df):,} [{time.time() - t0:.0f}s]")
    y = np.concatenate(ys); h = np.concatenate(hours)
    overlap = np.zeros(len(h), bool)
    for a, b in TRAIN_WINDOWS:
        overlap |= (h >= a) & (h < b)
    res = {"n_messages": int(len(y)), "train_windows_s": TRAIN_WINDOWS, "seeds": {}}
    for s in seeds:
        pr_ = np.concatenate(preds[s])
        res["seeds"][str(s)] = {"external_non_overlapping_hours": metrics(y[~overlap], pr_[~overlap], classes),
                                "overlapping_hours": metrics(y[overlap], pr_[overlap], classes) if overlap.any() else None,
                                "all_hours": metrics(y, pr_, classes)}
    agg = {}
    for part in ["external_non_overlapping_hours", "overlapping_hours", "all_hours"]:
        vals = [res["seeds"][str(s)][part] for s in seeds if res["seeds"][str(s)][part]]
        if vals:
            agg[part] = {k: [float(np.mean([v[k] for v in vals])), float(np.std([v[k] for v in vals]))]
                         for k in ["macro_f1", "accuracy", "binary_f1", "genuine_fpr", "attack_miss_rate"]}
            agg[part]["n"] = vals[0]["n"]
    res["mean_std"] = agg
    res["labels"] = label
    if label == "vehicle" and main_features:
        ref = main_reference_vehicle_labels(main_features, results, classes, seeds)
        if ref:
            res["main_test_vehicle_labels"] = ref
            agg["MAIN test set (same labels)"] = {k: [float(np.mean([r[k] for r in ref.values()])),
                                                      float(np.std([r[k] for r in ref.values()]))]
                                                  for k in ["macro_f1", "accuracy", "binary_f1", "genuine_fpr", "attack_miss_rate"]}
            agg["MAIN test set (same labels)"]["n"] = next(iter(ref.values()))["n"]
    with open(os.path.join(out, f"mixall_external_{label}labels.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    p(f"\n=== MixAll_0024 external test, {label}-level labels (proposed model, unchanged; mean ± std over seeds) ===")
    for part, a in agg.items():
        p(f"{part:32s} n={a['n']:,}  macroF1={a['macro_f1'][0] * 100:.2f}±{a['macro_f1'][1] * 100:.2f}  "
          f"acc={a['accuracy'][0] * 100:.2f}  binF1={a['binary_f1'][0] * 100:.2f}  "
          f"genFPR={a['genuine_fpr'][0] * 100:.2f}%  missed attacks={a['attack_miss_rate'][0] * 100:.2f}%")
    pc = res["seeds"][str(seeds[0])]["external_non_overlapping_hours"]["per_class"]
    if "main_test_vehicle_labels" in res:
        mpc = res["main_test_vehicle_labels"][str(seeds[0])]["per_class"]
        p("per-class F1 / recall, seed %d: MixAll (non-overlapping hours) vs MAIN test, same labels:" % seeds[0])
        for c, v in pc.items():
            m = mpc.get(c, {})
            p(f"   class {c:>2}: MixAll F1={v['f1'] * 100:5.1f} R={v['recall'] * 100:5.1f} n={v['support']:>10,} | "
              f"main F1={m.get('f1', float('nan')) * 100:5.1f} R={m.get('recall', float('nan')) * 100:5.1f}")
    p(f"per-class (seed {seeds[0]}, non-overlapping hours): " +
      ", ".join(f"{c}: F1={v['f1'] * 100:.1f} R={v['recall'] * 100:.1f} n={v['support']:,}" for c, v in pc.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--roadmap", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--label", choices=["msg", "vehicle"], default="msg",
                    help="vehicle: label every message of an attacker vehicle with its attack type "
                         "(MixAll's ground-truth file logs the transmitted content, so message labels are impossible)")
    ap.add_argument("--main-features", default=None, help="main feature folder, for the same-label reference")
    args = ap.parse_args()
    raw_dir = os.path.join(args.work, "raw")
    feat_dir = os.path.join(args.work, "features")
    os.makedirs(args.out, exist_ok=True)
    p("=== Stage 1: extract + raw tables ===")
    rep = stage_raw(args.zip, args.work, raw_dir)
    p("=== Stage 2: features (training road map) ===")
    stage_features(raw_dir, feat_dir, args.roadmap)
    p("=== Stage 3: evaluation ===")
    stage_eval(feat_dir, args.cache, args.results, args.out, [int(s) for s in args.seeds.split(",")],
               label=args.label, main_features=args.main_features)
    p("\nHours processed:", json.dumps(rep))
    rep_name = f"mixall_report_{args.label}labels.txt"
    open(os.path.join(args.out, rep_name), "w", encoding="utf-8").write("\n".join(LOG))
    p(f"Saved {os.path.join(args.out, rep_name)}")


if __name__ == "__main__":
    main()
