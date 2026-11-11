"""
Step 8c - Why does the proposed model transfer worse to MixAll than XGBoost?

Hypothesis (scene-level shortcut): in VeReMi Extension every simulation contains ONE
attack type, so the neighbourhood of a receiver reveals which attack is "in the scene".
MixAll mixes all attacks in one simulation and breaks that cue. If the hypothesis holds,
models WITHOUT the neighbourhood branch should lose less on MixAll.

This script (no training) evaluates every saved variant on MixAll and on the main test
set with the SAME vehicle-level labels:
  full model (5 seeds), no-neighbourhood (3), no-history (3), Bi-LSTM (3).
It also reports binary ROC-AUC (attack vs genuine) and the miss rate at the false-alarm
rate of XGBoost (3.86%), so operating points can be compared fairly, and a covariate-shift
check of the 26 inputs on genuine messages.

Usage:
    python step8c_mixall_variants.py --features F:\\mixall\\features --main-features F:\\veremi_features_v3 ^
        --cache F:\\dl_cache --final F:\\results_final --extra F:\\results_extra --out F:\\results_extra
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from step5b_train_model import LinkNeighbourNet
from step8_mixall import TRAIN_WINDOWS, make_data, metrics

VARIANTS = [("tmodel", "final", [0, 1, 2, 3, 4]), ("tmodel_noNeigh", "final", [0, 1, 2]),
            ("tmodel_noHist", "final", [0, 1, 2]), ("bilstm", "extra", [0, 1, 2]),
            ("tmix", "mixed", [0, 1, 2]), ("tmix_nodens", "mixed", [0, 1, 2]),
            ("tmix_nodens_noNeigh", "mixed", [0, 1, 2]),          # final regime, M = 0
            ("tmix_nodens_q25", "mixed", [0]), ("tmix_nodens_q75", "mixed", [0])]   # q sensitivity
LOG = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


@torch.no_grad()
def predict_proba_g(model, data, cfg, g, bs=16384):
    """argmax class and P(genuine)."""
    preds, pg = [], []
    for i in range(0, data.n, bs):
        rows = torch.arange(i, min(i + bs, data.n), device=data.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=data.device.type == "cuda"):
            logits = model(*data.batch(rows, cfg["history"], cfg["neighbors"], cfg["window"])).float()
        pr = torch.softmax(logits, -1)
        preds.append(pr.argmax(-1).cpu()); pg.append(pr[:, g].cpu())
    return torch.cat(preds).numpy().astype(np.int8), torch.cat(pg).numpy().astype(np.float32)


def binary_extra(y, pg, fpr_target):
    att = y != 0
    score = 1.0 - pg
    auc = roc_auc_score(att, score)
    thr = np.quantile(score[~att], 1 - fpr_target)          # threshold giving the target FPR
    return {"binary_auc": float(auc), "fpr_target": fpr_target,
            "miss_rate_at_target_fpr": float((score[att] <= thr).mean())}


def load_model(info, path_json, path_w, device):
    cfg = json.load(open(path_json))["config"]
    m = LinkNeighbourNet(len(info["features"]), len(info["classes"]), cfg["history"], cfg["neighbors"],
                         d=cfg["d"], layers=cfg["layers"], encoder=cfg.get("encoder", "transformer"))
    m.load_state_dict(torch.load(path_w, map_location="cpu"))
    return m.to(device).eval(), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--main-features", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--final", required=True)
    ap.add_argument("--extra", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mixed", default="", help="results folder of step9 (mixed-scene training)")
    ap.add_argument("--xgb-fpr", type=float, default=0.0386)
    args = ap.parse_args()
    info = json.load(open(os.path.join(args.cache, "info.json")))
    classes = info["classes"]
    g = classes.index(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dirs = {"final": args.final, "extra": args.extra, "mixed": args.mixed or args.extra}

    # ---------------- MixAll ----------------
    cols = info["features"] + ["aux_pos_x", "aux_pos_y", "label_vehicle", "receiver", "senderPseudo",
                               "rcvTime", "sim_hour"]
    models = {}
    for name, d, seeds in VARIANTS:
        for s in seeds:
            j = os.path.join(dirs[d], f"{name}_multi_vehicle_seed{s}.json")
            w = os.path.join(dirs[d], f"{name}_multi_vehicle_seed{s}_weights.pt")
            if os.path.exists(j) and os.path.exists(w):
                models[(name, s)] = load_model(info, j, w, device)
            elif not name.startswith("tmix") or args.mixed:
                p(f"missing {name} seed {s} - skipped")
    ys, hs, preds, pgs = [], [], {k: [] for k in models}, {k: [] for k in models}
    Xg_mix = []
    for f in sorted(glob.glob(os.path.join(args.features, "*.parquet"))):
        t0 = time.time()
        df = pd.read_parquet(f, columns=cols)
        if len(df) == 0:
            continue
        df["label_msg"] = df["label_vehicle"]
        data, y = make_data(df, info, device)
        ys.append(y.astype(np.int8)); hs.append(df["sim_hour"].to_numpy(np.int32))
        gen = np.where(y == 0)[0]
        if len(gen):
            sel = np.random.default_rng(0).choice(gen, min(100_000, len(gen)), replace=False)
            Xg_mix.append(data.X[torch.from_numpy(sel).to(device)].float().cpu().numpy())
        for k, (m, cfg) in models.items():
            zf = [info["features"].index(f) for f in cfg.get("zero_features", "").split(",") if f]
            if zf:
                saved = data.X[:, zf].clone(); data.X[:, zf] = 0
            pr, pg = predict_proba_g(m, data, cfg, g)
            if zf:
                data.X[:, zf] = saved
            preds[k].append(pr); pgs[k].append(pg)
        del data
        if device.type == "cuda":
            torch.cuda.empty_cache()
        p(f"MixAll {os.path.basename(f)} rows={len(df):,} [{time.time() - t0:.0f}s]")
    y = np.concatenate(ys).astype(np.int64); h = np.concatenate(hs)
    ov = np.zeros(len(h), bool)
    for a, b in TRAIN_WINDOWS:
        ov |= (h >= a) & (h < b)
    keep = ~ov

    # ---------------- main test set, same vehicle labels ----------------
    files = sorted(glob.glob(os.path.join(args.main_features, "*.parquet")))
    lv = []
    for f in files:
        d = pd.read_parquet(f, columns=["label_vehicle", "split"])
        lv.append(d.loc[d["split"] == "test", "label_vehicle"].to_numpy())
    y_main = np.searchsorted(classes, np.concatenate(lv)).astype(np.int64)

    res = {"n_mixall_non_overlap": int(keep.sum()), "variants": {}}
    p("\n=== MixAll (non-overlapping hours) vs MAIN test, vehicle-level labels; mean over seeds ===")
    p(f"{'variant':16s} {'seeds':>5s} | {'MixAll macroF1':>14s} {'binF1':>6s} {'FPR%':>5s} {'miss%':>6s} {'AUC':>6s} "
      f"{'miss@XGB-FPR':>12s} | {'MAIN macroF1':>12s} {'drop':>6s}")
    for name, d, seeds in VARIANTS:
        ks = [k for k in models if k[0] == name]
        if not ks:
            continue
        mix, main_, bx = [], [], []
        for k in ks:
            pr = np.concatenate(preds[k]).astype(np.int64); pg = np.concatenate(pgs[k])
            mix.append(metrics(y[keep], pr[keep], classes))
            bx.append(binary_extra(y[keep], pg[keep], args.xgb_fpr))
            pm = os.path.join(dirs[d], f"{name}_multi_vehicle_seed{k[1]}_pred.npy")
            if os.path.exists(pm):
                pmain = np.load(pm).astype(np.int64)
                if len(pmain) == len(y_main):
                    main_.append(metrics(y_main, pmain, classes))
                else:
                    p(f"{name} seed {k[1]}: main prediction length differs - main score skipped")
        if name.startswith("tmix_nodens"):                     # per-seed lines (q sensitivity uses seed 0)
            for i, k in enumerate(ks):
                mm = main_[i]["macro_f1"] * 100 if i < len(main_) else float("nan")
                p(f"   per-seed {name} seed {k[1]}: MixAll macroF1={mix[i]['macro_f1'] * 100:.2f} "
                  f"binF1={mix[i]['binary_f1'] * 100:.2f} FPR={mix[i]['genuine_fpr'] * 100:.2f} | MAIN={mm:.2f}")
        mean = lambda L, key: float(np.mean([v[key] for v in L])) if L else float("nan")
        r = {"seeds": [k[1] for k in ks],
             "mixall": {kk: mean(mix, kk) for kk in ["macro_f1", "binary_f1", "genuine_fpr", "attack_miss_rate"]},
             "mixall_binary": {kk: mean(bx, kk) for kk in ["binary_auc", "miss_rate_at_target_fpr"]},
             "main": {kk: mean(main_, kk) for kk in ["macro_f1", "binary_f1", "genuine_fpr", "attack_miss_rate"]},
             "mixall_per_class_f1": {c: float(np.mean([v["per_class"][c]["f1"] for v in mix])) for c in mix[0]["per_class"]}}
        res["variants"][name] = r
        mx, mb, mn = r["mixall"], r["mixall_binary"], r["main"]
        p(f"{name:16s} {len(ks):5d} | {mx['macro_f1'] * 100:14.2f} {mx['binary_f1'] * 100:6.2f} "
          f"{mx['genuine_fpr'] * 100:5.2f} {mx['attack_miss_rate'] * 100:6.2f} {mb['binary_auc']:6.4f} "
          f"{mb['miss_rate_at_target_fpr'] * 100:12.2f} | {mn['macro_f1'] * 100:12.2f} "
          f"{(mn['macro_f1'] - mx['macro_f1']) * 100:6.2f}")
    p("XGBoost (from run_mixall_xgb): MixAll macroF1=78.41 binF1=94.29 FPR=3.86% miss=5.79% | MAIN 89.35 drop 10.94")
    p("\nPer-class F1 on MixAll (non-overlapping), variants side by side:")
    names = [n for n in res["variants"]]
    for c in res["variants"][names[0]]["mixall_per_class_f1"]:
        p(f"  class {c:>2}: " + "  ".join(f"{n}={res['variants'][n]['mixall_per_class_f1'][c] * 100:5.1f}" for n in names))

    # ---------------- covariate shift on genuine messages ----------------
    X = np.load(os.path.join(args.cache, "X.npy"), mmap_mode="r")
    yc = np.load(os.path.join(args.cache, "y.npy"))
    sp = np.load(os.path.join(args.cache, "split.npy"))
    gm = np.where((sp == 2) & (yc == g))[0]
    gm = np.sort(np.random.default_rng(0).choice(gm, min(1_000_000, len(gm)), replace=False))
    Xm = np.asarray(X[gm], dtype=np.float32)
    Xx = np.concatenate(Xg_mix)
    smd = (Xx.mean(0) - Xm.mean(0)) / np.sqrt(0.5 * (Xx.var(0) + Xm.var(0)) + 1e-9)
    order = np.argsort(-np.abs(smd))
    p("\nCovariate shift on GENUINE messages (standardised mean difference, MixAll - main test):")
    for i in order[:12]:
        p(f"  {info['features'][i]:26s} SMD={smd[i]:+.2f}  mean main={Xm[:, i].mean():+.2f} MixAll={Xx[:, i].mean():+.2f}")
    res["covariate_shift_smd"] = {info["features"][i]: float(smd[i]) for i in range(len(smd))}
    with open(os.path.join(args.out, "mixall_variants.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    open(os.path.join(args.out, "mixall_variants.txt"), "w", encoding="utf-8").write("\n".join(LOG))
    p(f"\nSaved {os.path.join(args.out, 'mixall_variants.txt')}")


if __name__ == "__main__":
    main()
