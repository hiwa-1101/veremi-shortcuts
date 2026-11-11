"""
Step 6a - Statistical tests between the proposed model and XGBoost (test split).
No training: uses the saved per-message predictions of the final runs.

  * McNemar test (with continuity correction) on every paired seed:
    messages that one model classifies correctly and the other does not.
  * Per-class McNemar on the messages of each true class.
  * Vehicle-cluster bootstrap (1,000 resamples) of the macro-F1 difference:
    resamples whole vehicles, so the confidence interval respects that
    messages of one vehicle are not independent.

Usage:
    python step6_stats.py --cache F:\\dl_cache --results F:\\results_final --out F:\\results_extra
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import chi2


def mcnemar(ca, cb):
    b = int((ca & ~cb).sum())      # A right, B wrong
    c = int((~ca & cb).sum())      # A wrong, B right
    stat = (abs(b - c) - 1) ** 2 / (b + c) if b + c > 0 else 0.0
    return {"a_only_correct": b, "b_only_correct": c, "chi2": float(stat),
            "p_value": float(chi2.sf(stat, 1))}


def macro_f1_from_cm(cm):
    """cm: [..., C, C] (true x pred); classes with no true messages are ignored."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    support = cm.sum(-1)
    predicted = cm.sum(-2)
    f1 = np.where(support + predicted > 0, 2 * tp / np.maximum(support + predicted, 1), 0.0)
    present = support > 0
    return (f1 * present).sum(-1) / present.sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--model", default="tmodel", help="deep model name (tmodel, tmix)")
    ap.add_argument("--results-a", default="", help="folder of the deep model's predictions if not --results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    info = json.load(open(os.path.join(args.cache, "info.json")))
    classes = info["classes"]
    C = len(classes)
    split = np.load(os.path.join(args.cache, "split.npy"))
    rows = np.where(split == 2)[0]
    y = np.load(os.path.join(args.cache, "y.npy"))[rows].astype(np.int64)
    sender = np.load(os.path.join(args.cache, "sender.npy"))[rows]
    period = pd.read_parquet(os.path.join(args.cache, "meta.parquet"), columns=["period"])["period"]
    period = period.astype(str).to_numpy()[rows]
    vehicle = pd.factorize(pd.MultiIndex.from_arrays([period, sender]))[0]
    V = vehicle.max() + 1
    print(f"test messages={len(y):,}  vehicles={V:,}")

    out = {"n_test": int(len(y)), "n_vehicles": int(V), "seeds": {}}
    for s in [int(x) for x in args.seeds.split(",")]:
        pa = np.load(os.path.join(args.results_a or args.results, f"{args.model}_multi_vehicle_seed{s}_pred.npy")).astype(np.int64)
        pb = np.load(os.path.join(args.results, f"xgb_multi_vehicle_seed{s}_pred.npy")).astype(np.int64)
        yx = np.load(os.path.join(args.results, f"ytest_multi_vehicle_seed{s}.npy")).astype(np.int64)
        if not (len(pa) == len(pb) == len(y)) or not np.array_equal(yx, y):
            raise SystemExit("Row order of XGBoost and the deep model differ - stop and report this.")
        ca, cb = pa == y, pb == y
        res = {"overall": mcnemar(ca, cb), "per_class": {}}
        for k in range(C):
            m = y == k
            if m.any():
                res["per_class"][str(classes[k])] = mcnemar(ca[m], cb[m])

        # vehicle-cluster bootstrap of the macro-F1 difference
        cma = np.zeros((V, C * C), np.float64)
        cmb = np.zeros((V, C * C), np.float64)
        np.add.at(cma, (vehicle, y * C + pa), 1)
        np.add.at(cmb, (vehicle, y * C + pb), 1)
        rng = np.random.default_rng(s)
        diffs, fa, fb = [], [], []
        for i in range(0, args.boot, 100):
            n = min(100, args.boot - i)
            W = np.stack([np.bincount(rng.integers(0, V, V), minlength=V) for _ in range(n)]).astype(np.float64)
            A = macro_f1_from_cm((W @ cma).reshape(n, C, C))
            B = macro_f1_from_cm((W @ cmb).reshape(n, C, C))
            diffs.append(A - B); fa.append(A); fb.append(B)
        diffs, fa, fb = map(np.concatenate, (diffs, fa, fb))
        res["bootstrap_vehicle"] = {
            "resamples": int(args.boot),
            "macro_f1_proposed": [float(np.mean(fa)), *map(float, np.percentile(fa, [2.5, 97.5]))],
            "macro_f1_xgb": [float(np.mean(fb)), *map(float, np.percentile(fb, [2.5, 97.5]))],
            "diff_mean": float(diffs.mean()),
            "diff_ci95": [float(x) for x in np.percentile(diffs, [2.5, 97.5])],
            "share_resamples_proposed_better": float((diffs > 0).mean())}
        out["seeds"][str(s)] = res
        o = res["overall"]
        print(f"seed {s}: McNemar chi2={o['chi2']:.1f} p={o['p_value']:.2e} "
              f"(proposed-only correct={o['a_only_correct']:,}, xgb-only correct={o['b_only_correct']:,}); "
              f"bootstrap diff={res['bootstrap_vehicle']['diff_mean'] * 100:.2f} "
              f"CI95={[round(x * 100, 2) for x in res['bootstrap_vehicle']['diff_ci95']]}")
    with open(os.path.join(args.out, ("stats_mcnemar_bootstrap.json" if args.model == "tmodel" else f"stats_mcnemar_bootstrap_{args.model}.json")), "w") as f:
        json.dump(out, f, indent=1)
    print("saved statistics json")


if __name__ == "__main__":
    main()
