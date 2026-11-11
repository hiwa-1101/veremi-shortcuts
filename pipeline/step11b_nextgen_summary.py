"""
Step 11b - Results on VeReMi NextGen (test split = separate geographic area).

Compares, on the same NextGen test messages:
  ngmix    : proposed model (mixed-scene training, density + map inputs zeroed), 3 seeds
  ngbilstm : Bi-LSTM on the same link histories, 3 seeds
  xgb      : XGBoost on the same features (map input dropped), 3 seeds
Reports macro-F1, accuracy, binary F1, genuine FPR, missed attacks, per-class F1,
per-scenario scores, Welch t (seeds), Wilcoxon (classes) and a vehicle-cluster bootstrap
of the macro-F1 difference proposed - XGBoost (seed 0, 1,000 resamples).

Usage:
    python step11b_nextgen_summary.py --cache F:\\hiwa\\nextgen_cache --results F:\\hiwa\\nextgen_results --work F:\\hiwa\\nextgen
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

MODELS = [("Proposed (mixed-scene)", "ngmix_multi_vehicle_seed{}_pred.npy"),
          ("Bi-LSTM", "ngbilstm_multi_vehicle_seed{}_pred.npy"),
          ("XGBoost", "xgb_multi_vehicle_seed{}_pred.npy")]
L = []


def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)


def conf(y, pr, C):
    return np.bincount(y * C + pr, minlength=C * C).reshape(C, C)


def scores(cm, present):
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1); rec = tp / np.maximum(cm.sum(1), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
    att_tp = cm[1:, 1:].sum(); fp = cm[0, 1:].sum(); fn = cm[1:, 0].sum()
    return {"macro_f1": float(f1[present].mean()), "accuracy": float(tp.sum() / cm.sum()),
            "binary_f1": float(2 * att_tp / max(2 * att_tp + fp + fn, 1)),
            "genuine_fpr": float(fp / max(cm[0].sum(), 1)), "missed": float(fn / max(cm[1:].sum(), 1)),
            "f1": f1, "recall": rec, "precision": prec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--work", required=True)
    args = ap.parse_args()
    info = json.load(open(os.path.join(args.cache, "info.json")))
    classes = info["classes"]; C = len(classes)
    names = json.load(open(os.path.join(args.work, "nextgen_classes.json")))
    cname = [names.get(str(c), str(c)) for c in classes]
    split = np.load(os.path.join(args.cache, "split.npy"))
    te = np.where(split == 2)[0]
    y = np.load(os.path.join(args.cache, "y.npy")).astype(np.int64)[te]
    sender = np.load(os.path.join(args.cache, "sender.npy"))[te]
    period = pd.read_parquet(os.path.join(args.cache, "meta.parquet"), columns=["period"])["period"].astype(str).to_numpy()[te]
    present = np.unique(y)
    p(f"NextGen test: {len(y):,} messages, {len(np.unique(sender)):,} vehicles, classes present: {len(present)}/{C}")
    p("test messages per class: " + ", ".join(f"{cname[c]}={int((y == c).sum()):,}" for c in present))

    res = {}
    for label, pat in MODELS:
        preds = []
        for s in range(5):
            f = os.path.join(args.results, pat.format(s))
            if os.path.exists(f):
                pr = np.load(f).astype(np.int64)
                if len(pr) != len(y):
                    p(f"! {label} seed {s}: {len(pr)} predictions vs {len(y)} test rows - skipped"); continue
                preds.append(pr)
        if preds:
            res[label] = preds
    if "XGBoost" in res:                          # same row order check
        yx = os.path.join(args.results, "ytest_multi_vehicle_seed0.npy")
        if os.path.exists(yx):
            yxv = np.load(yx).astype(np.int64)
            p("row-order check XGBoost vs cache:", "OK" if np.array_equal(yxv, y) else "MISMATCH - do not use")

    p("\n=== Overall (mean ± std over seeds) ===")
    S = {}
    for label, preds in res.items():
        sc = [scores(conf(y, pr, C), present) for pr in preds]
        S[label] = sc
        m = lambda k: np.mean([v[k] for v in sc]) * 100
        sd = np.std([v["macro_f1"] for v in sc], ddof=1) * 100 if len(sc) > 1 else 0.0
        p(f"{label:24s} seeds={len(sc)} macroF1={m('macro_f1'):.2f}±{sd:.2f} acc={m('accuracy'):.2f} "
          f"binF1={m('binary_f1'):.2f} genFPR={m('genuine_fpr'):.2f}% missed={m('missed'):.2f}%")

    prop = "Proposed (mixed-scene)"
    for other in [k for k in S if k != prop]:
        if prop not in S:
            break
        a = np.array([v["macro_f1"] for v in S[prop]]) * 100
        b = np.array([v["macro_f1"] for v in S[other]]) * 100
        pa = np.mean([v["f1"] for v in S[prop]], 0)[present] * 100
        pb = np.mean([v["f1"] for v in S[other]], 0)[present] * 100
        t = stats.ttest_ind(a, b, equal_var=False) if len(a) > 1 and len(b) > 1 else None
        w = stats.wilcoxon(pa, pb)
        p(f"\nProposed vs {other}: {a.mean() - b.mean():+.2f} points"
          + (f"  Welch t={t.statistic:.1f} p={t.pvalue:.2g}" if t is not None else "")
          + f"  Wilcoxon over classes p={w.pvalue:.2g}  better on {(pa > pb).sum()}/{len(present)} classes")

    if prop in res and "XGBoost" in res:
        p("\n=== Vehicle-cluster bootstrap, proposed - XGBoost, seed 0 (1,000 resamples) ===")
        veh, inv = np.unique(sender, return_inverse=True)
        V = len(veh)
        ca = np.zeros((V, C * C), np.int64); cb = np.zeros((V, C * C), np.int64)
        np.add.at(ca, (inv, y * C + res[prop][0]), 1)
        np.add.at(cb, (inv, y * C + res["XGBoost"][0]), 1)
        rng = np.random.default_rng(0)
        diffs = []
        for _ in range(1000):
            wgt = np.bincount(rng.integers(0, V, V), minlength=V)
            ma = (wgt @ ca).reshape(C, C); mb = (wgt @ cb).reshape(C, C)
            pres = np.where(ma.sum(1) > 0)[0]
            diffs.append(scores(ma, pres)["macro_f1"] - scores(mb, pres)["macro_f1"])
        d = np.array(diffs) * 100
        p(f"diff={d.mean():.2f}  CI95=[{np.percentile(d, 2.5):.2f}, {np.percentile(d, 97.5):.2f}]  "
          f"P(proposed better)={(d > 0).mean():.3f}  vehicles={V:,}")

    p("\n=== Per-class F1 (mean over seeds) ===")
    p(f"{'class':28s}" + "".join(f"{k[:12]:>14s}" for k in S))
    for c in present:
        p(f"{cname[c]:28s}" + "".join(f"{np.mean([v['f1'][c] for v in S[k]]) * 100:14.1f}" for k in S))
    p(f"{'recall of proposed':28s}" + (" ".join(f"{cname[c]}={np.mean([v['recall'][c] for v in S[prop]]) * 100:.1f}"
                                               for c in present) if prop in S else ""))

    p("\n=== Per scenario (macro-F1 / binary F1, mean over seeds) ===")
    for sc_name in sorted(set(period)):
        m = period == sc_name
        pres = np.unique(y[m])
        row = []
        for label, preds in res.items():
            v = [scores(conf(y[m], pr[m], C), pres) for pr in preds]
            row.append(f"{label[:12]}: {np.mean([x['macro_f1'] for x in v]) * 100:.2f} / {np.mean([x['binary_f1'] for x in v]) * 100:.2f}")
        p(f"{sc_name:12s} n={int(m.sum()):,}  " + " | ".join(row))

    out = os.path.join(args.results, "nextgen_summary.txt")
    open(out, "w", encoding="utf-8").write("\n".join(L))
    print("\nSaved", out)


if __name__ == "__main__":
    main()
