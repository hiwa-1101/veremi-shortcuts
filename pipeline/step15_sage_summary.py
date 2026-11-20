"""
Step 15 - Does mixed-scene training generalise to a second context architecture?

Second context model ("lsage"): Bi-LSTM over the link history + GraphSAGE-style mean/max
aggregation of the receiver's 1-s neighbourhood (no attention). Three regimes, 3 seeds each:
  lsage        original training, 26 inputs
  lsage_mix    mixed-scene training (q = 0.5), 26 inputs
  lsage_mixnd  mixed-scene training + the two scene-density inputs set to zero
This script summarises the main TEST split (message-level labels) from the step5b/step9 JSON
files, next to the reference models. MixAll results come from step8c (mixall_variants.txt).

Usage:
    python step15_sage_summary.py --sage F:\\hiwa\\results_sage --final F:\\hiwa\\results_final ^
        --mixed F:\\hiwa\\results_mixed --extra F:\\hiwa\\results_extra
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import stats


def load(folder, name):
    rs = []
    for f in sorted(glob.glob(os.path.join(folder, f"{name}_multi_vehicle_seed*.json"))):
        if "_val_" in f:
            continue
        rs.append(json.load(open(f)))
    return rs


def summ(rs):
    f1 = np.array([r["macro_f1"] for r in rs]) * 100
    fpr = [1 - np.array(r["confusion_matrix"])[0, 0] / np.array(r["confusion_matrix"])[0].sum() for r in rs]
    b = [r.get("binary_from_multi", {}).get("f1", np.nan) for r in rs]
    return f1, np.mean(fpr) * 100, np.nanmean(b) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sage", required=True)
    ap.add_argument("--final", required=True)
    ap.add_argument("--mixed", required=True)
    ap.add_argument("--extra", required=True)
    args = ap.parse_args()
    rows = [("lsage", args.sage, "Bi-LSTM + SAGE, original training (26 inputs)"),
            ("lsage_mix", args.sage, "Bi-LSTM + SAGE, mixed-scene training (26 inputs)"),
            ("lsage_mixnd", args.sage, "Bi-LSTM + SAGE, mixed-scene, no density inputs"),
            ("bilstm", args.extra, "reference: Bi-LSTM, no neighbourhood"),
            ("tmodel", args.final, "reference: Transformer, original training"),
            ("tmix", args.mixed, "reference: Transformer, mixed-scene (26 inputs)"),
            ("tmix_nodens", args.mixed, "reference: Transformer, mixed-scene, no density (proposed)")]
    L = ["=== MAIN test split (message-level labels), mean ± std over seeds ==="]
    res = {}
    for name, folder, desc in rows:
        rs = load(folder, name)
        if not rs:
            L.append(f"{name:13s} missing in {folder}"); continue
        f1, fpr, bf1 = summ(rs)
        res[name] = (f1, rs)
        sd = f1.std(ddof=1) if len(f1) > 1 else 0.0
        L.append(f"{name:13s} seeds={len(f1)} macroF1={f1.mean():.2f}±{sd:.2f} binF1={bf1:.2f} genFPR={fpr:.2f}%  ({desc})")
    if "lsage_mixnd" in res and "bilstm" in res:
        a, b = res["lsage_mixnd"][0], res["bilstm"][0]
        t = stats.ttest_ind(a, b, equal_var=False)
        L.append(f"\nSAGE neighbourhood (final regime) vs Bi-LSTM without neighbourhood: {a.mean() - b.mean():+.2f} points "
                 f"(Welch t={t.statistic:.1f}, p={t.pvalue:.2g})")
    if "lsage_mix" in res and "lsage" in res:
        a, b = res["lsage_mix"][0], res["lsage"][0]
        L.append(f"Mixed-scene training cost in distribution (lsage_mix - lsage): {a.mean() - b.mean():+.2f} points")
    rs = res.get("lsage_mixnd", (None, []))[1]
    if rs:
        nc = len(rs[0]["classes"])
        pc = np.mean([[r["per_class"][str(c)]["f1-score"] for c in range(nc)] for r in rs], 0) * 100
        L.append("lsage_mixnd per-class F1: " + ", ".join(f"{rs[0]['classes'][c]}={v:.1f}" for c, v in enumerate(pc)))
    L.append("\nMixAll results: see mixall_variants.txt in the same folder (step8c).")
    out = os.path.join(args.sage, "sage_summary.txt")
    open(out, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L)); print("\nSaved", out)


if __name__ == "__main__":
    main()
