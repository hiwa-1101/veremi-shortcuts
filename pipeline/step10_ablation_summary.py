"""
Step 10 - Summary of the two reviewer-requested checks on the FINAL training regime:
  (a) branch ablation: final model (tmix_nodens) vs the same regime without the neighbourhood
      branch (tmix_nodens_noNeigh, M = 0), 3 seeds each, main test set;
  (b) q sensitivity: q = 0.25 / 0.5 / 0.75, seed 0, main test set.
MixAll numbers for the same runs are in mixall_variants.txt (step8c).

Usage:
    python step10_ablation_summary.py F:\\hiwa\\results_mixed
Writes <folder>\\ablation_q_summary.txt
"""
import json
import os
import sys

import numpy as np
from scipy import stats

NAMES = ["Genuine", "ConstPos", "ConstPosOffset", "RandomPos", "RandomPosOffset", "ConstSpeed",
         "ConstSpeedOffset", "RandomSpeed", "RandomSpeedOffset", "EventualStop", "Disruptive",
         "DataReplay", "DelayedMessages", "DoS", "DoSRandom", "DoSDisruptive", "GridSybil",
         "DataReplaySybil", "DoSRandomSybil", "DoSDisruptiveSybil"]
D = sys.argv[1]
L = []
p = L.append


def load(name, seeds):
    out = []
    for s in seeds:
        f = os.path.join(D, f"{name}_multi_vehicle_seed{s}.json")
        if os.path.exists(f):
            out.append(json.load(open(f)))
    return out


def row(name, rs):
    f1 = np.array([r["macro_f1"] for r in rs]) * 100
    fpr, miss = [], []
    for r in rs:
        cm = np.array(r["confusion_matrix"])
        fpr.append(1 - cm[0, 0] / cm[0].sum()); miss.append(cm[1:, 0].sum() / cm[1:].sum())
    sd = f1.std(ddof=1) if len(f1) > 1 else 0.0
    p(f"{name:22s} seeds={len(rs)} macroF1={f1.mean():.2f}±{sd:.2f} "
      f"acc={np.mean([r['accuracy'] for r in rs]) * 100:.2f} "
      f"binF1={np.mean([r['binary_from_multi']['f1'] for r in rs]) * 100:.2f} "
      f"genFPR={np.mean(fpr) * 100:.2f}% missed={np.mean(miss) * 100:.2f}% "
      f"params={rs[0].get('params')} latency={rs[0].get('latency')}")
    return f1, np.mean([[r["per_class"][str(c)]["f1-score"] for c in range(20)] for r in rs], 0) * 100


full = load("tmix_nodens", [0, 1, 2])
nn = load("tmix_nodens_noNeigh", [0, 1, 2])
p("=== (a) Neighbourhood ablation under the final regime (main test set) ===")
if full and nn:
    f_full, pc_full = row("final (M=32)", full)
    f_nn, pc_nn = row("final, M=0", nn)
    t = stats.ttest_ind(f_full, f_nn, equal_var=False)
    w = stats.wilcoxon(pc_full, pc_nn)
    p(f"difference = {f_full.mean() - f_nn.mean():+.2f} points  Welch t={t.statistic:.1f} p={t.pvalue:.2g}  "
      f"Wilcoxon over 20 classes p={w.pvalue:.2g}  better on {(pc_full > pc_nn).sum()}/20 classes")
    p("per-class F1 (final | M=0 | delta):")
    for c in np.argsort(pc_nn - pc_full):
        p(f"   {NAMES[c]:20s} {pc_full[c]:6.1f} {pc_nn[c]:6.1f} {pc_full[c] - pc_nn[c]:+6.1f}")
else:
    p(f"missing runs: final={len(full)} noNeigh={len(nn)}")

p("\n=== (b) Sensitivity to the swap probability q (seed 0, main test set) ===")
qs = [("q=0.25", load("tmix_nodens_q25", [0])), ("q=0.50 (final)", full[:1]), ("q=0.75", load("tmix_nodens_q75", [0]))]
pcs = {}
for name, rs in qs:
    if rs:
        _, pcs[name] = row(name, rs)
    else:
        p(f"{name}: missing")
if len(pcs) == 3:
    p("per-class F1 where q matters most (q=0.25 | 0.50 | 0.75):")
    a = np.array(list(pcs.values()))
    for c in np.argsort(-(a.max(0) - a.min(0)))[:6]:
        p(f"   {NAMES[c]:20s} " + " ".join(f"{v:6.1f}" for v in a[:, c]))

txt = "\n".join(L)
open(os.path.join(D, "ablation_q_summary.txt"), "w", encoding="utf-8").write(txt)
print(txt)
print("\nSaved", os.path.join(D, "ablation_q_summary.txt"))
