"""
Step 12 - Tune XGBoost on val-dev, then re-test (answers "the baseline was not tuned").

Protocol (no test data in any decision):
  1. grid over max_depth x learning_rate x min_child_weight, seed 0, trained on the training
     split with early stopping on the early-stopping half of validation, scored on the
     OTHER half (val-dev) - exactly the protocol used for every other design decision;
  2. the configuration with the best val-dev macro-F1 is retrained with seeds 0,1,2 and
     evaluated ONCE on the test split;
  3. the summary compares it with default XGBoost (89.07) and the proposed model.
Every run is a separate step4 call, so finished runs are skipped after a power cut.

Usage:
    python step12_xgb_tune.py --features F:\\hiwa\\veremi_features_v3 --out F:\\hiwa\\results_xgbtune ^
        --proposed F:\\hiwa\\results_mixed
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
from scipy import stats

GRID = [  # (max_depth, learning_rate, min_child_weight); (8, 0.1, 1) = setting used so far
    (6, 0.1, 1), (8, 0.1, 1), (10, 0.1, 1), (12, 0.1, 1),
    (8, 0.05, 1), (10, 0.05, 1), (10, 0.1, 10),
]


def tag(d, lr, mcw):
    return f"_d{d}_lr{str(lr).replace('.', 'p')}_mcw{mcw}"


def run(features, out, d, lr, mcw, eval_split, seeds):
    env = dict(os.environ, XGB_MAX_DEPTH=str(d), XGB_LR=str(lr), XGB_MIN_CHILD=str(mcw))
    cmd = [sys.executable, "step4_baselines.py", "--features", features, "--out", out, "--protocol", "vehicle",
           "--task", "multi", "--models", "xgb", "--seeds", seeds, "--eval-split", eval_split, "--tag", tag(d, lr, mcw)]
    print(">>", " ".join(cmd), f"(depth={d} lr={lr} min_child_weight={mcw})", flush=True)
    subprocess.run(cmd, env=env, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--proposed", required=True, help="folder with tmix_nodens_multi_vehicle_seed*.json")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    L = []
    p = lambda s: (print(s, flush=True), L.append(s))

    for d, lr, mcw in GRID:
        run(args.features, args.out, d, lr, mcw, "val", "0")
    val = []
    for d, lr, mcw in GRID:
        r = json.load(open(os.path.join(args.out, f"xgb{tag(d, lr, mcw)}_multi_vehicle_val_seed0.json")))
        val.append((r["macro_f1"], d, lr, mcw, r.get("train_seconds")))
    p("=== XGBoost grid, val-dev macro-F1 (seed 0) ===")
    for f1, d, lr, mcw, ts in sorted(val, reverse=True):
        p(f"depth={d:2d} lr={lr:<5} min_child_weight={mcw:<3} val-dev macroF1={f1 * 100:.2f}  train={ts}s"
          + ("   <- setting used in the paper" if (d, lr, mcw) == (8, 0.1, 1) else ""))
    best = max(val)
    _, d, lr, mcw, _ = best
    p(f"\nselected on val-dev: depth={d} lr={lr} min_child_weight={mcw}")

    run(args.features, args.out, d, lr, mcw, "test", "0,1,2")
    rs = [json.load(open(x)) for x in sorted(glob.glob(os.path.join(args.out, f"xgb{tag(d, lr, mcw)}_multi_vehicle_seed*.json")))]
    a = np.array([r["macro_f1"] for r in rs]) * 100
    fpr = [1 - np.array(r["confusion_matrix"])[0, 0] / np.array(r["confusion_matrix"])[0].sum() for r in rs]
    p(f"\n=== Tuned XGBoost on the TEST split ({len(rs)} seeds) ===")
    p(f"macroF1={a.mean():.2f}±{a.std(ddof=1) if len(a) > 1 else 0:.2f} acc={np.mean([r['accuracy'] for r in rs]) * 100:.2f} "
      f"binF1={np.mean([r['binary_from_multi']['f1'] for r in rs]) * 100:.2f} genFPR={np.mean(fpr) * 100:.2f}% "
      f"size={rs[0]['model_size_mb']}MB latency={rs[0]['latency']}")
    nc = len(rs[0]["classes"])
    pc = np.mean([[r["per_class"][str(c)]["f1-score"] for c in range(nc)] for r in rs], 0) * 100
    p("per-class F1: " + ", ".join(f"{c}={v:.1f}" for c, v in enumerate(pc)))

    prop = [json.load(open(x)) for x in sorted(glob.glob(os.path.join(args.proposed, "tmix_nodens_multi_vehicle_seed*.json")))]
    if prop:
        b = np.array([r["macro_f1"] for r in prop]) * 100
        pp = np.mean([[r["per_class"][str(c)]["f1-score"] for c in range(nc)] for r in prop], 0) * 100
        t = stats.ttest_ind(b, a, equal_var=False)
        w = stats.wilcoxon(pp, pc)
        p(f"\nProposed {b.mean():.2f} vs tuned XGBoost {a.mean():.2f}: {b.mean() - a.mean():+.2f} points "
          f"(Welch t={t.statistic:.1f} p={t.pvalue:.2g}; Wilcoxon over classes p={w.pvalue:.2g}; "
          f"proposed better on {(pp > pc).sum()}/{nc} classes)")
    p("Default XGBoost in the paper: 89.07 ± 0.01 (depth 8, lr 0.1)")
    open(os.path.join(args.out, "xgb_tuning_summary.txt"), "w", encoding="utf-8").write("\n".join(L))
    print("\nSaved", os.path.join(args.out, "xgb_tuning_summary.txt"))


if __name__ == "__main__":
    main()
