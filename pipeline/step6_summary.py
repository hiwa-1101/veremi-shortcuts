"""
Step 6d - Collect every additional experiment into one short text file to send back.

Usage:
    python step6_summary.py --extra F:\\results_extra
Writes <extra>\\extra_summary.txt
"""

import argparse
import glob
import json
import os
import re

import numpy as np

NAMES = {0: "Genuine", 1: "ConstPos", 2: "ConstPosOffset", 3: "RandomPos", 4: "RandomPosOffset",
         5: "ConstSpeed", 6: "ConstSpeedOffset", 7: "RandomSpeed", 8: "RandomSpeedOffset",
         9: "EventualStop", 10: "Disruptive", 11: "DataReplay", 12: "DelayedMessages", 13: "DoS",
         14: "DoSRandom", 15: "DoSDisruptive", 16: "GridSybil", 17: "DataReplaySybil",
         18: "DoSRandomSybil", 19: "DoSDisruptiveSybil"}


def load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", required=True)
    args = ap.parse_args()
    L = []
    p = L.append
    runs = {os.path.basename(f)[:-5]: load(f) for f in glob.glob(os.path.join(args.extra, "*.json"))}
    runs = {k: v for k, v in runs.items() if v}

    fin = [r for k, r in sorted(runs.items()) if re.fullmatch(r"tmix_nodens_multi_vehicle_seed\d+", k)]
    if fin:
        p("=== 0. FINAL model (tmix_nodens) on the main test set ===")
        f1 = np.array([r["macro_f1"] for r in fin]) * 100
        fpr, miss = [], []
        for r in fin:
            cm = np.array(r["confusion_matrix"]); fpr.append(1 - cm[0, 0] / cm[0].sum()); miss.append(cm[1:, 0].sum() / cm[1:].sum())
        p(f"seeds={len(fin)} macroF1={f1.mean():.2f}±{f1.std(ddof=1) if len(f1) > 1 else 0:.2f} "
          f"acc={np.mean([r['accuracy'] for r in fin]) * 100:.2f} binF1={np.mean([r['binary_from_multi']['f1'] for r in fin]) * 100:.2f} "
          f"AUC={np.mean([r['roc_auc_macro'] for r in fin]):.4f} genFPR={np.mean(fpr) * 100:.2f}% missed={np.mean(miss) * 100:.2f}% "
          f"latency={fin[0].get('latency')}")
        pc = np.mean([[r["per_class"][str(c)]["f1-score"] for c in range(20)] for r in fin], 0) * 100
        rc = np.mean([[r["per_class"][str(c)]["recall"] for c in range(20)] for r in fin], 0) * 100
        p("   per-class F1: " + ", ".join(f"{NAMES[c]}={pc[c]:.1f}" for c in range(20)))
        p("   per-class recall: " + ", ".join(f"{NAMES[c]}={rc[c]:.1f}" for c in range(20)))
        p("")
    p("=== 1. Leave-one-attack-out (test split; attack never seen in training) ===")
    p(f"{'attack':22s} {'model':7s} {'detect%':>8s} {'genFPR%':>8s} {'macroF1':>8s}  top predicted-as")
    for k in range(1, 20):
        for mdl in ("tmodel", "tmix", "tmix_nodens", "xgb"):
            r = runs.get(f"{mdl}_loao{k}_multi_vehicle_seed0")
            if not r or "unseen" not in r:
                continue
            u = r["unseen"]
            top = sorted(u["predicted_as"].items(), key=lambda x: -x[1])[:3]
            top = ", ".join(f"{NAMES.get(int(c), c)}:{n}" for c, n in top)
            p(f"{NAMES[k]:22s} {mdl:7s} {u['detection_rate'] * 100:8.2f} {u['genuine_fpr'] * 100:8.2f} "
              f"{r['macro_f1'] * 100:8.2f}  {top}")

    p("\n=== 2. Traffic-density transfer (train period -> test period) ===")
    for k, r in sorted(runs.items()):
        m = re.match(r"(tmodel|tmix_nodens|tmix|xgb)_d(\d{4})to(\d{4})_multi_vehicle_seed0", k)
        if m:
            p(f"{m.group(1):7s} {m.group(2)} -> {m.group(3)}  macroF1={r['macro_f1'] * 100:.2f}  "
              f"acc={r['accuracy'] * 100:.2f}  binF1={r['binary_from_multi']['f1'] * 100:.2f}  n_test={r['n_test']:,}")

    p("\n=== 3. Literature baselines under our protocol (test split) ===")
    groups = {}
    for k, r in runs.items():
        m = re.match(r"(bilstm.*?)_multi_vehicle_seed(\d+)$", k)
        if m:
            groups.setdefault(m.group(1), []).append(r)
    for g, rs in groups.items():
        f1 = np.array([r["macro_f1"] for r in rs]) * 100
        r0 = rs[0]
        p(f"{g}: seeds={len(rs)} macroF1={f1.mean():.2f}±{f1.std(ddof=1) if len(f1) > 1 else 0:.2f} "
          f"acc={np.mean([r['accuracy'] for r in rs]) * 100:.2f} params={r0.get('params')} "
          f"size={r0.get('model_size_mb')}MB latency={r0.get('latency')}")
        pc = np.mean([[r["per_class"][str(c)]["f1-score"] for c in range(20)] for r in rs], 0) * 100
        p("   per-class F1: " + ", ".join(f"{NAMES[c]}={pc[c]:.1f}" for c in range(20)))

    p("\n=== 4. Literature protocol (XGBoost + absolute position, pseudonym ID, time) ===")
    for k, r in sorted(runs.items()):
        if k.startswith("xgb_lit"):
            p(f"{k}: macroF1={r['macro_f1'] * 100:.2f} acc={r['accuracy'] * 100:.2f} "
              f"binF1={(r.get('binary_from_multi') or {}).get('f1', float('nan')) * 100:.2f}")

    for stname in ["stats_mcnemar_bootstrap", "stats_mcnemar_bootstrap_tmix", "stats_mcnemar_bootstrap_tmix_nodens"]:
      st = runs.get(stname)
      if st:
        p(f"\n=== 5. McNemar + vehicle-cluster bootstrap ({stname}: proposed vs XGBoost) ===")
        for s, r in st["seeds"].items():
            o, b = r["overall"], r["bootstrap_vehicle"]
            p(f"seed {s}: chi2={o['chi2']:.1f} p={o['p_value']:.3g} proposed-only={o['a_only_correct']:,} "
              f"xgb-only={o['b_only_correct']:,} | diff={b['diff_mean'] * 100:.2f} "
              f"CI95=[{b['diff_ci95'][0] * 100:.2f}, {b['diff_ci95'][1] * 100:.2f}] "
              f"P(better)={b['share_resamples_proposed_better']:.3f}")
            worse = [(c, v) for c, v in r["per_class"].items() if v["b_only_correct"] > v["a_only_correct"]]
            p("   classes where XGBoost is right more often: "
              + ", ".join(f"{NAMES[int(c)]}(p={v['p_value']:.2g})" for c, v in worse))
            ns = [NAMES[int(c)] for c, v in r["per_class"].items() if v["p_value"] >= 0.05]
            p("   classes with no significant difference (p>=0.05): " + (", ".join(ns) or "none"))
        p(f"   test messages={st['n_test']:,} vehicles={st['n_vehicles']:,}")

    for k, r in sorted(runs.items()):
        if k.startswith("quant_int8"):
            p("\n=== 6. int8 quantisation (full test split, CPU) ===")
            for m in ("fp32_cpu", "int8_cpu"):
                q = r[m]
                p(f"{m}: macroF1={q['macro_f1'] * 100:.2f} acc={q['accuracy'] * 100:.2f} "
                  f"size={q['model_size_mb']:.3f}MB latency={q['latency']}")
            p(f"agreement int8 vs fp32: {r['int8_cpu']['agreement_with_fp32'] * 100:.3f}%")

    for k, r in sorted(runs.items()):
        if k.startswith("importance"):
            p("\n=== 7. Permutation importance (macro-F1 drop, points) ===")
            p(f"sample={r['n_sample']:,} base macroF1={r['base_macro_f1'] * 100:.2f}")
            items = sorted(r["tests"].items(), key=lambda x: -x[1]["macro_f1_drop"])
            for name, t in items:
                pc = np.array(t["per_class_f1_drop"]) * 100
                top = np.argsort(-pc)[:3]
                p(f"{name:42s} {t['macro_f1_drop'] * 100:6.2f}±{t['macro_f1_drop_std'] * 100:.2f}  "
                  + ", ".join(f"{NAMES[c]}:{pc[c]:.1f}" for c in top))
            p("EventualStop (class 9) - five largest drops:")
            es = sorted(((t["per_class_f1_drop"][9] * 100, n) for n, t in r["tests"].items()), reverse=True)[:5]
            p("   " + "; ".join(f"{n}={v:.1f}" for v, n in es))

    txt = "\n".join(L)
    path = os.path.join(args.extra, "extra_summary.txt")
    open(path, "w", encoding="utf-8").write(txt)
    print(txt)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
