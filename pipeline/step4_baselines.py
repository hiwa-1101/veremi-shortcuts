"""
Step 4 - Classical baselines (Random Forest, XGBoost, LightGBM) under two protocols.

Protocols:
  vehicle : our leakage-free split (each physical vehicle in exactly one of
            train/val/test, identical across all attack folders) -> main results
  random  : the common practice in the literature (all folders pooled, messages
            split at random 70/15/15). Used ONLY to measure how much a leaky
            protocol inflates the scores (contribution C1).

Tasks:
  multi  : label_msg as is (genuine + every attack type)
  binary : genuine (0) vs any attack (1)

Training details:
  * Features: only the receiver-observable groups k_, r_, c_, n_ (step 3).
  * Genuine training messages are randomly down-sampled (default 3,000,000)
    because the same genuine traffic is repeated in every attack folder;
    all attack messages are kept. Remaining imbalance -> balanced sample weights.
  * Early stopping (XGBoost, LightGBM) on a random 1,000,000-message subset
    of the validation split. The test split is never used for any decision.
  * 5 seeds by default; every run stores metrics, per-class report, confusion
    matrix, CPU latency, model size, and test predictions (for McNemar tests).

Usage:
    python step4_baselines.py --features F:\\veremi_features --out F:\\results --protocol vehicle --task multi
    python step4_baselines.py --features F:\\veremi_features --out F:\\results --protocol random  --task multi --models xgb
Options: --models rf,xgb,lgbm   --seeds 0,1,2,3,4   --max-genuine-train 3000000
"""

import argparse
import glob
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

FEATURE_PREFIXES = ("k_", "r_", "c_", "n_", "m_")


# --------------------------------------------------------------------------
def load_features(folder, drop, extra=()):
    files = sorted(glob.glob(os.path.join(folder, "*.parquet")))
    first = pd.read_parquet(files[0])
    feats = [c for c in first.columns if c.startswith(FEATURE_PREFIXES)]
    feats = [c for c in feats if not any(c.startswith(d) for d in drop)]
    feats = feats + [c for c in extra if c not in feats]
    cols = feats + [c for c in ["label_msg", "split", "attack_folder", "period", "sender", "label_vehicle"]
                    if c not in feats and c in first.columns]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    df = df[df["split"].notna()].reset_index(drop=True)
    df["period"] = df["period"].astype(str).str.zfill(4)
    print(f"Loaded {len(df):,} messages from {len(files)} files, {len(feats)} features")
    return df, feats


def make_protocol_split(df, protocol, seed):
    if protocol == "vehicle":
        return df["split"].to_numpy()
    # Leaky literature protocol: random message-level split of the pooled data
    rng = np.random.default_rng(1000 + seed)
    u = rng.random(len(df))
    return np.where(u < 0.70, "train", np.where(u < 0.85, "val", "test"))


def downsample_genuine(idx, y, max_genuine, seed):
    rng = np.random.default_rng(seed)
    gen = idx[y[idx] == 0]
    att = idx[y[idx] != 0]
    if len(gen) > max_genuine:
        gen = rng.choice(gen, max_genuine, replace=False)
    out = np.concatenate([gen, att])
    rng.shuffle(out)
    return out


def build_model(name, n_classes, seed):
    if name == "rf":
        # 100 trees, depth 20: keeps memory/size manageable on ~23 M training messages
        return RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=2,
                                      max_samples=0.2, class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=seed)
    if name == "xgb":
        import xgboost as xgb
        obj = "binary:logistic" if n_classes == 2 else "multi:softprob"
        # defaults = the settings of all reported runs; XGB_* environment variables are used
        # only by the val-dev tuning run (step12_xgb_tune.py)
        return xgb.XGBClassifier(n_estimators=2000, learning_rate=float(os.environ.get("XGB_LR", 0.1)),
                                 max_depth=int(os.environ.get("XGB_MAX_DEPTH", 8)),
                                 min_child_weight=float(os.environ.get("XGB_MIN_CHILD", 1)),
                                 subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                 device=os.environ.get("XGB_DEVICE","cuda"), objective=obj, eval_metric="logloss" if n_classes == 2 else "mlogloss",
                                 early_stopping_rounds=50, random_state=seed)
    if name == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.1, num_leaves=127,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                  class_weight="balanced", n_jobs=-1, random_state=seed, verbose=-1)
    raise ValueError(name)


def fit_model(name, model, Xtr, ytr, wtr, Xva, yva):
    if name == "rf":
        model.fit(Xtr, ytr)  # class weights handled internally
    elif name == "xgb":
        model.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva, yva)], verbose=False)
    elif name == "lgbm":
        import lightgbm as lgb
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    return model


def predict_in_chunks(model, X, idx, chunk=500_000):
    """predict_proba in chunks: Random Forest allocates (rows x classes) float64 per
    tree and thread, which exhausts RAM on ~11 M test messages if done at once."""
    parts = [model.predict_proba(X[idx[i:i + chunk]]).astype(np.float32)
             for i in range(0, len(idx), chunk)]
    return np.concatenate(parts)


TRAIN_WINDOWS = [(25200, 32400), (50400, 57600)]   # 07-09 h and 14-16 h


def ext_metrics(y, pred, n_classes, names):
    present = np.unique(y)
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=list(range(n_classes)), zero_division=0)
    att, patt = y != 0, pred != 0
    tp = int((att & patt).sum()); fp = int((~att & patt).sum()); fn = int((att & ~patt).sum())
    return {"n": int(len(y)), "macro_f1": float(np.mean(f1[present])), "accuracy": float((y == pred).mean()),
            "binary_f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "genuine_fpr": float(patt[~att].mean()) if (~att).any() else None,
            "attack_miss_rate": float((~patt[att]).mean()) if att.any() else None,
            "per_class": {names[c]: {"precision": float(pr[c]), "recall": float(rc[c]), "f1": float(f1[c]),
                                     "support": int(sup[c])} for c in present}}


def external_eval(model, present, n_classes, le, feats, folder):
    """Apply a trained model to an external feature folder (e.g. MixAll) with
    VEHICLE-level labels (MixAll's ground truth logs the transmitted content)."""
    names = [str(c) for c in le.classes_]
    ys, ps, hs = [], [], []
    for f in sorted(glob.glob(os.path.join(folder, "*.parquet"))):
        d = pd.read_parquet(f, columns=feats + ["label_vehicle", "sim_hour"])
        if len(d) == 0:
            continue
        Xe = d[feats].to_numpy(np.float32)
        pr = predict_in_chunks(model, Xe, np.arange(len(Xe)))
        if pr.shape[1] < n_classes:
            full = np.zeros((len(pr), n_classes), pr.dtype); full[:, present] = pr; pr = full
        ps.append(pr.argmax(1).astype(np.int8))
        ys.append(le.transform(d["label_vehicle"].to_numpy()).astype(np.int8))
        hs.append(d["sim_hour"].to_numpy(np.int32))
        print(f"  external {os.path.basename(f)} rows={len(d):,}", flush=True)
    y, pred, h = (np.concatenate(a).astype(np.int64) for a in (ys, ps, hs))
    ov = np.zeros(len(h), bool)
    for a, b in TRAIN_WINDOWS:
        ov |= (h >= a) & (h < b)
    return {"labels": "vehicle",
            "external_non_overlapping_hours": ext_metrics(y[~ov], pred[~ov], n_classes, names),
            "overlapping_hours": ext_metrics(y[ov], pred[ov], n_classes, names),
            "all_hours": ext_metrics(y, pred, n_classes, names)}


def binary_from_multi(yt, proba, le):
    """Genuine-vs-attack metrics derived from a multi-class model (class 0 = genuine)."""
    g = int(np.where(le.classes_ == 0)[0][0])
    yb = (yt != g).astype(int)
    pb = 1.0 - proba[:, g]
    pr = (pb >= 0.5).astype(int)
    return {"accuracy": accuracy_score(yb, pr), "f1": f1_score(yb, pr), "roc_auc": float(roc_auc_score(yb, pb))}


def unseen_report(yt, pred, le, attack):
    """Leave-one-attack-out: how the never-seen attack is handled at test time."""
    names = [str(c) for c in le.classes_]
    g = int(np.where(le.classes_ == 0)[0][0])
    k = int(np.where(le.classes_ == attack)[0][0])
    m = yt == k
    pk = pred[m]
    return {"attack": int(attack), "n_test_msgs": int(m.sum()),
            "detection_rate": float((pk != g).mean()) if m.any() else None,
            "predicted_as": {names[c]: int(n) for c, n in zip(*np.unique(pk, return_counts=True))},
            "genuine_fpr": float((pred[yt == g] != g).mean())}


def cpu_latency(name, model, X):
    """Median per-message latency on CPU: single message and batch of 1024."""
    # one CPU thread: closer to an on-board unit and avoids thread start-up overhead
    old_jobs = model.get_params().get("n_jobs")
    model.set_params(n_jobs=1)
    if name == "xgb":
        model.set_params(device="cpu")
    single = X[:1]
    batch = X[:1024]
    t = []
    for _ in range(50):
        s = time.perf_counter(); model.predict(single); t.append(time.perf_counter() - s)
    tb = []
    for _ in range(10):
        s = time.perf_counter(); model.predict(batch); tb.append(time.perf_counter() - s)
    model.set_params(n_jobs=old_jobs)
    if name == "xgb":
        model.set_params(device=os.environ.get("XGB_DEVICE","cuda"))
    return {"single_msg_ms": float(np.median(t) * 1e3),
            "batch1024_per_msg_us": float(np.median(tb) / len(batch) * 1e6)}


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocol", choices=["vehicle", "random"], default="vehicle")
    ap.add_argument("--task", choices=["multi", "binary"], default="multi")
    ap.add_argument("--models", default="xgb,lgbm,rf")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--max-genuine-train", type=int, default=3_000_000)
    ap.add_argument("--eval-split", choices=["test", "val"], default="test",
                    help="'val' for all design decisions; 'test' only for final reported results")
    ap.add_argument("--drop-features", default="",
                    help="comma-separated feature names/prefixes to exclude (ablation)")
    ap.add_argument("--tag", default="", help="extra name for this configuration")
    ap.add_argument("--extra-cols", default="",
                    help="extra columns used as features (literature protocol: aux_pos_x,aux_pos_y,senderPseudo,rcvTime)")
    ap.add_argument("--train-period", default="", help="train only on this period, e.g. 0709")
    ap.add_argument("--test-period", default="", help="evaluate only on this period, e.g. 1416")
    ap.add_argument("--external-features", default="",
                    help="also evaluate on this external feature folder (vehicle-level labels), e.g. F:\\mixall\\features")
    ap.add_argument("--exclude-attack", type=int, default=-1,
                    help="leave-one-attack-out: remove this attack label from training/early stopping")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    drop = [d for d in args.drop_features.split(",") if d]
    extra = [c for c in args.extra_cols.split(",") if c]
    df, feats = load_features(args.features, drop, extra)
    print("Periods in data:", sorted(df["period"].unique()))
    for p_ in (args.train_period, args.test_period):
        if p_ and p_ not in set(df["period"]):
            raise SystemExit(f"period {p_} not found; available: {sorted(df['period'].unique())}")
    print("Features used:", feats)
    X = df[feats].to_numpy(dtype=np.float32)
    y_raw = df["label_msg"].to_numpy()
    if args.task == "binary":
        y_raw = (y_raw != 0).astype(int)
    le = LabelEncoder().fit(y_raw)
    y = le.transform(y_raw)
    n_classes = len(le.classes_)
    print(f"Task={args.task} protocol={args.protocol} classes={list(le.classes_)}")

    summary = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        split = make_protocol_split(df, args.protocol, seed)
        period = df["period"].to_numpy()
        keep_tr = split == "train"
        if args.train_period:
            keep_tr &= period == args.train_period
        if args.exclude_attack >= 0:
            keep_tr &= y_raw != args.exclude_attack
        idx_tr = downsample_genuine(np.where(keep_tr)[0], y, args.max_genuine_train, seed)
        # Validation vehicles are split in two halves: one half for early stopping,
        # the other half ("val_dev") for development decisions. The test split is
        # touched only when --eval-split test.
        # (vehicle IDs in VeReMi are all odd numbers, so a parity rule cannot be used;
        #  validation vehicles are randomly halved with a fixed seed instead)
        sender = df["sender"].to_numpy()
        val_vehicles = np.unique(sender[split == "val"])
        es_vehicles = np.random.default_rng(12345).permutation(val_vehicles)[: len(val_vehicles) // 2]
        half = np.isin(sender, es_vehicles)
        keep_va = (split == "val") & half
        if args.train_period:
            keep_va &= period == args.train_period
        if args.exclude_attack >= 0:
            keep_va &= y_raw != args.exclude_attack
        idx_va_all = np.where(keep_va)[0]
        rng = np.random.default_rng(seed)
        idx_va = rng.choice(idx_va_all, min(1_000_000, len(idx_va_all)), replace=False)
        if args.eval_split == "test":
            idx_te = np.where(split == "test")[0]
        else:
            idx_te = np.where((split == "val") & ~half)[0]
        if args.test_period:
            idx_te = idx_te[period[idx_te] == args.test_period]
        wtr = compute_sample_weight("balanced", y[idx_tr])

        for name in args.models.split(","):
            ev = "" if args.eval_split == "test" else "_val"
            tag = f"{name}{args.tag}_{args.task}_{args.protocol}{ev}_seed{seed}"
            if os.path.exists(os.path.join(args.out, tag + ".json")):
                print(f"[skip] {tag} exists"); continue
            print(f"\n=== {tag}: train={len(idx_tr):,} val={len(idx_va):,} test={len(idx_te):,}")
            # Leave-one-attack-out: XGBoost needs contiguous labels, so the removed
            # class is squeezed out for training and re-inserted (probability 0) after.
            # XGBoost needs contiguous labels: classes absent from training (an excluded
            # attack, or an attack missing from one traffic period) are squeezed out for
            # training and re-inserted with probability 0 afterwards.
            present = present_tr = np.unique(y[idx_tr])
            remap = np.full(n_classes, -1, np.int64); remap[present] = np.arange(len(present))
            idx_va = idx_va[remap[y[idx_va]] >= 0]
            if len(present) < n_classes:
                print("Classes absent from training:", sorted(set(range(n_classes)) - set(present.tolist())))
            model = build_model(name, len(present) if n_classes > 2 else 2, seed)
            t0 = time.time()
            model = fit_model(name, model, X[idx_tr], remap[y[idx_tr]], wtr, X[idx_va], remap[y[idx_va]])
            train_s = time.time() - t0

            pr = predict_in_chunks(model, X, idx_te)
            if pr.shape[1] < n_classes:
                proba = np.zeros((len(pr), n_classes), pr.dtype); proba[:, present] = pr
            else:
                proba = pr
            pred = proba.argmax(1)
            yt = y[idx_te]
            if n_classes == 2:
                auc = roc_auc_score(yt, proba[:, 1])
            else:
                present = np.unique(yt)
                auc = roc_auc_score(yt, proba[:, present] / proba[:, present].sum(1, keepdims=True),
                                    multi_class="ovr", average="macro", labels=present)
            names = [str(c) for c in le.classes_]
            res = {
                "model": name + args.tag, "task": args.task, "protocol": args.protocol, "seed": seed,
                "eval_split": args.eval_split, "features": feats,
                "classes": names, "n_train": int(len(idx_tr)), "n_test": int(len(idx_te)),
                "train_seconds": round(train_s, 1),
                "accuracy": accuracy_score(yt, pred),
                "macro_f1": f1_score(yt, pred, average="macro"),
                "weighted_f1": f1_score(yt, pred, average="weighted"),
                "roc_auc_macro": float(auc),
                "per_class": classification_report(yt, pred, labels=range(n_classes),
                                                   target_names=names, output_dict=True, zero_division=0),
                "confusion_matrix": confusion_matrix(yt, pred, labels=range(n_classes)).tolist(),
                "model_size_mb": round(len(pickle.dumps(model)) / 1e6, 2),
                "binary_from_multi": binary_from_multi(yt, proba, le) if n_classes > 2 else None,
                "latency": cpu_latency(name, model, X[idx_te]),
                "config": {k: v for k, v in vars(args).items()},
            }
            if args.exclude_attack >= 0 and n_classes > 2:
                res["unseen"] = unseen_report(yt, pred, le, args.exclude_attack)
            if args.external_features:
                res["external"] = external_eval(model, present_tr, n_classes, le, feats, args.external_features)
                if "label_vehicle" in df.columns:
                    yv = le.transform(df["label_vehicle"].to_numpy()[idx_te])
                    res["main_test_vehicle_labels"] = ext_metrics(yv, pred, n_classes, names)
                for part in ["external_non_overlapping_hours", "overlapping_hours", "all_hours"]:
                    e = res["external"][part]
                    print(f"EXTERNAL {part:32s} n={e['n']:,} macroF1={e['macro_f1'] * 100:.2f} acc={e['accuracy'] * 100:.2f} "
                          f"binF1={e['binary_f1'] * 100:.2f} genFPR={e['genuine_fpr'] * 100:.2f}% missed={e['attack_miss_rate'] * 100:.2f}%")
                if "main_test_vehicle_labels" in res:
                    e = res["main_test_vehicle_labels"]
                    print(f"MAIN test, vehicle labels: macroF1={e['macro_f1'] * 100:.2f} binF1={e['binary_f1'] * 100:.2f} "
                          f"genFPR={e['genuine_fpr'] * 100:.2f}% missed={e['attack_miss_rate'] * 100:.2f}%")
                print("EXTERNAL per-class F1 (non-overlapping): " + ", ".join(
                    f"{c}={v['f1'] * 100:.1f}" for c, v in res["external"]["external_non_overlapping_hours"]["per_class"].items()))
            with open(os.path.join(args.out, tag + ".json"), "w") as f:
                json.dump(res, f, indent=1)
            np.save(os.path.join(args.out, tag + "_pred.npy"), pred.astype(np.int8))
            np.save(os.path.join(args.out, f"ytest{args.tag}_{args.task}_{args.protocol}{ev}_seed{seed}.npy"), yt.astype(np.int8))
            print(f"acc={res['accuracy']:.4f} macroF1={res['macro_f1']:.4f} AUC={res['roc_auc_macro']:.4f} "
                  f"size={res['model_size_mb']}MB latency={res['latency']}")
            summary.append({k: res[k] for k in ["model", "task", "protocol", "seed", "accuracy",
                                                 "macro_f1", "weighted_f1", "roc_auc_macro",
                                                 "train_seconds", "model_size_mb"]})
            del model

    # Aggregate every result file present (mean ± std across seeds)
    rows = []
    for f in glob.glob(os.path.join(args.out, "*.json")):
        r = json.load(open(f))
        if not isinstance(r, dict) or "model" not in r or "task" not in r:
            continue          # other result files (statistics, quantisation, importance)
        row = {k: r[k] for k in ["model", "task", "protocol", "seed", "accuracy",
                                  "macro_f1", "weighted_f1", "roc_auc_macro"]}
        row["eval_split"] = r.get("eval_split", "test")
        rows.append(row)
    if rows:
        allr = pd.DataFrame(rows)
        agg = allr.groupby(["eval_split", "task", "protocol", "model"]).agg(["mean", "std", "count"])
        agg = agg.drop(columns="seed", level=0)
        agg.to_csv(os.path.join(args.out, "summary_baselines.csv"))
        pd.set_option("display.width", 250)
        pd.set_option("display.max_columns", 50)
        print("\n", agg.round(4))
        print(f"\nSaved: {os.path.join(args.out, 'summary_baselines.csv')}")


if __name__ == "__main__":
    main()
