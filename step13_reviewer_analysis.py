"""
Step 13 - Two analyses requested in internal review (no training; uses saved models only).

Part A - Unseen self-consistent attacks (leave-one-attack-out models of the final regime).
  For each withheld attack K (default: 2 ConstPosOffset, 16 GridSybil, 12 DelayedMessages):
   A1  single-input separability on the test split: ROC-AUC of each of the 26 inputs alone,
       attack K vs genuine (is the evidence in the inputs at all?);
   A2  scores of the model that never saw K:
         - 1 - P(genuine)            (attack score of the closed-set classifier)
         - 1 - max softmax (MSP)     (standard open-set score)
         - Mahalanobis distance of the penultimate representation to the TRAINING genuine
           messages (is K separated in the representation?)
       each as ROC-AUC (K vs genuine) and as detection rate at the false-alarm rate the
       model already has (its genuine FPR under argmax);
   A3  the same scores for the final model trained WITH K (reference).
  Interpretation: high A2 AUC but low argmax detection -> the information is there and the
  closed-set head maps K to genuine; low A1 and A2 AUC -> not separable with these inputs.

Part B - Sybil variants on MixAll (proposed model, final regime, seeds 0-2).
   B1  where do DataReplaySybil (17) and DoSDisruptiveSybil (19) messages go: predicted-class
       distribution on MixAll (non-overlapping hours) vs the main test set, both with
       vehicle-level labels;
   B2  what the 1-s neighbourhood looks like for these messages: number of neighbour messages
       and pseudonyms, share of neighbour messages from attacker vehicles, number of distinct
       attack types among neighbours, share from the same attack type - MixAll vs main test.

Usage (Windows example):
    python step13_reviewer_analysis.py --cache F:\\hiwa\\dl_cache --mixed F:\\hiwa\\results_mixed ^
        --main-features F:\\hiwa\\veremi_features_v3 --mixall-features F:\\hiwa\\mixall\\features ^
        --out F:\\hiwa\\results_mixed
    (add --skip-b or --skip-a to run one part only)
"""
import argparse
import glob
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import step5b_train_model as s5
from step8_mixall import TRAIN_WINDOWS, make_data

NAMES = {0: "Genuine", 1: "ConstPos", 2: "ConstPosOffset", 3: "RandomPos", 4: "RandomPosOffset", 5: "ConstSpeed",
         6: "ConstSpeedOffset", 7: "RandomSpeed", 8: "RandomSpeedOffset", 9: "EventualStop", 10: "Disruptive",
         11: "DataReplay", 12: "DelayedMessages", 13: "DoS", 14: "DoSRandom", 15: "DoSDisruptive",
         16: "GridSybil", 17: "DataReplaySybil", 18: "DoSRandomSybil", 19: "DoSDisruptiveSybil"}
L = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    L.append(s)


def load_model(info, base, device):
    res = json.load(open(base + ".json"))
    cfg = res["config"]
    m = s5.LinkNeighbourNet(len(info["features"]), len(info["classes"]), cfg["history"], cfg["neighbors"],
                            d=cfg["d"], layers=cfg["layers"], encoder=cfg.get("encoder", "transformer"))
    m.load_state_dict(torch.load(base + "_weights.pt", map_location="cpu"))
    zf = [info["features"].index(f) for f in (cfg.get("zero_features") or "").split(",") if f]
    return m.to(device).eval(), cfg, zf, res


@torch.no_grad()
def run(model, cfg, data, rows, bs=8192):
    """softmax probabilities and penultimate representation (input of the last linear layer)."""
    feats = []
    h = model.head[4].register_forward_hook(lambda mod, i, o: feats.append(i[0].float().cpu()))
    probs = []
    for i in range(0, len(rows), bs):
        rb = rows[i:i + bs]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=data.device.type == "cuda"):
            logits = model(*data.batch(rb, cfg["history"], cfg["neighbors"], cfg["window"]))
        probs.append(F.softmax(logits.float(), -1).cpu())
    h.remove()
    return torch.cat(probs).numpy(), torch.cat(feats).numpy()


def auc(pos, neg):
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(roc_auc_score(y, np.r_[pos, neg]))


def recall_at(pos, neg, fpr):
    thr = np.quantile(neg, 1 - fpr)
    return float((pos > thr).mean())


class Zero:
    """temporarily zero the inputs a model was trained without."""
    def __init__(self, data, idx): self.d, self.i = data, idx
    def __enter__(self):
        if self.i:
            self.saved = self.d.X[:, self.i].clone(); self.d.X[:, self.i] = 0
    def __exit__(self, *a):
        if self.i:
            self.d.X[:, self.i] = self.saved


def part_a(args, info, device):
    classes = info["classes"]; g = classes.index(0)
    split = np.load(os.path.join(args.cache, "split.npy"))
    y = np.load(os.path.join(args.cache, "y.npy")).astype(np.int64)
    rng = np.random.default_rng(0)
    te_g = np.sort(rng.choice(np.where((split == 2) & (y == g))[0], args.n_genuine, replace=False))
    tr_g = np.sort(rng.choice(np.where((split == 0) & (y == g))[0], args.n_ref, replace=False))
    Xmm = np.load(os.path.join(args.cache, "X.npy"), mmap_mode="r")
    Xg = np.asarray(Xmm[te_g], dtype=np.float32)
    data = s5.GpuData(args.cache, device)
    T = lambda a: torch.from_numpy(a).to(device)
    feats = info["features"]

    p("=" * 100)
    p("PART A - unseen self-consistent attacks (leave-one-attack-out, final regime, seed 0)")
    p(f"genuine test sample: {len(te_g):,} messages; genuine training reference: {len(tr_g):,} messages")
    for K in [int(k) for k in args.attacks.split(",")]:
        kx = classes.index(K)
        rows_k = np.where((split == 2) & (y == kx))[0]
        if len(rows_k) > args.n_attack:
            rows_k = np.sort(rng.choice(rows_k, args.n_attack, replace=False))
        p(f"\n--- withheld attack {K} {NAMES.get(K, K)}: {len(rows_k):,} test messages ---")
        # A1 single inputs
        Xk = np.asarray(Xmm[rows_k], dtype=np.float32)
        a1 = []
        for j, f in enumerate(feats):
            a = auc(Xk[:, j], Xg[:, j])
            a1.append((max(a, 1 - a), f, "higher" if a >= 0.5 else "lower"))
        a1.sort(reverse=True)
        p("A1 single-input ROC-AUC (attack vs genuine), top 6: " +
          "; ".join(f"{f}={v:.3f} ({d} for attack)" for v, f, d in a1[:6]))
        for v, f, d in a1:
            if f.startswith("m_road"):
                p(f"   map input {f}: AUC={v:.3f} ({d} for attack)")
        # A2 / A3
        for label, base in [("A2 model WITHOUT this attack",
                              os.path.join(args.mixed, f"{args.name}_loao{K}_multi_vehicle_seed0")),
                             ("A3 final model WITH this attack (reference)",
                              os.path.join(args.mixed, f"{args.name}_multi_vehicle_seed0"))]:
            if not os.path.exists(base + "_weights.pt"):
                p(f"{label}: missing {base}_weights.pt - skipped"); continue
            m, cfg, zf, res = load_model(info, base, device)
            with Zero(data, zf):
                pk, hk = run(m, cfg, data, T(rows_k))
                pg, hg = run(m, cfg, data, T(te_g))
                _, hr = run(m, cfg, data, T(tr_g))
            fpr = float((pg.argmax(1) != g).mean())
            det = float((pk.argmax(1) != g).mean())
            mu = hr.mean(0); cov = np.cov(hr, rowvar=False) + 1e-3 * np.eye(hr.shape[1])
            P = np.linalg.inv(cov)
            md = lambda H: np.einsum("ij,jk,ik->i", H - mu, P, H - mu)
            scores = {"1-P(genuine)": (1 - pk[:, g], 1 - pg[:, g]),
                      "1-max softmax": (1 - pk.max(1), 1 - pg.max(1)),
                      "Mahalanobis (repr.)": (md(hk), md(hg))}
            top = np.bincount(pk.argmax(1), minlength=len(classes)) / len(pk)
            topc = np.argsort(-top)[:3]
            p(f"{label}: argmax detection={det * 100:.1f}% at genuine FPR={fpr * 100:.2f}%; predicted as " +
              ", ".join(f"{NAMES.get(classes[c], classes[c])} {top[c] * 100:.1f}%" for c in topc))
            for sname, (sp_, sn_) in scores.items():
                p(f"   {sname:22s} AUC={auc(sp_, sn_):.3f}  detection at same FPR={recall_at(sp_, sn_, max(fpr, 1e-4)) * 100:5.1f}%"
                  f"  at FPR 5%={recall_at(sp_, sn_, 0.05) * 100:5.1f}%")
            del m
            if device.type == "cuda":
                torch.cuda.empty_cache()
    del data
    if device.type == "cuda":
        torch.cuda.empty_cache()


def neigh_stats(data, rows, M, W, ylab):
    """composition of the 1-s neighbourhood (same rule as the model) for target rows."""
    out = []
    for i in range(0, len(rows), 8192):
        rb = rows[i:i + 8192]
        k2 = torch.arange(1, M + 1, device=data.device)
        r2 = data.rank_rx[rb].unsqueeze(1) - k2
        ok2 = r2 >= 0
        idx2 = data.order_rx[r2.clamp(min=0)]
        ok2 &= data.rx_id[idx2] == data.rx_id[rb].unsqueeze(1)
        dt2 = data.time[rb].unsqueeze(1) - data.time[idx2]
        ok2 &= (dt2 >= 0) & (dt2 <= W)
        ok2 &= data.pseudo[idx2] != data.pseudo[rb].unsqueeze(1)
        lab = ylab[idx2]; tgt = ylab[rb].unsqueeze(1)
        n = ok2.sum(1).float()
        att = ((lab != 0) & ok2).sum(1).float()
        same = ((lab == tgt) & ok2).sum(1).float()
        ps = torch.where(ok2, data.pseudo[idx2], torch.full_like(idx2, -1))
        npseudo = torch.tensor([len(set(r.tolist()) - {-1}) for r in ps.cpu()], dtype=torch.float32)
        la = torch.where(ok2 & (lab != 0), lab, torch.full_like(lab, -1)).cpu()
        ntypes = torch.tensor([len(set(r.tolist()) - {-1}) for r in la], dtype=torch.float32)
        out.append(torch.stack([n.cpu(), npseudo, (att / n.clamp(min=1)).cpu(), ntypes,
                                (same / n.clamp(min=1)).cpu()], 1))
    return torch.cat(out).numpy()


def fmt_ns(a):
    m = a.mean(0)
    return (f"neighbour msgs={m[0]:.1f}  pseudonyms={m[1]:.1f}  attacker share={m[2] * 100:.1f}%  "
            f"attack types={m[3]:.2f}  same-type share={m[4] * 100:.1f}%")


def pred_dist(pr, classes, k=4):
    c = np.bincount(pr, minlength=len(classes)) / max(len(pr), 1)
    o = np.argsort(-c)[:k]
    return ", ".join(f"{NAMES.get(classes[i], classes[i])} {c[i] * 100:.1f}%" for i in o)


def part_b(args, info, device):
    classes = info["classes"]
    targets = [classes.index(k) for k in (17, 19, 18, 16) if k in classes]
    seeds = [int(s) for s in args.seeds.split(",")]
    models = {}
    for s in seeds:
        base = os.path.join(args.mixed, f"{args.name}_multi_vehicle_seed{s}")
        if os.path.exists(base + "_weights.pt"):
            models[s] = load_model(info, base, device)
    if not models:
        p("PART B: no models found"); return
    p("\n" + "=" * 100)
    p(f"PART B - Sybil variants on MixAll (vehicle-level labels), {args.name}, seeds {list(models)}")

    # ---- main test set: vehicle labels for every cache row (same row order as step5a) ----
    files = sorted(glob.glob(os.path.join(args.main_features, "*.parquet")))
    lv_all, lv_test = [], []
    for f in files:
        d = pd.read_parquet(f, columns=["label_vehicle", "split"])
        d = d[d["split"].notna()]
        lv_all.append(d["label_vehicle"].to_numpy()); lv_test.append(d.loc[d["split"] == "test", "label_vehicle"].to_numpy())
    yv_all = np.searchsorted(classes, np.concatenate(lv_all)).astype(np.int64)
    yv_test = np.searchsorted(classes, np.concatenate(lv_test)).astype(np.int64)
    split = np.load(os.path.join(args.cache, "split.npy"))
    if len(yv_all) != len(split):
        p(f"! vehicle labels ({len(yv_all):,}) and cache rows ({len(split):,}) differ - main neighbourhood skipped")
        yv_all = None
    main_pred = {}
    for s in models:
        f = os.path.join(args.mixed, f"{args.name}_multi_vehicle_seed{s}_pred.npy")
        if os.path.exists(f):
            pr = np.load(f).astype(np.int64)
            if len(pr) == len(yv_test):
                main_pred[s] = pr
    cfg0 = next(iter(models.values()))[1]
    M, W = cfg0["neighbors"], cfg0["window"]
    rng = np.random.default_rng(0)
    main_ns = {}
    if yv_all is not None:
        data = s5.GpuData(args.cache, device)
        ylab = torch.from_numpy(yv_all).to(device)
        te = np.where(split == 2)[0]
        for c in targets:
            r = te[yv_all[te] == c]
            r = np.sort(rng.choice(r, min(args.n_attack, len(r)), replace=False))
            main_ns[c] = neigh_stats(data, torch.from_numpy(r).to(device), M, W, ylab)
        del data, ylab
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- MixAll ----
    cols = info["features"] + ["aux_pos_x", "aux_pos_y", "label_vehicle", "receiver", "senderPseudo",
                               "rcvTime", "sim_hour"]
    mix_pred = {s: {c: [] for c in targets} for s in models}
    mix_ns = {c: [] for c in targets}
    for f in sorted(glob.glob(os.path.join(args.mixall_features, "*.parquet"))):
        t0 = time.time()
        df = pd.read_parquet(f, columns=cols)
        if len(df) == 0:
            continue
        h = df["sim_hour"].to_numpy()
        ov = np.zeros(len(h), bool)
        for a, b in TRAIN_WINDOWS:
            ov |= (h >= a) & (h < b)
        df["label_msg"] = df["label_vehicle"]
        data, y = make_data(df, info, device)
        for c in targets:
            r = np.where((y == c) & ~ov)[0]
            if not len(r):
                continue
            rt = torch.from_numpy(r).to(device)
            for s, (m, cfg, zf, _) in models.items():
                with Zero(data, zf):
                    pr, _ = run(m, cfg, data, rt)
                mix_pred[s][c].append(pr.argmax(1))
            rs = np.sort(rng.choice(r, min(args.n_attack // 10, len(r)), replace=False))
            mix_ns[c].append(neigh_stats(data, torch.from_numpy(rs).to(device), M, W, data.y))
        del data
        if device.type == "cuda":
            torch.cuda.empty_cache()
        p(f"MixAll {os.path.basename(f)} [{time.time() - t0:.0f}s]")

    for c in targets:
        p(f"\n--- class {classes[c]} {NAMES.get(classes[c], classes[c])} ---")
        mp = [np.concatenate(mix_pred[s][c]) for s in models if mix_pred[s][c]]
        if mp:
            p(f"B1 MixAll  ({len(mp[0]):,} msgs) predicted as: " + pred_dist(np.concatenate(mp), classes))
        if main_pred:
            mm = np.concatenate([pr[yv_test == c] for pr in main_pred.values()])
            p(f"B1 main    ({int((yv_test == c).sum()):,} msgs) predicted as: " + pred_dist(mm, classes))
        if mix_ns[c]:
            p("B2 MixAll  neighbourhood: " + fmt_ns(np.concatenate(mix_ns[c])))
        if c in main_ns:
            p("B2 main    neighbourhood: " + fmt_ns(main_ns[c]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--mixed", required=True, help="results folder with tmix_nodens models and LOAO runs")
    ap.add_argument("--main-features", required=True)
    ap.add_argument("--mixall-features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="tmix_nodens")
    ap.add_argument("--attacks", default="2,16,12")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n-genuine", type=int, default=1_000_000)
    ap.add_argument("--n-ref", type=int, default=300_000)
    ap.add_argument("--n-attack", type=int, default=200_000)
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-b", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = json.load(open(os.path.join(args.cache, "info.json")))
    if not args.skip_a:
        part_a(args, info, device)
    if not args.skip_b:
        part_b(args, info, device)
    out = os.path.join(args.out, "reviewer_analysis.txt")
    open(out, "w", encoding="utf-8").write("\n".join(L))
    print("\nSaved", out)


if __name__ == "__main__":
    main()
