"""
Step 6c - Permutation importance of the final proposed model (shortcut check).

For each input feature (and each feature group) the values are shuffled across
ALL messages, so every link-history and neighbour token sees the shuffled value,
and the drop in macro-F1 and per-class F1 on a stratified test sample is measured.
Also tested: the neighbours' relative geometry (positions shuffled).
A large drop for a feature means the model relies on it; the question for
the paper is whether EventualStop, GridSybil etc. rely on physically meaningful
evidence (kinematic residuals, co-location) rather than on artefacts.

Model-agnostic and exact for this architecture (unlike gradient SHAP on attention).

Usage:
    python step6_importance.py --cache F:\\dl_cache --results F:\\results_final --out F:\\results_extra --seed 0
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from step5b_train_model import GpuData, LinkNeighbourNet

GROUPS = {"k_ kinematics": "k_", "r_ receiver-relative": "r_", "c_ link consistency": "c_",
          "n_ neighbourhood": "n_", "m_ map plausibility": "m_"}


@torch.no_grad()
def predict(model, data, rows, cfg, bs=16384):
    out = []
    for i in range(0, len(rows), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=data.device.type == "cuda"):
            logits = model(*data.batch(rows[i:i + bs], cfg["history"], cfg["neighbors"], cfg["window"]))
        out.append(logits.float().argmax(-1).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-class", type=int, default=50_000, help="max test messages per class")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--model", default="tmodel")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tag = f"{args.model}_multi_vehicle_seed{args.seed}"
    cfg = json.load(open(os.path.join(args.results, tag + ".json")))["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = json.load(open(os.path.join(args.cache, "info.json")))
    classes, feats = info["classes"], info["features"]
    C = len(classes)
    data = GpuData(args.cache, device)
    model = LinkNeighbourNet(data.f, C, cfg["history"], cfg["neighbors"], d=cfg["d"], layers=cfg["layers"])
    model.load_state_dict(torch.load(os.path.join(args.results, tag + "_weights.pt"), map_location="cpu"))
    model.to(device).eval()
    zf = [feats.index(f) for f in cfg.get("zero_features", "").split(",") if f]
    if zf:                                   # model trained without these inputs
        data.X[:, zf] = 0
        print("inputs set to zero:", [feats[i] for i in zf])

    split = np.load(os.path.join(args.cache, "split.npy"))
    y = np.load(os.path.join(args.cache, "y.npy")).astype(np.int64)
    rng = np.random.default_rng(args.seed)
    test = np.where(split == 2)[0]
    rows_np = np.concatenate([rng.choice(test[y[test] == k], min(args.per_class, (y[test] == k).sum()), replace=False)
                              for k in range(C) if (y[test] == k).any()])
    rows = torch.from_numpy(rows_np).to(device)
    yt = y[rows_np]
    labels = list(range(C))

    def score(pred):
        return f1_score(yt, pred, average="macro"), f1_score(yt, pred, labels=labels, average=None, zero_division=0)

    base_macro, base_pc = score(predict(model, data, rows, cfg))
    print(f"sample={len(rows_np):,} base macroF1={base_macro:.4f}")

    tests = {f: [i] for i, f in enumerate(feats) if i not in zf}
    for g, pre in GROUPS.items():
        idx = [i for i, f in enumerate(feats) if f.startswith(pre) and i not in zf]
        if idx:
            tests["GROUP " + g] = idx
    tests["GROUP neighbour geometry (positions)"] = "pos"

    out = {"seed": args.seed, "n_sample": int(len(rows_np)), "classes": [str(c) for c in classes],
           "base_macro_f1": base_macro, "base_per_class_f1": base_pc.tolist(), "tests": {}}
    for name, cols in tests.items():
        t0 = time.time()
        drops, pc_drops = [], []
        for r in range(args.repeats):
            g = torch.Generator(device=device).manual_seed(1000 * args.seed + r)
            perm = torch.randperm(data.n, device=device, generator=g)
            if cols == "pos":
                saved = data.pos.clone(); data.pos = data.pos[perm]
            else:
                ci = torch.tensor(cols, device=device)
                saved = data.X[:, ci].clone(); data.X[:, ci] = saved[perm]
            m, pc = score(predict(model, data, rows, cfg))
            if cols == "pos":
                data.pos = saved
            else:
                data.X[:, ci] = saved
            del saved, perm
            drops.append(base_macro - m); pc_drops.append(base_pc - pc)
        out["tests"][name] = {"macro_f1_drop": float(np.mean(drops)), "macro_f1_drop_std": float(np.std(drops)),
                              "per_class_f1_drop": np.mean(pc_drops, 0).tolist()}
        top = np.argsort(-np.mean(pc_drops, 0))[:3]
        print(f"{name:45s} drop={np.mean(drops) * 100:6.2f}  most hurt: "
              + ", ".join(f"{classes[k]}({np.mean(pc_drops, 0)[k] * 100:.1f})" for k in top)
              + f"  [{time.time() - t0:.0f}s]", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    with open(os.path.join(args.out, (f"importance_seed{args.seed}.json" if args.model == "tmodel" else f"importance_{args.model}_seed{args.seed}.json")), "w") as f:
        json.dump(out, f, indent=1)
    print("saved", (f"importance_seed{args.seed}.json" if args.model == "tmodel" else f"importance_{args.model}_seed{args.seed}.json"))


if __name__ == "__main__":
    main()
