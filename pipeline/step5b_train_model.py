"""
Step 5b - Proposed model: Link-history Transformer + Neighbourhood cross-attention.

For every received message (the "target") the model sees only what the receiver
has observed up to that moment (online, causal):
  1. Link branch   : the last L messages of the SAME pseudonym on this receiver
                     (including the target), each = engineered features + time gap.
                     -> small Transformer encoder, output at the target token.
                     Captures temporal patterns (replay, delays, eventual stop).
  2. Neighbour branch: the last M messages the receiver got from OTHER pseudonyms
                     within W seconds before the target, each = features +
                     position RELATIVE to the target + time gap.
                     -> cross-attention from the target to its neighbours.
                     Captures Sybil ghosts, co-location, DoS load, replay sources.
  3. Fusion MLP -> class probabilities (genuine + attack types).

Evaluation and outputs follow step 4 exactly (same splits, same validation
halving, same JSON format), so all models are directly comparable.

Usage (development, on validation):
    python step5b_train_model.py --cache F:\\dl_cache --out F:\\results_dev --eval-split val --seeds 0
Ablations:
    --history 1        (no link history)     --neighbors 0   (no neighbour branch)
Final (test, 5 seeds) only after all design decisions are frozen:
    python step5b_train_model.py --cache F:\\dl_cache --out F:\\results_final --eval-split test --seeds 0,1,2,3,4
"""

import argparse
import io
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, roc_auc_score)


# --------------------------------------------------------------------------
# Data on the GPU and batch assembly
# --------------------------------------------------------------------------
class GpuData:
    def __init__(self, cache, device):
        def load(name, dtype=None):
            a = np.load(os.path.join(cache, f"{name}.npy"))
            t = torch.from_numpy(a)
            return (t.to(dtype) if dtype is not None else t).to(device)
        self.X = load("X")                       # float16 [N, F]
        self.y = load("y", torch.long)
        self.link_id = load("link_id")
        self.rx_id = load("rx_id")
        self.pseudo = load("pseudo")
        self.time = load("time")
        self.pos = load("pos")
        self.order_link = load("order_link")
        self.rank_link = load("rank_link")
        self.order_rx = load("order_rx")
        self.rank_rx = load("rank_rx")
        self.n, self.f = self.X.shape
        self.device = device

    def batch(self, rows, L, M, W):
        """rows: LongTensor [B] of target rows. Returns model inputs."""
        B = rows.shape[0]
        dev = self.device
        # ---- link history (k = 0 is the target itself) ----
        k = torch.arange(L, device=dev)
        r = self.rank_link[rows].unsqueeze(1) - k                      # [B, L]
        ok = r >= 0
        idx = self.order_link[r.clamp(min=0)]
        ok &= self.link_id[idx] == self.link_id[rows].unsqueeze(1)
        dt = (self.time[rows].unsqueeze(1) - self.time[idx]).clamp(min=0)
        link_tok = torch.cat([self.X[idx].float(), torch.log1p(dt).unsqueeze(-1).float()], -1)
        link_tok = link_tok * ok.unsqueeze(-1)
        if M == 0:
            return link_tok, ok, None, None
        # ---- neighbours: previous M receptions from other pseudonyms within W s ----
        k2 = torch.arange(1, M + 1, device=dev)
        r2 = self.rank_rx[rows].unsqueeze(1) - k2                       # [B, M]
        ok2 = r2 >= 0
        idx2 = self.order_rx[r2.clamp(min=0)]
        ok2 &= self.rx_id[idx2] == self.rx_id[rows].unsqueeze(1)
        dt2 = (self.time[rows].unsqueeze(1) - self.time[idx2])
        ok2 &= (dt2 >= 0) & (dt2 <= W)
        ok2 &= self.pseudo[idx2] != self.pseudo[rows].unsqueeze(1)
        rel = self.pos[idx2] - self.pos[rows].unsqueeze(1)              # [B, M, 2]
        dist = torch.linalg.norm(rel, dim=-1, keepdim=True)
        rel_feats = torch.cat([(rel / 100.0).clamp(-5, 5), torch.log1p(dist) / 5.0,
                               (dt2 / W).unsqueeze(-1)], -1)
        nb_tok = torch.cat([self.X[idx2].float(), rel_feats.float()], -1) * ok2.unsqueeze(-1)
        return link_tok, ok, nb_tok, ok2


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class LinkNeighbourNet(nn.Module):
    def __init__(self, f, n_classes, L, M, d=64, heads=4, layers=2, dropout=0.1, encoder="transformer"):
        super().__init__()
        self.L, self.M, self.encoder = L, M, encoder
        self.link_in = nn.Linear(f + 1, d)
        self.age_emb = nn.Parameter(torch.zeros(L, d))
        if encoder == "lstm":
            # literature-style baseline (e.g. VeMisNet): Bi-LSTM over the link history.
            # The history holds only past messages, so the model stays causal.
            self.lstm = nn.LSTM(d, d // 2, num_layers=layers, batch_first=True,
                                bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        else:
            enc_layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, batch_first=True,
                                                   norm_first=True)
            self.link_enc = nn.TransformerEncoder(enc_layer, layers, enable_nested_tensor=False)
        if M > 0:
            self.nb_in = nn.Sequential(nn.Linear(f + 4, d), nn.GELU(), nn.Linear(d, d))
            self.null_tok = nn.Parameter(torch.zeros(1, 1, d))   # always-valid "nobody" token
            self.cross = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
            self.cross_norm = nn.LayerNorm(d)
        fuse_in = 2 * d if M > 0 else d
        self.head = nn.Sequential(nn.LayerNorm(fuse_in), nn.Linear(fuse_in, d), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d, n_classes))
        nn.init.normal_(self.age_emb, std=0.02)

    def forward(self, link_tok, link_ok, nb_tok=None, nb_ok=None):
        if self.encoder == "lstm":
            h = self.link_in(link_tok)
            # valid history tokens form a prefix (target first, then older messages);
            # feed them oldest -> newest so the target is the last step.
            lengths = link_ok.sum(1).clamp(min=1)
            idx = (lengths.unsqueeze(1) - 1 - torch.arange(h.shape[1], device=h.device)).clamp(min=0)
            h = torch.gather(h, 1, idx.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
            packed = nn.utils.rnn.pack_padded_sequence(h.float(), lengths.cpu(), batch_first=True,
                                                       enforce_sorted=False)
            with torch.autocast(h.device.type, enabled=False):   # cuDNN LSTM: run in fp32
                out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=h.shape[1])
            tgt = out[torch.arange(out.shape[0], device=out.device), lengths - 1]
        else:
            h = self.link_in(link_tok) + self.age_emb
            h = self.link_enc(h, src_key_padding_mask=~link_ok)
            tgt = h[:, 0]                                        # target token
        if self.M == 0:
            return self.head(tgt)
        nb = self.nb_in(nb_tok)
        B = nb.shape[0]
        nb = torch.cat([self.null_tok.expand(B, 1, -1), nb], 1)
        mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=nb.device), ~nb_ok], 1)
        att, _ = self.cross(tgt.unsqueeze(1), nb, nb, key_padding_mask=mask)
        ctx = self.cross_norm(att.squeeze(1) + tgt)
        return self.head(torch.cat([tgt, ctx], -1))


# --------------------------------------------------------------------------
@torch.no_grad()
def predict(model, data, rows, args, bs=8192):
    model.eval()
    out = []
    for i in range(0, len(rows), bs):
        rb = rows[i:i + bs]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=data.device.type == "cuda"):
            logits = model(*data.batch(rb, args.history, args.neighbors, args.window))
        out.append(F.softmax(logits.float(), -1).cpu())
    return torch.cat(out).numpy()


def model_cost(model, data, args, sample_rows):
    """Parameters, size, FLOPs per message, and single-thread CPU latency."""
    params = sum(p.numel() for p in model.parameters())
    buf = io.BytesIO(); torch.save(model.state_dict(), buf)
    inputs = [t.cpu() if t is not None else None
              for t in data.batch(sample_rows[:1024], args.history, args.neighbors, args.window)]
    m = model.cpu().eval()
    flops = None
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc:
            m(*[t[:1] if t is not None else None for t in inputs])
        flops = int(fc.get_total_flops())
    except Exception:
        pass
    torch.set_num_threads(1)
    with torch.no_grad():
        single = [t[:1] if t is not None else None for t in inputs]
        ts = []
        for _ in range(100):
            s = time.perf_counter(); m(*single); ts.append(time.perf_counter() - s)
        tb = []
        for _ in range(10):
            s = time.perf_counter(); m(*inputs); tb.append(time.perf_counter() - s)
    torch.set_num_threads(os.cpu_count() or 1)
    model.to(data.device)
    return {"params": int(params), "model_size_mb": round(buf.getbuffer().nbytes / 1e6, 3),
            "flops_per_msg": flops,
            "latency": {"single_msg_ms": float(np.median(ts) * 1e3),
                        "batch1024_per_msg_us": float(np.median(tb) / 1024 * 1e6)}}


def metrics(yt, proba, classes):
    pred = proba.argmax(1)
    present = np.unique(yt)
    p = proba[:, present]
    p = p / np.clip(p.sum(1, keepdims=True), 1e-12, None)
    names = [str(c) for c in classes]
    yb = (yt != 0).astype(int)
    pb = 1.0 - proba[:, 0]
    return pred, {
        "accuracy": accuracy_score(yt, pred),
        "macro_f1": f1_score(yt, pred, average="macro"),
        "weighted_f1": f1_score(yt, pred, average="weighted"),
        "roc_auc_macro": float(roc_auc_score(yt, p, multi_class="ovr", average="macro", labels=present)),
        "per_class": classification_report(yt, pred, labels=range(len(classes)), target_names=names,
                                           output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(yt, pred, labels=range(len(classes))).tolist(),
        "binary_from_multi": {"accuracy": accuracy_score(yb, (pb >= 0.5).astype(int)),
                              "f1": f1_score(yb, (pb >= 0.5).astype(int)),
                              "roc_auc": float(roc_auc_score(yb, pb))},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-split", choices=["val", "test"], default="val")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--history", type=int, default=16, help="L: messages of the same link")
    ap.add_argument("--neighbors", type=int, default=32, help="M: messages from other pseudonyms")
    ap.add_argument("--window", type=float, default=1.0, help="W: neighbour look-back (s)")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-genuine-train", type=int, default=3_000_000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--name", default="tmodel", help="model name in the output file (e.g. bilstm)")
    ap.add_argument("--encoder", choices=["transformer", "lstm"], default="transformer")
    ap.add_argument("--exclude-attack", type=int, default=-1,
                    help="leave-one-attack-out: attack label removed from training and early stopping")
    ap.add_argument("--train-period", default="", help="train only on this period, e.g. 0709")
    ap.add_argument("--test-period", default="", help="evaluate only on this period, e.g. 1416")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = json.load(open(os.path.join(args.cache, "info.json")))
    classes = info["classes"]
    C = len(classes)
    split = np.load(os.path.join(args.cache, "split.npy"))
    sender = np.load(os.path.join(args.cache, "sender.npy"))
    y_np = np.load(os.path.join(args.cache, "y.npy")).astype(np.int64)
    data = GpuData(args.cache, device)
    print(f"Device={device} rows={data.n:,} features={data.f} classes={classes}")

    # Same validation halving as step4 (early-stopping half / development half)
    val_vehicles = np.unique(sender[split == 1])
    es_vehicles = np.random.default_rng(12345).permutation(val_vehicles)[: len(val_vehicles) // 2]
    half = np.isin(sender, es_vehicles)
    import pandas as pd
    period = pd.read_parquet(os.path.join(args.cache, "meta.parquet"), columns=["period"])["period"]
    period = period.astype(str).str.zfill(4).to_numpy()
    print("Periods in data:", sorted(set(period)))
    for p_ in (args.train_period, args.test_period):
        if p_ and p_ not in set(period):
            raise SystemExit(f"period {p_} not found; available: {sorted(set(period))}")
    keep = np.ones(len(split), bool)                 # rows allowed for training / early stopping
    if args.train_period:
        keep &= period == args.train_period
    kx = -1
    if args.exclude_attack >= 0:
        kx = classes.index(args.exclude_attack)
        keep &= y_np != kx
    rows_es_all = np.where((split == 1) & half & keep)[0]
    rows_eval = np.where(split == 2)[0] if args.eval_split == "test" else np.where((split == 1) & ~half)[0]
    if args.test_period:
        rows_eval = rows_eval[period[rows_eval] == args.test_period]

    for seed in [int(s) for s in args.seeds.split(",")]:
        ev = "" if args.eval_split == "test" else "_val"
        tag = f"{args.name}{args.tag}_multi_vehicle{ev}_seed{seed}"
        if os.path.exists(os.path.join(args.out, tag + ".json")):
            print(f"[skip] {tag}"); continue
        torch.manual_seed(seed); np.random.seed(seed)
        rng = np.random.default_rng(seed)

        tr = np.where((split == 0) & keep)[0]
        gen, att = tr[y_np[tr] == 0], tr[y_np[tr] != 0]
        if len(gen) > args.max_genuine_train:
            gen = rng.choice(gen, args.max_genuine_train, replace=False)
        rows_tr = np.concatenate([gen, att])
        rows_es = rng.choice(rows_es_all, min(500_000, len(rows_es_all)), replace=False)

        counts = np.bincount(y_np[rows_tr], minlength=C).astype(np.float64)
        w = np.sqrt(counts.sum() / (C * np.maximum(counts, 1)))      # sqrt-balanced weights
        weight = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)

        model = LinkNeighbourNet(data.f, C, args.history, args.neighbors, d=args.d,
                                 layers=args.layers, encoder=args.encoder).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        steps = args.epochs * math.ceil(len(rows_tr) / args.batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
        rows_tr_t = torch.from_numpy(rows_tr).to(device)

        print(f"\n=== {tag}: train={len(rows_tr):,} es={len(rows_es):,} eval={len(rows_eval):,} "
              f"params={sum(p.numel() for p in model.parameters()):,}")
        best_f1, best_state, t0 = -1.0, None, time.time()
        for epoch in range(args.epochs):
            model.train()
            perm = rows_tr_t[torch.randperm(len(rows_tr_t), device=device)]
            tot = 0.0
            for i in range(0, len(perm), args.batch):
                rb = perm[i:i + args.batch]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = model(*data.batch(rb, args.history, args.neighbors, args.window))
                    loss = F.cross_entropy(logits.float(), data.y[rb], weight=weight)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step()
                tot += loss.item() * len(rb)
            proba = predict(model, data, torch.from_numpy(rows_es).to(device), args)
            f1 = f1_score(y_np[rows_es], proba.argmax(1), average="macro")
            print(f"epoch {epoch + 1}/{args.epochs} loss={tot / len(perm):.4f} es_macroF1={f1:.4f} "
                  f"({time.time() - t0:.0f}s)")
            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        train_s = time.time() - t0
        model.load_state_dict(best_state)

        proba = predict(model, data, torch.from_numpy(rows_eval).to(device), args)
        yt = y_np[rows_eval]
        pred, res = metrics(yt, proba, classes)
        if kx >= 0:
            g = classes.index(0)
            m = yt == kx
            pk = pred[m]
            res["unseen"] = {"attack": int(args.exclude_attack), "n_test_msgs": int(m.sum()),
                             "detection_rate": float((pk != g).mean()) if m.any() else None,
                             "predicted_as": {str(classes[c]): int(n) for c, n in zip(*np.unique(pk, return_counts=True))},
                             "genuine_fpr": float((pred[yt == g] != g).mean())}
            print("unseen attack:", res["unseen"])
        res.update({"model": args.name + args.tag, "task": "multi", "protocol": "vehicle",
                    "seed": seed, "eval_split": args.eval_split, "classes": [str(c) for c in classes],
                    "features": info["features"], "config": vars(args),
                    "n_train": int(len(rows_tr)), "n_test": int(len(rows_eval)),
                    "train_seconds": round(train_s, 1), "best_es_macro_f1": best_f1})
        res.update(model_cost(model, data, args, torch.from_numpy(rows_eval[:2048]).to(device)))
        with open(os.path.join(args.out, tag + ".json"), "w") as f:
            json.dump(res, f, indent=1)
        np.save(os.path.join(args.out, tag + "_pred.npy"), pred.astype(np.int8))
        torch.save(best_state, os.path.join(args.out, tag + "_weights.pt"))
        print(f"acc={res['accuracy']:.4f} macroF1={res['macro_f1']:.4f} AUC={res['roc_auc_macro']:.4f} "
              f"params={res['params']:,} size={res['model_size_mb']}MB latency={res['latency']}")


if __name__ == "__main__":
    main()
