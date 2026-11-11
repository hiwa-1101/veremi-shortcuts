"""
Step 6b - int8 quantisation of the final proposed model (CPU deployment check).

Loads the saved weights of a final run, applies PyTorch dynamic int8 quantisation
to all Linear layers, and compares fp32 vs int8 on the FULL test split:
macro-F1, accuracy, model size and single-thread CPU latency.
Input batches are built on the GPU (same code as training) and run on the CPU.

Usage:
    python step6_quantize.py --cache F:\\dl_cache --results F:\\results_final --out F:\\results_extra --seed 0
"""

import argparse
import copy
import io
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from step5b_train_model import GpuData, LinkNeighbourNet


def size_mb(model):
    buf = io.BytesIO(); torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


@torch.no_grad()
def run_cpu(model, data, rows, cfg, bs=16384):
    preds = []
    for i in range(0, len(rows), bs):
        inp = [t.cpu().float() if t is not None and t.is_floating_point() else (t.cpu() if t is not None else None)
               for t in data.batch(rows[i:i + bs], cfg["history"], cfg["neighbors"], cfg["window"])]
        preds.append(model(*inp).argmax(-1))
        if (i // bs) % 100 == 0:
            print(f"  {i:,}/{len(rows):,}", flush=True)
    return torch.cat(preds).numpy()


@torch.no_grad()
def latency(model, data, rows, cfg):
    inp = [t.cpu().float() if t is not None and t.is_floating_point() else (t.cpu() if t is not None else None)
           for t in data.batch(rows[:1024], cfg["history"], cfg["neighbors"], cfg["window"])]
    torch.set_num_threads(1)
    single = [t[:1] if t is not None else None for t in inp]
    for _ in range(10):
        model(*single)
    ts = []
    for _ in range(200):
        s = time.perf_counter(); model(*single); ts.append(time.perf_counter() - s)
    tb = []
    for _ in range(10):
        s = time.perf_counter(); model(*inp); tb.append(time.perf_counter() - s)
    torch.set_num_threads(os.cpu_count() or 1)
    return {"single_msg_ms": float(np.median(ts) * 1e3),
            "batch1024_per_msg_us": float(np.median(tb) / 1024 * 1e6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="tmodel")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    try:
        torch.backends.mha.set_fastpath_enabled(False)   # quantised layers cannot use the fused path
    except Exception:
        pass

    tag = f"{args.model}_multi_vehicle_seed{args.seed}"
    res0 = json.load(open(os.path.join(args.results, tag + ".json")))
    cfg = res0["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = json.load(open(os.path.join(args.cache, "info.json")))
    C = len(info["classes"])
    data = GpuData(args.cache, device)
    split = np.load(os.path.join(args.cache, "split.npy"))
    y = np.load(os.path.join(args.cache, "y.npy")).astype(np.int64)
    rows_np = np.where(split == 2)[0]
    rows = torch.from_numpy(rows_np).to(device)
    yt = y[rows_np]
    zf = [info["features"].index(f) for f in cfg.get("zero_features", "").split(",") if f]
    if zf:
        data.X[:, zf] = 0

    fp32 = LinkNeighbourNet(data.f, C, cfg["history"], cfg["neighbors"], d=cfg["d"], layers=cfg["layers"])
    fp32.load_state_dict(torch.load(os.path.join(args.results, tag + "_weights.pt"), map_location="cpu"))
    fp32.eval()
    int8 = torch.ao.quantization.quantize_dynamic(copy.deepcopy(fp32), {nn.Linear}, dtype=torch.qint8)
    int8.eval()

    out = {"seed": args.seed, "n_test": int(len(rows_np))}
    for name, m in (("fp32_cpu", fp32), ("int8_cpu", int8)):
        print(f"== {name}: predicting {len(rows_np):,} test messages on CPU")
        t0 = time.time()
        pred = run_cpu(m, data, rows, cfg)
        out[name] = {"macro_f1": f1_score(yt, pred, average="macro"),
                     "accuracy": accuracy_score(yt, pred),
                     "per_class_f1": f1_score(yt, pred, average=None).tolist(),
                     "model_size_mb": size_mb(m),
                     "latency": latency(m, data, rows, cfg),
                     "total_seconds": time.time() - t0}
        if name == "int8_cpu":
            out[name]["agreement_with_fp32"] = float((pred == out["_p"]).mean())
        else:
            out["_p"] = pred
        print(f"   macroF1={out[name]['macro_f1']:.4f} size={out[name]['model_size_mb']:.3f}MB "
              f"latency={out[name]['latency']}")
    out.pop("_p")
    with open(os.path.join(args.out, (f"quant_int8_seed{args.seed}.json" if args.model == "tmodel" else f"quant_int8_{args.model}_seed{args.seed}.json")), "w") as f:
        json.dump(out, f, indent=1)
    print("saved", (f"quant_int8_seed{args.seed}.json" if args.model == "tmodel" else f"quant_int8_{args.model}_seed{args.seed}.json"))


if __name__ == "__main__":
    main()
