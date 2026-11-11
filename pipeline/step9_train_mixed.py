"""
Step 9 - Mixed-scene training (fix for the scene-level shortcut found on MixAll).

Problem: every VeReMi Extension simulation contains ONE attack type, so the neighbourhood
branch can learn "which attack is in this scene" instead of "is THIS message wrong".
MixAll, where all attacks co-occur, breaks that cue (macro-F1 95.5 -> 60.3).

Fix (training only; the model and inference are unchanged): all attack folders replay the
same traffic (A4). The same receiver hears the same sender at the same time in every folder;
the sender is genuine in some folders and an attacker (of that folder's type) in others.
While training, each neighbour token is, with probability q, replaced by the version of the
SAME reception (same receiver, same sender, |dt| <= 0.5 s) taken from a RANDOMLY chosen
folder. Each neighbour therefore takes the role it has in a random folder: genuine in most,
an attacker of some type in a few - a realistic mixed scene with a natural attacker share.
  * No labels are used to build the scene; only senders of the TRAINING split are swapped.
  * Only neighbour tokens change; the target message and its own link history do not.
  * Active only while gradients are computed (training steps). Early stopping and every
    evaluation see the real scenes. q is fixed in advance (default 0.5).
  * MixAll is never used for training, tuning or model selection.

Usage (same options as step5b, plus --mix-q):
    python step9_train_mixed.py --cache F:\\hiwa\\dl_cache --out F:\\hiwa\\results_mixed ^
        --eval-split test --seeds 0,1,2 --lr 0.001 --name tmix --mix-q 0.5
"""

import os
import sys

import numpy as np
import torch

import step5b_train_model as s5

MIX_Q = 0.5
for i, a in enumerate(list(sys.argv)):
    if a == "--mix-q":
        MIX_Q = float(sys.argv[i + 1])
        del sys.argv[i:i + 2]
        break

ZERO_FEATURES = []
for i, a in enumerate(list(sys.argv)):
    if a == "--zero-features":              # control run: these inputs are set to 0 everywhere
        ZERO_FEATURES = [f for f in sys.argv[i + 1].split(",") if f]
        del sys.argv[i:i + 2]
        break

TIME_SPAN = 1e5            # simulation times are < 1e5 s
HALF_WIN = 0.5             # versions of one reception differ by < 0.5 s (1 Hz beacons)


class MixGpuData(s5.GpuData):
    def __init__(self, cache, device):
        super().__init__(cache, device)
        if ZERO_FEATURES:
            import json
            feats = json.load(open(os.path.join(cache, "info.json")))["features"]
            idx = [feats.index(f) for f in ZERO_FEATURES]
            self.X[:, idx] = 0
            print("inputs set to zero:", ZERO_FEATURES)
        if not os.path.exists(os.path.join(cache, "gs_id.npy")):
            raise SystemExit("gs_id.npy missing: run step9a_scene_arrays.py first")
        gs = np.load(os.path.join(cache, "gs_id.npy"))
        split = np.load(os.path.join(cache, "split.npy"))
        t = np.load(os.path.join(cache, "time.npy"))
        self.gs = torch.from_numpy(gs).to(device)
        pool = split == 0                                      # messages of TRAINING-split senders
        if "--exclude-attack" in sys.argv:                     # leave-one-attack-out: the withheld
            import json                                        # attack must not enter any scene
            k = int(sys.argv[sys.argv.index("--exclude-attack") + 1])
            if k >= 0:
                classes = json.load(open(os.path.join(cache, "info.json")))["classes"]
                pool &= np.load(os.path.join(cache, "y.npy")) != classes.index(k)
        pool = np.where(pool)[0]
        key = gs[pool].astype(np.float64) * TIME_SPAN + t[pool]
        o = np.argsort(key, kind="stable")
        self.pool_key = torch.from_numpy(key[o]).to(device)
        self.pool_rows = torch.from_numpy(pool[o].astype(np.int64)).to(device)
        del gs, split, t, key, o
        self.n_swapped = 0
        self.n_tokens = 0
        print(f"mixed-scene pool: {len(pool):,} receptions of training-split senders, q={MIX_Q}")

    def batch(self, rows, L, M, W):
        link_tok, ok, nb_tok, ok2 = super().batch(rows, L, M, W)
        if M == 0 or not torch.is_grad_enabled() or MIX_Q <= 0:
            return link_tok, ok, nb_tok, ok2
        dev = self.device
        k2 = torch.arange(1, M + 1, device=dev)                 # same indices as the parent
        r2 = self.rank_rx[rows].unsqueeze(1) - k2
        idx2 = self.order_rx[r2.clamp(min=0)]
        key = self.gs[idx2].double() * TIME_SPAN + self.time[idx2]
        lo = torch.searchsorted(self.pool_key, key - HALF_WIN)
        hi = torch.searchsorted(self.pool_key, key + HALF_WIN, right=True)
        cnt = hi - lo
        j = (lo + (torch.rand(idx2.shape, device=dev) * cnt).long()).clamp(max=len(self.pool_rows) - 1)
        f = self.pool_rows[j]
        swap = ok2 & (cnt > 0) & (torch.rand(idx2.shape, device=dev) < MIX_Q)
        self.n_swapped += int((swap & (f != idx2)).sum()); self.n_tokens += int(ok2.sum())
        if not swap.any():
            return link_tok, ok, nb_tok, ok2
        idx2 = torch.where(swap, f, idx2)
        dt2 = (self.time[rows].unsqueeze(1) - self.time[idx2]).clamp(min=0)
        rel = self.pos[idx2] - self.pos[rows].unsqueeze(1)
        dist = torch.linalg.norm(rel, dim=-1, keepdim=True)
        rel_feats = torch.cat([(rel / 100.0).clamp(-5, 5), torch.log1p(dist) / 5.0,
                               (dt2 / W).clamp(max=1.5).unsqueeze(-1)], -1)
        nb_tok = torch.cat([self.X[idx2].float(), rel_feats.float()], -1) * ok2.unsqueeze(-1)
        return link_tok, ok, nb_tok, ok2


if __name__ == "__main__":
    s5.GpuData = MixGpuData
    if ZERO_FEATURES:                       # record it in the run's config
        _orig = s5.argparse.ArgumentParser.parse_args
        def _parse(self, *a, **k):
            ns = _orig(self, *a, **k)
            ns.zero_features = ",".join(ZERO_FEATURES)
            return ns
        s5.argparse.ArgumentParser.parse_args = _parse
    s5.main()
