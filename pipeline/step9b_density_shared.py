"""Shared-class macro-F1 for the traffic-period transfer runs (classes present in BOTH the
training and the test period; finding A10). Usage:
    python step9b_density_shared.py F:\\hiwa\\results_mixed F:\\hiwa\\results_extra
"""
import json
import os
import sys

import numpy as np

absent = {"0709": {2, 16}, "1416": {8}}      # classes without labelled messages in a period (A10/A9)
for folder in sys.argv[1:]:
    for m in ["tmix_nodens", "tmix", "tmodel", "xgb"]:
        for a, b in [("0709", "1416"), ("1416", "0709")]:
            fc = os.path.join(folder, f"{m}_d{a}to{b}_multi_vehicle_seed0.json")
            fi = os.path.join(folder, f"{m}_d{b}to{b}_multi_vehicle_seed0.json")
            if not (os.path.exists(fc) and os.path.exists(fi)):
                continue
            rc, ri = json.load(open(fc)), json.load(open(fi))
            cls = [c for c in range(20) if c not in absent[a] | absent[b]]
            f1 = lambda r: np.mean([r["per_class"][str(c)]["f1-score"] for c in cls]) * 100
            print(f"{m:6s} {a}->{b}: shared-class macroF1 cross={f1(rc):.2f} in-domain={f1(ri):.2f} "
                  f"drop={f1(ri) - f1(rc):.2f} | binF1 cross={rc['binary_from_multi']['f1'] * 100:.2f}")
