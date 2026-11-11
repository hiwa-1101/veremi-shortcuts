"""
Step 7 - Two quick checks (about 10-20 minutes in total, no GPU needed).

(A) Timestamp/simulation-identity check on VeReMi Extension.
    Why does the "literature protocol" XGBoost (absolute position, pseudonym ID,
    receive time) still reach 93.9 macro-F1 under a vehicle-disjoint split?
    Hypothesis: every attack folder is a separate simulation run, so rcvTime and
    pseudonym values identify the run, i.e. the attack type, without looking at
    any misbehaviour. Here we print the rcvTime range of each folder and train
    small trees that see ONLY rcvTime / ONLY senderPseudo / ONLY position.

(B) Structure of the newly downloaded data (MixAll_0024 and VeReMi NextGen):
    lists the archive contents and prints the first records, so the loaders
    for the external tests can be written.

Usage:
    python step7_inspect.py --features F:\\veremi_features_v3 --mixall F:\\mixall_zip ^
                            --nextgen F:\\nextgen_zips --out F:\\results_extra
Writes <out>\\inspect_report.txt
"""

import argparse
import glob
import io
import json
import os
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

LOG = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def part_a(folder):
    p("=== A. Can the run (attack type) be read from non-behavioural fields? ===")
    files = sorted(glob.glob(os.path.join(folder, "*.parquet")))
    cols = ["rcvTime", "senderPseudo", "aux_pos_x", "aux_pos_y", "label_msg", "split", "attack_folder", "period"]
    df = pd.concat([pd.read_parquet(f, columns=cols).assign(file=np.int16(i)) for i, f in enumerate(files)],
                   ignore_index=True)
    df["attack_folder"] = df["attack_folder"].astype("category")
    df = df[df["split"].notna()]
    df["period"] = df["period"].astype(str).str.zfill(4)
    g = df.groupby(["attack_folder", "period", "file"], observed=True)["rcvTime"].agg(["min", "max", "size"])
    p("rcvTime range per simulation file (first 30):")
    p(g.head(30).round(1).to_string())
    starts = g["min"].round(0)
    p(f"files={len(g)}  distinct rounded start times={starts.nunique()}")
    p("Attack folders present per period:")
    for per, sub in df.groupby("period"):
        p(f"  {per}: {sorted(sub['attack_folder'].unique())}")

    tr = (df["split"] == "train").to_numpy()
    te = (df["split"] == "test").to_numpy()
    y = df["label_msg"].to_numpy()
    rng = np.random.default_rng(0)
    tri = np.where(tr)[0]
    tri = rng.choice(tri, min(3_000_000, len(tri)), replace=False)
    tei = np.where(te)[0]
    for name, feats in [("rcvTime only", ["rcvTime"]),
                        ("senderPseudo only", ["senderPseudo"]),
                        ("absolute position only", ["aux_pos_x", "aux_pos_y"]),
                        ("rcvTime + senderPseudo + position", ["rcvTime", "senderPseudo", "aux_pos_x", "aux_pos_y"])]:
        X = df[feats].to_numpy(np.float64)
        clf = DecisionTreeClassifier(max_depth=24, min_samples_leaf=20, random_state=0)
        clf.fit(X[tri], y[tri])
        pred = clf.predict(X[tei])
        att = y[tei] > 0
        # Among ATTACK messages: can the tree name the right attack type?
        type_acc = float((pred[att] == y[tei][att]).mean())
        p(f"[vehicle-disjoint test] {name:36s} macroF1={f1_score(y[tei], pred, average='macro') * 100:6.2f}  "
          f"attack-type accuracy on attack msgs={type_acc * 100:6.2f}")
    # Same, but: can these fields name the attack FOLDER of any message (genuine included)?
    folder = pd.factorize(df["attack_folder"])[0]
    X = df[["rcvTime"]].to_numpy(np.float64)
    clf = DecisionTreeClassifier(max_depth=24, min_samples_leaf=20, random_state=0).fit(X[tri], folder[tri])
    p(f"[vehicle-disjoint test] rcvTime -> attack folder of ANY message: accuracy="
      f"{(clf.predict(X[tei]) == folder[tei]).mean() * 100:.2f}% (chance {100 / (folder.max() + 1):.1f}%)")


def head_bytes(zf, name, n=1500):
    with zf.open(name) as f:
        return f.read(n).decode("utf-8", "replace")


def show_zip(path, max_list=15):
    p(f"\n--- {os.path.basename(path)}  ({os.path.getsize(path) / 1e6:.1f} MB)")
    try:
        zf = zipfile.ZipFile(path)
    except Exception as e:
        p("   cannot open:", e)
        return
    names = zf.namelist()
    files = [n for n in names if not n.endswith("/")]
    exts = pd.Series([os.path.splitext(n)[1] for n in files]).value_counts().to_dict()
    p(f"   entries={len(names)} files={len(files)} by extension={exts}")
    for n in files[:max_list]:
        p("   ", n, zf.getinfo(n).file_size)
    inner = [n for n in files if n.endswith(".zip")]
    data = [n for n in files if not n.endswith(".zip")]
    for n in data[:2]:
        p(f"   head of {n}:")
        p("   " + head_bytes(zf, n).replace("\n", "\n   ")[:1500])
    if inner:
        small = min(inner, key=lambda n: zf.getinfo(n).file_size)
        p(f"   contains {len(inner)} inner zips; testing CRC of all, then opening the smallest: {small}")
        bad = zf.testzip()
        p(f"   CRC test: {'OK' if bad is None else 'CORRUPT, first bad member: ' + bad}")
        izf = zipfile.ZipFile(io.BytesIO(zf.read(small)))
        ifiles = [n for n in izf.namelist() if not n.endswith("/")]
        p(f"   inner files={len(ifiles)}; first: {ifiles[:8]}")
        for n in ifiles[:2]:
            p(f"   head of inner {n}:")
            p("   " + head_bytes(izf, n).replace("\n", "\n   ")[:1500])


def show_json_head(path):
    p(f"\n--- {os.path.basename(path)}  ({os.path.getsize(path) / 1e6:.1f} MB)")
    with open(path, "rb") as f:
        p("   " + f.read(1500).decode("utf-8", "replace").replace("\n", "\n   "))


def part_b(mixall, nextgen):
    p("\n=== B. Structure of the new data ===")
    for z in sorted(glob.glob(os.path.join(mixall, "*.zip"))):
        try:
            show_zip(z)
        except Exception as e:
            p("   ERROR:", repr(e))
    p("\nNextGen files:", len(glob.glob(os.path.join(nextgen, "*"))))
    for z in ["InTAS_urban_2_dataReplay.zip", "InTAS_urban_2.zip"]:
        if os.path.exists(os.path.join(nextgen, z)):
            try:
                show_zip(os.path.join(nextgen, z))
            except Exception as e:
                p("   ERROR:", repr(e))
    for j in ["InTAS_urban_2_Train_groundTruth.json", "InTAS_urban_2_Test_ground_truth.json"]:
        if os.path.exists(os.path.join(nextgen, j)):
            show_json_head(os.path.join(nextgen, j))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--mixall", required=True)
    ap.add_argument("--nextgen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-a", action="store_true", help="only inspect the new data (part B)")
    args = ap.parse_args()
    try:
        part_b(args.mixall, args.nextgen)
    except Exception as e:
        p("PART B ERROR:", repr(e))
    if args.skip_a:
        pass
    else:
      try:
        part_a(args.features)
      except Exception as e:
        p("PART A ERROR:", repr(e))
    path = os.path.join(args.out, "inspect_report_B.txt" if args.skip_a else "inspect_report.txt")
    open(path, "w", encoding="utf-8").write("\n".join(LOG))
    print("\nSaved", path)


if __name__ == "__main__":
    main()
