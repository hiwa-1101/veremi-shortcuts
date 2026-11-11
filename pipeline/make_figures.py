"""
make_figures.py - Publication figures (PDF vector + PNG 600 dpi) and tables
(CSV + LaTeX) for the paper, generated directly from the experiment outputs.

Every figure is built only from files you actually produced; anything whose
input is missing is skipped with a message, so the script can be re-run as
more results arrive (dev results now, final 5-seed test results later).

Usage:
    python make_figures.py --results F:\\results_dev_v3 F:\\results_dev_lr1e3 ^
        --eval-split val --counts step3_class_split_counts.csv ^
        --audit step2c_audit.json --out F:\\paper_figures

Figures
  fig_framework          study pipeline (audit -> protocol -> features -> model -> evaluation)
  fig_architecture       proposed link-neighbourhood model
  fig_class_distribution messages per class (log scale)
  fig_audit_rate         median message rate per link and class (data audit)
  fig_per_class_f1       per-class F1, baseline vs proposed (dumbbell)
  fig_confusion          row-normalised confusion matrix of the proposed model
  fig_ablation           macro-F1 of model variants
  fig_efficiency         model size vs macro-F1 (log x)
  fig_leakage            per-class F1, random vs vehicle split (needs random-protocol runs)
Tables
  tab_main_results, tab_per_class_f1, tab_ablation  (.csv and .tex)
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# --------------------------------------------------------------------------
# Style: IEEE column widths, small serif type, recessive axes, fixed colours
# --------------------------------------------------------------------------
COL1, COL2 = 3.5, 7.16          # inches: single / double column
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# Colour follows the model, never its rank (validated categorical slots 1-4)
MODEL_COLOR = {"proposed": "#2a78d6", "xgb": "#eb6834", "lgbm": "#1baf7a", "rf": "#eda100"}
SEQ = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "TeX Gyre Termes", "Liberation Serif", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})

CLASS_NAMES = {
    0: "Genuine", 1: "ConstPos", 2: "ConstPosOffset", 3: "RandomPos", 4: "RandomPosOffset",
    5: "ConstSpeed", 6: "ConstSpeedOffset", 7: "RandomSpeed", 8: "RandomSpeedOffset",
    9: "EventualStop", 10: "Disruptive", 11: "DataReplay", 12: "DelayedMessages", 13: "DoS",
    14: "DoSRandom", 15: "DoSDisruptive", 16: "GridSybil", 17: "DataReplaySybil",
    18: "DoSRandomSybil", 19: "DoSDisruptiveSybil",
}
MAIN = "tmix_nodens"                  # final model: mixed-scene training, no density inputs
DISPLAY = {"tmix_nodens": "Proposed", "tmix": "Mixed-scene, with density inputs",
           "tmodel": "Original training", "bilstm": "Bi-LSTM", "xgb": "XGBoost", "lgbm": "LightGBM", "rf": "Random Forest",
           "tmodel_noHist": "w/o link history (L=1)", "tmodel_noNeigh": "w/o neighbourhood (M=0)",
           "tmodel_M64W2": "M=64, W=2 s", "tmodel_L32": "L=32",
           "xgb_v1feat": "XGBoost w/o map & replay features"}


ABLATIONS = ["tmodel", "tmodel_noHist", "tmodel_noNeigh", "tmodel_M64W2", "tmodel_L32"]


def is_aux(model):
    """leave-one-attack-out, period-transfer and literature-protocol runs are not main results."""
    return "_loao" in model or "_d0" in model or "_d1" in model or model.startswith("xgb_")


def family(model):
    if model.startswith("tmodel") or model.startswith("tmix"):
        return "proposed"
    return model.split("_")[0]


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=600)
    plt.close(fig)
    print(f"  saved {name}.pdf / .png")


# --------------------------------------------------------------------------
# Loading results
# --------------------------------------------------------------------------
def load_results(dirs, eval_split):
    rows = []
    for d in dirs:
        for f in glob.glob(os.path.join(d, "*.json")):
            try:
                r = json.load(open(f))
            except Exception:
                continue
            if "per_class" not in r or r.get("eval_split", "test") != eval_split:
                continue
            r["_file"] = f
            rows.append(r)
    # the same configuration found in several folders: keep the newest file
    latest = {}
    for r in rows:
        key = (r["model"], r["protocol"], r["task"], r["seed"])
        if key not in latest or os.path.getmtime(r["_file"]) > os.path.getmtime(latest[key]["_file"]):
            latest[key] = r
    return list(latest.values())


def per_class_f1(r):
    return {int(c): r["per_class"][c]["f1-score"] for c in r["classes"]}


def mean_std(values):
    v = np.asarray(values, dtype=float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else np.nan)


def fmt(m, s, pct=True):
    k = 100 if pct else 1
    return f"{m * k:.2f}" if np.isnan(s) else f"{m * k:.2f} ± {s * k:.2f}"


# --------------------------------------------------------------------------
# Diagrams
# --------------------------------------------------------------------------
def box(ax, x, y, w, h, text, fill="#f4f3ef", edge=INK2, bold=False, size=7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015",
                                fc=fill, ec=edge, lw=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                 lw=0.8, color=INK2))


def fig_framework(out):
    fig, ax = plt.subplots(figsize=(COL2, 1.55))
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(0, 1); ax.axis("off")
    steps = [
        ("VeReMi\nExtension\n19 attacks,\n74.5 M msgs", "#f4f3ef"),
        ("Data audit\nreuse, labels,\nshortcuts\n(A1\u2013A12)", "#e6effa"),
        ("Leakage-free\nprotocol\nvehicle-disjoint,\nobservable inputs", "#e6effa"),
        ("Link\u2013\nneighbourhood\nTransformer\n(0.1 M params)", "#e6effa"),
        ("Mixed-scene\ntraining\n(removes the\nscene shortcut)", "#e6effa"),
        ("Evaluation\nin-distribution,\nunseen attacks,\nMixAll (external)", "#f4f3ef"),
    ]
    n, gap = len(steps), 0.035
    w = (1 - gap * (n - 1)) / n
    for i, (t, c) in enumerate(steps):
        x = i * (w + gap)
        box(ax, x, 0.20, w, 0.72, t, fill=c, size=6.3)
        if i < n - 1:
            arrow(ax, x + w + 0.004, 0.56, x + w + gap - 0.004, 0.56)
    ax.text(0.5, 0.03, "Shaded blue: contributions of this work", ha="center", fontsize=6.3, color=INK2)
    save(fig, out, "fig_framework")


def fig_architecture(out):
    fig, ax = plt.subplots(figsize=(COL2, 3.0))
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.13, 1); ax.axis("off")
    # inputs
    box(ax, 0.005, 0.60, 0.215, 0.30, "Link history\nlast L=16 messages\nof the same pseudonym\n(24 features + time gap)", size=6.2)
    box(ax, 0.005, 0.08, 0.215, 0.30, "Neighbourhood\nlast M=32 messages from\nother pseudonyms (W=1 s)\n(features + rel. position, dt)", size=6.2)
    # training-time augmentation
    box(ax, 0.005, -0.125, 0.62, 0.13, "Training only: each neighbour is replaced (p = 0.5) by the same reception\n"
        "from a random attack folder; evaluation and inference use the real scene", fill="#fdf1e6",
        edge=MODEL_COLOR["xgb"], size=5.6)
    arrow(ax, 0.11, 0.025, 0.11, 0.08)
    # link branch
    box(ax, 0.255, 0.66, 0.13, 0.18, "Linear +\nage embedding", fill="#e6effa", size=6.2)
    box(ax, 0.415, 0.66, 0.19, 0.18, "Transformer encoder\n2 layers, d=64,\n4 heads, pre-norm", fill="#e6effa", size=6.2)
    box(ax, 0.635, 0.66, 0.10, 0.18, "Target\ntoken", fill="#e6effa", size=6.2)
    # neighbour branch
    box(ax, 0.255, 0.14, 0.13, 0.18, "MLP +\nnull token", fill="#e6effa", size=6.2)
    box(ax, 0.475, 0.14, 0.26, 0.18, "Cross-attention\nquery = target token\nkeys / values = neighbours", fill="#e6effa", size=6.2)
    # fusion / head
    box(ax, 0.775, 0.40, 0.095, 0.22, "Concat\n+ MLP", fill="#e6effa", size=6.2)
    box(ax, 0.895, 0.40, 0.10, 0.22, "Softmax\n20 classes", size=6.2)
    for a in [(0.22, 0.75, 0.255, 0.75), (0.385, 0.75, 0.415, 0.75), (0.605, 0.75, 0.635, 0.75),
              (0.22, 0.23, 0.255, 0.23), (0.385, 0.23, 0.475, 0.23), (0.685, 0.66, 0.62, 0.32),
              (0.735, 0.75, 0.79, 0.62), (0.735, 0.23, 0.79, 0.40), (0.87, 0.51, 0.895, 0.51)]:
        arrow(ax, *a)
    ax.text(0.665, 0.47, "query", fontsize=6, color=INK2)
    ax.text(0.5, 0.96, "Causal / online: only messages received before the target are used",
            ha="center", fontsize=6.5, color=INK2)
    save(fig, out, "fig_architecture")


def fig_mixed_scene(out):
    """Why mixed-scene training: one attack type per simulation vs. a realistic mixed scene."""
    rng = np.random.default_rng(3)
    fig, axs = plt.subplots(1, 2, figsize=(COL2, 2.0))
    pts = rng.uniform(-1, 1, (26, 2)); pts = pts[np.linalg.norm(pts, axis=1) < 0.95][:20]
    att = np.zeros(len(pts), int); att[[1, 5, 9, 13, 17]] = 1
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8f63c9"]
    titles = ["(a) VeReMi Extension: one attack type per simulation",
              "(b) Mixed-scene training / MixAll: attack types co-occur"]
    for k, ax in enumerate(axs):
        ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.2, 1.15)
        ax.add_patch(plt.Circle((0, 0), 1.0, fc="none", ec=GRID, lw=0.8, ls="--"))
        for i, (x, y) in enumerate(pts):
            if att[i]:
                c = pal[1] if k == 0 else pal[1 + (i // 4) % 4]
                ax.scatter(x, y, s=36, color=c, edgecolor="white", linewidth=0.6, zorder=3, marker="^")
            else:
                ax.scatter(x, y, s=22, color="#bfbdb7", edgecolor="white", linewidth=0.5, zorder=2)
        ax.scatter(0, 0, s=70, color=INK, marker="s", zorder=4)
        ax.text(0, -0.17, "receiver", ha="center", va="top", fontsize=6, color=INK)
        ax.text(0, 1.1, titles[k], ha="center", fontsize=6.6, color=INK)
    axs[0].text(0, -1.17, "The neighbourhood reveals which attack is \"in the scene\" \u2192 shortcut",
                ha="center", fontsize=6, color=INK2)
    axs[1].text(0, -1.17, "Each neighbour takes its role from a random folder (finding A4)",
                ha="center", fontsize=6, color=INK2)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", ls="", mfc="#bfbdb7", mec="white", label="genuine sender"),
                        Line2D([], [], marker="^", ls="", mfc=pal[1], mec="white", label="attacker (colour = attack type)")],
               loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=6)
    save(fig, out, "fig_mixed_scene")


# --------------------------------------------------------------------------
# Data figures
# --------------------------------------------------------------------------
def fig_class_distribution(counts_csv, out):
    if not counts_csv or not os.path.exists(counts_csv):
        print("  skip fig_class_distribution (no counts file)"); return
    c = pd.read_csv(counts_csv)
    c["total"] = c[[x for x in ("train", "val", "test") if x in c.columns]].sum(axis=1)
    c["name"] = c["label_msg"].map(CLASS_NAMES)
    c = c.sort_values("total")
    fig, ax = plt.subplots(figsize=(COL1, 0.17 * len(c) + 0.5))
    ax.barh(c["name"], c["total"], color=MODEL_COLOR["proposed"], height=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Messages (log scale)")
    ax.grid(axis="x"); ax.set_axisbelow(True)
    for y, v in enumerate(c["total"]):
        ax.text(v * 1.08, y, f"{v / 1e6:.2f} M" if v >= 1e6 else f"{v / 1e3:.0f} k",
                va="center", fontsize=6, color=INK2)
    ax.set_xlim(right=c["total"].max() * 4)
    save(fig, out, "fig_class_distribution")


def fig_audit_rate(audit_json, out):
    if not audit_json or not os.path.exists(audit_json):
        print("  skip fig_audit_rate (no audit file)"); return
    rate = json.load(open(audit_json)).get("E_rate", {})
    rows = []
    for k, v in rate.items():
        if v is None:
            continue
        folder, lab = k.split("|")
        rows.append((int(lab), v["median_msgs_per_s"]))
    df = pd.DataFrame(rows, columns=["label", "rate"]).groupby("label")["rate"].median().reset_index()
    df["name"] = df["label"].map(CLASS_NAMES)
    df = df.sort_values("rate")
    fig, ax = plt.subplots(figsize=(COL1, 0.17 * len(df) + 0.5))
    colors = [INK2 if l == 0 else MODEL_COLOR["proposed"] for l in df["label"]]
    ax.barh(df["name"], df["rate"], color=colors, height=0.7)
    gen = df.loc[df["label"] == 0, "rate"]
    if len(gen):
        ax.axvline(gen.iloc[0], color=INK2, lw=0.6, ls="--")
    ax.set_xlabel("Median messages per second per (receiver, pseudonym) link")
    ax.grid(axis="x"); ax.set_axisbelow(True)
    for y, v in enumerate(df["rate"]):
        ax.text(v + 0.03, y, f"{v:.2f}", va="center", fontsize=6, color=INK2)
    save(fig, out, "fig_audit_rate")


# --------------------------------------------------------------------------
# Result figures and tables
# --------------------------------------------------------------------------
def avg_per_class(runs):
    f = pd.DataFrame([per_class_f1(r) for r in runs])
    return f.mean(), f.std(ddof=1) if len(runs) > 1 else f.mean() * np.nan


def fig_per_class_f1(res, out, base="xgb", main=MAIN):
    b = [r for r in res if r["model"] == base and r["protocol"] == "vehicle" and r["task"] == "multi"]
    m = [r for r in res if r["model"] == main and r["protocol"] == "vehicle" and r["task"] == "multi"]
    if not b or not m:
        print(f"  skip fig_per_class_f1 (need {base} and {main})"); return
    fb, _ = avg_per_class(b)
    fm, _ = avg_per_class(m)
    cls = sorted(set(fb.index) & set(fm.index), key=lambda c: fm[c] - fb[c], reverse=True)
    y = np.arange(len(cls))
    fig, ax = plt.subplots(figsize=(COL1, 0.17 * len(cls) + 0.6))
    for i, c in enumerate(cls):
        ax.plot([fb[c], fm[c]], [i, i], color=GRID, lw=2, zorder=1)
    ax.scatter([fb[c] for c in cls], y, s=18, color=MODEL_COLOR["xgb"], zorder=2,
               edgecolor="white", linewidth=0.6, label=DISPLAY.get(base, base))
    ax.scatter([fm[c] for c in cls], y, s=18, color=MODEL_COLOR["proposed"], zorder=3,
               edgecolor="white", linewidth=0.6, label=DISPLAY.get(main, main))
    ax.set_yticks(y, [CLASS_NAMES[c] for c in cls])
    ax.set_xlabel("F1-score")
    lo = min(fb.min(), fm.min())
    ax.set_xlim(max(0, np.floor(lo * 20) / 20 - 0.02), 1.005)
    ax.grid(axis="x"); ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handletextpad=0.2, borderaxespad=0.2)
    ax.invert_yaxis()
    save(fig, out, "fig_per_class_f1")


def fig_confusion(res, out, main=MAIN):
    m = [r for r in res if r["model"] == main and r["protocol"] == "vehicle" and r["task"] == "multi"]
    if not m:
        print("  skip fig_confusion"); return
    cm = np.sum([np.array(r["confusion_matrix"], dtype=float) for r in m], axis=0)
    cmn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    labels = [CLASS_NAMES[int(c)] for c in m[0]["classes"]]
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    fig, ax = plt.subplots(figsize=(COL2 * 0.62, COL2 * 0.58))
    im = ax.imshow(cmn, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cmn[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v * 100:.0f}" if v >= 0.095 else f"{v * 100:.1f}",
                        ha="center", va="center", fontsize=4.6,
                        color="white" if v > 0.55 else INK)
    ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Share of true class (%)"); cb.ax.tick_params(labelsize=6)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1], labels=["0", "25", "50", "75", "100"])
    save(fig, out, "fig_confusion")


def fig_ablation(res, out):
    abl = [r for r in res if r["model"] in ABLATIONS and r["protocol"] == "vehicle"]
    if len({r["model"] for r in abl}) < 2:
        print("  skip fig_ablation (need >= 2 model variants)"); return
    df = pd.DataFrame([{"model": r["model"], "f1": r["macro_f1"]} for r in abl])
    agg = df.groupby("model")["f1"].agg(["mean", "std", "count"]).reset_index()
    agg["name"] = agg["model"].map(lambda x: DISPLAY.get(x, x))
    agg = agg.sort_values("mean")
    full = agg.loc[agg["model"] == "tmodel", "mean"]
    fig, ax = plt.subplots(figsize=(COL1, 0.28 * len(agg) + 0.5))
    colors = [MODEL_COLOR["proposed"] if mdl == "tmodel" else "#9ec5f4" for mdl in agg["model"]]
    ax.barh(agg["name"], agg["mean"] * 100, color=colors, height=0.6,
            xerr=agg["std"].fillna(0) * 100 if agg["count"].max() > 1 else None,
            error_kw={"lw": 0.6, "capsize": 2, "ecolor": INK2})
    lo = agg["mean"].min() * 100
    ax.set_xlim(max(0, np.floor(lo / 5) * 5 - 2), 100)
    for y, (v, s, mdl) in enumerate(zip(agg["mean"], agg["std"].fillna(0), agg["model"])):
        d = "" if mdl == "tmodel" or not len(full) else f"  ({(v - full.iloc[0]) * 100:+.1f})"
        ax.text((v + s) * 100 + 0.4, y, f"{v * 100:.1f}{d}", va="center", fontsize=6, color=INK2)
    ax.set_xlabel("Macro-F1 (%)")
    ax.grid(axis="x"); ax.set_axisbelow(True)
    save(fig, out, "fig_ablation")


def fig_efficiency(res, out):
    rows = []
    for r in res:
        if r["protocol"] != "vehicle" or r["task"] != "multi" or ("_" in r["model"] and r["model"] != MAIN) \
                or r["model"] in ("tmix", "tmodel"):
            continue
        size = r.get("model_size_mb")
        lat = r.get("latency", {}).get("single_msg_ms")
        if size:
            rows.append({"model": r["model"], "size": size, "f1": r["macro_f1"], "lat": lat})
    if len({x["model"] for x in rows}) < 2:
        print("  skip fig_efficiency (need >= 2 models)"); return
    df = pd.DataFrame(rows).groupby("model").mean().reset_index()
    fig, ax = plt.subplots(figsize=(COL1, 2.2))
    for _, r in df.iterrows():
        c = MODEL_COLOR.get(family(r["model"]), INK2)
        ax.scatter(r["size"], r["f1"] * 100, s=40, color=c, edgecolor="white", linewidth=0.8, zorder=3)
        lab = DISPLAY.get(r["model"], r["model"])
        if r["lat"] is not None:
            lab += f"\n{r['size']:.2f} MB, {r['lat']:.2f} ms/msg"
        right = r["size"] > np.sqrt(df["size"].min() * df["size"].max())
        dy = {"rf": 9, "lgbm": -9}.get(r["model"], 0)
        ax.annotate(lab, (r["size"], r["f1"] * 100), xytext=(-7 if right else 7, dy),
                    textcoords="offset points", fontsize=6, color=INK, va="center",
                    ha="right" if right else "left")
    ax.set_xscale("log")
    ax.set_xlabel("Model size (MB, log scale)")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_xlim(df["size"].min() / 4, df["size"].max() * 4)
    lo, hi = df["f1"].min() * 100, df["f1"].max() * 100
    pad = max(0.5, (hi - lo) * 0.15)
    ax.set_ylim(lo - 2.2 * pad, hi + pad)
    ax.grid(True, which="major"); ax.set_axisbelow(True)
    ax.text(0.0, -0.32, "Latency: single message, one CPU thread.", transform=ax.transAxes,
            fontsize=6, color=INK2)
    save(fig, out, "fig_efficiency")


def fig_leakage(res, out, model="xgb"):
    v = [r for r in res if r["model"] == model and r["protocol"] == "vehicle" and r["task"] == "multi"]
    rd = [r for r in res if r["model"] == model and r["protocol"] == "random" and r["task"] == "multi"]
    if not v or not rd:
        print(f"  skip fig_leakage (need {model} with both protocols on this split)"); return
    fv, _ = avg_per_class(v)
    fr, _ = avg_per_class(rd)
    cls = sorted(set(fv.index) & set(fr.index), key=lambda c: fr[c] - fv[c], reverse=True)
    y = np.arange(len(cls))
    fig, ax = plt.subplots(figsize=(COL1, 0.17 * len(cls) + 0.6))
    for i, c in enumerate(cls):
        ax.plot([fv[c], fr[c]], [i, i], color=GRID, lw=2, zorder=1)
    ax.scatter([fv[c] for c in cls], y, s=18, color=MODEL_COLOR["proposed"], zorder=3,
               edgecolor="white", linewidth=0.6, label="Vehicle-disjoint split (ours)")
    ax.scatter([fr[c] for c in cls], y, s=18, color=INK2, zorder=2, marker="D",
               edgecolor="white", linewidth=0.6, label="Random message split")
    ax.set_yticks(y, [CLASS_NAMES[c] for c in cls])
    ax.set_xlabel(f"F1-score ({DISPLAY.get(model, model)})")
    ax.grid(axis="x"); ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, handletextpad=0.2, borderaxespad=0.2)
    ax.invert_yaxis()
    save(fig, out, "fig_leakage")


MIX_ORDER = [("tmodel", "Original training"), ("tmodel_noNeigh", "Original, w/o neighbourhood"),
             ("tmodel_noHist", "Original, w/o link history"), ("bilstm", "Bi-LSTM"), ("xgb", "XGBoost"),
             ("tmix", "Mixed-scene, with density inputs"), ("tmix_nodens", "Proposed (mixed-scene)")]
XGB_MIXALL = {"main": 0.8935, "mixall": 0.7841}      # from run_mixall_xgb (vehicle-level labels)


def fig_mixall(mix_json, out):
    if not mix_json or not os.path.exists(mix_json):
        print("  skip fig_mixall (no mixall_variants.json)"); return
    v = json.load(open(mix_json))["variants"]
    rows = []
    for k, lab in MIX_ORDER:
        if k == "xgb":
            rows.append((lab, XGB_MIXALL["main"], XGB_MIXALL["mixall"], k))
        elif k in v:
            rows.append((lab, v[k]["main"]["macro_f1"], v[k]["mixall"]["macro_f1"], k))
    fig, ax = plt.subplots(figsize=(COL1, 0.27 * len(rows) + 0.7))
    for i, (lab, a, b, k) in enumerate(rows):
        c = MODEL_COLOR["proposed"] if k == MAIN else (MODEL_COLOR["xgb"] if k == "xgb" else INK2)
        ax.plot([b * 100, a * 100], [i, i], color=GRID, lw=2.2, zorder=1)
        ax.scatter(a * 100, i, s=20, color="white", edgecolor=c, linewidth=1.0, zorder=3)
        ax.scatter(b * 100, i, s=22, color=c, edgecolor="white", linewidth=0.6, zorder=3)
        ax.text(b * 100 - 1.2, i, f"{b * 100:.1f}", ha="right", va="center", fontsize=5.8, color=c)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Macro-F1 (%), vehicle-level labels")
    ax.set_xlim(25, 100)
    ax.grid(axis="x"); ax.set_axisbelow(True)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, label="Main test set"),
                       Line2D([], [], marker="o", ls="", mfc=INK2, mec="white", label="MixAll (external)")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handletextpad=0.2, borderaxespad=0.2)
    save(fig, out, "fig_mixall")


def fig_loao(dirs, out, a="tmodel", b=MAIN):
    def load(m):
        d = {}
        for folder in dirs:
            for k in range(1, 20):
                f = os.path.join(folder, f"{m}_loao{k}_multi_vehicle_seed0.json")
                if os.path.exists(f):
                    d[k] = json.load(open(f))["unseen"]["detection_rate"]
        return d
    da, db = load(a), load(b)
    ks = [k for k in range(1, 20) if k in da and k in db]
    if not ks:
        print("  skip fig_loao (no leave-one-attack-out results for both models)"); return
    ks = sorted(ks, key=lambda k: db[k])
    fig, ax = plt.subplots(figsize=(COL1, 0.17 * len(ks) + 0.6))
    for i, k in enumerate(ks):
        ax.plot([da[k] * 100, db[k] * 100], [i, i], color=GRID, lw=2, zorder=1)
    ax.scatter([da[k] * 100 for k in ks], range(len(ks)), s=16, color=INK2, marker="D", zorder=2,
               edgecolor="white", linewidth=0.5, label=DISPLAY.get(a, a))
    ax.scatter([db[k] * 100 for k in ks], range(len(ks)), s=18, color=MODEL_COLOR["proposed"], zorder=3,
               edgecolor="white", linewidth=0.6, label=DISPLAY.get(b, b))
    ax.set_yticks(range(len(ks)), [CLASS_NAMES[k] for k in ks])
    ax.set_xlabel("Withheld attack detected as an attack (%)")
    ax.set_xlim(0, 101)
    ax.grid(axis="x"); ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handletextpad=0.2, borderaxespad=0.2)
    save(fig, out, "fig_loao")


def fig_importance(dirs, out, model=MAIN, top=14):
    f = None
    for folder in dirs:
        for name in (f"importance_{model}_seed0.json", "importance_seed0.json" if model == "tmodel" else ""):
            if name and os.path.exists(os.path.join(folder, name)):
                f = os.path.join(folder, name)
    if not f:
        print(f"  skip fig_importance (no importance file for {model})"); return
    t = json.load(open(f))["tests"]
    # a group with a single member duplicates that feature: drop the group entry
    t = {k: v for k, v in t.items() if not (k.startswith("GROUP m_"))}
    items = sorted(t.items(), key=lambda x: -x[1]["macro_f1_drop"])[:top][::-1]
    fig, ax = plt.subplots(figsize=(COL1, 0.17 * len(items) + 0.5))
    names = [k.replace("GROUP ", "[group] ") for k, _ in items]
    vals = [x["macro_f1_drop"] * 100 for _, x in items]
    cols = ["#9ec5f4" if k.startswith("GROUP") else MODEL_COLOR["proposed"] for k, _ in items]
    ax.barh(names, vals, color=cols, height=0.65,
            xerr=[x["macro_f1_drop_std"] * 100 for _, x in items], error_kw={"lw": 0.5, "ecolor": INK2})
    for y, v in enumerate(vals):
        ax.text(v + 0.6, y, f"{v:.1f}", va="center", fontsize=5.8, color=INK2)
    ax.set_xlabel("Macro-F1 drop when shuffled (points)")
    ax.grid(axis="x"); ax.set_axisbelow(True)
    save(fig, out, "fig_importance")


def write_table(df, out, name, caption, label):
    df.to_csv(os.path.join(out, f"{name}.csv"), index=False)
    tex = df.to_latex(index=False, escape=True, column_format="l" + "c" * (df.shape[1] - 1))
    tex = ("\\begin{table}[t]\n\\centering\n\\caption{" + caption + "}\n\\label{" + label + "}\n"
           "\\footnotesize\n" + tex + "\\end{table}\n")
    open(os.path.join(out, f"{name}.tex"), "w", encoding="utf-8").write(tex)
    print(f"  saved {name}.csv / .tex")


def tables(res, out, eval_split):
    main = [r for r in res if r["protocol"] == "vehicle" and r["task"] == "multi" and not is_aux(r["model"])]
    if not main:
        print("  skip tables"); return
    rows = []
    for model in sorted({r["model"] for r in main}, key=lambda x: (family(x) != "proposed", x)):
        rr = [r for r in main if r["model"] == model]
        row = {"Model": DISPLAY.get(model, model), "Seeds": len(rr)}
        for k, lab in [("accuracy", "Accuracy (%)"), ("macro_f1", "Macro-F1 (%)"),
                       ("weighted_f1", "Weighted-F1 (%)"), ("roc_auc_macro", "AUC (%)")]:
            row[lab] = fmt(*mean_std([r[k] for r in rr]))
        row["Size (MB)"] = f"{np.mean([r.get('model_size_mb', np.nan) for r in rr]):.2f}"
        lat = [r.get("latency", {}).get("single_msg_ms", np.nan) for r in rr]
        row["Latency (ms)"] = f"{np.nanmean(lat):.2f}"
        rows.append(row)
    write_table(pd.DataFrame(rows), out, "tab_main_results",
                f"Overall results ({eval_split} split, vehicle-disjoint protocol). Mean $\\pm$ std over seeds.",
                "tab:main")

    pcs = {}
    for model in sorted({r["model"] for r in main}):
        if "_" in model and model != MAIN:
            continue
        f, _ = avg_per_class([r for r in main if r["model"] == model])
        pcs[DISPLAY.get(model, model)] = f
    if pcs:
        t = pd.DataFrame(pcs)
        t.insert(0, "Class", [CLASS_NAMES[int(c)] for c in t.index])
        for c in t.columns[1:]:
            t[c] = (t[c] * 100).map(lambda x: f"{x:.1f}")
        write_table(t, out, "tab_per_class_f1", "Per-class F1 (%).", "tab:perclass")

    abl = [r for r in main if r["model"] in ABLATIONS]
    if len({r["model"] for r in abl}) > 1:
        t = []
        for model in sorted({r["model"] for r in abl}):
            rr = [r for r in abl if r["model"] == model]
            t.append({"Variant": DISPLAY.get(model, model),
                      "Macro-F1 (%)": fmt(*mean_std([r["macro_f1"] for r in rr])),
                      "Accuracy (%)": fmt(*mean_std([r["accuracy"] for r in rr])),
                      "Params": f"{rr[0].get('params', 0):,}"})
        write_table(pd.DataFrame(t), out, "tab_ablation", "Ablation of the proposed model.", "tab:ablation")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="*", default=[])
    ap.add_argument("--eval-split", choices=["val", "test"], default="val")
    ap.add_argument("--counts", default="step3_class_split_counts.csv")
    ap.add_argument("--audit", default="step2c_audit.json")
    ap.add_argument("--out", default="paper_figures")
    ap.add_argument("--mixall", default="", help="mixall_variants.json from step8c")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    res = load_results(args.results, args.eval_split)
    print(f"Loaded {len(res)} result files for split '{args.eval_split}': "
          f"{sorted({(r['model'], r['protocol']) for r in res})}")
    fig_framework(args.out)
    fig_architecture(args.out)
    fig_mixed_scene(args.out)
    fig_class_distribution(args.counts, args.out)
    fig_audit_rate(args.audit, args.out)
    fig_per_class_f1(res, args.out)
    fig_confusion(res, args.out)
    fig_ablation(res, args.out)
    fig_efficiency(res, args.out)
    fig_leakage(res, args.out)
    fig_mixall(args.mixall, args.out)
    fig_loao(args.results, args.out)
    fig_importance(args.results, args.out)
    tables(res, args.out, args.eval_split)
    print(f"\nAll outputs in {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
