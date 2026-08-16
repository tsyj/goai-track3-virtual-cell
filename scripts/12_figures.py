"""Render the results-pack figures from the CSVs written by the analysis scripts."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcell.viz import GRID, INK, INK2, MUTED, S1, S2, S3, caption, strip, use_style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figs")
os.makedirs(FIG, exist_ok=True)
use_style()


def have(*names):
    return all(os.path.exists(os.path.join(RES, n)) for n in names)


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p)


# --------------------------------------------------------------- fig 1
if have("variance_decomposition.csv", "ceiling.csv"):
    vd = pd.read_csv(os.path.join(RES, "variance_decomposition.csv"))
    vd = vd.sort_values("var_frac")
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.6),
                               gridspec_kw={"width_ratios": [1.25, 1]})
    lbl = {"Yeast_cell_plate": "plate (batch)", "instrument": "instrument",
           "data_source": "data source", "compound": "compound",
           "Strains": "strain", "Medium": "medium", "pert_time": "time",
           "Temperature": "temperature"}
    names = [lbl.get(x, x) for x in vd.factor]
    colors = [S2 if x == "compound" else (S1 if x in
              ("Yeast_cell_plate", "instrument", "data_source") else MUTED)
              for x in vd.factor]
    a.barh(names, vd.var_frac, color=colors, height=0.62)
    for y, v in enumerate(vd.var_frac):
        a.text(v + 0.012, y, f"{v:.0%}", va="center", fontsize=8.5, color=INK2)
    a.set_xlim(0, 1.02)
    a.set_xlabel("share of log2 proteome variance explained alone")
    a.set_title("Batch structure dominates the signal")
    a.grid(axis="y", visible=False)
    strip(a)

    budget = [("biological\nperturbation effect", 0.146, S2),
              ("measurement noise\nper sample", 0.260, MUTED),
              ("strain baseline\n(unseen strain error)", 0.47, S1),
              ("plate level", 0.656, S1)]
    b.barh([x[0] for x in budget], [x[1] for x in budget],
           color=[x[2] for x in budget], height=0.6)
    for y, (_, v, _) in enumerate(budget):
        b.text(v + 0.012, y, f"{v:.2f}", va="center", fontsize=8.5, color=INK2)
    b.set_xlabel("rms magnitude (log2 units)")
    b.set_title("The effect is smaller than the noise")
    b.set_xlim(0, 0.78)
    b.grid(axis="y", visible=False)
    strip(b)
    caption(fig, "Left: variance explained by each design factor alone, after per-sample "
                 "loading normalisation (results/variance_decomposition.csv).\nRight: "
                 "effect and noise magnitudes implied by WAYB's three replicate batches "
                 "(replicate Delta correlation 0.136).")
    save(fig, "fig1_data_budget.png")

# --------------------------------------------------------------- fig 2
if have("information_ladder.csv"):
    il = pd.read_csv(os.path.join(RES, "information_ladder.csv")).set_index("model")
    mods = [("M1_abs(20%)", "M1 absolute\nfidelity (20%)"),
            ("M2_rawFC(25%)", "M2 matched-control\nfold change (25%)"),
            ("M3_ctx(20%)", "M3 context-mean\nresidual (20%)"),
            ("M4_drug(20%)", "M4 drug-mean\nresidual (20%)"),
            ("M5_bt(10%)", "M5 both-unseen\n+ time (10%)"),
            ("M6_DEP(5%)", "M6 high-effect\ndetection (5%)")]
    rows = ["L1 batch level, zero drug effect", "L2 batch + modelled drug effect",
            "L3 batch + TRUE drug effect"]
    names = ["no perturbation model", "our model", "oracle: true effect"]
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    y = np.arange(len(mods))
    h = 0.24
    for i, (r, nm, c) in enumerate(zip(rows, names, (MUTED, S1, S2))):
        v = [il.loc[r, k] for k, _ in mods]
        ax.barh(y + (1 - i) * h, v, height=h * 0.88, color=c, label=nm)
        for yy, vv in zip(y + (1 - i) * h, v):
            ax.text(vv + 0.008, yy, f"{vv:.2f}", va="center", fontsize=7.6, color=INK2)
    ax.set_yticks(y, [m[1] for m in mods])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("module score")
    ax.set_title("How much of each module a perturbation model can actually move")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncols=3)
    strip(ax)
    caption(fig, "The gap between the grey and orange bars is the module's entire "
                 "perturbation-sensitive range. M1 barely moves: it is a batch-modelling "
                 "task.\nM3 does not move at all for our model, because an unseen "
                 "compound receives no compound term (results/information_ladder.csv).",
            y=-0.14)
    save(fig, "fig2_module_headroom.png")

# --------------------------------------------------------------- fig 3
if have("m3_clean.csv", "ceiling.csv", "control_noise.csv"):
    cn = pd.read_csv(os.path.join(RES, "control_noise.csv"))
    m3c = pd.read_csv(os.path.join(RES, "m3_clean.csv"))
    ce = pd.read_csv(os.path.join(RES, "ceiling.csv")).set_index("model")
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 3.9),
                               gridspec_kw={"width_ratios": [1, 1.05], "wspace": 0.42})

    w = cn[cn.k > 0].sort_values("mean_wells")
    a.plot(w.mean_wells, w.TOTAL, color=S1, marker="o", label="weighted TOTAL")
    a.plot(w.mean_wells, w["M2_rawFC(25%)"], color=S2, marker="o",
           label="M2 matched-control fold change")
    for _, r in w.iterrows():
        a.annotate(f"{r.TOTAL:.3f}", (r.mean_wells, r.TOTAL + 0.018),
                   fontsize=8, color=INK2, ha="center")
    a.set_xlabel("control wells averaged into the reference")
    a.set_ylabel("score")
    a.set_ylim(0.30, 0.66)
    a.set_title("A better-measured control lowers the score")
    a.legend(loc="lower left")
    strip(a, which=("top", "right"))

    bars = [("no compound\ninformation at all", 0.2125, MUTED),
            ("an independent\nrepeat measurement", 
             float(ce.loc["ORACLE replicate-measurement", "M3_ctx(20%)"]), S2),
            ("the metric's own\nreference", float(m3c["M3_ctx(20%)"].iloc[-1]), S1)]
    b.barh([x[0] for x in bars], [x[1] for x in bars],
           color=[x[2] for x in bars], height=0.55)
    for y, (_, v, _) in enumerate(bars):
        b.text(max(v, 0) + 0.006, y, f"{v:.3f}", va="center", fontsize=9, color=INK2)
    b.invert_yaxis()
    b.set_xlim(-0.02, 0.28)
    b.set_xlabel("M3 context-mean residual score")
    b.set_title("What M3 actually rewards")
    b.grid(axis="y", visible=False)
    strip(b)
    caption(fig, "Left: one fixed set of predictions, scored against control "
                 "references of increasing quality (results/control_noise.csv).\n"
                 "Right: M3 for three predictors. A model containing no compound "
                 "information outscores a real repeat measurement by 2.3x.")
    save(fig, "fig3_metric_artefact.png")

# --------------------------------------------------------------- fig 4
if have("compound_reliability.csv", "compound_pair_similarity.csv"):
    rel = pd.read_csv(os.path.join(RES, "compound_reliability.csv"))
    pair = pd.read_csv(os.path.join(RES, "compound_pair_similarity.csv"))
    pair = pair[pair.same_source]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.8, 4.0),
                               gridspec_kw={"width_ratios": [1, 1.05]})
    r = rel.sort_values("reliability")
    a.hlines(np.arange(len(r)), 0, r.reliability, color=GRID, linewidth=1.2)
    a.plot(r.reliability, np.arange(len(r)), "o", color=S1, markersize=4.5,
           linestyle="none")
    a.axvline(0.5, color=S2, linewidth=1.4, linestyle=(0, (4, 3)))
    a.text(0.51, len(r) - 1.5, "reliability 0.5", color=S2, fontsize=8, va="top")
    a.set_yticks(np.arange(len(r)), [c[:26] for c in r.compound], fontsize=6.4)
    a.set_xlabel("split-half reliability of the compound's response vector")
    a.set_title("Compound responses are reliably measured")
    a.set_xlim(0, 1.0)
    a.grid(axis="y", visible=False)
    strip(a)

    for flag, c, lab, z in [(False, MUTED, "different mechanism", 1),
                            (True, S2, "same mechanism", 3)]:
        s = pair[pair.same_moa == flag]
        a2 = 26 if flag else 12
        b.scatter(s.tanimoto, s.r_disatt, s=a2, c=c, alpha=0.75 if flag else 0.45,
                  linewidths=0, label=lab, zorder=z)
    ok = np.isfinite(pair.tanimoto) & np.isfinite(pair.r_disatt)
    z = np.polyfit(pair.tanimoto[ok], pair.r_disatt[ok], 1)
    xs = np.linspace(0, pair.tanimoto.max(), 20)
    b.plot(xs, np.polyval(z, xs), color=S1, linewidth=1.8, zorder=2,
           label="linear fit (Spearman ρ = −0.02, p = 0.73)")
    b.axhline(0, color=GRID, linewidth=1.0, zorder=0)
    b.set_xlabel("structural similarity (Tanimoto, Morgan fingerprint)")
    b.set_ylabel("response-vector correlation (noise-corrected)")
    b.set_title("…but chemistry does not predict the response")
    b.legend(loc="upper right")
    strip(b, which=("top", "right"))
    caption(fig, "Left: each compound's effect vector estimated twice from disjoint "
                 "halves of its own samples; 35 of 37 exceed 0.5.\nRight: all 456 "
                 "within-data-source compound pairs. Same-mechanism pairs are not more "
                 "alike than different-mechanism pairs (+0.007, permutation p = 0.95).")
    save(fig, "fig4_chemistry.png")

# --------------------------------------------------------------- fig 5
if have("biology_check.csv"):
    b = pd.read_csv(os.path.join(RES, "biology_check.csv"))
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    lim = max(b.measured_effect.abs().max(), b.model_effect.abs().max()) * 1.15
    ax.plot([-lim, lim], [-lim, lim], color=GRID, linewidth=1.4, zorder=0)
    ax.axhline(0, color=GRID, linewidth=1.0, zorder=0)
    ax.axvline(0, color=GRID, linewidth=1.0, zorder=0)
    for vis, c, lab, mk in [(True, S1, "compound seen in training", "o"),
                            (False, S2, "compound held out", "D")]:
        g = b[b.visible_to_model == vis]
        ax.scatter(g.measured_effect, g.model_effect, s=70, c=c, marker=mk,
                   linewidths=0, label=lab, zorder=3)
    # push overlapping labels apart vertically, then draw a leader line
    lab = b.sort_values("model_effect").reset_index(drop=True)
    ys, MIN_GAP = [], 0.075
    for _, r in lab.iterrows():
        y = r.model_effect
        if ys and y - ys[-1] < MIN_GAP:
            y = ys[-1] + MIN_GAP
        ys.append(y)
    for (_, r), y in zip(lab.iterrows(), ys):
        right = r.measured_effect < 0.8
        dx = 0.055 if right else -0.055
        ax.plot([r.measured_effect, r.measured_effect + dx * 0.75],
                [r.model_effect, y], color=GRID, linewidth=0.8, zorder=1)
        ax.annotate(f"{r.compound.split()[0]} / {r.markers}",
                    (r.measured_effect + dx, y), fontsize=7.2, color=INK2,
                    va="center", ha="left" if right else "right")
    ax.set_xlim(-0.55, lim)
    ax.set_ylim(-0.55, lim)
    ax.set_xlabel("measured effect on the marker set (log2, vs all other proteins)")
    ax.set_ylabel("effect recovered by the model")
    ax.set_title("The model recovers known biology — for compounds it has seen")
    ax.legend(loc="upper left")
    strip(ax, which=("top", "right"))
    caption(fig, "Ten pre-registered yeast responses. Points on the diagonal are "
                 "recovered at the right magnitude; points on the horizontal axis "
                 "are effects\nthe model returns as zero. Every held-out compound "
                 "falls on the axis — hydroxyurea's RNR induction is the largest "
                 "effect measured (+1.31)\nand the model predicts -0.01 "
                 "(results/biology_check.csv).")
    save(fig, "fig5_biology.png")

print("\nfigures in", FIG)
