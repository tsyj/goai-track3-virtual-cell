"""Can a held-out strain's baseline be predicted from its genome?

The official deck points at the 1002 Genomes project, and the strain codes turn
out to be its isolate names: BAH / BAI / CEK / CGD / CRD are all in the 1,011
isolate panel (DHY210 is the S288c-background lab strain and is not).

That gives genomic relatedness for the *unseen* strain -- the one quantity we
previously argued was unknowable.  And the relatedness is not uniform:

    CRD -> CGD  0.398 %      CRD -> CEK  1.475 %
    CRD -> BAI  1.478 %      CRD -> BAH  2.099 %

so CRD has a close relative among the training strains.

Decisive test, done on strains whose labels we actually hold: take a strain out,
estimate its per-protein baseline deviation as a genome-similarity-weighted
average of the others, and compare against the zero the additive model uses.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcell.harness import build_fold                                      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")
GEN = os.path.join(ROOT, "data", "genomes")
pd.set_option("display.width", 260)

D = pd.read_csv(os.path.join(GEN, "1011DistanceMatrixBasedOnSNPs.tab.gz"),
                sep="\t", index_col=0)
# strains that are BOTH in the genome panel AND have measured labels.  CRD is in
# the panel but has no labels at all (it is the held-out test strain), so it can
# only ever be a prediction target at submission time, never a validation case.
PANEL = ["BAH", "BAI", "CEK", "CGD", "CRD"]
IN_PANEL = ["BAH", "BAI", "CEK", "CGD"]

f = build_fold()
meta, Y = f.meta, f.Y
train_like = (~meta["is_qc"]).to_numpy()

# Per-strain, per-protein baseline profile, from control wells only so the
# compound composition of a strain's wells cannot confound it.
#
# Centring matters.  If the grand mean includes the held-out strain, that
# strain's deviation is mechanically about -1/(n-1) times the sum of the others,
# so ANY average-of-others estimator is negatively correlated with it by
# construction and the test is rigged against the hypothesis.  The model centres
# on the VISIBLE strains only, so the test must too: for each held-out target,
# the centre is recomputed from the donors alone.
ctrl = meta["is_control"].to_numpy() & train_like
prof = {}
for s in sorted(set(meta["Strains"])):
    rows = ctrl & (meta["Strains"] == s).to_numpy()
    prof[s] = np.nanmean(Y[rows], axis=0)
print(f"per-strain baseline profile computed on {ctrl.sum()} control wells")


def deviations(donors, target):
    """Profiles re-centred on the donor set, exactly as the model centres them."""
    centre = np.nanmean([prof[s] for s in donors], axis=0)
    return {s: prof[s] - centre for s in donors + [target]}


def weighted_estimate(dev, target, donors, tau):
    d = D.loc[target, donors].to_numpy(float)
    w = np.exp(-d / tau)
    w /= w.sum()
    return sum(wi * dev[s] for wi, s in zip(w, donors)), dict(zip(donors, w.round(3)))


print("\n=== leave-one-strain-out: predict a strain's baseline deviation ===")
print(f"{'held out':10s} {'estimator':26s} {'corr':>7s} {'rmse':>7s}   weights")
rows_out = []
for target in IN_PANEL:
    donors = [s for s in IN_PANEL if s != target]
    dev = deviations(donors, target)
    truth = dev[target]
    m = np.isfinite(truth)
    variants = [("zero (what we do now)", np.zeros_like(truth), {}),
                ("unweighted mean of others",
                 np.nanmean([dev[s] for s in donors], 0), {})]
    for tau in (0.3, 0.6, 1.2):
        est, w = weighted_estimate(dev, target, donors, tau)
        variants.append((f"genome-weighted tau={tau}", est, w))
    nearest = D.loc[target, donors].idxmin()
    variants.append((f"nearest relative ({nearest})", dev[nearest],
                     {nearest: 1.0}))
    for name, est, w in variants:
        mm = m & np.isfinite(est)
        c = np.corrcoef(truth[mm], est[mm])[0, 1] if est[mm].std() > 0 else np.nan
        r = float(np.sqrt(np.nanmean((truth[mm] - est[mm]) ** 2)))
        rows_out.append({"held_out": target, "estimator": name, "corr": c, "rmse": r})
        print(f"{target:10s} {name:26s} {c:7.3f} {r:7.3f}   "
              f"{ {k: v for k, v in w.items() if v > 0.05} }")
    print()

df = pd.DataFrame(rows_out)
df.to_csv(os.path.join(OUT, "genome_strain_prior.csv"), index=False)
piv = df.pivot_table(index="estimator", values=["corr", "rmse"], aggfunc="mean")
print("=== mean over the held-out strains ===")
print(piv.sort_values("rmse").round(4).to_string())
best = piv.rmse.idxmin()
zero = piv.loc["zero (what we do now)", "rmse"]
print(f"\nbest: {best}  rmse {piv.loc[best,'rmse']:.4f} vs zero {zero:.4f}  "
      f"({100*(1-piv.loc[best,'rmse']/zero):.1f}% lower)")

# what this implies for the actual submission
print("\n=== applied to CRD (the real held-out strain) ===")
donors = IN_PANEL
for tau in (0.3, 0.6, 1.2):
    d = D.loc["CRD", donors].to_numpy(float)
    w = np.exp(-d / tau); w /= w.sum()
    print(f"  tau={tau}: " + ", ".join(f"{s} {wi:.3f}" for s, wi in zip(donors, w)))
print(f"  nearest relative of CRD among training strains: "
      f"{D.loc['CRD', donors].idxmin()} "
      f"(SNP distance {D.loc['CRD', donors].min():.3f}%)")

# can the validation panel speak to CRD's regime at all?
print("\n=== is the CRD-CGD closeness inside the range we can validate? ===")
for t in IN_PANEL:
    dn = [s for s in IN_PANEL if s != t]
    print(f"  {t}: nearest donor {D.loc[t, dn].idxmin()} at {D.loc[t, dn].min():.3f}%")
print(f"  CRD: nearest donor CGD at {D.loc['CRD', IN_PANEL].min():.3f}%  "
      "<-- far closer than any validatable case")
