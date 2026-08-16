"""Does a genome-informed strain prior improve the actual score?

scripts/31 showed that a held-out strain's baseline correlates with its nearest
genomic relative (r = 0.27) once the centring artefact is removed, but that using
that relative's profile raw is too noisy to help much on rmse.  The natural fix is
to shrink it: add alpha * (nearest visible relative's strain term) for a strain
with no visible data.

Measured end-to-end on the mirrors, because rmse on a baseline profile is not the
objective -- the six weighted modules are.

Caveat carried into the write-up: on the val mirror the held-out strain is BAI,
whose nearest visible relative sits at 1.36% SNP divergence.  CRD's nearest
relative CGD sits at 0.398%.  No validatable case is anywhere near that regime,
so whatever alpha we fit here is being extrapolated for the real submission.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcell.harness import (INNER, VAL, build_fold, evaluate,              # noqa: E402
                           make_inner_splits, summary_row)
from vcell.models import (BATCH_FACTORS, PERT_FACTORS, ResidualBooster,   # noqa: E402
                          UnifiedBackfit, interaction_codes)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")
D = pd.read_csv(os.path.join(ROOT, "data", "genomes",
                             "1011DistanceMatrixBasedOnSNPs.tab.gz"),
                sep="\t", index_col=0)
CFG = {"lam_plate": 0.3, "lam_plate_x_strain": 2.0}
pd.set_option("display.width", 260)


def fit_once(fo):
    """Fit once.  The strain prior is a post-hoc additive term, so sweeping alpha
    needs no refitting -- the earlier version refit for every alpha and wasted 5x
    the compute."""
    um = UnifiedBackfit(
        batch_factors=[(a, c, CFG.get(f"lam_{a}", l)) for a, c, l in BATCH_FACTORS],
        pert_factors=[(a, c, CFG.get(f"lam_{a}", l)) for a, c, l in PERT_FACTORS],
    ).fit(fo.meta, fo.Y_obs, fo.obs_mask)
    P = um.predict()
    rb = ResidualBooster(n_comp=96, n_estimators=800, learning_rate=0.03, n_jobs=16)
    rb.fit(fo.meta, fo.Y_obs, fo.obs_mask, P)
    P = P + rb.predict()

    codes, _ = interaction_codes(fo.meta, ["Strains"])
    levels = pd.Series(fo.meta["Strains"].to_numpy()).groupby(codes).first().to_numpy()
    visible = {levels[c] for c in np.unique(codes[fo.obs_mask])}
    donors = [s for s in visible if s in D.index]
    prior, note = np.zeros_like(P), {}
    for c, s in enumerate(levels):
        if s in visible or s not in D.index or not donors:
            continue
        near = D.loc[s, donors].idxmin()
        prior[codes == c] = um.terms["strain"][list(levels).index(near)]
        note[s] = (near, float(D.loc[s, near]))
    return P, prior, note


folds = [("val mirror (BAI held out)", build_fold(), VAL)]
base_meta = folds[0][1].meta
for seed, strain in [(0, "CEK"), (1, "CGD"), (3, "BAH")]:
    sp = make_inner_splits(base_meta, hold_strain=strain, seed=seed)
    folds.append((f"inner ({strain} held out)", build_fold(splits=sp), INNER))

rows = []
for name, fo, which in folds:
    print(f"\n=== {name} ===")
    P0, prior, note = fit_once(fo)
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        P = P0 + alpha * prior
        r = summary_row(f"alpha={alpha}", evaluate(fo, P, which))
        rows.append({"fold": name, "alpha": alpha, **r})
        tag = "" if not note else "  donor " + ", ".join(
            f"{k}<-{v[0]} ({v[1]:.2f}%)" for k, v in note.items())
        print(f"  alpha={alpha:<5} TOTAL={r['TOTAL']:.4f}  M2={r['M2_rawFC(25%)']:.3f} "
              f"M4={r['M4_drug(20%)']:.3f}  FC[strain_only]="
              f"{r.get('FC[strain_only]', float('nan')):.3f}{tag}")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "strain_prior_endtoend.csv"), index=False)
print("\n=== paired delta vs alpha=0, averaged over folds ===")
piv = df.pivot_table(index="alpha", columns="fold", values="TOTAL")
delta = piv.subtract(piv.loc[0.0], axis=1)
print(delta.round(4).to_string())
print("\nmean delta:", delta.mean(axis=1).round(4).to_dict())
