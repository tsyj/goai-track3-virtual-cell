"""Is the plate term actually low-rank, and where does its signal stop?

A fast spectrum is not enough on its own -- a pure-noise matrix also has a decaying
spectrum.  The question that matters is which singular directions are *reproducible*.
So: split the training wells in half at random, fit the batch model separately on
each half, and compare the two independently estimated plate terms direction by
direction.  Directions where the halves agree carry batch structure; where they
stop agreeing, the fitted term is estimation noise and truncating it can only help.

Cheap (two additive fits, no booster) -- run it before spending an hour on a rank
search.

Jiao Xinyuan 2026-08-16
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcell.harness import build_fold                                       # noqa: E402
from vcell.models import BATCH_FACTORS, PERT_FACTORS, UnifiedBackfit       # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
CFG = {"lam_plate": 0.3, "lam_plate_x_strain": 2.0}
rng = np.random.default_rng(0)


def fit_terms(fo, use):
    um = UnifiedBackfit(
        batch_factors=[(a, c, CFG.get(f"lam_{a}", l)) for a, c, l in BATCH_FACTORS],
        pert_factors=[(a, c, CFG.get(f"lam_{a}", l)) for a, c, l in PERT_FACTORS],
        n_pass=6,
    ).fit(fo.meta, np.where(use[:, None], fo.Y_obs, np.nan), use)
    return um


fo = build_fold()
tr = np.where(fo.obs_mask)[0]
half = rng.permutation(len(tr))
A = np.zeros(fo.n, bool); A[tr[half[: len(tr) // 2]]] = True
B = np.zeros(fo.n, bool); B[tr[half[len(tr) // 2:]]] = True
print(f"train rows {len(tr)} -> halves {A.sum()} / {B.sum()}", flush=True)

umA, umB = fit_terms(fo, A), fit_terms(fo, B)
rows = []
for term in ("plate", "plate_x_strain", "strain", "compound"):
    if term not in umA.terms:
        continue
    TA, TB = umA.terms[term], umB.terms[term]
    # only levels seen in both halves are comparable
    ok = (np.abs(TA).sum(1) > 0) & (np.abs(TB).sum(1) > 0)
    TA, TB = TA[ok], TB[ok]
    if TA.shape[0] < 4:
        continue
    U, S, Vt = np.linalg.svd(TA, full_matrices=False)
    energy = np.cumsum(S ** 2) / (S ** 2).sum()
    # project BOTH halves on half-A's directions and correlate the loadings
    a = TA @ Vt.T
    b = TB @ Vt.T
    k = min(TA.shape[0], 80)
    corr = [float(np.corrcoef(a[:, j], b[:, j])[0, 1]) for j in range(k)]
    rows.append({"term": term, "levels": int(ok.sum()),
                 **{f"cum_energy@{r}": float(energy[min(r, len(energy)) - 1])
                    for r in (4, 8, 16, 32, 64)},
                 **{f"halfcorr@{r}": float(np.mean(corr[max(r - 4, 0):r]))
                    for r in (4, 8, 16, 32, 64) if r <= k}})
    print(f"\n--- {term}: {int(ok.sum())} levels, {TA.shape[1]} proteins", flush=True)
    print("  direction :   " + " ".join(f"{j:>6d}" for j in range(0, min(k, 40), 4)))
    print("  half-half r:  " + " ".join(f"{corr[j]:6.2f}" for j in range(0, min(k, 40), 4)))
    last = next((j for j in range(k) if corr[j] < 0.3), k)
    print(f"  first direction with half-half r < 0.30: {last}"
          f"   (cumulative energy there: {energy[min(last, len(energy)-1)]:.3f})")

df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print("\n=== summary ===")
print(df.round(3).to_string(index=False))
df.to_csv(os.path.join(OUT, "plate_rank.csv"), index=False)
