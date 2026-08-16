"""The baselines the handbook promises but has not released.

"组委会提供均值模型、随机森林、梯度提升等基础基线" -- and the cross-direction
ranking is defined on scores *normalised relative to those baselines*, so without
them no submission's number means anything.  We implement the three ourselves,
in the most natural way, and report our model relative to them.

Naive form on purpose: metadata features -> proteome, with no batch decomposition
and no perturbation/batch split.  Because 5,243 outputs is impractical for a
per-protein forest, all three predict the leading principal components of the
log2 matrix and reconstruct -- which is the standard way this baseline is built
and is, if anything, generous to it.
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcell.harness import build_fold, evaluate, summary_row               # noqa: E402
from vcell.models import (BATCH_FACTORS, PERT_FACTORS, ResidualBooster,   # noqa: E402
                          UnifiedBackfit)

import lightgbm as lgb                                     # noqa: E402
from sklearn.ensemble import RandomForestRegressor         # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
pd.set_option("display.width", 280)
CFG = {"lam_plate": 0.3, "lam_plate_x_strain": 2.0}
CAT = ["Strains", "compound", "Medium", "data_source", "instrument",
       "Yeast_cell_plate", "well_row"]

f = build_fold(vehicle="both")
meta, n = f.meta, len(f.meta)
vis = f.obs_mask

m = meta.copy()
m["well_row"] = m["protein_well"].str.extract(r"^([A-H])")[0]
X = pd.DataFrame({c: pd.Categorical(m[c].astype(str)).codes for c in CAT})
X["Temperature"] = m["Temperature"].astype(float)
X["pert_time"] = np.log2(m["pert_time"].astype(float))
X["well_col"] = m["protein_well"].str.extract(r"(\d+)$")[0].astype(float)

mu = np.nanmean(np.where(vis[:, None], f.Y, np.nan), 0).astype(np.float32)
mu = np.where(np.isfinite(mu), mu, np.nanpercentile(f.Y[vis], 5)).astype(np.float32)
Yc = np.nan_to_num(np.where(np.isfinite(f.Y) & vis[:, None], f.Y - mu, np.nan))
U, S, Vt = np.linalg.svd(Yc[vis], full_matrices=False)
K = 64
V = Vt[:K]
Z = Yc[vis] @ V.T
print(f"baseline target: top-{K} PCs, {(S[:K]**2).sum()/(S**2).sum():.1%} of variance")

rows, t0 = [], time.time()


def add(name, P):
    r = summary_row(name, evaluate(f, P))
    rows.append(r)
    print(f"  {name:34s} TOTAL={r['TOTAL']:.4f} | M1={r['M1_abs(20%)']:.3f} "
          f"M2={r['M2_rawFC(25%)']:.3f} M3={r['M3_ctx(20%)']:.3f} "
          f"M4={r['M4_drug(20%)']:.3f}  ({time.time()-t0:.0f}s)")
    return r


print("\n=== official-style baselines ===")
b_mean = add("baseline 1: mean model", np.tile(mu, (n, 1)))

rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=32,
                           random_state=0)
rf.fit(X[vis], Z)
b_rf = add("baseline 2: random forest", mu + rf.predict(X) @ V)

Zh = np.zeros((n, K), np.float32)
for k in range(K):
    g = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=31,
                          min_child_samples=20, verbose=-1, n_jobs=32)
    g.fit(X[vis], Z[:, k], categorical_feature=CAT)
    Zh[:, k] = g.predict(X)
b_gbm = add("baseline 3: gradient boosting", mu + Zh @ V)

print("\n=== ours ===")
um = UnifiedBackfit(
    batch_factors=[(a, c, CFG.get(f"lam_{a}", l)) for a, c, l in BATCH_FACTORS],
    pert_factors=[(a, c, CFG.get(f"lam_{a}", l)) for a, c, l in PERT_FACTORS],
).fit(meta, f.Y_obs, vis)
P = um.predict()
add("ours: additive only", P)
rb = ResidualBooster(n_comp=96, n_estimators=800, learning_rate=0.03, n_jobs=32)
rb.fit(meta, f.Y_obs, vis, P)
ours = add("ours: additive + booster", P + rb.predict())

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "official_baselines.csv"), index=False)
keep = ["model", "TOTAL", "M1_abs(20%)", "M2_rawFC(25%)", "M3_ctx(20%)",
        "M4_drug(20%)", "M5_bt(10%)", "M6_DEP(5%)"]
print("\n" + df[keep].to_string(index=False))

best_base = max(b_mean["TOTAL"], b_rf["TOTAL"], b_gbm["TOTAL"])
print(f"\nstrongest official-style baseline: {best_base:.4f}")
print(f"ours: {ours['TOTAL']:.4f}   ->  +{ours['TOTAL']-best_base:.4f} "
      f"({100*(ours['TOTAL']/best_base - 1):.1f}% relative)")
