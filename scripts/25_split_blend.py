"""Per-regime estimator choice for the unseen-strain split.

For a held-out strain the pooled empirical Delta (same compound, same
medium/temperature/time/source, averaged over the visible strains) reaches 0.373
raw fold-change correlation against our model's 0.346 -- but it scores *worse* on
M4, because M4 subtracts mu_drug and the pooled estimate sits almost on top of it.

The two estimators therefore trade off against each other, so blend them and let
the inner mirror pick the weight.  Different generalisation regimes genuinely
carry different information, so using a different estimator per regime is a
modelling choice, not metric-chasing -- but the weight is selected on the inner
mirror and reported either way.
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcell.harness import (INNER, build_fold, evaluate, make_inner_splits,  # noqa: E402
                           summary_row)
from vcell.models import (BATCH_FACTORS, PERT_FACTORS, ResidualBooster,     # noqa: E402
                          UnifiedBackfit)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
pd.set_option("display.width", 280)
CFG = {"lam_plate": 0.3, "lam_plate_x_strain": 2.0}
POOL = ["compound", "Medium", "Temperature", "pert_time", "data_source"]

base_fold = build_fold()
folds = [build_fold(splits=make_inner_splits(base_fold.meta, hold_strain=s, seed=i))
         for i, s in [(0, "CEK"), (1, "CGD")]]

prepped = []
for fo in folds:
    um = UnifiedBackfit(
        batch_factors=[(a, c, CFG.get(f"lam_{a}", l)) for a, c, l in BATCH_FACTORS],
        pert_factors=[(a, c, CFG.get(f"lam_{a}", l)) for a, c, l in PERT_FACTORS],
    ).fit(fo.meta, fo.Y_obs, fo.obs_mask)
    B = um.predict()
    rb = ResidualBooster(n_comp=96, n_estimators=800, learning_rate=0.03)
    rb.fit(fo.meta, fo.Y_obs, fo.obs_mask, B)
    ours = B + rb.predict()

    D = np.where(np.isfinite(fo.C_obs), fo.Y_obs - fo.C_obs, np.nan).astype(np.float32)
    vis = fo.obs_mask & (~fo.meta["is_control"]).to_numpy() \
        & (~fo.meta["is_qc"]).to_numpy()
    k = fo.meta[POOL].astype(str).agg("|".join, axis=1).to_numpy()
    pooled = np.full_like(D, np.nan)
    for g in np.unique(k):
        rows = np.where(k == g)[0]
        src = rows[vis[rows]]
        if len(src):
            pooled[rows] = np.nanmean(D[src], axis=0)
    tgt = fo.meta["split_final"].isin(["in_strain_only", "in_both"]).to_numpy() \
        & np.isfinite(pooled).any(1)
    prepped.append((fo, ours, B, np.nan_to_num(pooled), tgt))
    print(f"fold ready; pooled Delta available for {tgt.sum()} unseen-strain rows")

t0, rows = time.time(), []
for w in (0.0, 0.25, 0.5, 0.75, 1.0):
    acc = []
    for fo, ours, B, pooled, tgt in prepped:
        P = ours.copy()
        P[tgt] = (1 - w) * ours[tgt] + w * (B[tgt] + pooled[tgt])
        acc.append(summary_row(f"w={w}", evaluate(fo, P, INNER)))
    s = pd.DataFrame(acc).mean(numeric_only=True)
    rows.append({"w": w, **s.to_dict()})
    print(f"  w={w:<5} TOTAL={s['TOTAL']:.4f}  M2={s['M2_rawFC(25%)']:.3f} "
          f"M4={s['M4_drug(20%)']:.3f}  FC[str]={s['FC[strain_only]']:.3f} "
          f"FC[both]={s['FC[both]']:.3f}  ({time.time()-t0:.0f}s)")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "split_blend.csv"), index=False)
best = df.loc[df.TOTAL.idxmax()]
print(f"\nbest blend weight = {best.w}  (inner TOTAL {best.TOTAL:.4f}, "
      f"vs {df[df.w == 0].TOTAL.iloc[0]:.4f} without)")
