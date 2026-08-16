"""How far does the residual learner go, and what is it actually using?

Part 1 sweeps the number of residual components (the gain was still rising at 48).
Part 2 ablates feature groups, because a 96-well layout is fixed per panel, so
well position is partly a proxy for compound identity and it matters whether the
gain survives without it.
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
from vcell.models import BATCH_FACTORS, PERT_FACTORS, UnifiedBackfit        # noqa: E402

import lightgbm as lgb  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
pd.set_option("display.width", 260)
CFG = {"lam_plate": 0.3, "lam_plate_x_strain": 2.0}

ALL_CAT = ["Strains", "compound", "Medium", "data_source", "instrument",
           "Yeast_cell_plate", "well_row"]
ALL_NUM = ["Temperature", "pert_time", "well_col"]


def prep(fo):
    m = fo.meta.copy()
    m["well_row"] = m["protein_well"].str.extract(r"^([A-H])")[0]
    m["well_col"] = m["protein_well"].str.extract(r"(\d+)$")[0].astype(float)
    m["pert_time"] = np.log2(m["pert_time"].astype(float))
    m["Temperature"] = m["Temperature"].astype(float)
    um = UnifiedBackfit(
        batch_factors=[(n, c, CFG.get(f"lam_{n}", l)) for n, c, l in BATCH_FACTORS],
        pert_factors=[(n, c, CFG.get(f"lam_{n}", l)) for n, c, l in PERT_FACTORS],
    ).fit(fo.meta, fo.Y_obs, fo.obs_mask)
    base = um.predict()
    vis = fo.obs_mask
    Rv = np.nan_to_num(np.where(np.isfinite(fo.Y_obs) & vis[:, None],
                                fo.Y_obs - base, np.nan)[vis]).astype(np.float32)
    return m, base, Rv, vis


def residual_pred(m, Rv, vis, n_comp, cats, nums, n_estimators=800, lr=0.03):
    U, S, Vt = np.linalg.svd(Rv, full_matrices=False)
    V = Vt[:n_comp]
    Z = Rv @ V.T
    X = pd.DataFrame(index=m.index)
    for c in cats:
        X[c] = pd.Categorical(m[c].astype(str)).codes
    for c in nums:
        X[c] = m[c].astype(float)
    Zhat = np.zeros((len(m), n_comp), np.float32)
    for k in range(n_comp):
        g = lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=lr,
                              num_leaves=31, min_child_samples=30, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.9, reg_lambda=1.0,
                              verbose=-1, n_jobs=8)
        g.fit(X[vis], Z[:, k], categorical_feature=[c for c in cats])
        Zhat[:, k] = g.predict(X)
    return Zhat @ V


base_fold = build_fold()
folds = [build_fold(splits=make_inner_splits(base_fold.meta, hold_strain=s, seed=i))
         for i, s in [(0, "CEK"), (1, "CGD")]]
prepped = [prep(fo) for fo in folds]
t0, rows = time.time(), []


def trial(tag, n_comp, cats, nums):
    acc = []
    for fo, (m, base, Rv, vis) in zip(folds, prepped):
        add = residual_pred(m, Rv, vis, n_comp, cats, nums)
        acc.append(summary_row(tag, evaluate(fo, base + add, INNER)))
    s = pd.DataFrame(acc).mean(numeric_only=True)
    rows.append({"trial": tag, **s.to_dict()})
    print(f"  {tag:36s} TOTAL={s['TOTAL']:.4f}  M2={s['M2_rawFC(25%)']:.3f} "
          f"M3={s['M3_ctx(20%)']:.3f} M4={s['M4_drug(20%)']:.3f}  ({time.time()-t0:.0f}s)")


acc = []
for fo, (m, base, Rv, vis) in zip(folds, prepped):
    acc.append(summary_row("additive only", evaluate(fo, base, INNER)))
s = pd.DataFrame(acc).mean(numeric_only=True)
rows.append({"trial": "additive only", **s.to_dict()})
print(f"  {'additive only':36s} TOTAL={s['TOTAL']:.4f}  M2={s['M2_rawFC(25%)']:.3f} "
      f"M3={s['M3_ctx(20%)']:.3f} M4={s['M4_drug(20%)']:.3f}")

print("\n=== number of residual components ===")
for k in (48, 96, 160):
    trial(f"{k} components", k, ALL_CAT, ALL_NUM)

print("\n=== feature ablations (96 components) ===")
trial("no well position", 96, [c for c in ALL_CAT if c != "well_row"],
      [n for n in ALL_NUM if n != "well_col"])
trial("no compound id", 96, [c for c in ALL_CAT if c != "compound"], ALL_NUM)
trial("no plate id", 96, [c for c in ALL_CAT if c != "Yeast_cell_plate"], ALL_NUM)
trial("context only (no cmpd/well)", 96,
      ["Strains", "Medium", "data_source", "instrument", "Yeast_cell_plate"],
      ["Temperature", "pert_time"])

df = pd.DataFrame(rows).sort_values("TOTAL", ascending=False)
df.to_csv(os.path.join(OUT, "residual_sweep.csv"), index=False)
print("\n" + df[["trial", "TOTAL", "M1_abs(20%)", "M2_rawFC(25%)", "M3_ctx(20%)",
                 "M4_drug(20%)", "M6_DEP(5%)"]].to_string(index=False))
