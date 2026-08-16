"""Is there signal the additive model leaves behind that ML can recover?

Decisive test.  Fit the additive model, take its residual on the visible rows,
compress the residual to its leading principal components (the proteome is
strongly low-rank, so 5,243 targets collapse to a few dozen), and train gradient
boosting on the design metadata to predict those components.  Add the recovered
residual back and re-score.

If the score improves, the additive family was too narrow.  If it does not, the
"extra capacity does not pay" conclusion finally covers a second model family
rather than just variations of the first.

Run on the inner mirror, which holds out a whole strain and a set of compounds,
so a model that only memorises categories is caught.
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcell.harness import (INNER, build_fold, evaluate, make_inner_splits,  # noqa: E402
                           summary_row)
from vcell.models import UnifiedBackfit                                     # noqa: E402

import lightgbm as lgb  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
pd.set_option("display.width", 260)

CFG = {"lam_plate": 0.3, "lam_plate_x_strain": 2.0}
CAT = ["Strains", "compound", "Medium", "data_source", "instrument",
       "Yeast_cell_plate", "well_row"]
NUM = ["Temperature", "pert_time", "well_col"]


def featurise(meta):
    X = pd.DataFrame(index=meta.index)
    for c in CAT:
        X[c] = pd.Categorical(meta[c].astype(str)).codes
    X["Temperature"] = meta["Temperature"].astype(float)
    X["pert_time"] = np.log2(meta["pert_time"].astype(float))
    X["well_col"] = meta["protein_well"].str.extract(r"(\d+)$")[0].astype(float)
    return X[CAT + NUM]


def fit_additive(fo):
    from vcell.models import BATCH_FACTORS, PERT_FACTORS
    return UnifiedBackfit(
        batch_factors=[(n, c, CFG.get(f"lam_{n}", l)) for n, c, l in BATCH_FACTORS],
        pert_factors=[(n, c, CFG.get(f"lam_{n}", l)) for n, c, l in PERT_FACTORS],
    ).fit(fo.meta, fo.Y_obs, fo.obs_mask)


def residual_ml(fo, n_comp=48, n_estimators=300, learning_rate=0.05, num_leaves=31):
    um = fit_additive(fo)
    base = um.predict()
    vis = fo.obs_mask
    R = np.where(np.isfinite(fo.Y_obs) & vis[:, None], fo.Y_obs - base, np.nan)

    # PCA on the visible residual, mean-imputed inside the SVD only
    Rv = np.nan_to_num(R[vis]).astype(np.float32)
    U, S, Vt = np.linalg.svd(Rv, full_matrices=False)
    V = Vt[:n_comp]                                   # (k, p)
    Z = Rv @ V.T                                      # (n_vis, k) target scores
    ev = (S ** 2 / (S ** 2).sum())[:n_comp].sum()
    meta = fo.meta.copy()
    meta["well_row"] = meta["protein_well"].str.extract(r"^([A-H])")[0]
    X = featurise(meta)
    Xv = X[vis]

    Zhat = np.zeros((len(meta), n_comp), np.float32)
    for k in range(n_comp):
        m = lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
                              num_leaves=num_leaves, min_child_samples=30,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
                              reg_lambda=1.0, verbose=-1, n_jobs=8)
        m.fit(Xv, Z[:, k], categorical_feature=CAT)
        Zhat[:, k] = m.predict(X)
    return um, base, Zhat @ V, ev


t0 = time.time()
base_fold = build_fold()
folds = [build_fold(splits=make_inner_splits(base_fold.meta, hold_strain=s, seed=i))
         for i, s in [(0, "CEK"), (1, "CGD")]]

rows = []
for tag, kw in [("48 comps, 300 trees", dict(n_comp=48)),
                ("16 comps, 300 trees", dict(n_comp=16)),
                ("48 comps, 800 trees, lr .03",
                 dict(n_comp=48, n_estimators=800, learning_rate=0.03))]:
    accs = {"additive only": [], "+ GBDT residual": [], "+ 0.5x GBDT residual": []}
    for fo in folds:
        um, base, add, ev = residual_ml(fo, **kw)
        accs["additive only"].append(summary_row("a", evaluate(fo, base, INNER)))
        accs["+ GBDT residual"].append(
            summary_row("b", evaluate(fo, base + add, INNER)))
        accs["+ 0.5x GBDT residual"].append(
            summary_row("c", evaluate(fo, base + 0.5 * add, INNER)))
    print(f"\n--- {tag}  (residual variance captured by comps: {ev:.1%}) ---")
    for k, v in accs.items():
        m = pd.DataFrame(v).mean(numeric_only=True)
        rows.append({"setting": tag, "model": k, **m.to_dict()})
        print(f"  {k:24s} TOTAL={m['TOTAL']:.4f}  M1={m['M1_abs(20%)']:.3f} "
              f"M2={m['M2_rawFC(25%)']:.3f} M3={m['M3_ctx(20%)']:.3f} "
              f"M4={m['M4_drug(20%)']:.3f}  ({time.time()-t0:.0f}s)")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "residual_ml.csv"), index=False)
print("\n" + df[["setting", "model", "TOTAL", "M1_abs(20%)", "M2_rawFC(25%)",
                 "M3_ctx(20%)", "M4_drug(20%)", "M6_DEP(5%)"]].to_string(index=False))
