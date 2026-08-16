"""A clean version of the M3 demonstration.

The earlier gamma-sweep added a generic drug response *on top of* the plate term,
which already contains the average treated level -- so it double-counted, and the
decline it produced was not purely a metric artefact.

This version compares two estimators of E[y_treat | unseen compound, cell] that
are both legitimate and both use only training labels:

  A. the model's smoothed plate x strain estimate (lower variance, slight bias)
  B. m_ctx, the empirical mean of the *training compounds measured in that exact
     cell* -- unbiased, conditioned on the exact context, higher variance

and interpolates between them.  B is the quantity the metric itself subtracts, so
if the score falls as the prediction moves toward B, the score is rewarding
*failure to reproduce the reference's noise*, not prediction quality.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcell.harness import build_fold, evaluate, summary_row     # noqa: E402
from vcell.models import UnifiedBackfit                         # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
pd.set_option("display.width", 260)

f = build_fold(vehicle="both")
meta = f.meta
um = UnifiedBackfit().fit(meta, f.Y_obs, f.obs_mask)
P0 = um.predict()

# m_ctx: mean measured value of the TRAINING compounds in the same
# (source, strain, medium, temp, time, plate) cell -- exactly the grouping the
# scorer uses for mu_ctx, and computable from training labels alone.
ctx = meta[list(f.scorer.cfg.ctx_cols)].astype(str).agg("|".join, axis=1).to_numpy()
train_treated = f.obs_mask & (~meta["is_control"]).to_numpy() & (~meta["is_qc"]).to_numpy()
m_ctx = np.full_like(P0, np.nan)
for c in np.unique(ctx):
    rows = np.where(ctx == c)[0]
    src = rows[train_treated[rows]]
    if len(src):
        m_ctx[rows] = np.nanmean(f.Y[src], axis=0)
have = np.isfinite(m_ctx).any(1)
print(f"cells with a usable empirical context mean: {have.mean():.1%} of rows")

unseen = ~meta["compound"].isin(
    meta.loc[f.obs_mask & ~meta["is_control"], "compound"].unique()).to_numpy()
print(f"rows with an unseen compound: {unseen.sum()}")
target = unseen & have
Mc = np.where(np.isfinite(m_ctx), m_ctx, P0)

rows = []
print("\nlam = 0 : model's smoothed estimate      lam = 1 : empirical context mean")
for lam in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
    P = P0.copy()
    P[target] = (1 - lam) * P0[target] + lam * Mc[target]
    r = summary_row(f"lam={lam}", evaluate(f, P))
    r["lam"] = lam
    # how far the prediction is from the reference the metric subtracts
    d = np.sqrt(np.nanmean((P[target] - Mc[target]) ** 2))
    r["rms_to_reference"] = d
    rows.append(r)
    print(f"  lam={lam:<4} |pred - m_ctx| = {d:.4f}   M3={r['M3_ctx(20%)']:.4f}   "
          f"M2={r['M2_rawFC(25%)']:.4f}   TOTAL={r['TOTAL']:.4f}")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "m3_clean.csv"), index=False)
print("\nAs the prediction approaches the metric's own reference -- an unbiased, "
      "context-conditional\nestimate built only from training labels -- M3 falls "
      "toward zero.")

# the same question for M4 / mu_drug on the unseen-strain split
print("\n=== same test for M4 on the unseen strain ===")
drug = meta["compound"].astype(str).to_numpy()
m_drug = np.full_like(P0, np.nan)
D = np.where(np.isfinite(f.C_true), f.Y - f.C_true, np.nan)
for c in np.unique(drug):
    rows_ = np.where(drug == c)[0]
    src = rows_[train_treated[rows_]]
    if len(src):
        m_drug[rows_] = np.nanmean(D[src], axis=0)
s2 = (meta["split_final"] == "val_strain_only").to_numpy() & np.isfinite(m_drug).any(1)
out = []
for lam in (0.0, 0.5, 1.0):
    P = P0.copy()
    ref = np.where(np.isfinite(f.C_obs), f.C_obs, um.predict(0.0)) + np.nan_to_num(m_drug)
    P[s2] = (1 - lam) * P0[s2] + lam * ref[s2]
    r = summary_row(f"m4 lam={lam}", evaluate(f, P))
    out.append({"lam": lam, **r})
    print(f"  lam={lam:<4} M4={r['M4_drug(20%)']:.4f}  TOTAL={r['TOTAL']:.4f}")
pd.DataFrame(out).to_csv(os.path.join(OUT, "m4_clean.csv"), index=False)
