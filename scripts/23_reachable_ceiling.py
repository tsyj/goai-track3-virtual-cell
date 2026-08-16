"""How much is actually left, per split?

The "batch + true Delta" oracle in the information ladder is unreachable: it uses
the target sample's own measurement.  This builds a *reference estimator* from
information a model is allowed to have, as a sanity check on whether our model is
leaving obvious signal behind:

  unseen strain (S2)  : that compound's measured Delta in the same
                        medium/temperature/time/source, averaged over the
                        strains that ARE visible
  time extrapolation  : that compound+strain's measured Delta at the OTHER time
                        points
  unseen compound(S1) : nothing compound-specific is knowable, so the reachable
                        Delta is the context mean -- i.e. what we already predict
  both unseen (S3)    : likewise nothing

The baseline is our own batch model in every case, so the unseen-strain baseline
error is charged to the reference too.

This is a reference, NOT a ceiling: on the time split our model beats it, because
pooling a compound's Delta over *other* time points is a poor estimator when the
response is strongly time-dependent.  Only the unseen-strain row is informative.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcell.harness import build_fold, evaluate, summary_row               # noqa: E402
from vcell.models import (BATCH_FACTORS, PERT_FACTORS, ResidualBooster,   # noqa: E402
                          UnifiedBackfit)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
pd.set_option("display.width", 280)
CFG = {"lam_plate": 0.3, "lam_plate_x_strain": 2.0}

f = build_fold(vehicle="both")
meta, n = f.meta, len(f.meta)
um = UnifiedBackfit(
    batch_factors=[(a, c, CFG.get(f"lam_{a}", l)) for a, c, l in BATCH_FACTORS],
    pert_factors=[(a, c, CFG.get(f"lam_{a}", l)) for a, c, l in PERT_FACTORS],
).fit(meta, f.Y_obs, f.obs_mask)
B = um.predict()
rb = ResidualBooster(n_comp=96, n_estimators=800, learning_rate=0.03)
rb.fit(meta, f.Y_obs, f.obs_mask, B)
ours = B + rb.predict()

D = np.where(np.isfinite(f.C_true), f.Y - f.C_true, np.nan).astype(np.float32)
vis = f.obs_mask & (~meta["is_control"]).to_numpy() & (~meta["is_qc"]).to_numpy()


def group_mean_excluding(keys, exclude_key):
    """Mean visible Delta per group, where the group deliberately drops one field."""
    out = np.full_like(D, np.nan)
    k = meta[keys].astype(str).agg("|".join, axis=1).to_numpy()
    for g in np.unique(k):
        rows = np.where(k == g)[0]
        src = rows[vis[rows]]
        if len(src):
            out[rows] = np.nanmean(D[src], axis=0)
    return out


# S2: same compound + same medium/temp/time/source, pooled over visible strains
d_strain = group_mean_excluding(
    ["compound", "Medium", "Temperature", "pert_time", "data_source"], "Strains")
# time: same compound + strain + medium/temp/source, pooled over the other times
d_time = group_mean_excluding(
    ["compound", "Strains", "Medium", "Temperature", "data_source"], "pert_time")

reach = np.zeros_like(D)
s2 = (meta["split_final"] == "val_strain_only").to_numpy()
st = (meta["split_final"] == "val_time").to_numpy()
reach[s2] = np.nan_to_num(d_strain[s2])
reach[st] = np.nan_to_num(d_time[st])
print("reachable Delta available for: strain_only %.1f%%, time %.1f%%"
      % (100 * np.isfinite(d_strain[s2]).any(1).mean(),
         100 * np.isfinite(d_time[st]).any(1).mean()))

rows = [summary_row("our model", evaluate(f, ours)),
        summary_row("reference estimator (pooled Delta)", evaluate(f, B + reach)),
        summary_row("UNREACHABLE oracle (true Delta)", evaluate(f, B + np.nan_to_num(D)))]
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "reachable_ceiling.csv"), index=False)
print("\n" + df[["model", "TOTAL", "M2_rawFC(25%)", "M4_drug(20%)",
                 "FC[strain_only]", "FC[time]"]].to_string(index=False))

lo = summary_row("floor", evaluate(f, B))
print("\n=== our model vs the pooled reference estimator ===")
for c in ("FC[strain_only]", "FC[time]"):
    a, b, o = lo[c], df.iloc[0][c], df.iloc[1][c]
    verdict = ("reference is better -- signal left behind" if o > b
               else "our model is better -- reference is not a ceiling")
    print(f"  {c:16s} batch-only={a:.3f}  ours={b:.3f}  reference={o:.3f}   {verdict}")
