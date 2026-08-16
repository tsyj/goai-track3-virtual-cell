"""A compound x context term that is POOLED OVER STRAINS.

scripts/23 found that a plain empirical mean of a compound's Delta over the
*visible* strains, within the same medium/temperature/time/source, scores 0.373
on the unseen-strain split -- better than our model's 0.346.  That estimator is
available for a held-out strain, and the additive model cannot represent it: it
only has compound main effects and two-way compound x context tables, not the
full compound x medium x temperature x time x source cell.

Earlier (scripts/17) a cell term keyed on (compound, strain, medium, temp, time)
was tested and rejected -- but that key *includes strain*, so it evaluates to zero
for exactly the split that needed it.  Dropping strain from the key is the fix.
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
CTX = ("compound", "Medium", "Temperature", "pert_time", "data_source")

base_fold = build_fold()
folds = [build_fold(splits=make_inner_splits(base_fold.meta, hold_strain=s, seed=i))
         for i, s in [(0, "CEK"), (1, "CGD")]]
t0, rows = time.time(), []


def run(tag, pert, boost=False):
    acc = []
    for fo in folds:
        um = UnifiedBackfit(
            batch_factors=[(a, c, CFG.get(f"lam_{a}", l)) for a, c, l in BATCH_FACTORS],
            pert_factors=pert).fit(fo.meta, fo.Y_obs, fo.obs_mask)
        P = um.predict()
        if boost:
            rb = ResidualBooster(n_comp=96, n_estimators=800, learning_rate=0.03)
            rb.fit(fo.meta, fo.Y_obs, fo.obs_mask, P)
            P = P + rb.predict()
        acc.append(summary_row(tag, evaluate(fo, P, INNER)))
    s = pd.DataFrame(acc).mean(numeric_only=True)
    rows.append({"model": tag, **s.to_dict()})
    print(f"  {tag:44s} TOTAL={s['TOTAL']:.4f}  M2={s['M2_rawFC(25%)']:.3f} "
          f"M3={s['M3_ctx(20%)']:.3f} M4={s['M4_drug(20%)']:.3f} "
          f"FC[str]={s['FC[strain_only]']:.3f}  ({time.time()-t0:.0f}s)")
    return s


print("=== additive model only ===")
run("baseline", PERT_FACTORS)
for lam in (1.0, 4.0, 10.0, 25.0):
    run(f"+ cmpd x ctx (no strain), lam={lam}",
        PERT_FACTORS + [("cmpd_x_ctx", CTX, lam)])

df = pd.DataFrame(rows)
best = df.iloc[1:]["TOTAL"].idxmax()
best_lam = float(df.loc[best, "model"].split("lam=")[1])
print(f"\nbest lambda = {best_lam}")

print("\n=== with the residual booster on top ===")
run("baseline + booster", PERT_FACTORS, boost=True)
run(f"+ cmpd x ctx lam={best_lam} + booster",
    PERT_FACTORS + [("cmpd_x_ctx", CTX, best_lam)], boost=True)

df = pd.DataFrame(rows).sort_values("TOTAL", ascending=False)
df.to_csv(os.path.join(OUT, "cmpd_context_term.csv"), index=False)
print("\n" + df[["model", "TOTAL", "M2_rawFC(25%)", "M3_ctx(20%)", "M4_drug(20%)",
                 "M5_bt(10%)", "FC[strain_only]", "FC[both]", "FC[time]"]]
      .to_string(index=False))
