"""Third model family (embedding MLP) + GBDT tuning + the ensemble of the two.

All on the inner mirror.  The official val mirror stays untouched until
scripts/14_final_eval.py runs once at the end.
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
from vcell.nn import ResidualNN                                             # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
pd.set_option("display.width", 260)
CFG = {"lam_plate": 0.3, "lam_plate_x_strain": 2.0}

base_fold = build_fold()
folds = [build_fold(splits=make_inner_splits(base_fold.meta, hold_strain=s, seed=i))
         for i, s in [(0, "CEK"), (1, "CGD")]]

prepped = []
for fo in folds:
    um = UnifiedBackfit(
        batch_factors=[(n, c, CFG.get(f"lam_{n}", l)) for n, c, l in BATCH_FACTORS],
        pert_factors=[(n, c, CFG.get(f"lam_{n}", l)) for n, c, l in PERT_FACTORS],
    ).fit(fo.meta, fo.Y_obs, fo.obs_mask)
    prepped.append((fo, um.predict()))
print("additive base fitted for both inner folds")

t0, rows = time.time(), []


def report(tag, preds):
    acc = [summary_row(tag, evaluate(fo, p, INNER)) for (fo, _), p in zip(prepped, preds)]
    s = pd.DataFrame(acc).mean(numeric_only=True)
    rows.append({"model": tag, **s.to_dict()})
    print(f"  {tag:34s} TOTAL={s['TOTAL']:.4f}  M2={s['M2_rawFC(25%)']:.3f} "
          f"M3={s['M3_ctx(20%)']:.3f} M4={s['M4_drug(20%)']:.3f} "
          f"M6={s['M6_DEP(5%)']:.3f}  ({time.time()-t0:.0f}s)")
    return s


report("additive only", [b for _, b in prepped])

# ---------------------------------------------------------------- GBDT (reference)
gbdt = []
for fo, b in prepped:
    rb = ResidualBooster(n_comp=96, n_estimators=800, learning_rate=0.03)
    rb.fit(fo.meta, fo.Y_obs, fo.obs_mask, b)
    gbdt.append(rb.predict())
report("+ GBDT residual", [b + g for (_, b), g in zip(prepped, gbdt)])

# ------------------------------------------------------------------------- GBDT tuning
print("\n=== GBDT capacity ===")
for leaves, mcs in [(63, 20), (127, 20), (31, 10)]:
    preds = []
    for fo, b in prepped:
        rb = ResidualBooster(n_comp=96, n_estimators=800, learning_rate=0.03,
                             num_leaves=leaves)
        rb.n_jobs = 8
        rb.fit(fo.meta, fo.Y_obs, fo.obs_mask, b)
        preds.append(rb.predict())
    report(f"GBDT leaves={leaves}", [b + p for (_, b), p in zip(prepped, preds)])

# --------------------------------------------------------------------------- MLP
print("\n=== embedding MLP ===")
best_nn, best_tot = None, -9
for tag, kw in [("MLP h256 e400", dict(hidden=256, epochs=400)),
                ("MLP h512 e400", dict(hidden=512, epochs=400)),
                ("MLP h256 e800", dict(hidden=256, epochs=800)),
                ("MLP h256 e400 wd1e-3", dict(hidden=256, epochs=400,
                                              weight_decay=1e-3))]:
    preds = []
    for fo, b in prepped:
        nn_ = ResidualNN(n_comp=96, n_seeds=3, **kw)
        nn_.fit(fo.meta, fo.Y_obs, fo.obs_mask, b)
        preds.append(nn_.predict())
    s = report(tag, [b + p for (_, b), p in zip(prepped, preds)])
    if s["TOTAL"] > best_tot:
        best_tot, best_nn = s["TOTAL"], preds

# ---------------------------------------------------------------------- ensemble
print("\n=== ensemble of GBDT and MLP residuals ===")
for w in (0.25, 0.5, 0.75):
    report(f"ensemble {1-w:.2f}*GBDT + {w:.2f}*MLP",
           [b + (1 - w) * g + w * m for (_, b), g, m in zip(prepped, gbdt, best_nn)])

df = pd.DataFrame(rows).sort_values("TOTAL", ascending=False)
df.to_csv(os.path.join(OUT, "nn_ensemble.csv"), index=False)
print("\n" + df[["model", "TOTAL", "M1_abs(20%)", "M2_rawFC(25%)", "M3_ctx(20%)",
                 "M4_drug(20%)", "M5_bt(10%)", "M6_DEP(5%)"]].to_string(index=False))
