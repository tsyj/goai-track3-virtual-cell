"""Exploit the replicate design: a full condition-cell interaction term.

WAYB was run three times over the same grid, so a sample at
(compound, strain, medium, temperature, time) usually has one or two siblings in
another data_source that ARE labelled.  An additive model cannot use them beyond
its main effects; a cell term can, and it is exactly the residual
compound x strain x context interaction the additive terms leave behind.

WAYC contributes one sample per cell, so its cell term would fit only that
sample's own noise -- which is what the shrinkage strength has to control.
Tested on the inner mirror.
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
from vcell.models import BATCH_FACTORS, PERT_FACTORS, UnifiedBackfit        # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
pd.set_option("display.width", 260)
CELL = ("compound", "Strains", "Medium", "Temperature", "pert_time")

base = build_fold()
key = base.meta[list(CELL)].astype(str).agg("|".join, axis=1)
vis = base.meta["split_final"].to_numpy() == "train"
sib = pd.DataFrame({"key": key, "vis": vis}).groupby("key")["vis"].sum()
n_sib = key.map(sib).to_numpy() - vis
print("labelled siblings available per row (same condition, other batch):")
print(pd.Series(n_sib).value_counts().sort_index().to_string())
print("rows with >=1 labelled sibling: %.1f%%" % (100 * (n_sib >= 1).mean()))

folds = []
for seed, strain in [(0, "CEK"), (1, "CGD")]:
    folds.append(build_fold(splits=make_inner_splits(base.meta, hold_strain=strain,
                                                     seed=seed)))

t0, rows = time.time(), []
for name, lam in [("baseline (no cell term)", None), ("cell lam=1", 1.0),
                  ("cell lam=3", 3.0), ("cell lam=8", 8.0), ("cell lam=20", 20.0)]:
    pf = PERT_FACTORS if lam is None else PERT_FACTORS + [("cell", CELL, lam)]
    acc = []
    for fo in folds:
        um = UnifiedBackfit(batch_factors=BATCH_FACTORS, pert_factors=pf).fit(
            fo.meta, fo.Y_obs, fo.obs_mask)
        acc.append(summary_row(name, evaluate(fo, um.predict(), INNER)))
    m = pd.DataFrame(acc).mean(numeric_only=True)
    rows.append({"trial": name, **m.to_dict()})
    print(f"  {name:24s} inner TOTAL={m['TOTAL']:.4f}  M1={m['M1_abs(20%)']:.3f} "
          f"M2={m['M2_rawFC(25%)']:.3f} M3={m['M3_ctx(20%)']:.3f} "
          f"M4={m['M4_drug(20%)']:.3f} M5={m['M5_bt(10%)']:.3f}  ({time.time()-t0:.0f}s)")

df = pd.DataFrame(rows).sort_values("TOTAL", ascending=False)
df.to_csv(os.path.join(OUT, "cell_term.csv"), index=False)
print("\n" + df[["trial", "TOTAL", "M1_abs(20%)", "M2_rawFC(25%)", "M3_ctx(20%)",
                 "M4_drug(20%)", "M5_bt(10%)", "M6_DEP(5%)"]].to_string(index=False))
