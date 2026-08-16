"""Two extra structural ideas, tested on the inner mirror (never on val).

1. plate position.  On a 96-well plate, edge wells evaporate and load differently.
   The layout is fixed across plates of the same panel, so a global well factor is
   partly confounded with strain x compound -- shrinkage has to sort that out, and
   whether it helps is an empirical question, not an assumption.
2. a bare 'treated vs control' indicator, which is the only perturbation term an
   unseen compound can receive.  It is the Bayes-optimal guess with no compound
   information, so it is worth knowing what it costs under the official metric.
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

base = build_fold()
base.meta["well_row"] = base.meta["protein_well"].str.extract(r"^([A-H])")[0]
base.meta["treated"] = (~base.meta["is_control"] & ~base.meta["is_qc"]).astype(str)

folds = []
for seed, strain in [(0, "CEK"), (1, "CGD")]:
    sp = make_inner_splits(base.meta, hold_strain=strain, seed=seed)
    fo = build_fold(splits=sp)
    fo.meta["well_row"] = fo.meta["protein_well"].str.extract(r"^([A-H])")[0]
    fo.meta["treated"] = (~fo.meta["is_control"] & ~fo.meta["is_qc"]).astype(str)
    folds.append((f"{strain}/s{seed}", fo))

TRIALS = {
    "baseline": (BATCH_FACTORS, PERT_FACTORS),
    "+ well position": (BATCH_FACTORS + [("well", ("protein_well",), 20.0)],
                        PERT_FACTORS),
    "+ well row": (BATCH_FACTORS + [("wellrow", ("well_row",), 10.0)], PERT_FACTORS),
    "+ treated indicator": (BATCH_FACTORS,
                            [("treated", ("treated",), 4.0)] + PERT_FACTORS),
    "+ treated x time": (BATCH_FACTORS,
                         [("treated", ("treated",), 4.0),
                          ("treated_x_time", ("treated", "pert_time"), 8.0)]
                         + PERT_FACTORS),
}

t0, rows = time.time(), []
for name, (bf, pf) in TRIALS.items():
    accum = []
    for _, fo in folds:
        um = UnifiedBackfit(batch_factors=bf, pert_factors=pf).fit(
            fo.meta, fo.Y_obs, fo.obs_mask)
        accum.append(summary_row(name, evaluate(fo, um.predict(), INNER)))
    m = pd.DataFrame(accum).mean(numeric_only=True)
    rows.append({"trial": name, **m.to_dict()})
    print(f"  {name:22s} inner TOTAL={m['TOTAL']:.4f}  M1={m['M1_abs(20%)']:.3f} "
          f"M2={m['M2_rawFC(25%)']:.3f} M3={m['M3_ctx(20%)']:.3f} "
          f"M4={m['M4_drug(20%)']:.3f}  ({time.time()-t0:.0f}s)")

df = pd.DataFrame(rows).sort_values("TOTAL", ascending=False)
df.to_csv(os.path.join(OUT, "extra_factors.csv"), index=False)
print("\n" + df[["trial", "TOTAL", "M1_abs(20%)", "M2_rawFC(25%)", "M3_ctx(20%)",
                 "M4_drug(20%)", "M6_DEP(5%)"]].to_string(index=False))
