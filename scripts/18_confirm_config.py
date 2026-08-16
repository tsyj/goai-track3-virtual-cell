"""Fine sweep of the plate shrinkage, then confirm the merged config.

The plate term carries 88% of the variance and the coarse grid in
scripts/08_tune_inner.py preferred its weakest setting, so it is worth resolving
properly.  Still on the inner mirror; val is scored once, afterwards.
"""
import json
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
folds = [build_fold(splits=make_inner_splits(base.meta, hold_strain=s, seed=i))
         for i, s in [(0, "CEK"), (1, "CGD"), (2, "DHY210")]]


def run(cfg):
    acc = []
    for fo in folds:
        um = UnifiedBackfit(
            batch_factors=[(n, c, cfg.get(f"lam_{n}", l)) for n, c, l in BATCH_FACTORS],
            pert_factors=[(n, c, cfg.get(f"lam_{n}", l)) for n, c, l in PERT_FACTORS],
            n_pass=cfg.get("n_pass", 6)).fit(fo.meta, fo.Y_obs, fo.obs_mask)
        acc.append(summary_row("x", evaluate(fo, um.predict(), INNER)))
    return pd.DataFrame(acc).mean(numeric_only=True)


t0, rows = time.time(), []
print("=== plate shrinkage ===")
for lp in (0.02, 0.1, 0.3, 0.6, 1.0):
    m = run({"lam_plate": lp})
    rows.append({"trial": f"lam_plate={lp}", "lam_plate": lp, **m.to_dict()})
    print(f"  lam_plate={lp:<5} inner TOTAL={m['TOTAL']:.4f}  M1={m['M1_abs(20%)']:.3f} "
          f"M2={m['M2_rawFC(25%)']:.3f} M4={m['M4_drug(20%)']:.3f}  ({time.time()-t0:.0f}s)")

best_lp = max(rows, key=lambda r: r["TOTAL"])["lam_plate"]
print(f"\nbest plate shrinkage: {best_lp}")

print("\n=== merged candidates ===")
cands = {
    "plate only": {"lam_plate": best_lp},
    "+ strain 1.0": {"lam_plate": best_lp, "lam_strain": 1.0},
    "+ plate_x_strain 2": {"lam_plate": best_lp, "lam_plate_x_strain": 2.0},
    "+ compound 4": {"lam_plate": best_lp, "lam_compound": 4.0},
    "+ n_pass 10": {"lam_plate": best_lp, "n_pass": 10},
}
best, best_cfg = -9, {"lam_plate": best_lp}
for name, cfg in cands.items():
    m = run(cfg)
    rows.append({"trial": name, **m.to_dict()})
    print(f"  {name:22s} inner TOTAL={m['TOTAL']:.4f}  ({time.time()-t0:.0f}s)")
    if m["TOTAL"] > best:
        best, best_cfg = m["TOTAL"], cfg

df = pd.DataFrame(rows).sort_values("TOTAL", ascending=False)
df.to_csv(os.path.join(OUT, "confirm_config.csv"), index=False)
print("\n" + df[["trial", "TOTAL", "M1_abs(20%)", "M2_rawFC(25%)", "M3_ctx(20%)",
                 "M4_drug(20%)", "M6_DEP(5%)"]].to_string(index=False))
print("\nFINAL config:", json.dumps(best_cfg), f"inner TOTAL={best:.4f}")
json.dump({"config": best_cfg, "inner_total": best,
           "selected_by": "scripts/18_confirm_config.py, 3 inner folds",
           "rejected": ["low-rank truncation", "empirical-Bayes per-protein shrinkage",
                        "plate-position factor", "condition-cell interaction term"]},
          open(os.path.join(OUT, "best_config.json"), "w"), indent=1)
