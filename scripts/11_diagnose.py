"""Where is the remaining headroom?  Per-split diagnostics + an achievable ceiling.

The 'true Delta' oracle in the information ladder is not achievable: it contains
the individual sample's own measurement noise.  The honest ceiling is the
*denoised* effect -- the mean Delta over all replicate samples sharing
(compound, strain, medium, temperature, time) -- which is the best any model
could infer from the design.  Everything between that and our model is real,
recoverable headroom; everything above it is noise-fitting.
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
pd.set_option("display.width", 300)

f = build_fold(vehicle="both")
meta = f.meta
um = UnifiedBackfit().fit(meta, f.Y_obs, f.obs_mask)
batch = um.predict(pert_scale=0.0)

rows = [summary_row("batch only", evaluate(f, batch)),
        summary_row("our model", evaluate(f, um.predict())),
        summary_row("+ EB shrinkage",
                    evaluate(f, UnifiedBackfit(eb=True)
                             .fit(meta, f.Y_obs, f.obs_mask).predict()))]

# ---- achievable oracle: denoised Delta from replicate cells -----------------
cell = meta[["compound", "Strains", "Medium", "Temperature", "pert_time"]] \
    .astype(str).agg("|".join, axis=1).to_numpy()
D = np.where(np.isfinite(f.C_true), f.Y - f.C_true, np.nan).astype(np.float32)
den = np.full_like(D, np.nan)
for c in np.unique(cell):
    idx = np.where(cell == c)[0]
    den[idx] = np.nanmean(D[idx], axis=0)
n_rep = pd.Series(cell).value_counts()
print("replicate cells: %d, median samples per cell = %.0f"
      % (len(n_rep), n_rep.median()))
rows.append(summary_row("ORACLE denoised Delta (achievable)",
                        evaluate(f, batch + np.nan_to_num(den))))
rows.append(summary_row("ORACLE true Delta (not achievable)",
                        evaluate(f, batch + np.nan_to_num(D))))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "diagnostics.csv"), index=False)
cols = [c for c in df.columns if c.startswith(("model", "TOTAL", "M", "FC["))]
print("\n" + df[cols].to_string(index=False))

# ---- how much of each split's headroom have we captured? -------------------
print("\n=== fraction of achievable Delta-signal captured, per split ===")
lo = df[df.model == "batch only"].iloc[0]
mid = df[df.model == "our model"].iloc[0]
hi = df[df.model == "ORACLE denoised Delta (achievable)"].iloc[0]
for c in [c for c in df.columns if c.startswith("FC[")]:
    gap = hi[c] - lo[c]
    got = mid[c] - lo[c]
    print(f"  {c:22s} floor={lo[c]:.3f}  ours={mid[c]:.3f}  ceiling={hi[c]:.3f}  "
          f"captured={100*got/gap if gap > 1e-6 else float('nan'):5.1f}%")
