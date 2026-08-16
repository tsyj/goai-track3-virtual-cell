"""Noise ceiling + metric geometry.

Two questions that decide how to read every other number:

1. WAYB was run three times (WAYB / WAYB_rep1 / WAYB_rep2) over the same
   strain x compound x medium x temperature x time grid.  Feeding one replicate's
   *measured* proteome as the prediction for another gives an empirical upper
   bound: no model can beat the experiment's own reproducibility.

2. Every Delta-based module subtracts a quantity that is itself measured, so
   prediction and truth share noise terms.  We quantify how much of each module's
   score is signal and how much is that shared term, by scoring degenerate
   predictors that contain no information at all.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcell.harness import build_fold, evaluate, summary_row     # noqa: E402
from vcell.metrics import _pcc_rows                             # noqa: E402
from vcell.models import ControlBaseline                        # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
pd.set_option("display.width", 260)

f = build_fold(vehicle="both")
meta, Y, n = f.meta, f.Y, len(f.meta)
treated = (~meta["is_control"] & ~meta["is_qc"]).to_numpy()

# ---------------------------------------------------------------- effect sizes
D = f.Y - f.C_true
print("=== effect-size budget (log2) ===")
for tag, msk in [("all treated", treated),
                 ("val_chem_only", (meta.split_final == "val_chem_only").to_numpy()),
                 ("val_strain_only", (meta.split_final == "val_strain_only").to_numpy())]:
    d = D[msk]
    print(f"  {tag:16s} rms|Delta|={np.sqrt(np.nanmean(d**2)):.3f}  "
          f"frac |Delta|>1: {np.nanmean(np.abs(d) > 1):.3f}")

# ---------------------------------------------------------- replicate matching
rep_key = ["Strains", "compound", "Medium", "Temperature", "pert_time"]
wayb = meta["data_source"].str.startswith("WAYB").to_numpy()
sub = meta[wayb].copy()
sub["row"] = np.where(wayb)[0]
groups = sub.groupby(rep_key)["row"].apply(list)
pairs = [(a, b) for rows in groups if len(rows) >= 2
         for i, a in enumerate(rows) for b in rows[i + 1:]]
print(f"\n=== replicate pairs within WAYB/rep1/rep2: {len(pairs)} ===")
A = np.array([a for a, _ in pairs])
B_ = np.array([b for _, b in pairs])

print("  raw log2 profile   PCC = %.4f" % np.nanmean(_pcc_rows(Y[A], Y[B_])))
tr = treated[A] & treated[B_]
print("  Delta vs own ctrl  PCC = %.4f   (<- ceiling for module M2)"
      % np.nanmean(_pcc_rows(D[A][tr], D[B_][tr])))
d1, d2 = D[A][tr], D[B_][tr]
hi = np.isfinite(d1) & np.isfinite(d2) & (np.abs(d2) > 1)
with np.errstate(invalid="ignore"):
    acc = ((np.sign(d1) == np.sign(d2)) & hi).sum(1) / np.maximum(hi.sum(1), 1)
print("  high-effect direction agreement = %.4f   (<- ceiling for M6 dirAcc)"
      % np.nanmean(np.where(hi.sum(1) > 0, acc, np.nan)))

# --------------------------------------------- replicate-oracle as a prediction
# Predict each sample with a *different replicate's measured proteome*.
partner = np.full(n, -1)
for rows in groups:
    for i, a in enumerate(rows):
        partner[a] = rows[(i + 1) % len(rows)] if len(rows) > 1 else -1
cb = ControlBaseline().fit(meta, f.Y_obs)
Bbase = cb.predict()
P_rep = Bbase.copy()
has = partner >= 0
P_rep[has] = Y[partner[has]]
print("\nreplicate-oracle covers %d/%d rows (WAYB only)" % (has.sum(), n))

rows_out = []
rows_out.append(summary_row("ORACLE truth", evaluate(f, Y)))
rows_out.append(summary_row("ORACLE replicate-measurement", evaluate(f, P_rep)))

# ----------------------------------------------------- degenerate / null models
gm = np.nanmean(np.where(f.obs_mask[:, None], Y, np.nan), 0).astype(np.float32)
rows_out.append(summary_row("NULL protein-mean", evaluate(f, np.tile(gm, (n, 1)))))
rows_out.append(summary_row("NULL plate x strain (Delta=0)", evaluate(f, Bbase)))

rng = np.random.default_rng(0)
perm = rng.permutation(n)
rows_out.append(summary_row("NULL shuffled-sample truth", evaluate(f, Y[perm])))

df = pd.DataFrame(rows_out)
df.to_csv(os.path.join(OUT, "ceiling.csv"), index=False)
print("\n" + df.to_string(index=False))

print("""
Read-out
--------
* 'ORACLE replicate-measurement' is what a perfect model of the biology would
  score if it still had to be read out by this assay.  Anything above it is
  fitting measurement noise.
* 'NULL plate x strain' contains no perturbation information at all.  Whatever it
  scores on a module is that module's floor, not a result.
""")
