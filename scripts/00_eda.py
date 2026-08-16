"""EDA: missingness, batch structure, strain signatures, variance decomposition."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcell.harness import build_fold  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 220)

f = build_fold()
Y, m = f.Y, f.meta
print(f"samples={Y.shape[0]}  proteins={Y.shape[1]}  NaN={np.isnan(Y).mean():.3f}")

# ---- missingness -----------------------------------------------------------
per_s = np.isnan(Y).mean(1)
per_p = np.isnan(Y).mean(0)
print("\nmissing per sample  q10/50/90:", np.percentile(per_s, [10, 50, 90]).round(3))
print("missing per protein q10/50/90:", np.percentile(per_p, [10, 50, 90]).round(3))
print("proteins never missing:", (per_p == 0).sum(),
      "| >90% missing:", (per_p > 0.9).sum())
obs_mean = np.nanmean(Y, axis=0)
ok = np.isfinite(obs_mean)
print("corr(protein missing-rate, mean log2 abundance) =",
      round(float(np.corrcoef(per_p[ok], obs_mean[ok])[0, 1]), 3),
      " -> negative == missing-not-at-random / below detection")

# ---- sample-level loading offsets -----------------------------------------
off = np.nanmedian(Y, axis=1)
m2 = m.copy()
m2["offset"] = off
print("\nsample median log2 offset: sd=%.3f  range=%.2f..%.2f"
      % (off.std(), off.min(), off.max()))
for k in ["data_source", "instrument", "Yeast_cell_plate"]:
    g = m2.groupby(k)["offset"].mean()
    print(f"  between-{k:17s} sd={g.std():.3f}")
resid = off - m2.groupby("Yeast_cell_plate")["offset"].transform("mean")
print(f"  within-plate residual  sd={resid.std():.3f}"
      "   <- unpredictable part of the per-sample offset")

# ---- variance decomposition on centred log2 --------------------------------
Z = Y - np.nanmean(Y, axis=0, keepdims=True)
Z = Z - np.nanmedian(Z, axis=1, keepdims=True)          # remove loading offset
tot = np.nanvar(Z)
rows = []
for k in ["data_source", "instrument", "Yeast_cell_plate", "Strains", "Medium",
          "Temperature", "pert_time", "compound"]:
    codes = pd.factorize(m[k].astype(str))[0]
    ss = 0.0
    for c in np.unique(codes):
        sel = codes == c
        gm = np.nanmean(Z[sel], axis=0)
        ss += sel.sum() * np.nansum(gm ** 2)
    rows.append({"factor": k, "var_frac": ss / (tot * Z.shape[0] * Z.shape[1])})
vd = pd.DataFrame(rows).sort_values("var_frac", ascending=False)
print("\nmarginal variance explained (each factor alone, log2 after loading-norm):")
print(vd.to_string(index=False))
vd.to_csv(os.path.join(OUT, "variance_decomposition.csv"), index=False)

# ---- strain signatures: is a strain missing a protein entirely? ------------
print("\n--- strain-specific dropouts (knock-out signature test) ---")
hits = []
for s in sorted(m["Strains"].unique()):
    a = m["Strains"].to_numpy() == s
    miss_in = np.isnan(Y[a]).mean(0)
    miss_out = np.isnan(Y[~a]).mean(0)
    cand = np.where((miss_in > 0.95) & (miss_out < 0.30))[0]
    for c in cand:
        hits.append({"strain": s, "protein": f.proteins[c],
                     "miss_in_strain": round(float(miss_in[c]), 3),
                     "miss_elsewhere": round(float(miss_out[c]), 3)})
    lo = np.nanmean(Y[a], 0) - np.nanmean(Y[~a], 0)
    k = np.argsort(lo)[:5]
    print(f"{s:7s} n={a.sum():5d}  full-dropouts={len(cand):3d}  "
          f"most-depleted={[ (f.proteins[i], round(float(lo[i]),2)) for i in k ]}")
pd.DataFrame(hits).to_csv(os.path.join(OUT, "strain_dropouts.csv"), index=False)
print("strain-specific dropout candidates written:", len(hits))
