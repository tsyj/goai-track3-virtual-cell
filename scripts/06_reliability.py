"""Is a compound's response vector reliable enough to be transferable at all?

Split-half reliability: estimate each compound's effect vector twice from disjoint
halves of its own samples and correlate.  This is the ceiling on any
compound-level transfer -- a kernel cannot recover signal the estimate does not
contain.  Reported both for the raw effect and after low-rank denoising.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcell.chem import load_pubchem, similarity_matrix, transfer_weights  # noqa: E402
from vcell.harness import build_fold                                      # noqa: E402
from vcell.models import UnifiedBackfit                                   # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
pd.set_option("display.width", 260)
rng = np.random.default_rng(0)

f = build_fold(vehicle="both")
meta, obs = f.meta, f.obs_mask
treated = (~meta["is_control"] & ~meta["is_qc"]).to_numpy()

# Delta relative to the *modelled* batch level, so plate/strain are removed.
um = UnifiedBackfit(pert_factors=[]).fit(meta, f.Y_obs, obs)      # batch terms only
R = np.where((obs & treated)[:, None], f.Y - um.predict(), np.nan).astype(np.float32)

comp = meta["compound"].to_numpy()
src = meta["data_source"].to_numpy()
half = rng.integers(0, 2, len(meta))

rows = []
for c in sorted(set(comp[obs & treated])):
    idx = np.where((comp == c) & obs & treated)[0]
    if len(idx) < 20:
        continue
    a = np.nanmean(R[idx[half[idx] == 0]], 0)
    b = np.nanmean(R[idx[half[idx] == 1]], 0)
    m = np.isfinite(a) & np.isfinite(b)
    r = np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 100 else np.nan
    full = np.nanmean(R[idx], 0)
    rows.append({"compound": c, "n": len(idx), "source": pd.Series(src[idx]).mode()[0],
                 "effect_rms": float(np.sqrt(np.nanmean(full ** 2))),
                 "split_half_r": r,
                 # Spearman-Brown: reliability of the full-sample estimate
                 "reliability": (2 * r / (1 + r)) if np.isfinite(r) else np.nan})
rel = pd.DataFrame(rows).sort_values("effect_rms", ascending=False)
print("=== split-half reliability of per-compound effect vectors ===")
print(rel.to_string(index=False))
rel.to_csv(os.path.join(OUT, "compound_reliability.csv"), index=False)
print("\nmedian split-half r = %.3f | median full-estimate reliability = %.3f"
      % (rel.split_half_r.median(), rel.reliability.median()))
print("compounds with reliability > 0.5: %d / %d"
      % ((rel.reliability > 0.5).sum(), len(rel)))

# ---------------------------------------------------------------- low-rank space
A = np.stack([np.nanmean(R[np.where((comp == c) & obs & treated)[0]], 0)
              for c in rel.compound])
A = np.where(np.isfinite(A), A, 0.0)
keep = np.isfinite(f.Y).mean(0) > 0.5            # well-measured proteins only
Ak = A[:, keep]
Ak = Ak - Ak.mean(0)
U, S, Vt = np.linalg.svd(Ak, full_matrices=False)
ev = S ** 2 / (S ** 2).sum()
print("\n=== compound-effect spectrum (%d compounds x %d proteins) ==="
      % Ak.shape)
print("variance explained by first 10 PCs:", np.round(ev[:10], 3))
print("cumulative:", np.round(np.cumsum(ev[:10]), 3))

# how much of the split-half agreement survives at rank k?
for k in (1, 2, 3, 5, 10, 20):
    Ar = U[:, :k] @ np.diag(S[:k]) @ Vt[:k]
    rr = []
    for i, c in enumerate(rel.compound):
        idx = np.where((comp == c) & obs & treated)[0]
        b = np.nanmean(R[idx[half[idx] == 1]], 0)[keep]
        m = np.isfinite(b)
        rr.append(np.corrcoef(Ar[i][m], b[m])[0, 1])
    print(f"  rank {k:2d}: mean corr(rank-k full estimate, held-out half) = "
          f"{np.nanmean(rr):.3f}")

# ------------------------------------------------- chemistry -> PC coordinates
print("\n=== LOCO transfer in the low-rank space ===")
chem = load_pubchem().set_index("name").reindex(rel.compound).reset_index()
chem["moa"] = chem["moa"].fillna("unknown")
is_ctrl = rel.compound.isin(["Water", "DMSO", "Quality Control"]).to_numpy()
donors = np.where(~is_ctrl)[0]
w = rel.reliability.to_numpy()

for k in (3, 5, 10):
    Ar = (U[:, :k] * S[:k]) @ Vt[:k]
    for ws, wm in [(1.0, 0.0), (0.5, 0.5), (0.0, 1.0)]:
        Smat = similarity_matrix(chem, ws, wm)
        W = transfer_weights(Smat, donors, donors, topk=5, temp=0.3)
        pred = W @ Ar[donors]
        rr = []
        for i, t in enumerate(donors):
            idx = np.where((comp == rel.compound.iloc[t]) & obs & treated)[0]
            b = np.nanmean(R[idx[half[idx] == 1]], 0)[keep]
            m = np.isfinite(b) & np.isfinite(pred[i])
            rr.append(np.corrcoef(pred[i][m], b[m])[0, 1] if pred[i][m].std() > 0
                      else np.nan)
        rr = np.array(rr)
        print(f"  rank={k:2d} w_struct={ws} w_moa={wm}: mean r={np.nanmean(rr):+.3f} "
              f"frac>0={np.nanmean(rr > 0):.2f}")
