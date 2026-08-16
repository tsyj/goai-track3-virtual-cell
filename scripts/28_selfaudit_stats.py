"""Adversarial review, track 1: attack our own numbers.

(a) Unit-test the metric primitives against scipy/sklearn.  Every result in this
    repo is measured with our re-implementation of the official scorer; if that
    is wrong, nothing else means anything.
(b) Paired bootstrap over evaluation samples.  We reported 0.4711 -> 0.4811 from
    adding the residual booster and never checked whether that gap survives
    resampling.  Same resample for both models, so the comparison is paired.
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcell.harness import VAL, build_fold, summary_row                    # noqa: E402
from vcell.metrics import _average_precision, _pcc_rows, _r2_rows         # noqa: E402
from vcell.models import (BATCH_FACTORS, PERT_FACTORS, ResidualBooster,   # noqa: E402
                          UnifiedBackfit)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
pd.set_option("display.width", 280)
CFG = {"lam_plate": 0.3, "lam_plate_x_strain": 2.0}

# ---------------------------------------------------------------- (a) unit tests
print("=== metric primitives vs scipy / sklearn ===")
from scipy.stats import pearsonr                                          # noqa: E402
from sklearn.metrics import average_precision_score, r2_score             # noqa: E402

rng = np.random.default_rng(0)
A = rng.normal(size=(40, 300))
B = 0.4 * A + rng.normal(size=(40, 300))
A[rng.random(A.shape) < 0.25] = np.nan          # NaN on one side
B[rng.random(B.shape) < 0.25] = np.nan          # and the other

ours = _pcc_rows(A, B)
ref = []
for i in range(len(A)):
    m = np.isfinite(A[i]) & np.isfinite(B[i])
    ref.append(pearsonr(A[i][m], B[i][m])[0])
d = np.nanmax(np.abs(ours - np.array(ref)))
print(f"  PCC   max |ours - scipy|            = {d:.2e}   {'PASS' if d < 1e-9 else 'FAIL'}")

ours = _r2_rows(A, B)
ref = []
for i in range(len(A)):
    m = np.isfinite(A[i]) & np.isfinite(B[i])
    ref.append(r2_score(B[i][m], A[i][m]))      # r2_score(y_true, y_pred)
d = np.nanmax(np.abs(ours - np.array(ref)))
print(f"  R^2   max |ours - sklearn|          = {d:.2e}   {'PASS' if d < 1e-9 else 'FAIL'}")

sc = rng.random(500)
lb = (rng.random(500) < 0.2).astype(float)
d = abs(_average_precision(sc, lb) - average_precision_score(lb, sc))
print(f"  AP    |ours - sklearn|              = {d:.2e}   {'PASS' if d < 1e-9 else 'FAIL'}")

# degenerate cases the official scorer will also hit
z = np.zeros((3, 50))
print(f"  PCC of a constant row -> NaN        : {np.all(np.isnan(_pcc_rows(z, A[:3, :50])))}")
print(f"  PCC with <3 finite pairs -> NaN     : "
      f"{np.isnan(_pcc_rows(np.array([[1., 2., np.nan]]), np.array([[1., 2., np.nan]])))[0]}")

# ------------------------------------------------------- (b) paired bootstrap
print("\n=== paired bootstrap over evaluation samples ===")
f = build_fold(vehicle="both")
meta = f.meta
um = UnifiedBackfit(
    batch_factors=[(a, c, CFG.get(f"lam_{a}", l)) for a, c, l in BATCH_FACTORS],
    pert_factors=[(a, c, CFG.get(f"lam_{a}", l)) for a, c, l in PERT_FACTORS],
).fit(meta, f.Y_obs, f.obs_mask)
P_add = um.predict()
rb = ResidualBooster(n_comp=96, n_estimators=800, learning_rate=0.03, n_jobs=32)
rb.fit(meta, f.Y_obs, f.obs_mask, P_add)
P_boost = P_add + rb.predict()
print("  point estimates: additive %.4f | + booster %.4f"
      % (summary_row("a", f.scorer.report(P_add, **VAL))["TOTAL"],
         summary_row("b", f.scorer.report(P_boost, **VAL))["TOTAL"]))

sc = f.scorer
splits = {k: np.where(sc.eval_masks[v])[0] for k, v in VAL.items()}
B_ITER = int(os.environ.get("VCELL_BOOT", 200))
t0 = time.time()


def total_on(P, rows_by_split):
    """Recompute the weighted total on a given (possibly resampled) row set."""
    allr = np.concatenate([rows_by_split[k] for k in ("s1", "s2", "s3", "stime")])
    out = {"M1_absolute": sc.m1_absolute(P, allr), "M2_rawFC": sc.m2_rawfc(P, allr),
           "M3_ctx_resid": sc.m3_ctx_resid(P, rows_by_split["s1"]),
           "M4_drug_resid": sc.m4_drug_resid(P, rows_by_split["s2"]),
           "M6_high_effect": sc.m6_high_effect(P, allr)}
    both, tm = rows_by_split["s3"], rows_by_split["stime"]
    out["M5_both_time"] = {"score": np.mean([
        sc.m2_rawfc(P, both)["score"], sc.m1_absolute(P, both)["score"],
        sc.m1_absolute(P, tm)["score"], sc.m2_rawfc(P, tm)["score"]])}
    return float(sum(sc.cfg.weights[k] * out[k]["score"] for k in sc.cfg.weights))


boot = np.zeros((B_ITER, 3))
brng = np.random.default_rng(12345)
for b in range(B_ITER):
    rs = {k: brng.choice(v, size=len(v), replace=True) for k, v in splits.items()}
    a, c = total_on(P_add, rs), total_on(P_boost, rs)
    boot[b] = (a, c, c - a)
    if (b + 1) % 50 == 0:
        print(f"    {b+1}/{B_ITER} ({time.time()-t0:.0f}s)", flush=True)

lo_a, hi_a = np.percentile(boot[:, 0], [2.5, 97.5])
lo_c, hi_c = np.percentile(boot[:, 1], [2.5, 97.5])
lo_d, hi_d = np.percentile(boot[:, 2], [2.5, 97.5])
print(f"\n  additive          {boot[:,0].mean():.4f}  95% CI [{lo_a:.4f}, {hi_a:.4f}]")
print(f"  additive+booster  {boot[:,1].mean():.4f}  95% CI [{lo_c:.4f}, {hi_c:.4f}]")
print(f"  paired difference {boot[:,2].mean():+.4f} 95% CI [{lo_d:+.4f}, {hi_d:+.4f}]"
      f"   {'REAL (excludes 0)' if lo_d > 0 else 'NOT SIGNIFICANT'}")
print(f"\n  -> a single val score carries about +-{(hi_a-lo_a)/2:.3f}; differences "
      f"smaller than that\n     cannot be read off one number, only from the paired "
      f"comparison.")
pd.DataFrame(boot, columns=["additive", "booster", "diff"]).to_csv(
    os.path.join(OUT, "bootstrap_val.csv"), index=False)
