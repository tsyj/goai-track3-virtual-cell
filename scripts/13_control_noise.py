"""Decisive test of the shared-control-noise artefact.

One fixed model (UnifiedBackfit never looks at control wells, so the prediction
is identical throughout).  Only the *scoring* changes:

  (a) how many matched control wells the reference averages over -- more wells
      means a less noisy y_control;
  (b) whether Delta_pred is built from the measured control (the handbook's
      notation) or from the model's own prediction for the control sample.

If the score rises as the control gets noisier, the metric is partly rewarding
noise-sharing rather than prediction quality.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcell.design import control_reference, match_controls    # noqa: E402
from vcell.harness import VAL, build_fold, summary_row         # noqa: E402
from vcell.metrics import ScoreConfig, Scorer                  # noqa: E402
from vcell.models import UnifiedBackfit                        # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
pd.set_option("display.width", 240)

f = build_fold(vehicle="both")
meta = f.meta
um = UnifiedBackfit().fit(meta, f.Y_obs, f.obs_mask)
P = um.predict()
obs = f.obs_mask
masks = {s: (meta["split_final"] == s).to_numpy() for s in meta["split_final"].unique()}

full = match_controls(meta, strategy="both")
rng = np.random.default_rng(0)


def truncated(k):
    """Keep at most k matched control wells per sample (deterministic subsample)."""
    return pd.Series([rows if k is None or len(rows) <= k
                      else list(rng.permutation(rows)[:k]) for rows in full],
                     index=full.index)


rows = []
print("=== score vs number of control wells averaged (identical prediction) ===")
for k in (1, 2, 3, None):
    cr = truncated(k)
    C = control_reference(f.Y, cr)
    navg = np.mean([len(r) for r in cr])
    # how noisy is this reference?  spread of single wells around their own mean
    resid = []
    for i, r in enumerate(cr):
        if len(r) >= 2:
            resid.append(np.nanstd(f.Y[r], axis=0, ddof=1))
    noise = float(np.nanmean(np.concatenate(resid))) if resid else float("nan")
    sc = Scorer(meta, f.Y, C, obs, masks, ScoreConfig(), control_rows=cr)
    r = summary_row(f"controls<= {k}", sc.report(P, **VAL))
    r.update({"k": k or 99, "mean_wells": navg, "well_sd": noise})
    rows.append(r)
    print(f"  wells<= {str(k):4s} (mean {navg:.2f}/sample, between-well sd {noise:.3f})"
          f"  TOTAL={r['TOTAL']:.4f}  M2={r['M2_rawFC(25%)']:.4f}  "
          f"M4={r['M4_drug(20%)']:.4f}  M6={r['M6_DEP(5%)']:.4f}")

print("\n=== Delta_pred built from the measured vs the predicted control ===")
for mode in ("measured", "predicted"):
    sc = Scorer(meta, f.Y, f.C_true, obs, masks,
                ScoreConfig(delta_mode=mode), control_rows=full)
    r = summary_row(f"delta_mode={mode}", sc.report(P, **VAL))
    r.update({"k": -1, "mean_wells": np.nan, "well_sd": np.nan})
    rows.append(r)
    print(f"  {mode:10s} TOTAL={r['TOTAL']:.4f}  M2={r['M2_rawFC(25%)']:.4f}  "
          f"M3={r['M3_ctx(20%)']:.4f}  M4={r['M4_drug(20%)']:.4f}  "
          f"M6={r['M6_DEP(5%)']:.4f}")

    # under the 'predicted' reading, does a better perturbation model now help?
    for tag, Q in [("batch only", um.predict(pert_scale=0.0)), ("with pert", P)]:
        rr = summary_row(tag, sc.report(Q, **VAL))
        print(f"     {tag:12s} TOTAL={rr['TOTAL']:.4f} M2={rr['M2_rawFC(25%)']:.4f} "
              f"M3={rr['M3_ctx(20%)']:.4f} M4={rr['M4_drug(20%)']:.4f}")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "control_noise.csv"), index=False)
print("\nwrote results/control_noise.csv")
