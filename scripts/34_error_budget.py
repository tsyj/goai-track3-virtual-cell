"""Where does the remaining prediction error actually live?

Every "oracle gap" in RESULTS.md is stated against an oracle that uses the target
sample's own measurement, so it cannot say how much of the gap is *reachable*.
This asks the reachable question directly, in log2 units rather than in
correlation units:

    err = y_hat - y   on held-out rows, decomposed into

      per-protein bias   mean_i err[i, p]      -- systematic, correctable by
                                                  recalibration
      per-sample offset  mean_p err[i, p]      -- the protein-loading nuisance the
                                                  model deliberately sets to zero
                                                  on held-out rows; it cancels out
                                                  of every correlation module but
                                                  NOT out of the R^2 half of M1
      residual           what is left

and compares the total against the replicate-derived measurement noise
(rms 0.26 log2 per sample, effect rms 0.146 -- scripts/02_ceiling.py).  If the
residual already sits at the noise floor, no amount of modelling moves it and the
remaining oracle gaps are structural.

Single job, meant to run alongside a parallel search.

Jiao Xinyuan 2026-08-16
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcell.harness import VAL, build_fold, evaluate, summary_row           # noqa: E402
from vcell.models import (BATCH_FACTORS, PERT_FACTORS, ResidualBooster,    # noqa: E402
                          UnifiedBackfit)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
CFG = {"lam_plate": 0.3, "lam_plate_x_strain": 2.0}
NOISE_RMS = 0.26        # single-sample measurement noise, from the WAYB triplicates
EFFECT_RMS = 0.146      # true perturbation effect

VAL_NAMES = ["val_chem_only", "val_strain_only", "val_both", "val_time"]


def rms(x):
    x = np.asarray(x, dtype=np.float64)
    m = np.isfinite(x)
    return float(np.sqrt((x[m] ** 2).mean())) if m.any() else float("nan")


fo = build_fold()
print(f"rows {fo.n}  proteins {fo.Y.shape[1]}  train rows {int(fo.obs_mask.sum())}", flush=True)

um = UnifiedBackfit(
    batch_factors=[(a, c, CFG.get(f"lam_{a}", l)) for a, c, l in BATCH_FACTORS],
    pert_factors=[(a, c, CFG.get(f"lam_{a}", l)) for a, c, l in PERT_FACTORS],
    n_pass=6,
).fit(fo.meta, fo.Y_obs, fo.obs_mask)
P_batch = um.predict(pert_scale=0.0)
P_add = um.predict()
rb = ResidualBooster(n_comp=96, n_estimators=800, learning_rate=0.03, n_jobs=8)
rb.fit(fo.meta, fo.Y_obs, fo.obs_mask, P_add)
P = P_add + rb.predict()
print("fitted; scoring", flush=True)
print(summary_row("full", evaluate(fo, P)), flush=True)

rows = []
split = fo.meta["split_final"].to_numpy()
for name in ["ALL VAL"] + VAL_NAMES:
    sel = np.isin(split, VAL_NAMES) if name == "ALL VAL" else (split == name)
    idx = np.where(sel)[0]
    if len(idx) == 0:
        continue
    E = P[idx] - fo.Y[idx]                       # (n_sel, p) prediction error
    D_true = fo.Y[idx] - fo.C_true[idx]          # metric's Delta_true
    D_hat = P[idx] - fo.C_true[idx]              # metric's Delta_pred
    pert = (P_add - P_batch)[idx]                # the model's own perturbation term
    boost = rb.predict()[idx]

    with np.errstate(invalid="ignore"):
        bias_p = np.nanmean(E, axis=0)           # per-protein systematic bias
        off_i = np.nanmean(E, axis=1)            # per-sample loading offset
        resid = E - bias_p[None, :] - off_i[:, None]

    rows.append({
        "split": name, "n": len(idx),
        "rms_err": rms(E),
        "rms_protein_bias": rms(bias_p),
        "rms_sample_offset": rms(off_i),
        "rms_residual": rms(resid),
        "rms_delta_true": rms(D_true),
        "rms_delta_hat": rms(D_hat),
        "rms_pert_term": rms(pert),
        "rms_booster": rms(boost),
        "noise_floor": NOISE_RMS,
        "err_over_floor": rms(E) / NOISE_RMS,
    })

df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print("\n=== error budget on held-out rows (log2 units) ===")
print(df.round(4).to_string(index=False))
df.to_csv(os.path.join(OUT, "error_budget.csv"), index=False)

# --- what a perfect per-sample offset would be worth ------------------------
val_idx = np.where(np.isin(split, VAL_NAMES))[0]
with np.errstate(invalid="ignore"):
    off = np.nanmean(P[val_idx] - fo.Y[val_idx], axis=1)
P_off = P.copy()
P_off[val_idx] -= off[:, None]
print("\n=== oracle: subtract each held-out row's true loading offset ===")
print("(unreachable -- it uses the target's own measurement -- but it prices the term)")
print(summary_row("full", evaluate(fo, P)))
print(summary_row("+ oracle per-sample offset", evaluate(fo, P_off)))

# --- what a perfect per-protein bias correction would be worth --------------
with np.errstate(invalid="ignore"):
    bp = np.nanmean(P[val_idx] - fo.Y[val_idx], axis=0)
P_bp = P.copy()
P_bp[val_idx] -= bp[None, :]
print(summary_row("+ oracle per-protein bias", evaluate(fo, P_bp)))
P_both = P.copy()
P_both[val_idx] -= bp[None, :]
with np.errstate(invalid="ignore"):
    off2 = np.nanmean(P_both[val_idx] - fo.Y[val_idx], axis=1)
P_both[val_idx] -= off2[:, None]
print(summary_row("+ both", evaluate(fo, P_both)))

# --- per-protein slope on the perturbation term ----------------------------
# beta_p fitted on the *held-out* rows is an oracle too; it prices whether a
# per-protein rescaling of Delta_hat could matter at all before we spend a
# nested cross-fit on estimating it honestly.
pert_val = (P_add - P_batch)[val_idx] + rb.predict()[val_idx]
d_true = fo.Y[val_idx] - fo.C_true[val_idx]
base = P_batch[val_idx]
tgt = d_true - (base - fo.C_true[val_idx])       # what the pert term should equal
num = np.nansum(pert_val * tgt, axis=0)
den = np.nansum(pert_val * pert_val, axis=0)
beta = np.where(den > 1e-9, num / np.maximum(den, 1e-9), 1.0)
beta = np.clip(beta, -2, 3)
P_beta = P.copy()
P_beta[val_idx] = base + beta[None, :] * pert_val
print("\n=== oracle: per-protein slope on the perturbation term ===")
print(f"beta quantiles 5/25/50/75/95: {np.nanpercentile(beta, [5,25,50,75,95]).round(3)}")
print(summary_row("+ oracle per-protein beta", evaluate(fo, P_beta)))
