"""Audit of the six official scoring modules.

Every Delta-based module compares two quantities that are built from the *same*
measured control (and, for M3, the same measured context mean).  Shared
measurement noise therefore contributes to the score independently of how good
the model is.  This script isolates that contribution by scoring predictors that
contain a controlled amount of information, from none at all up to the truth.

Findings are reported as a table so they can be handed to the organisers with the
submission; the sensitivity sweep also covers the two places where the handbook
leaves the implementation underspecified (the vehicle-matching rule and the
context key used for mu_ctx).
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcell.harness import build_fold, evaluate, summary_row     # noqa: E402
from vcell.metrics import ScoreConfig                           # noqa: E402
from vcell.models import UnifiedBackfit                         # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
pd.set_option("display.width", 260)

f = build_fold(vehicle="both")
meta, n = f.meta, len(f.meta)
um = UnifiedBackfit().fit(meta, f.Y_obs, f.obs_mask)
batch = um.predict(pert_scale=0.0)
pert = um.pert_part()
Ytrue = f.Y

# ------------------------------------------------------- information ladder
print("=== information ladder: how much of each module is model, how much is "
      "shared measurement noise? ===")
rows = []
ladder = [
    ("L0 constant vector", np.tile(np.nanmean(Ytrue, 0), (n, 1))),
    ("L1 batch level, zero drug effect", batch),
    ("L2 batch + modelled drug effect", batch + pert),
    ("L3 batch + TRUE drug effect", batch + np.where(
        np.isfinite(f.C_true), Ytrue - f.C_true, 0.0)),
    ("L4 truth", Ytrue),
]
for name, P in ladder:
    r = summary_row(name, evaluate(f, P))
    rows.append(r)
    print(f"  {name:34s} TOTAL={r['TOTAL']:.4f} | M1={r['M1_abs(20%)']:.3f} "
          f"M2={r['M2_rawFC(25%)']:.3f} M3={r['M3_ctx(20%)']:.3f} "
          f"M4={r['M4_drug(20%)']:.3f} M6={r['M6_DEP(5%)']:.3f}")
pd.DataFrame(rows).to_csv(os.path.join(OUT, "information_ladder.csv"), index=False)

# ------------------------------------------------- M3: the monotonicity check
print("\n=== M3 monotonicity: does predicting the drug effect *better* help? ===")
print("gamma scales a generic drug-response vector added for UNSEEN compounds;")
print("the Bayes-optimal choice with no compound information is gamma = 1.")
tr = f.obs_mask & (~meta["is_control"]).to_numpy() & (~meta["is_qc"]).to_numpy()
generic = np.nanmean(np.where(tr[:, None], Ytrue - f.C_obs, np.nan), 0).astype(np.float32)
unseen = (~meta["compound"].isin(
    meta.loc[f.obs_mask & ~meta["is_control"], "compound"].unique()).to_numpy())
print(f"  rows with an unseen compound: {unseen.sum()}")
m3 = []
for g in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
    P = batch + pert + g * unseen[:, None] * generic
    r = summary_row(f"gamma={g}", evaluate(f, P))
    m3.append({"gamma": g, **r})
    print(f"  gamma={g:<5} TOTAL={r['TOTAL']:.4f}  M3={r['M3_ctx(20%)']:.4f}  "
          f"M2={r['M2_rawFC(25%)']:.4f}  M6={r['M6_DEP(5%)']:.4f}")
pd.DataFrame(m3).to_csv(os.path.join(OUT, "m3_monotonicity.csv"), index=False)

# ------------------------------------------------------- spec sensitivity
print("\n=== sensitivity to the two underspecified rules ===")
sens = []
for veh in ("both", "dmso", "water", "curated"):
    fo = build_fold(vehicle=veh)
    um2 = UnifiedBackfit().fit(fo.meta, fo.Y_obs, fo.obs_mask)
    r = summary_row(f"vehicle={veh}", evaluate(fo, um2.predict()))
    sens.append({"knob": "vehicle", "value": veh, **r})
    print(f"  vehicle={veh:8s} TOTAL={r['TOTAL']:.4f} M2={r['M2_rawFC(25%)']:.3f} "
          f"M3={r['M3_ctx(20%)']:.3f} M4={r['M4_drug(20%)']:.3f}")

for ctx in [("data_source", "Strains", "Medium", "Temperature", "pert_time",
             "Yeast_cell_plate"),
            ("data_source", "Strains", "Medium", "Temperature", "pert_time"),
            ("Strains", "Medium", "Temperature", "pert_time"),
            ("Strains",)]:
    fo = build_fold(vehicle="both", cfg=ScoreConfig(ctx_cols=ctx))
    r = summary_row("ctx=" + "+".join(c[:4] for c in ctx), evaluate(fo, um.predict()))
    sens.append({"knob": "mu_ctx key", "value": "+".join(ctx), **r})
    print(f"  mu_ctx over {len(ctx)} keys: M3={r['M3_ctx(20%)']:.4f}  TOTAL={r['TOTAL']:.4f}")

for agg in ("pcc_only", "with_r2"):
    fo = build_fold(vehicle="both", cfg=ScoreConfig(m1_aggregate=agg))
    r = summary_row(f"M1={agg}", evaluate(fo, um.predict()))
    sens.append({"knob": "M1 aggregation", "value": agg, **r})
    print(f"  M1 aggregation={agg:9s} M1={r['M1_abs(20%)']:.4f} TOTAL={r['TOTAL']:.4f}")

pd.DataFrame(sens).to_csv(os.path.join(OUT, "spec_sensitivity.csv"), index=False)
print("\nwrote results/{information_ladder,m3_monotonicity,spec_sensitivity}.csv")
