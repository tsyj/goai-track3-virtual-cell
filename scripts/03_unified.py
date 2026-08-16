"""Unified backfit model + the perturbation-scale sweep that exposes M3's inversion."""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcell.harness import build_fold, evaluate, summary_row                  # noqa: E402
from vcell.models import ControlBaseline, DeltaBackfit, UnifiedBackfit       # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
pd.set_option("display.width", 260)

t0 = time.time()
f = build_fold(vehicle="both")
meta, Yo, obs = f.meta, f.Y_obs, f.obs_mask
n = len(meta)
treated = (~meta["is_control"] & ~meta["is_qc"]).to_numpy()
rows = []


def run(name, P, keep=True):
    r = summary_row(name, evaluate(f, P))
    if keep:
        rows.append(r)
    print(f"{name:36s} TOTAL={r['TOTAL']:.4f} | M1={r['M1_abs(20%)']:.3f} "
          f"M2={r['M2_rawFC(25%)']:.3f} M3={r['M3_ctx(20%)']:.3f} "
          f"M4={r['M4_drug(20%)']:.3f} M5={r['M5_bt(10%)']:.3f} M6={r['M6_DEP(5%)']:.3f}")
    return r


# reference: previous best (controls-only baseline + separate Delta backfit)
cb = ControlBaseline().fit(meta, Yo)
B = cb.predict()
D_obs = np.where(obs[:, None], f.Y - f.C_obs, np.nan).astype(np.float32)
use_d = obs & treated & np.isfinite(D_obs).any(1)
bf = DeltaBackfit().fit(meta, D_obs, use_d)
run("prev: ctrl-only + Delta backfit", B + bf.predict())

# unified
print(f"\nfitting unified model on {obs.sum()} visible samples ...")
um = UnifiedBackfit().fit(meta, Yo, obs)
print(f"  fit in {time.time()-t0:.0f}s")
run("unified (batch only)", um.predict(pert_scale=0.0))
best = run("unified (batch + pert)", um.predict())
print("\nterm magnitudes:")
print(um.term_norms().to_string(index=False))

# ------------------------------------------------------------------ pert scale
print("\n=== perturbation-scale sweep ===")
print("a coefficient on the predicted perturbation effect; 0 = 'no drug does "
      "anything', 1 = the model's own estimate")
sweep = []
for s in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
    r = run(f"  pert_scale={s}", um.predict(pert_scale=s), keep=False)
    r["pert_scale"] = s
    sweep.append(r)
sw = pd.DataFrame(sweep)
sw.to_csv(os.path.join(OUT, "pert_scale_sweep.csv"), index=False)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "unified.csv"), index=False)
print("\n" + df.to_string(index=False))
print(f"\ntotal {time.time()-t0:.0f}s")
