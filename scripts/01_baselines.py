"""Baseline ladder on the local val mirror of the official test protocol."""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcell.harness import build_fold, evaluate, summary_row            # noqa: E402
from vcell.models import ControlBaseline, DeltaBackfit                 # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
pd.set_option("display.width", 260)

t0 = time.time()
f = build_fold(vehicle="both")
meta, Yo, obs = f.meta, f.Y_obs, f.obs_mask
n, p = f.Y.shape
treated = (~meta["is_control"] & ~meta["is_qc"]).to_numpy()
strain_hidden = (meta["split_final"].isin(["val_strain_only", "val_both"])).to_numpy()
print(f"fold built in {time.time()-t0:.1f}s   n={n} p={p}")

cb = ControlBaseline().fit(meta, Yo)
B = cb.predict()
print("control-level model:  rmse vs measured matched control (log2)")
for tag, msk in [("visible-strain rows", np.isfinite(f.C_obs).any(1)),
                 ("hidden-strain rows", strain_hidden)]:
    e = (B - f.C_true)[msk]
    print(f"   {tag:22s} {np.sqrt(np.nanmean(e**2)):.3f}")

# Delta target: measured treated minus measured matched control, visible rows only
D_obs = np.where(obs[:, None], f.Y - f.C_obs, np.nan).astype(np.float32)
use = obs & treated & np.isfinite(D_obs).any(1)
print(f"rows usable for Delta fitting: {use.sum()}")

rows = []


def run(name, P):
    rep = evaluate(f, P)
    r = summary_row(name, rep)
    rows.append(r)
    print(f"{name:34s} TOTAL={r['TOTAL']:.4f} | M1={r['M1_abs(20%)']:.3f} "
          f"M2={r['M2_rawFC(25%)']:.3f} M3={r['M3_ctx(20%)']:.3f} "
          f"M4={r['M4_drug(20%)']:.3f} M5={r['M5_bt(10%)']:.3f} M6={r['M6_DEP(5%)']:.3f} "
          f"| sR2={r['sampR2']:.3f}")
    return rep


gm = np.nanmean(np.where(obs[:, None], f.Y, np.nan), 0).astype(np.float32)
run("B0 protein-mean", np.tile(gm, (n, 1)))
run("B1 plate x strain baseline", B)
run("B2 copy measured control", np.where(np.isfinite(f.C_obs), f.C_obs, B))

mu0 = np.nanmean(np.where(use[:, None], D_obs, np.nan), 0).astype(np.float32)
run("B3 baseline + global Delta", B + mu0)

bf_c = DeltaBackfit(factors=[("global", (), 0.0), ("compound", ("compound",), 8.0)]
                    ).fit(meta, D_obs, use)
run("B4 baseline + mu_drug", B + bf_c.predict())

bf = DeltaBackfit().fit(meta, D_obs, use)
run("B5 baseline + structured Delta", B + bf.predict())
run("B6 control-copy + structured", np.where(np.isfinite(f.C_obs), f.C_obs, B)
    + bf.predict())

print("\nterm magnitudes (rms over levels x proteins, log2):")
print(bf.term_norms().to_string(index=False))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "baselines.csv"), index=False)
print("\n" + df.to_string(index=False))
print(f"\ntotal {time.time()-t0:.1f}s")
