"""Final evaluation on the organisers' val_* mirror -- run once, after tuning.

Reports the headline table plus a robustness band: the same predictions scored
under every plausible reading of the parts of the scoring spec the handbook
leaves open.  A single number would be misleading, because that band is wider
than the difference between competing models.
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcell.harness import build_fold, evaluate, summary_row              # noqa: E402
from vcell.metrics import ScoreConfig                                    # noqa: E402
from vcell.models import (BATCH_FACTORS, PERT_FACTORS, ControlBaseline,  # noqa: E402
                          DeltaBackfit, ResidualBooster, UnifiedBackfit)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")
pd.set_option("display.width", 300)

cfgfile = os.path.join(OUT, "best_config.json")
cfg = json.load(open(cfgfile))["config"] if os.path.exists(cfgfile) else {}
print("config selected on the inner mirror:", json.dumps(cfg))

f = build_fold(vehicle="both")
meta, n = f.meta, len(f.meta)
treated = (~meta["is_control"] & ~meta["is_qc"]).to_numpy()


def fit(**kw):
    c = {**cfg, **kw}
    return UnifiedBackfit(
        batch_factors=[(nm, co, c.get(f"lam_{nm}", l)) for nm, co, l in BATCH_FACTORS],
        pert_factors=[(nm, co, c.get(f"lam_{nm}", l)) for nm, co, l in PERT_FACTORS],
        n_pass=c.get("n_pass", 6), lowrank=c.get("lowrank", {}),
        eb=c.get("eb", False)).fit(meta, f.Y_obs, f.obs_mask)


rows = []


def add(name, P):
    r = summary_row(name, evaluate(f, P))
    rows.append(r)
    print(f"{name:38s} TOTAL={r['TOTAL']:.4f} | M1={r['M1_abs(20%)']:.3f} "
          f"M2={r['M2_rawFC(25%)']:.3f} M3={r['M3_ctx(20%)']:.3f} "
          f"M4={r['M4_drug(20%)']:.3f} M5={r['M5_bt(10%)']:.3f} M6={r['M6_DEP(5%)']:.3f}")
    return r


print("\n=== model ladder on the official val mirror ===")
gm = np.nanmean(np.where(f.obs_mask[:, None], f.Y, np.nan), 0).astype(np.float32)
add("R0 protein mean", np.tile(gm, (n, 1)))

cb = ControlBaseline().fit(meta, f.Y_obs)
B = cb.predict()
add("R1 controls-only plate x strain", B)

D_obs = np.where(f.obs_mask[:, None], f.Y - f.C_obs, np.nan).astype(np.float32)
bf = DeltaBackfit().fit(meta, D_obs, f.obs_mask & treated & np.isfinite(D_obs).any(1))
add("R2 R1 + separate Delta backfit", B + bf.predict())

um0 = UnifiedBackfit()
um0.fit(meta, f.Y_obs, f.obs_mask)
add("R3 unified, batch terms only", um0.predict(pert_scale=0.0))
add("R4 unified, default settings", um0.predict())

um = fit()
P_add = um.predict(cfg.get("pert_scale", 1.0))
add("R5 unified, tuned (additive only)", P_add)

boost = cfg.get("booster")
if boost:
    rb = ResidualBooster(**boost).fit(meta, f.Y_obs, f.obs_mask, P_add)
    P_add = P_add + rb.predict()
    print(f"    booster: {len(rb.models)} comps, {rb.explained:.1%} of residual var")
final = add("R6 + residual booster  <-- SUBMITTED", P_add)

print("\n=== reference points ===")
add("floor: shuffled truth",
    f.Y[np.random.default_rng(0).permutation(n)])
D = np.where(np.isfinite(f.C_true), f.Y - f.C_true, np.nan).astype(np.float32)
add("ceiling: batch + true Delta (oracle)", um.predict(0.0) + np.nan_to_num(D))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "final_val.csv"), index=False)
keep = ["model", "TOTAL", "M1_abs(20%)", "M2_rawFC(25%)", "M3_ctx(20%)",
        "M4_drug(20%)", "M5_bt(10%)", "M6_DEP(5%)"]
print("\n" + df[keep].to_string(index=False))
print("\nper-split matched-control fold-change PCC:")
print(df[["model"] + [c for c in df.columns if c.startswith("FC[")]].to_string(index=False))

# ------------------------------------------------------------------ robustness
print("\n=== robustness band: the same prediction under different readings "
      "of the spec ===")
P = P_add
band = []
variants = [("as specified (all controls, plate-level mu_ctx, R^2 in M1)",
             dict(vehicle="both", cfg=ScoreConfig())),
            ("vehicle = DMSO first", dict(vehicle="dmso", cfg=ScoreConfig())),
            ("vehicle = Water first", dict(vehicle="water", cfg=ScoreConfig())),
            ("vehicle = by solubility", dict(vehicle="curated", cfg=ScoreConfig())),
            ("mu_ctx over strain/medium/temp/time",
             dict(vehicle="both", cfg=ScoreConfig(
                 ctx_cols=("Strains", "Medium", "Temperature", "pert_time")))),
            ("M1 = correlations only",
             dict(vehicle="both", cfg=ScoreConfig(m1_aggregate="pcc_only"))),
            ("Delta from predicted control",
             dict(vehicle="both", cfg=ScoreConfig(delta_mode="predicted")))]
for name, kw in variants:
    fo = build_fold(**kw)
    r = summary_row(name, evaluate(fo, P))
    band.append(r)
    print(f"  {name:52s} TOTAL={r['TOTAL']:.4f}  M3={r['M3_ctx(20%)']:.3f}")

# the handbook does not say whether the control wells themselves are scored as
# OOD samples; for them Delta_true is a difference between two control wells, i.e.
# pure measurement noise
fo = build_fold(vehicle="both")
keep = (~fo.meta["is_control"] & ~fo.meta["is_qc"]).to_numpy()
n_ctrl = int(((~keep) & fo.meta["split_final"].str.startswith("val").to_numpy()).sum())
for s in list(fo.scorer.eval_masks):
    fo.scorer.eval_masks[s] = fo.scorer.eval_masks[s] & keep
r = summary_row(f"treated samples only ({n_ctrl} control rows dropped)",
                evaluate(fo, P))
band.append(r)
print(f"  {'treated samples only (%d control rows dropped)' % n_ctrl:52s} "
      f"TOTAL={r['TOTAL']:.4f}  M3={r['M3_ctx(20%)']:.3f}")
bd = pd.DataFrame(band)
bd.to_csv(os.path.join(OUT, "robustness_band.csv"), index=False)
print(f"\nTOTAL across readings: {bd.TOTAL.min():.4f} – {bd.TOTAL.max():.4f}  "
      f"(spread {bd.TOTAL.max()-bd.TOTAL.min():.4f})")
print(f"our modelling gain over the batch-only model, as specified: "
      f"{final['TOTAL'] - df[df.model=='R3 unified, batch terms only'].TOTAL.iloc[0]:+.4f}")
print("under the 'Delta from predicted control' reading, the batch-only model "
      "scores ~0 on M2 by construction, so essentially all of that number is the "
      "perturbation model.")
