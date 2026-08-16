"""Reproduce the organisers' own baseline diagnostics -- external validation.

The official interpretation deck (Guomics / Westlake, 周梓卓) publishes a table of
baseline numbers on the frozen validation splits.  Reproducing it end-to-end
checks our sample alignment, the protein filter, the log2 scale handling and the
control matching against a reference we did not produce.

Official rules implemented here, all of which we had wrong before:
  * protein filter: drop proteins missing in >= 80% of the *train* rows
    (5,243 -> expected 4,232)
  * metrics are mask-aware: computed only where truth and control are both present
  * the diagnostic subset is samples with an exact matched control

Official targets (log2 RMSE / Global R^2 / median per-protein R^2):
  double-unseen   n=266   mean 0.994/0.871/-0.064   control 0.382/0.980/0.809
  new compound    n=1015  mean 1.004/0.868/-0.038   control 0.379/0.980/0.836
  new strain      n=1293  mean 0.875/0.897/-0.036   control 0.399/0.978/0.726
  time            n=128   mean 0.869/0.900/-0.009   control 0.426/0.975/0.719
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcell.design import control_reference, match_controls                # noqa: E402
from vcell.io import load_proteome                                        # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
pd.set_option("display.width", 280)

OFFICIAL = {
    "val_both":        dict(name="双重未知", n=266,  mean=(0.994, 0.871, -0.064),
                            ctrl=(0.382, 0.980, 0.809)),
    "val_chem_only":   dict(name="新化合物", n=1015, mean=(1.004, 0.868, -0.038),
                            ctrl=(0.379, 0.980, 0.836)),
    "val_strain_only": dict(name="新菌株",   n=1293, mean=(0.875, 0.897, -0.036),
                            ctrl=(0.399, 0.978, 0.726)),
    "val_time":        dict(name="时间验证", n=128,  mean=(0.869, 0.900, -0.009),
                            ctrl=(0.426, 0.975, 0.719)),
}

P = load_proteome("train_val")
meta, Y = P.meta.reset_index(drop=True), P.X
train = (meta["split_final"] == "train").to_numpy()

# ---- official protein filter --------------------------------------------
missing_rate = np.isnan(Y[train]).mean(0)
keep = missing_rate < 0.80
print(f"protein filter: {len(keep)} -> {keep.sum()} kept, {(~keep).sum()} dropped "
      f"(official: 5,243 -> 4,232)   {'MATCH' if keep.sum() == 4232 else 'MISMATCH'}")
Yk = Y[:, keep]

ctrl_rows = match_controls(meta, strategy="both")
C = control_reference(Yk, ctrl_rows)
protein_mean = np.nanmean(np.where(train[:, None], Yk, np.nan), 0)


def diagnostics(truth, pred):
    """Mask-aware pooled RMSE, pooled R^2, and the median per-protein R^2."""
    m = np.isfinite(truth) & np.isfinite(pred)
    res = (truth - pred)[m]
    rmse = float(np.sqrt((res ** 2).mean()))
    gm = truth[m].mean()
    r2 = float(1 - (res ** 2).sum() / (((truth[m] - gm) ** 2).sum()))
    per = []
    for j in range(truth.shape[1]):
        mm = m[:, j]
        if mm.sum() < 3:
            continue
        t, p = truth[mm, j], pred[mm, j]
        ss = ((t - t.mean()) ** 2).sum()
        if ss <= 0:
            continue
        per.append(1 - ((t - p) ** 2).sum() / ss)
    return rmse, r2, float(np.median(per)) if per else np.nan


rows = []
print(f"\n{'scenario':12s} {'subset':28s} {'n':>5s}  "
      f"{'RMSE':>16s} {'GlobalR2':>16s} {'protR2med':>16s}")
for split, off in OFFICIAL.items():
    sel = (meta["split_final"] == split).to_numpy()
    treated = sel & (~meta["is_control"]).to_numpy() & (~meta["is_qc"]).to_numpy()
    has_ctrl = np.array([len(r) > 0 for r in ctrl_rows])
    subset = treated & has_ctrl & np.isfinite(C).any(1)
    idx = np.where(subset)[0]
    for tag, pred in [("mean", np.tile(protein_mean, (len(idx), 1))),
                      ("ctrl", C[idx])]:
        r = diagnostics(Yk[idx], pred)
        o = off[tag]
        rows.append({"scenario": off["name"], "baseline": tag, "n_ours": len(idx),
                     "n_official": off["n"], "rmse": r[0], "rmse_off": o[0],
                     "globalR2": r[1], "globalR2_off": o[1],
                     "protR2med": r[2], "protR2med_off": o[2]})
        print(f"{off['name']:12s} {'ours / official':28s} {len(idx):5d}  "
              f"{r[0]:7.3f} /{o[0]:7.3f} {r[1]:7.3f} /{o[1]:7.3f} "
              f"{r[2]:7.3f} /{o[2]:7.3f}")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "official_calibration.csv"), index=False)
for c, o in [("rmse", "rmse_off"), ("globalR2", "globalR2_off"),
             ("protR2med", "protR2med_off")]:
    rel = (df[c] - df[o]).abs() / df[o].abs().clip(lower=1e-6)
    print(f"\n{c:10s} max relative deviation from official: {rel.max():.1%}"
          f"   {'within 5%' if rel.max() < 0.05 else 'OUTSIDE 5%'}")
print(f"\nsample-count deviation: "
      f"{(df.n_ours - df.n_official).abs().max()} rows worst case")
