"""Fit on all of train_val, predict the 4,454 test samples, write prediction.csv.

The test proteome file is never opened -- test rows enter as metadata only.
Two files are written because the organisers have not published a submission
template yet (see docs/OPEN_QUESTIONS.md):
  prediction.csv       -- log2 abundance, index sample_ID, filtered protein columns

Rules taken from the official interpretation deck (Guomics / Westlake):
  * submit log2 intensity.  raw intensity and z-scored space are both explicitly
    disallowed, so the raw variant we used to emit has been removed.
  * "训练仅可使用训练集；验证集与测试集不得参与训练，包括用于估计归一化统计量" --
    so the final fit uses the 5,920 split_final=='train' rows ONLY.  The val_*
    rows stay held out even for the final submission.
  * proteins missing in >=80% of train rows are dropped before submission.
"""
import hashlib
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcell.harness import protein_keep_mask                               # noqa: E402
from vcell.io import load_combined                                        # noqa: E402
from vcell.models import (BATCH_FACTORS, PERT_FACTORS,                    # noqa: E402
                          ResidualBooster, UnifiedBackfit)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "submission")
os.makedirs(OUT, exist_ok=True)

cfgfile = os.path.join(ROOT, "results", "best_config.json")
cfg = json.load(open(cfgfile))["config"] if os.path.exists(cfgfile) else {}
print("config:", json.dumps(cfg))

t0 = time.time()
P = load_combined()
meta = P.meta
is_test = (meta["SET"] == "test").to_numpy()
# validation labels may NOT be used for training -- only split_final == 'train'
visible = (meta["split_final"] == "train").to_numpy()
held_back = (~is_test) & (~visible)
print(f"rows: {len(meta)}  trainable={visible.sum()}  "
      f"val rows withheld from training={held_back.sum()}  test={is_test.sum()}")

keep = protein_keep_mask(meta, P.X)
print(f"protein filter: {P.X.shape[1]} -> {keep.sum()} "
      f"(train-row missing rate < 0.80)")
P.X = P.X[:, keep]
P.proteins = P.proteins[keep]
print("unseen entities in test: strains=%s | compounds=%d"
      % (sorted(set(meta.loc[is_test, "Strains"]) - set(meta.loc[visible, "Strains"])),
         len(set(meta.loc[is_test, "compound"]) - set(meta.loc[visible, "compound"]))))

um = UnifiedBackfit(
    batch_factors=[(n, c, cfg.get(f"lam_{n}", l)) for n, c, l in BATCH_FACTORS],
    pert_factors=[(n, c, cfg.get(f"lam_{n}", l)) for n, c, l in PERT_FACTORS],
    n_pass=cfg.get("n_pass", 6), lowrank=cfg.get("lowrank", {}),
).fit(meta, P.X, visible)
Yhat = um.predict(cfg.get("pert_scale", 1.0))
print(f"additive model fitted in {time.time()-t0:.0f}s")

boost = cfg.get("booster")
resid = None
if boost:
    rb = ResidualBooster(**boost).fit(meta, P.X, visible, Yhat)
    resid = rb.predict()
    Yhat = Yhat + resid
    print(f"residual booster: {len(rb.models)} components, "
          f"{rb.explained:.1%} of residual variance, rms {np.sqrt((resid**2).mean()):.4f}"
          f"  ({time.time()-t0:.0f}s)")

# sanity: predictions must sit inside the observed dynamic range
obs = P.X[visible]
lo, hi = np.nanpercentile(obs, [0.01, 99.99])
pred = Yhat[is_test]
print(f"observed log2 range  [{np.nanmin(obs):.1f}, {np.nanmax(obs):.1f}]  "
      f"0.01-99.99 pct [{lo:.1f}, {hi:.1f}]")
print(f"predicted test range [{pred.min():.1f}, {pred.max():.1f}]  "
      f"mean={pred.mean():.2f}  nan={np.isnan(pred).sum()}")
assert np.isfinite(pred).all(), "non-finite prediction"

# per-split sanity on how much perturbation signal each test split receives
pp = um.pert_part()[is_test]
sp = meta.loc[is_test, "split_final"].to_numpy()
print("\nrms of the ADDITIVE perturbation term by test split "
      "(the booster contributes on top of this):")
for s in sorted(set(sp)):
    print(f"  {s:18s} n={int((sp == s).sum()):5d}  rms={np.sqrt((pp[sp == s]**2).mean()):.4f}")

ids = meta.loc[is_test, "sample_ID"].to_numpy()
cols = list(P.proteins)
df = pd.DataFrame(pred, columns=cols, index=pd.Index(ids, name="sample_ID"))
assert not df.isna().any().any() and np.isfinite(df.to_numpy()).all()
p_log2 = os.path.join(OUT, "prediction.csv")
df.to_csv(p_log2, float_format="%.5f")
print(f"submission: {df.shape[0]} rows x {df.shape[1]} protein columns, "
      f"scale=log2 (declare prediction_scale=log2)")

meta_out = {
    "n_test_rows": int(is_test.sum()), "n_proteins": len(cols),
    "config": cfg, "fitted_on": "split_final=='train' only (5,920 rows)",
    "prediction_scale": "log2", "protein_filter": "train-row missing rate < 0.80",
    "residual_booster": bool(boost),
    "test_labels_used": False,
    "files": {},
}
for p in (p_log2,):
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    meta_out["files"][os.path.basename(p)] = {
        "sha256_16": h, "mb": round(os.path.getsize(p) / 1e6, 1)}
    print(f"wrote {p}  ({meta_out['files'][os.path.basename(p)]['mb']} MB, sha {h})")
json.dump(meta_out, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
print(f"total {time.time()-t0:.0f}s")
