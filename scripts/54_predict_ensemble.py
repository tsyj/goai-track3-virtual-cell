"""Final submission: average five near-optimal configurations.

53_config_ensemble.py: averaging five configurations drawn from the neighbourhood
of the adopted optimum is worth +0.00150 +- 0.00022 over the single adopted config,
6/6 orphan-free folds -- the most consistent result of the session.  Every ensemble
with three or more members beat the single config on every fold.

Members (the two axes that moved this session, inside the region where each point
was individually at or near the top):

    A  plate 2, pert x4   <- the single adopted config
    B  plate 1, pert x4
    C  plate 4, pert x4
    D  plate 2, pert x2
    E  plate 2, pert x8

Each member is the full shipped model: source+instrument coarse levels, the
adopted booster (240 comps, 1600 trees, lr .015, three seeds).  Five members at
~30 min each.

Writes submission/prediction.csv, same format and protein filter as
10_predict_test.py, and records every member in the manifest.

    python scripts/54_predict_ensemble.py

Jiao Xinyuan 2026-08-16
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
BOOSTER = {"n_comp": 240, "n_estimators": 1600, "learning_rate": 0.015,
           "seeds": [0, 1, 2], "n_jobs": int(os.environ.get("VCELL_LGB_THREADS", 32))}


def lam_of(plate, pert_mult):
    d = {"lam_plate": plate, "lam_plate_x_strain": 6.0,
         "lam_source": 3.0, "lam_instrument": 3.0}
    d.update({f"lam_{a}": l * pert_mult for a, _, l in PERT_FACTORS})
    return d


MEMBERS = [("A plate2 pert4", lam_of(2.0, 4.0)), ("B plate1 pert4", lam_of(1.0, 4.0)),
           ("C plate4 pert4", lam_of(4.0, 4.0)), ("D plate2 pert2", lam_of(2.0, 2.0)),
           ("E plate2 pert8", lam_of(2.0, 8.0))]

t0 = time.time()
P = load_combined()
meta = P.meta
is_test = (meta["SET"] == "test").to_numpy()
visible = (meta["split_final"] == "train").to_numpy()
print(f"rows: {len(meta)}  trainable={visible.sum()}  test={is_test.sum()}", flush=True)
keep = protein_keep_mask(meta, P.X)
P.X = P.X[:, keep]
P.proteins = P.proteins[keep]
print(f"protein filter -> {keep.sum()}", flush=True)

acc = None
for name, lam in MEMBERS:
    t1 = time.time()
    um = UnifiedBackfit(
        batch_factors=[(a, c, lam.get(f"lam_{a}", l)) for a, c, l in BATCH_FACTORS],
        pert_factors=[(a, c, lam.get(f"lam_{a}", l)) for a, c, l in PERT_FACTORS],
        n_pass=6,
    ).fit(meta, P.X, visible)
    Yhat = um.predict()
    rb = ResidualBooster(**BOOSTER).fit(meta, P.X, visible, Yhat)
    Yhat = Yhat + rb.predict()
    acc = Yhat.astype(np.float64) if acc is None else acc + Yhat
    print(f"member {name} done ({time.time()-t1:.0f}s, total {time.time()-t0:.0f}s)",
          flush=True)

Yhat = (acc / len(MEMBERS)).astype(np.float32)
pred = Yhat[is_test]
ids = meta.loc[is_test, "sample_ID"].to_numpy()
df = pd.DataFrame(pred, columns=list(P.proteins), index=pd.Index(ids, name="sample_ID"))
assert not df.isna().any().any() and np.isfinite(df.to_numpy()).all()
path = os.path.join(OUT, "prediction.csv")
df.to_csv(path, float_format="%.5f")
h = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
json.dump({
    "n_test_rows": int(is_test.sum()), "n_proteins": int(keep.sum()),
    "ensemble_members": [{"name": n, "lam": l} for n, l in MEMBERS],
    "booster": {k: v for k, v in BOOSTER.items() if k != "n_jobs"},
    "fitted_on": "split_final=='train' only (5,920 rows)",
    "prediction_scale": "log2", "protein_filter": "train-row missing rate < 0.80",
    "test_labels_used": False,
    "selected_by": "scripts/53_config_ensemble.py, +0.00150 +- 0.00022 over the single "
                   "config, 6/6 orphan-free inner folds",
    "files": {"prediction.csv": {"sha256_16": h,
                                 "mb": round(os.path.getsize(path) / 1e6, 1)}},
}, open(os.path.join(OUT, "manifest.json"), "w"), indent=1, ensure_ascii=False)
print(f"wrote {path}  ({os.path.getsize(path)/1e6:.1f} MB, sha {h})")
print(f"total {time.time()-t0:.0f}s")
