"""Test-set prediction for ONE pool member (full shipped model), saved as an array.

54_predict_ensemble.py fits its five members sequentially in one process (~30 min
each, 2.5 h total).  This script fits a single member -- any name from
scripts/55_member_pool.py MEMBERS -- so members can run in parallel processes and
be combined afterwards by 58_compose_submission.py.

Writes submission/members/<member>.npy  (n_test x n_proteins float32, test rows in
the order of load_combined().meta[SET=='test'])  and <member>.json (ids, lam, sha).

    VCELL_LGB_THREADS=16 python scripts/57_predict_member.py A

Jiao Xinyuan 2026-08-16 (evening session)
"""
import hashlib
import importlib.util
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vcell.harness import protein_keep_mask                               # noqa: E402
from vcell.io import load_combined                                        # noqa: E402
from vcell.models import (BATCH_FACTORS, PERT_FACTORS,                    # noqa: E402
                          ResidualBooster, UnifiedBackfit)

_spec = importlib.util.spec_from_file_location(
    "pool55", os.path.join(ROOT, "scripts", "55_member_pool.py"))
pool55 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pool55)

OUT = os.path.join(ROOT, "submission", "members")
REAL = {"n_comp": 240, "n_estimators": 1600, "learning_rate": 0.015, "seeds": [0, 1, 2]}

member = sys.argv[1]
cfg = pool55.MEMBERS[member]
threads = int(os.environ.get("VCELL_LGB_THREADS", 16))
os.makedirs(OUT, exist_ok=True)

t0 = time.time()
P = load_combined()
meta = P.meta
is_test = (meta["SET"] == "test").to_numpy()
visible = (meta["split_final"] == "train").to_numpy()
keep = protein_keep_mask(meta, P.X)
P.X = P.X[:, keep]
P.proteins = P.proteins[keep]
print(f"[{member}] rows={len(meta)} trainable={visible.sum()} test={is_test.sum()} "
      f"proteins={keep.sum()}  cfg={cfg}", flush=True)

lam = {**pool55.BASE_LAM, **cfg.get("lam", {})}
batch = [pool55.BY_NAME[a] for a in cfg.get("order", pool55.CUR_ORDER)]
um = UnifiedBackfit(
    batch_factors=[(a, c, lam.get(f"lam_{a}", l)) for a, c, l in batch],
    pert_factors=[(a, c, lam.get(f"lam_{a}", l)) for a, c, l in PERT_FACTORS],
    n_pass=cfg.get("n_pass", 6), fit_offset=cfg.get("fit_offset", True),
).fit(meta, P.X, visible)
Yhat = um.predict()
print(f"[{member}] additive done ({time.time()-t0:.0f}s)", flush=True)
rb = ResidualBooster(n_jobs=threads, **cfg.get("booster", REAL)).fit(meta, P.X, visible, Yhat)
Yhat = (Yhat + rb.predict()).astype(np.float32)
pred = Yhat[is_test]
assert np.isfinite(pred).all() and pred.shape == (int(is_test.sum()), int(keep.sum()))

path = os.path.join(OUT, f"{member}.npy")
np.save(path + ".tmp.npy", pred)
os.replace(path + ".tmp.npy", path)
h = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
json.dump({
    "member": member, "cfg": cfg, "lam_effective": lam,
    "order": [a for a, _, _ in batch], "n_pass": cfg.get("n_pass", 6),
    "fit_offset": cfg.get("fit_offset", True), "booster": cfg.get("booster", REAL),
    "n_test_rows": int(is_test.sum()), "n_proteins": int(keep.sum()),
    "sample_ids": meta.loc[is_test, "sample_ID"].astype(str).tolist(),
    "proteins": [str(p) for p in P.proteins],
    "sha256_16": h, "secs": round(time.time() - t0),
}, open(os.path.join(OUT, f"{member}.json"), "w"), indent=1, ensure_ascii=False)
print(f"[{member}] wrote {path} sha {h}")
print(f"total {time.time()-t0:.0f}s")
