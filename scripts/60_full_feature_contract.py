"""Expand the submission to the full 5,243-protein feature contract.

The revised handbook (2026-08 修订版, docs/官方材料/新资料_0816) says the protein
columns of prediction.csv "须与官方 feature contract 完全一致 (名称、数量和顺序)".
The organisers' own retained list (4,232 proteins) cannot be reproduced from the
released data with the code they published (we get 4,422 at < 0.80; no threshold,
zero-handling or row subset gives 4,232 -- see docs/OPEN_QUESTIONS.md P0-7 and the
21:50 check in FINDINGS §6.6).  Since the scorer "只在...属于训练集保留蛋白列表的位置
计算", any name-based subsetting works on a superset, and the only superset we can
guarantee is the full 5,243-column proteome layout in its original column order.

This script keeps the validated ensemble prediction *bit-for-bit* on the 4,422
proteins it was built for, and fills the remaining 821 low-coverage proteins with
the additive model (config A) fitted on all 5,243 proteins -- a plate/strain-aware
baseline for columns that will most likely not be scored at all.

    python scripts/60_full_feature_contract.py --src submission/_candidates/ens12/prediction.csv \
        --out submission/_candidates/ens12_full5243/prediction.csv

Jiao Xinyuan 2026-08-16 (evening session)
"""
import argparse
import hashlib
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vcell.harness import protein_keep_mask                               # noqa: E402
from vcell.io import load_combined                                        # noqa: E402
from vcell.models import BATCH_FACTORS, PERT_FACTORS, UnifiedBackfit      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="validated 4,422-column prediction.csv")
ap.add_argument("--out", required=True)
args = ap.parse_args()

t0 = time.time()
P = load_combined()
meta = P.meta
is_test = (meta["SET"] == "test").to_numpy()
visible = (meta["split_final"] == "train").to_numpy()
all_proteins = [str(p) for p in P.proteins]
keep = protein_keep_mask(meta, P.X)
print(f"proteins total {len(all_proteins)}, kept by <0.80 filter {keep.sum()}, "
      f"extra {(~keep).sum()}", flush=True)

# additive model, config A (plate 2 / pxs 6, pert x4, source+instrument), ALL proteins
LAM = {"lam_plate": 2.0, "lam_plate_x_strain": 6.0}
LAM.update({f"lam_{a}": l * 4.0 for a, _, l in PERT_FACTORS})
um = UnifiedBackfit(
    batch_factors=[(a, c, LAM.get(f"lam_{a}", l)) for a, c, l in BATCH_FACTORS],
    pert_factors=[(a, c, LAM.get(f"lam_{a}", l)) for a, c, l in PERT_FACTORS],
    n_pass=6,
).fit(meta, P.X, visible)
add = um.predict()[is_test]                       # (n_test, 5243)
print(f"additive fit on all proteins done ({time.time()-t0:.0f}s)", flush=True)

src = pd.read_csv(args.src, index_col=0)
ids = meta.loc[is_test, "sample_ID"].astype(str).tolist()
assert [str(i) for i in src.index] == ids, "row order of src != test metadata order"
kept_names = [p for p, k in zip(all_proteins, keep) if k]
assert list(map(str, src.columns)) == kept_names, "src columns != kept protein list"

full = pd.DataFrame(add.astype(np.float32), index=pd.Index(ids, name="sample_ID"),
                    columns=all_proteins)
full.loc[:, kept_names] = src.to_numpy(np.float32)      # validated ensemble, unchanged
assert np.isfinite(full.to_numpy()).all()
# bit-for-bit check on the kept block
assert np.array_equal(full[kept_names].to_numpy(np.float32), src.to_numpy(np.float32))

os.makedirs(os.path.dirname(args.out), exist_ok=True)
tmp = args.out + ".tmp"
full.to_csv(tmp, float_format="%.5f")
os.replace(tmp, args.out)
h = hashlib.sha256(open(args.out, "rb").read()).hexdigest()[:16]
hs = hashlib.sha256(open(args.src, "rb").read()).hexdigest()[:16]
json.dump({
    "n_test_rows": len(ids), "n_proteins": len(all_proteins),
    "layout": "full 5,243-protein feature contract, original proteome column order",
    "kept_4422": {"source": os.path.relpath(args.src, ROOT), "sha256_16": hs,
                  "note": "validated 12-member ensemble, values unchanged"},
    "extra_821": {"model": "UnifiedBackfit config A (plate 2/pxs 6, pert x4, source+instrument), "
                           "fitted on all 5,243 proteins, train labels only",
                  "note": "low-coverage proteins (train missing rate >= 0.80); "
                          "never-observed proteins get the 5th-percentile fallback"},
    "fitted_on": "split_final=='train' only (5,920 rows)",
    "prediction_scale": "log2", "test_labels_used": False,
    "files": {os.path.basename(args.out): {
        "sha256_16": h, "mb": round(os.path.getsize(args.out) / 1e6, 1)}},
}, open(os.path.join(os.path.dirname(args.out), "manifest.json"), "w"),
    indent=1, ensure_ascii=False)
print(f"wrote {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB, sha {h})")
print(f"total {time.time()-t0:.0f}s")
