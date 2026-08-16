"""Compose submission/prediction.csv from saved member predictions.

    python scripts/58_compose_submission.py --members A,B,C,D,E,F_strain_early \
        [--csv path.csv:WEIGHT ...]  [--out submission/prediction.csv] [--dry]

Averages submission/members/<m>.npy (from 57_predict_member.py) with equal weight;
--csv adds an already-written prediction file with the given weight (e.g. the
5-member average from 54_predict_ensemble.py counts as weight 5).  Row/column
order is checked against the member json before averaging.  Writes the csv in
exactly 54's format and a manifest.json next to it.

Jiao Xinyuan 2026-08-16 (evening session)
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = os.path.join(ROOT, "submission")
MEM = os.path.join(SUB, "members")

ap = argparse.ArgumentParser()
ap.add_argument("--members", default="")
ap.add_argument("--csv", action="append", default=[], help="path:weight")
ap.add_argument("--out", default=os.path.join(SUB, "prediction.csv"))
ap.add_argument("--dry", action="store_true")
ap.add_argument("--note", default="")
args = ap.parse_args()

t0 = time.time()
members = [m for m in args.members.split(",") if m]
ids = proteins = None
acc, wsum, parts = None, 0.0, []
for m in members:
    info = json.load(open(os.path.join(MEM, f"{m}.json")))
    arr = np.load(os.path.join(MEM, f"{m}.npy")).astype(np.float64)
    if ids is None:
        ids, proteins = info["sample_ids"], info["proteins"]
    assert info["sample_ids"] == ids and info["proteins"] == proteins, f"{m}: order mismatch"
    assert arr.shape == (len(ids), len(proteins)) and np.isfinite(arr).all()
    acc = arr if acc is None else acc + arr
    wsum += 1.0
    parts.append({"member": m, "weight": 1.0, "sha256_16": info["sha256_16"],
                  "lam": info["lam_effective"], "order": info["order"],
                  "n_pass": info["n_pass"], "fit_offset": info["fit_offset"]})
    print(f"  member {m}  w=1  mean={arr.mean():.4f}", flush=True)
for spec in args.csv:
    path, w = spec.rsplit(":", 1)
    w = float(w)
    df = pd.read_csv(path, index_col=0)
    if ids is None:
        ids, proteins = [str(i) for i in df.index], [str(c) for c in df.columns]
    df = df.loc[ids, proteins]
    arr = df.to_numpy(np.float64)
    assert arr.shape == (len(ids), len(proteins)) and np.isfinite(arr).all()
    acc = arr * w if acc is None else acc + arr * w
    wsum += w
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    parts.append({"csv": os.path.relpath(path, ROOT), "weight": w, "sha256_16": h})
    print(f"  csv {path}  w={w:g}  mean={arr.mean():.4f}", flush=True)

assert wsum > 0, "nothing to average"
Y = (acc / wsum).astype(np.float32)
assert np.isfinite(Y).all()
print(f"composed {len(ids)} x {len(proteins)}, total weight {wsum:g}, "
      f"members {len(members)} + csv {len(args.csv)}")
if args.dry:
    sys.exit(0)

df = pd.DataFrame(Y, columns=proteins, index=pd.Index(ids, name="sample_ID"))
assert not df.isna().any().any()
tmp = args.out + ".tmp"
df.to_csv(tmp, float_format="%.5f")
os.replace(tmp, args.out)
h = hashlib.sha256(open(args.out, "rb").read()).hexdigest()[:16]
manifest = {
    "n_test_rows": len(ids), "n_proteins": len(proteins),
    "ensemble_parts": parts, "total_weight": wsum,
    "booster": {"n_comp": 240, "n_estimators": 1600, "learning_rate": 0.015,
                "seeds": [0, 1, 2]},
    "fitted_on": "split_final=='train' only (5,920 rows)",
    "prediction_scale": "log2", "protein_filter": "train-row missing rate < 0.80",
    "test_labels_used": False,
    "selected_by": args.note or "scripts/55_member_pool.py + 56_pool_eval.py",
    "files": {os.path.basename(args.out): {
        "sha256_16": h, "mb": round(os.path.getsize(args.out) / 1e6, 1)}},
}
json.dump(manifest, open(os.path.join(os.path.dirname(args.out), "manifest.json"), "w"),
          indent=1, ensure_ascii=False)
print(f"wrote {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB, sha {h})")
print(f"total {time.time()-t0:.0f}s")
