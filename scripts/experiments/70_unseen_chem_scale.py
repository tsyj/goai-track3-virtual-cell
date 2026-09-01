# -*- coding: utf-8 -*-
"""迭代 2：同一机制的第二个轴——未见化合物行的 booster 定标。

用 69 号缓存的 (加性, booster) 部件，对三类行独立扫缩放：
  仅未见化合物 (in_chem_only)、仅未见菌株 (in_strain_only)、双未知 (in_both)。
网格：k_strain × k_chem，both 行取 k_strain*k_chem 与单独 k_both 两种设计各评一次。

    VCELL_SCALE_TAG=cheap python scripts/70_unseen_chem_scale.py
"""
import os, sys, time, warnings
from multiprocessing import Pool
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vcell.harness import INNER, build_fold, make_inner_splits, evaluate, summary_row

OUT = os.path.join(ROOT, "results")
TAG = os.environ.get("VCELL_SCALE_TAG", "cheap")
CACHE = os.path.join(OUT, f"unseen_parts_{TAG}")
FOLDS = [(1, "CGD"), (5, "CGD"), (6, "CGD"), (3, "BAH"), (7, "BAH"), (8, "BAH")]
KS = [1.0, 1.2, 1.4, 1.6, 1.8]
KC = [1.0, 1.1, 1.2, 1.3, 1.4]


def one_fold(arg):
    seed, held = arg
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "2")
    base_meta = build_fold().meta
    fo = build_fold(splits=make_inner_splits(base_meta, hold_strain=held, seed=seed))
    P_add = np.load(os.path.join(CACHE, f"{seed}_{held}_add.npy"))
    B = np.load(os.path.join(CACHE, f"{seed}_{held}_boost.npy"))
    sp = fo.meta["split_final"].astype(str).to_numpy()
    r_strain = np.isin(sp, ["in_strain_only"])
    r_chem = np.isin(sp, ["in_chem_only"])
    r_both = np.isin(sp, ["in_both"])
    rows = []
    t0 = time.time()
    for ks in KS:
        for kc in KC:
            mult = np.ones(len(sp), np.float32)
            mult[r_strain] = ks
            mult[r_chem] = kc
            mult[r_both] = ks * kc
            r = summary_row("x", evaluate(fo, (P_add + mult[:, None] * B).astype(np.float32), INNER))
            r.update({"seed": seed, "strain": held, "ks": ks, "kc": kc})
            rows.append(r)
    print(f"  fold seed{seed} {held} done ({time.time()-t0:.0f}s)", flush=True)
    return rows


if __name__ == "__main__":
    with Pool(6) as pool:
        out = [r for rows in pool.imap_unordered(one_fold, FOLDS) for r in rows]
    d = pd.DataFrame(out)
    d.to_csv(os.path.join(OUT, f"unseen_chem_scale_{TAG}.csv"), index=False)
    piv = d.pivot_table(index=["seed", "strain"], columns=["ks", "kc"], values="TOTAL")
    base = piv[(1.0, 1.0)]
    print("\n=== (k_strain, k_chem) 网格，vs (1,1)，六折配对 ===")
    print("  ks/kc  " + "  ".join(f"{kc:>18.1f}" for kc in KC))
    for ks in KS:
        cells = []
        for kc in KC:
            dd = piv[(ks, kc)] - base
            cells.append(f"{dd.mean():+.5f}±{dd.sem():.5f}({int((dd>0).sum())})")
        print(f"{ks:>8.1f} " + "  ".join(f"{c:>18s}" for c in cells))
