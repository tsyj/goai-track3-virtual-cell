"""迭代 1c：只对「零标签菌株」的行缩放 booster 输出——一个没试过的结构轴。

booster 在训练集（几乎全是可见菌株）上学高阶交互；对留出菌株的行，它的输出可能系统性
过强或过弱。37 号测过的是加性扰动项的放大（撞 M2/M4 墙），booster 部分从未单独校准过。

对每个无孤儿折：拟合 A 配置的加性模型与便宜 booster，把两部分分开存，然后只对该折
留出菌株的行按 k 缩放 booster，逐折配对评估 k ∈ {0.6, 0.8, 1.0, 1.2, 1.4}。

    python scripts/69_unseen_booster_scale.py

Jiao Xinyuan 2026-09-02
"""
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vcell.harness import (INNER, build_fold, evaluate, make_inner_splits,  # noqa: E402
                           summary_row)
from vcell.models import (BATCH_FACTORS, PERT_FACTORS, ResidualBooster,     # noqa: E402
                          UnifiedBackfit)

OUT = os.path.join(ROOT, "results")
FOLDS = [(1, "CGD"), (5, "CGD"), (6, "CGD"), (3, "BAH"), (7, "BAH"), (8, "BAH")]
CHEAP = {"n_comp": 96, "n_estimators": 800, "learning_rate": 0.03}
REAL = {"n_comp": 240, "n_estimators": 1600, "learning_rate": 0.015, "seeds": [0, 1, 2]}
TAG = os.environ.get("VCELL_SCALE_TAG", "cheap")
BOOST = CHEAP if TAG == "cheap" else REAL
CACHE = os.path.join(OUT, f"unseen_parts_{TAG}")
PLATE = {"lam_plate": 2.0, "lam_plate_x_strain": 6.0}
PERT4 = {f"lam_{a}": l * 4.0 for a, _, l in PERT_FACTORS}
SCALES = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 3.0]


def one_fold(arg):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "2")
    seed, held = arg
    t0 = time.time()
    base_meta = build_fold().meta
    fo = build_fold(splits=make_inner_splits(base_meta, hold_strain=held, seed=seed))
    os.makedirs(CACHE, exist_ok=True)
    ca, cb = (os.path.join(CACHE, f"{seed}_{held}_{x}.npy") for x in ("add", "boost"))
    if os.path.exists(ca) and os.path.exists(cb):
        P_add, B = np.load(ca), np.load(cb)
    else:
        lam = {**PLATE, **PERT4}
        um = UnifiedBackfit(
            batch_factors=[(a, c, lam.get(f"lam_{a}", l)) for a, c, l in BATCH_FACTORS],
            pert_factors=[(a, c, lam.get(f"lam_{a}", l)) for a, c, l in PERT_FACTORS],
            n_pass=6).fit(fo.meta, fo.Y_obs, fo.obs_mask)
        P_add = um.predict()
        rb = ResidualBooster(n_jobs=6, **BOOST).fit(fo.meta, fo.Y_obs, fo.obs_mask, P_add)
        B = rb.predict().astype(np.float32)
        np.save(ca, P_add); np.save(cb, B)
    unseen = (fo.meta["Strains"] == held).to_numpy() & ~fo.obs_mask
    rows = []
    for k in SCALES:
        mult = np.where(unseen[:, None], k, 1.0).astype(np.float32)
        r = summary_row(f"k={k}", evaluate(fo, (P_add + mult * B).astype(np.float32), INNER))
        r.update({"seed": seed, "strain": held, "k": k})
        rows.append(r)
    print(f"  fold seed{seed} {held} done ({time.time()-t0:.0f}s, unseen rows {unseen.sum()})",
          flush=True)
    return rows


if __name__ == "__main__":
    t0 = time.time()
    with Pool(6) as pool:
        out = [r for rows in pool.imap_unordered(one_fold, FOLDS) for r in rows]
    d = pd.DataFrame(out)
    d.to_csv(os.path.join(OUT, f"unseen_booster_scale_{TAG}.csv"), index=False)
    piv = d.pivot_table(index=["seed", "strain"], columns="k", values="TOTAL")
    base = piv[1.0]
    print("\n=== 只缩放留出菌株行的 booster 输出（六折配对，vs k=1）===")
    for k in SCALES:
        dd = piv[k] - base
        print(f"  k={k:.1f}  delta={dd.mean():+.5f}  sem={dd.sem():.5f}  up={(dd>0).sum()}/6")
    for mod in ["M2_rawFC(25%)", "M4_drug(20%)", "M1_abs(20%)", "FC[strain_only]"]:
        if mod in d.columns:
            q = d.pivot_table(index=["seed", "strain"], columns="k", values=mod)
            print(f"  {mod:16s} " + "  ".join(f"k={k}:{(q[k]-q[1.0]).mean():+.5f}" for k in SCALES))
    print(f"\ntotal {time.time()-t0:.0f}s")
