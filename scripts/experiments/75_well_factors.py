# -*- coding: utf-8 -*-
"""迭代 5：孔位（板边缘）效应作为加性层的批次因子。

booster 有 well_row/well_col 特征，但加性层没有；系统性的、逐蛋白的板边缘偏差
用 240 维残差 SVD + 树来学是低效的。四个变体，六无孤儿折配对（便宜 booster）：
  base            : A 配置
  +row+col        : 加 well_row(8) 与 well_col(12) 主效应（plate×strain 之后）
  +row×inst+col×inst : 按仪器分开的行/列效应
  +edge           : 只加一个边缘标志（A/H 行或 1/12 列）
"""
import os, sys, time, warnings
from multiprocessing import Pool
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vcell.harness import INNER, build_fold, make_inner_splits, evaluate, summary_row
from vcell.models import BATCH_FACTORS, PERT_FACTORS, ResidualBooster, UnifiedBackfit

OUT = os.path.join(ROOT, "results")
FOLDS = [(1, "CGD"), (5, "CGD"), (6, "CGD"), (3, "BAH"), (7, "BAH"), (8, "BAH")]
CHEAP = {"n_comp": 96, "n_estimators": 800, "learning_rate": 0.03}
PLATE = {"lam_plate": 2.0, "lam_plate_x_strain": 6.0}
PERT4 = {f"lam_{a}": l * 4.0 for a, _, l in PERT_FACTORS}
VARIANTS = {
    "base": [],
    "+row+col": [("well_row", ("well_row",), 4.0), ("well_col", ("well_col",), 4.0)],
    "+rowXinst+colXinst": [("row_x_inst", ("well_row", "instrument"), 4.0),
                           ("col_x_inst", ("well_col", "instrument"), 4.0)],
    "+edge": [("edge", ("well_edge",), 4.0)],
}


def add_well_cols(meta):
    meta = meta.copy()
    w = meta["protein_well"].astype(str)
    meta["well_row"] = w.str[0]
    meta["well_col"] = w.str[1:].astype(int)
    meta["well_edge"] = (meta["well_row"].isin(["A", "H"]) | meta["well_col"].isin([1, 12])).astype(int)
    return meta


def one_fold(arg):
    seed, held = arg
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "2")
    t0 = time.time()
    base_meta = build_fold().meta
    fo = build_fold(splits=make_inner_splits(base_meta, hold_strain=held, seed=seed))
    meta = add_well_cols(fo.meta)
    lam = {**PLATE, **PERT4}
    batch0 = [(a, c, lam.get(f"lam_{a}", l)) for a, c, l in BATCH_FACTORS]
    # 在 plate_x_strain 之后插入孔位因子
    ins = [i for i, (a, _, _) in enumerate(batch0) if a == "plate_x_strain"][0] + 1
    rows = []
    for name, extra in VARIANTS.items():
        batch = batch0[:ins] + extra + batch0[ins:]
        um = UnifiedBackfit(batch_factors=batch,
                            pert_factors=[(a, c, lam.get(f"lam_{a}", l)) for a, c, l in PERT_FACTORS],
                            n_pass=6).fit(meta, fo.Y_obs, fo.obs_mask)
        P_add = um.predict()
        rb = ResidualBooster(n_jobs=6, **CHEAP).fit(fo.meta, fo.Y_obs, fo.obs_mask, P_add)
        P = (P_add + rb.predict()).astype(np.float32)
        r = summary_row(name, evaluate(fo, P, INNER)); r.update({"seed": seed, "strain": held, "variant": name})
        rows.append(r)
    print(f"  seed{seed} {held} done ({time.time()-t0:.0f}s)", flush=True)
    return rows


if __name__ == "__main__":
    with Pool(6) as pool:
        out = [r for rows in pool.imap_unordered(one_fold, FOLDS) for r in rows]
    d = pd.DataFrame(out); d.to_csv(os.path.join(OUT, "well_factors.csv"), index=False)
    piv = d.pivot_table(index=["seed", "strain"], columns="variant", values="TOTAL")
    print("\n=== 孔位因子（六折配对，vs base）===")
    for v in VARIANTS:
        dd = piv[v] - piv["base"]
        print(f"  {v:20s} delta={dd.mean():+.5f}  sem={dd.sem():.5f}  up={(dd>0).sum()}/6")
    for mod in ["M1_abs(20%)", "M2_rawFC(25%)", "M3_ctx(20%)", "M4_drug(20%)", "FC[chem_only]", "FC[strain_only]"]:
        q = d.pivot_table(index=["seed", "strain"], columns="variant", values=mod)
        print(f"  {mod:16s} " + "  ".join(f"{v}:{(q[v]-q['base']).mean():+.5f}" for v in VARIANTS if v != "base"))
