# -*- coding: utf-8 -*-
"""迭代 9：加性层的三阶交互与化合物×时间段中间层。

  +cmpd_x_tbin        : compound × {早(≤60min), 晚(>60)} 插在 compound 与 cmpd_x_time 之间（层级平滑）
  +strain_x_med_temp  : Strains × Medium × Temperature 三阶
  +cmpd_x_med_temp    : compound × Medium × Temperature 三阶
  +all                : 三者全加
六无孤儿折配对，便宜 booster。
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
TB = ("cmpd_x_tbin", ("compound", "tbin"), 24.0)
SMT = ("strain_x_med_temp", ("Strains", "Medium", "Temperature"), 6.0)
CMT = ("cmpd_x_med_temp", ("compound", "Medium", "Temperature"), 48.0)
VARIANTS = {"base": ([], []), "+cmpd_x_tbin": ([], [TB]), "+strain_x_med_temp": ([SMT], []),
            "+cmpd_x_med_temp": ([], [CMT]), "+all": ([SMT], [TB, CMT])}


def one_fold(arg):
    seed, held = arg
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "2")
    t0 = time.time()
    base_meta = build_fold().meta
    fo = build_fold(splits=make_inner_splits(base_meta, hold_strain=held, seed=seed))
    meta = fo.meta.copy(); meta["tbin"] = (meta["pert_time"].astype(float) > 60).astype(int)
    lam = {**PLATE, **PERT4}
    batch0 = [(a, c, lam.get(f"lam_{a}", l)) for a, c, l in BATCH_FACTORS]
    pert0 = [(a, c, lam.get(f"lam_{a}", l)) for a, c, l in PERT_FACTORS]
    rows = []
    for name, (bx, px) in VARIANTS.items():
        batch = batch0 + bx                      # 三阶菌株项放批次族末尾
        pert = pert0[:1] + [p for p in px if p[0] == "cmpd_x_tbin"] + pert0[1:] + [p for p in px if p[0] != "cmpd_x_tbin"]
        um = UnifiedBackfit(batch_factors=batch, pert_factors=pert, n_pass=6).fit(meta, fo.Y_obs, fo.obs_mask)
        P_add = um.predict()
        rb = ResidualBooster(n_jobs=6, **CHEAP).fit(fo.meta, fo.Y_obs, fo.obs_mask, P_add)
        r = summary_row(name, evaluate(fo, (P_add + rb.predict()).astype(np.float32), INNER))
        r.update({"seed": seed, "strain": held, "variant": name}); rows.append(r)
    print(f"  seed{seed} {held} done ({time.time()-t0:.0f}s)", flush=True)
    return rows


if __name__ == "__main__":
    with Pool(6) as pool:
        out = [r for rows in pool.imap_unordered(one_fold, FOLDS) for r in rows]
    d = pd.DataFrame(out); d.to_csv(os.path.join(OUT, "higher_order.csv"), index=False)
    piv = d.pivot_table(index=["seed", "strain"], columns="variant", values="TOTAL")
    print("\n=== 三阶交互 / 时间段中间层（六折配对，vs base）===")
    for v in VARIANTS:
        dd = piv[v] - piv["base"]
        print(f"  {v:20s} delta={dd.mean():+.5f}  sem={dd.sem():.5f}  up={(dd>0).sum()}/6")
    for mod in ["M2_rawFC(25%)", "M3_ctx(20%)", "M4_drug(20%)", "FC[time]", "FC[chem_only]", "FC[strain_only]"]:
        q = d.pivot_table(index=["seed", "strain"], columns="variant", values=mod)
        print(f"  {mod:16s} " + "  ".join(f"{v}:{(q[v]-q['base']).mean():+.5f}" for v in VARIANTS if v != "base"))
