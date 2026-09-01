"""Member pool: fit every candidate configuration once per orphan-free fold and
*save the prediction*, so that any single-config comparison and any ensemble
combination can be scored afterwards without refitting.

Why this shape.  53_config_ensemble.py refitted every member inside every
ensemble job (a 5-member job = five full fits), and singles / ensembles lived in
different scripts with different folds.  Here one (member, fold) job = one fit,
saved as results/pool_<tag>/<seed>_<strain>__<member>.npy (full n x p float32),
and scripts/56_pool_eval.py scores any subset -- greedy forward selection over
members included -- in seconds.

Candidates (all on top of the adopted base: source+instrument, plate 2 / pxs 6,
pert x4, n_pass 6, fit_offset on):

    A..E            the five shipped ensemble members (plate 1/2/4, pert x2/x4/x8)
    F_strain_early  strain fitted before plate (+0.0021, 3/6 in 51 -- diverse, not worse)
    G/H_npass4/3    fewer backfitting sweeps (12/24 were negative in 49, so try the
                    other direction; n_pass acts as extra implicit shrinkage)
    I_nooffset      fit_offset off (set once, never ablated)
    J/K_pxs3/12     plate_x_strain lambda neighbours (44/45 tuned it on orphan folds)
    S_<term>_x0.25 / _x4   each strain-family lambda alone (47 only ever moved the
                    whole family together, and found it flat)
    T_strainfam_x0.5       whole family x0.5 (47's marginal best at pert x4)

Booster is selected by VCELL_POOL_TAG: 'cheap' (96/800/.03, screening) or 'real'
(240/1600/.015, seeds x3, the shipped one).  VCELL_MEMBERS="A,B,F_strain_early"
restricts the member list (used for the real-booster confirmation).

    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    VCELL_WORKERS=22 VCELL_LGB_THREADS=8 VCELL_POOL_TAG=cheap \
      python scripts/55_member_pool.py

Jiao Xinyuan 2026-08-16 (evening session)
"""
import itertools
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
TAG = os.environ.get("VCELL_POOL_TAG", "cheap")
POOL_DIR = os.path.join(OUT, f"pool_{TAG}")
N_WORKERS = int(os.environ.get("VCELL_WORKERS", 22))
LGB_THREADS = int(os.environ.get("VCELL_LGB_THREADS", 8))

FOLDS = [(1, "CGD"), (5, "CGD"), (6, "CGD"), (3, "BAH"), (7, "BAH"), (8, "BAH")]

CHEAP = {"n_comp": 96, "n_estimators": 800, "learning_rate": 0.03}
REAL = {"n_comp": 240, "n_estimators": 1600, "learning_rate": 0.015, "seeds": [0, 1, 2]}
BOOSTER = CHEAP if TAG == "cheap" else REAL

PLATE = {"lam_plate": 2.0, "lam_plate_x_strain": 6.0}
PERT4 = {f"lam_{a}": l * 4.0 for a, _, l in PERT_FACTORS}
BASE_LAM = {**PLATE, **PERT4}
BY_NAME = {a: (a, c, l) for a, c, l in BATCH_FACTORS}
CUR_ORDER = [a for a, _, _ in BATCH_FACTORS]
STRAIN_EARLY = ["source", "instrument", "strain", "plate", "plate_x_strain",
                "strain_x_medium", "strain_x_temp", "strain_x_time", "strain_x_source"]
STRAIN_FAMILY = ["strain", "strain_x_medium", "strain_x_temp", "strain_x_time",
                 "strain_x_source"]
BASE_STRAIN_LAM = {a: l for a, _, l in BATCH_FACTORS if a in STRAIN_FAMILY}


def pert_mult(m):
    return {f"lam_{a}": l * m for a, _, l in PERT_FACTORS}


MEMBERS = {
    "A": {},
    "B": {"lam": {"lam_plate": 1.0}},
    "C": {"lam": {"lam_plate": 4.0}},
    "D": {"lam": pert_mult(2.0)},
    "E": {"lam": pert_mult(8.0)},
    "F_strain_early": {"order": STRAIN_EARLY},
    "G_npass4": {"n_pass": 4},
    "H_npass3": {"n_pass": 3},
    "I_nooffset": {"fit_offset": False},
    "J_pxs3": {"lam": {"lam_plate_x_strain": 3.0}},
    "K_pxs12": {"lam": {"lam_plate_x_strain": 12.0}},
    "T_strainfam_x0.5": {"lam": {f"lam_{a}": l * 0.5 for a, l in BASE_STRAIN_LAM.items()}},
}
for t in STRAIN_FAMILY:
    for m in (0.25, 4.0):
        MEMBERS[f"S_{t}_x{m:g}"] = {"lam": {f"lam_{t}": BASE_STRAIN_LAM[t] * m}}
# strain-early order crossed with the same lambda neighbourhood as B..E, so the
# ensemble can carry a whole "F family" (added after F alone came out +0.00077, 6/6)
MEMBERS.update({
    "FB_early_plate1": {"order": STRAIN_EARLY, "lam": {"lam_plate": 1.0}},
    "FC_early_plate4": {"order": STRAIN_EARLY, "lam": {"lam_plate": 4.0}},
    "FD_early_pert2": {"order": STRAIN_EARLY, "lam": pert_mult(2.0)},
    "FE_early_pert8": {"order": STRAIN_EARLY, "lam": pert_mult(8.0)},
})

# third round (20:20): structurally different candidates, screened for ensemble diversity
PLATE_STRAIN_PXS = ["source", "instrument", "plate", "strain", "plate_x_strain",
                    "strain_x_medium", "strain_x_temp", "strain_x_time", "strain_x_source"]
STRAIN_FIRST = ["strain", "source", "instrument", "plate", "plate_x_strain",
                "strain_x_medium", "strain_x_temp", "strain_x_time", "strain_x_source"]
MEMBERS.update({
    "G1_plate_strain_pxs": {"order": PLATE_STRAIN_PXS},
    "G2_strain_first": {"order": STRAIN_FIRST},
    "FS_early_strain_x0.25": {"order": STRAIN_EARLY, "lam": {"lam_strain": 0.75}},
    "FS_early_strain_x4": {"order": STRAIN_EARLY, "lam": {"lam_strain": 12.0}},
    "E16_pert16": {"lam": pert_mult(16.0)},
    "FE16_early_pert16": {"order": STRAIN_EARLY, "lam": pert_mult(16.0)},
})

# fourth round (2026-08-27, 复赛):结构轴上的多样性,不是 lambda 邻域。
# 08-16 的教训是 F 族(因子顺序)单独 3/6 不过线,做集成成员却 6/6 +0.0019,而 strain
# 族的 lambda 邻域做成员全零(56)。所以这一批只沿"结构"取点:再往外一档的 lambda 阶梯、
# 没试过的 n_pass 8、两个新的因子顺序,以及两个被单独否决过的 booster 容量点
# (num_leaves 127 = -0.0001, 320 成分 = +0.00012) —— 正是"高方差、均值近零"的成员画像。
PXS_LAST = ["source", "instrument", "plate", "strain", "strain_x_medium",
            "strain_x_temp", "strain_x_time", "strain_x_source", "plate_x_strain"]
INSTR_FIRST = ["instrument", "source", "plate", "plate_x_strain", "strain",
               "strain_x_medium", "strain_x_temp", "strain_x_time", "strain_x_source"]
MEMBERS.update({
    "L_pert32": {"lam": pert_mult(32.0)},
    "FL_early_pert32": {"order": STRAIN_EARLY, "lam": pert_mult(32.0)},
    "M_plate8": {"lam": {"lam_plate": 8.0}},
    "FM_early_plate8": {"order": STRAIN_EARLY, "lam": {"lam_plate": 8.0}},
    "N_npass8": {"n_pass": 8},
    "O_instr_first": {"order": INSTR_FIRST},
    "P_pxs_last": {"order": PXS_LAST},
    "FP_early_pxs_last": {"order": ["source", "instrument", "strain", "plate",
                                    "strain_x_medium", "strain_x_temp",
                                    "strain_x_time", "strain_x_source",
                                    "plate_x_strain"]},
    "Bq_leaves127": {"booster_over": {"num_leaves": 127}},
    "Br_ncomp320": {"booster_over": {"n_comp": 320}},
})
for _m, _cfg in MEMBERS.items():
    if "order" in _cfg:
        assert sorted(_cfg["order"]) == sorted(CUR_ORDER), _m

if os.environ.get("VCELL_MEMBERS"):
    want = [m.strip() for m in os.environ["VCELL_MEMBERS"].split(",") if m.strip()]
    missing = [m for m in want if m not in MEMBERS]
    assert not missing, f"unknown members {missing}"
    MEMBERS = {m: MEMBERS[m] for m in want}


def pred_path(seed, strain, member):
    return os.path.join(POOL_DIR, f"{seed}_{strain}__{member}.npy")


def one_job(arg):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "2")
    member, seed, strain = arg
    cfg = MEMBERS[member]
    t0 = time.time()
    base_meta = build_fold().meta
    fo = build_fold(splits=make_inner_splits(base_meta, hold_strain=strain, seed=seed))
    lam = {**BASE_LAM, **cfg.get("lam", {})}
    batch = [BY_NAME[a] for a in cfg.get("order", CUR_ORDER)]
    um = UnifiedBackfit(
        batch_factors=[(a, c, lam.get(f"lam_{a}", l)) for a, c, l in batch],
        pert_factors=[(a, c, lam.get(f"lam_{a}", l)) for a, c, l in PERT_FACTORS],
        n_pass=cfg.get("n_pass", 6), fit_offset=cfg.get("fit_offset", True),
    ).fit(fo.meta, fo.Y_obs, fo.obs_mask)
    P = um.predict()
    r_add = summary_row(member, evaluate(fo, P, INNER))["TOTAL"]
    boost = {**cfg.get("booster", BOOSTER), **cfg.get("booster_over", {})}
    rb = ResidualBooster(n_jobs=LGB_THREADS, **boost)
    rb.fit(fo.meta, fo.Y_obs, fo.obs_mask, P)
    P = (P + rb.predict()).astype(np.float32)
    tmp = pred_path(seed, strain, member) + ".tmp.npy"
    np.save(tmp, P)
    os.replace(tmp, pred_path(seed, strain, member))
    r = summary_row(member, evaluate(fo, P, INNER))
    r.update({"config": member, "seed": seed, "strain": strain, "tag": TAG,
              "TOTAL_additive_only": r_add, "secs": round(time.time() - t0, 1)})
    return r


if __name__ == "__main__":
    os.makedirs(POOL_DIR, exist_ok=True)
    jobs = [(m, s, st) for m, (s, st) in itertools.product(MEMBERS, FOLDS)
            if not os.path.exists(pred_path(s, st, m))]
    print(f"tag={TAG} booster={BOOSTER}", flush=True)
    print(f"{len(MEMBERS)} members x {len(FOLDS)} orphan-free folds = "
          f"{len(jobs)} jobs to run (existing predictions skipped)", flush=True)
    raw_path = os.path.join(OUT, f"pool_{TAG}_raw.csv")
    done = list(pd.read_csv(raw_path).to_dict("records")) if os.path.exists(raw_path) else []
    t0 = time.time()
    with Pool(N_WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(one_job, jobs), 1):
            done.append(r)
            print(f"  {i}/{len(jobs)} done ({time.time()-t0:.0f}s)  "
                  f"{r['config']} seed{r['seed']} {r['strain']} TOTAL={r['TOTAL']:.4f} "
                  f"({r['secs']:.0f}s)", flush=True)
            pd.DataFrame(done).to_csv(raw_path, index=False)
    pd.DataFrame(done).to_csv(raw_path, index=False)
    print(f"\ntotal {time.time()-t0:.0f}s")
    print("用 scripts/56_pool_eval.py 读（单配置 + 任意集成组合）")
