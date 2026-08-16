"""Backfitting order -- 41 proved it matters, and the current order is not coarse-to-fine.

41_instrument_level.py established that where a term sits in the backfit list can
be worth more than the term itself: moving `instrument` from before `plate` to
after it collapsed the gain from +0.0056 to +0.0013.  Coarse levels have to absorb
first, or the near-unshrunk plate term takes the coarse structure and leaves the
parent nothing.

The rest of the list was never examined in that light, and it is not sorted
coarse-to-fine:

    source(4) -> instrument(7) -> plate(144) -> plate_x_strain(381) -> strain(5)
        -> strain_x_medium -> strain_x_temp -> strain_x_time -> strain_x_source

`strain` has five levels but is fitted *after* the 144-level plate and the
381-level plate_x_strain.  By the mechanism 41 demonstrated, those two absorb the
strain-level structure first and `strain` only ever sees the remainder -- which
also means the strain term is estimated from whatever plate happens to leave it,
rather than from all wells of that strain.

That should matter most for exactly the rows that are hardest: a held-out strain
gets zero from every strain-indexed term, so what the coarse terms captured before
plate is all it has.

Also swept here, since it was likewise set once and never revisited: `n_pass`
(number of backfitting sweeps).  Adding two more levels to the ladder may simply
need more passes to converge.

Decision column: the three orphan-free folds.

    VCELL_WORKERS=14 python scripts/49_factor_order.py

Jiao Xinyuan 2026-08-16
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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcell.harness import (INNER, build_fold, evaluate, make_inner_splits,  # noqa: E402
                           summary_row)
from vcell.models import (BATCH_FACTORS, PERT_FACTORS, ResidualBooster,     # noqa: E402
                          UnifiedBackfit)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
N_WORKERS = int(os.environ.get("VCELL_WORKERS", 14))
LGB_THREADS = int(os.environ.get("VCELL_LGB_THREADS", 8))

FOLDS = [(0, "CEK"), (1, "CGD"), (2, "DHY210"), (3, "BAH"), (4, "CEK"), (5, "CGD")]
CHEAP = {"n_comp": 96, "n_estimators": 800, "learning_rate": 0.03}
PLATE = {"lam_plate": 2.0, "lam_plate_x_strain": 6.0}      # 44/45/46 winner

BY_NAME = {a: (a, c, l) for a, c, l in BATCH_FACTORS}
CUR = [a for a, _, _ in BATCH_FACTORS]
# coarse -> fine by number of levels: source 4, instrument 7, strain 5,
# strain_x_* 10-40, plate 144, plate_x_strain 381
FINE_LAST = ["source", "instrument", "strain", "strain_x_medium", "strain_x_temp",
             "strain_x_time", "strain_x_source", "plate", "plate_x_strain"]
STRAIN_EARLY = ["source", "instrument", "strain", "plate", "plate_x_strain",
                "strain_x_medium", "strain_x_temp", "strain_x_time", "strain_x_source"]

REF = "current order, n_pass 6"
CONFIGS = {
    REF:                          {"order": CUR, "n_pass": 6},
    "current order, n_pass 12":   {"order": CUR, "n_pass": 12},
    "current order, n_pass 24":   {"order": CUR, "n_pass": 24},
    "strain before plate":        {"order": STRAIN_EARLY, "n_pass": 6},
    "strain before plate, np12":  {"order": STRAIN_EARLY, "n_pass": 12},
    "fully coarse->fine":         {"order": FINE_LAST, "n_pass": 6},
    "fully coarse->fine, np12":   {"order": FINE_LAST, "n_pass": 12},
}


def one_job(arg):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "2"
    name, cfg, seed, strain = arg
    t0 = time.time()
    base_meta = build_fold().meta
    fo = build_fold(splits=make_inner_splits(base_meta, hold_strain=strain, seed=seed))
    batch = [BY_NAME[a] for a in cfg["order"]]
    um = UnifiedBackfit(
        batch_factors=[(a, c, PLATE.get(f"lam_{a}", l)) for a, c, l in batch],
        pert_factors=list(PERT_FACTORS),
        n_pass=cfg["n_pass"],
    ).fit(fo.meta, fo.Y_obs, fo.obs_mask)
    P = um.predict()
    r_add = summary_row(name, evaluate(fo, P, INNER))["TOTAL"]
    rb = ResidualBooster(n_jobs=LGB_THREADS, **CHEAP)
    rb.fit(fo.meta, fo.Y_obs, fo.obs_mask, P)
    r = summary_row(name, evaluate(fo, P + rb.predict(), INNER))
    r.update({"config": name, "seed": seed, "strain": strain,
              "TOTAL_additive_only": r_add, "secs": round(time.time() - t0, 1)})
    return r


if __name__ == "__main__":
    jobs = [(n, c, s, st) for (n, c), (s, st) in itertools.product(CONFIGS.items(), FOLDS)]
    print(f"{len(CONFIGS)} configs x {len(FOLDS)} folds = {len(jobs)} jobs", flush=True)
    t0 = time.time()
    done = []
    with Pool(N_WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(one_job, jobs), 1):
            done.append(r)
            if i % 6 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done ({time.time()-t0:.0f}s)", flush=True)
                pd.DataFrame(done).to_csv(os.path.join(OUT, "factor_order_raw.csv"), index=False)
    pd.DataFrame(done).to_csv(os.path.join(OUT, "factor_order_raw.csv"), index=False)
    print(f"\ntotal {time.time()-t0:.0f}s")
    print("用 scripts/analyze_paired.py 读, 决策看 free_delta 列")
