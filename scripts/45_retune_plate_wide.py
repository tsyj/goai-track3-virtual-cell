"""Re-tune the plate shrinkage -- both reasons to do it come from this session.

1. **The old tuning was contaminated.**  `lam_plate` (0.3) and `lam_plate_x_strain`
   (2.0) were picked on the six inner mirrors, three of which put ~32% of their
   evaluation rows on plates with no training label at all.  `lam_plate` is exactly
   the knob that decides how far a thinly-observed plate is pulled toward zero, so
   those rows had an outsized say -- and the official split has none of them
   (0 of 4,454 test rows).

2. **The model changed underneath it.**  data_source and instrument now sit above
   plate and absorb the coarse batch level first, so the plate term no longer
   carries it.  The amount of shrinkage that was right for "plate carries
   everything" is not the amount that is right for "plate carries the deviation
   from its instrument".

Decision column is the three orphan-free folds, as always
(`scripts/analyze_paired.py`).  All six are still run and reported so the size of
the contamination stays visible.

    VCELL_WORKERS=14 python scripts/44_retune_plate_lambda.py

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
REF = "prev best (plate 2, pxs 6)"

# BATCH_FACTORS already carries source + instrument as of 2026-08-16
GRID = [(p, x) for p in (2.0, 4.0, 8.0, 16.0, 32.0) for x in (6.0, 12.0, 24.0, 48.0)]
CONFIGS = {REF: {"lam_plate": 2.0, "lam_plate_x_strain": 6.0}}
for p, x in GRID:
    if (p, x) == (2.0, 6.0):
        continue
    CONFIGS[f"plate {p:g}, pxs {x:g}"] = {"lam_plate": p, "lam_plate_x_strain": x}


def one_job(arg):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "2"
    name, cfg, seed, strain = arg
    t0 = time.time()
    base_meta = build_fold().meta
    fo = build_fold(splits=make_inner_splits(base_meta, hold_strain=strain, seed=seed))
    um = UnifiedBackfit(
        batch_factors=[(a, c, cfg.get(f"lam_{a}", l)) for a, c, l in BATCH_FACTORS],
        pert_factors=[(a, c, cfg.get(f"lam_{a}", l)) for a, c, l in PERT_FACTORS],
        n_pass=6,
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
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done ({time.time()-t0:.0f}s)", flush=True)
                pd.DataFrame(done).to_csv(os.path.join(OUT, "retune_plate2_raw.csv"), index=False)
    raw = pd.DataFrame(done)
    raw.to_csv(os.path.join(OUT, "retune_plate2_raw.csv"), index=False)
    print(f"\ntotal {time.time()-t0:.0f}s")
    print("用 scripts/analyze_paired.py 读, 决策看 free_delta 列")
