"""The lambdas nobody ever tuned.

44/45 re-tuned lam_plate and lam_plate_x_strain and got +0.0052 -- the largest
single gain of the session.  Those two are also the *only* shrinkage parameters
that were ever tuned.  Everything else still carries the value it was first
written with:

    batch, strain family : strain 3.0, strain_x_medium 6.0, strain_x_temp 6.0,
                           strain_x_time 8.0, strain_x_source 8.0
    perturbation family  : compound 8.0, cmpd_x_time/temp/medium/source 12.0,
                           cmpd_x_strain 18.0

(`lam_compound` was probed at 3 and 20 in 08_tune_inner, on two folds, under the
old batch structure, and moved nothing -- that is not a tuning.)

There is no reason those initial values should be optimal, and less reason now
that source + instrument sit above plate and the plate lambdas have moved by
almost an order of magnitude.  Rather than a 10-dimensional search, sweep one
multiplier per family: it asks "is this whole family shrunk about right?", which
is the question the plate result says to ask.

Decision column is the three orphan-free folds, as always.

    VCELL_WORKERS=14 python scripts/47_family_lambdas.py

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
# 44/45 winner, now the incumbent for everything downstream
PLATE = {"lam_plate": 2.0, "lam_plate_x_strain": 6.0}
STRAIN_FAMILY = {"strain", "strain_x_medium", "strain_x_temp", "strain_x_time",
                 "strain_x_source"}
REF = "strain x1, pert x4"

CONFIGS = {}
for a in (0.25, 1.0):
    for b in (4.0, 6.0, 8.0, 12.0, 16.0, 24.0):
        CONFIGS[f"strain x{a:g}, pert x{b:g}"] = {"strain_mult": a, "pert_mult": b}


def one_job(arg):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "2"
    name, cfg, seed, strain = arg
    t0 = time.time()
    base_meta = build_fold().meta
    fo = build_fold(splits=make_inner_splits(base_meta, hold_strain=strain, seed=seed))

    def blam(a, l):
        l = PLATE.get(f"lam_{a}", l)
        return l * cfg["strain_mult"] if a in STRAIN_FAMILY else l

    um = UnifiedBackfit(
        batch_factors=[(a, c, blam(a, l)) for a, c, l in BATCH_FACTORS],
        pert_factors=[(a, c, l * cfg["pert_mult"]) for a, c, l in PERT_FACTORS],
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
                pd.DataFrame(done).to_csv(os.path.join(OUT, "family_lambda2_raw.csv"), index=False)
    pd.DataFrame(done).to_csv(os.path.join(OUT, "family_lambda2_raw.csv"), index=False)
    print(f"\ntotal {time.time()-t0:.0f}s")
    print("用 scripts/analyze_paired.py 读, 决策看 free_delta 列")
