"""Low-rank truncation of the *batch* terms -- the one structural knob never tried.

``UnifiedBackfit`` already supports rank truncation of any term, but every script
so far has only ever applied it to ``compound`` and the ``cmpd_x_*`` interactions
(08_tune_inner.py).  The plate term was never touched, which is odd, because it is
the term with by far the worst signal-to-noise:

  * plate explains 88% of total variance -- it is what the score mostly measures;
  * it carries 144 levels x 4,422 proteins of free parameters, each level
    estimated from only ~60 wells, so a large part of the fitted term is
    estimation noise;
  * batch effects are low-rank almost by construction -- that assumption is the
    basis of ComBat / RUV / surrogate-variable analysis.

Ridge shrinkage (lam_plate, tuned to 0.3) can only pull the whole term toward
zero uniformly; it cannot separate the few directions that carry real batch
structure from the many that carry noise.  Rank truncation can.

Same six inner mirrors and the same paired test as 26b / 33.

    VCELL_WORKERS=14 VCELL_LGB_THREADS=8 python scripts/35_lowrank_plate.py

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
TUNED = {"lam_plate": 0.3, "lam_plate_x_strain": 2.0}
BASE_BOOSTER = {"n_comp": 96, "n_estimators": 800, "learning_rate": 0.03}
REF = "REF (26b pick)"

CONFIGS = {
    REF:                          dict(TUNED),
    "plate rank 8":               {**TUNED, "lowrank": {"plate": 8}},
    "plate rank 16":              {**TUNED, "lowrank": {"plate": 16}},
    "plate rank 32":              {**TUNED, "lowrank": {"plate": 32}},
    "plate rank 64":              {**TUNED, "lowrank": {"plate": 64}},
    "plate_x_strain rank 8":      {**TUNED, "lowrank": {"plate_x_strain": 8}},
    "plate 32 + pxs 8":           {**TUNED, "lowrank": {"plate": 32, "plate_x_strain": 8}},
    "plate 32 + compound 10":     {**TUNED, "lowrank": {"plate": 32, "compound": 10}},
}


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
        n_pass=cfg.get("n_pass", 6), lowrank=cfg.get("lowrank", {}),
    ).fit(fo.meta, fo.Y_obs, fo.obs_mask)
    P = um.predict()
    r_add = summary_row(name, evaluate(fo, P, INNER))["TOTAL"]
    if not cfg.get("no_booster"):
        bcfg = {**BASE_BOOSTER, **cfg.get("booster", {})}
        rb = ResidualBooster(n_jobs=LGB_THREADS, **bcfg)
        rb.fit(fo.meta, fo.Y_obs, fo.obs_mask, P)
        P = P + rb.predict()
    r = summary_row(name, evaluate(fo, P, INNER))
    r.update({"config": name, "seed": seed, "strain": strain,
              "TOTAL_additive_only": r_add, "secs": round(time.time() - t0, 1)})
    return r


if __name__ == "__main__":
    jobs = [(n, c, s, st) for (n, c), (s, st) in itertools.product(CONFIGS.items(), FOLDS)]
    print(f"{len(CONFIGS)} configs x {len(FOLDS)} folds = {len(jobs)} jobs, "
          f"{N_WORKERS} workers x {LGB_THREADS} LightGBM threads", flush=True)
    t0 = time.time()
    done = []
    with Pool(N_WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(one_job, jobs), 1):
            done.append(r)
            if i % 5 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done ({time.time()-t0:.0f}s)", flush=True)
                pd.DataFrame(done).to_csv(os.path.join(OUT, "lowrank_plate_raw.csv"),
                                          index=False)
    raw = pd.DataFrame(done)
    raw.to_csv(os.path.join(OUT, "lowrank_plate_raw.csv"), index=False)

    pd.set_option("display.width", 240)
    for col, tag in [("TOTAL", "with booster"), ("TOTAL_additive_only", "additive only")]:
        g = raw.groupby("config")[col]
        summ = pd.DataFrame({"mean": g.mean(), "sem": g.sem(), "n": g.size()})
        piv = raw.pivot_table(index=["seed", "strain"], columns="config", values=col)
        cur = piv[REF]
        summ["delta_vs_ref"] = [(piv[c] - cur).mean() for c in summ.index]
        summ["delta_sem"] = [(piv[c] - cur).sem() for c in summ.index]
        summ["beats_ref"] = summ.delta_vs_ref > 2 * summ.delta_sem.replace(0, np.nan)
        summ = summ.sort_values("mean", ascending=False)
        summ.to_csv(os.path.join(OUT, f"lowrank_plate_{col}.csv"))
        print(f"\n=== {tag}: mean over 6 inner folds, paired against REF ===")
        print(summ.round(4).to_string())
    print(f"\ntotal {time.time()-t0:.0f}s")
