"""Booster knobs that were never swept, on the same six inner mirrors as 26b.

26b established that the additive model's lambdas are exhausted (everything within
+-0.003) and that the only configuration changes that cleared their own paired
standard error were two booster settings:

    booster lr .015 / 1600 trees   +0.0005 +- 0.0002
    booster 160 components         +0.0003 +- 0.0001

Neither was ever combined, and three knobs were never touched at all:

  * ``scale``      -- the booster is trained on the additive model's *in-sample*
                      residual, which is smaller than the residual it will meet on
                      a held-out compound.  If that mismatch matters the optimum
                      is scale < 1.  pert_scale peaked at exactly 1.0, but that is
                      a different estimator; this one has never been checked.
  * ``num_leaves``  -- 127 looked better than 31 on a single fold (+0.0006), which
                      is inside that fold's noise.
  * seed bagging    -- LightGBM here is stochastic (subsample .8, colsample .9,
                      random_state = seed + component).  Averaging several seeds
                      removes that variance for free; never tried.

Everything is judged by the paired fold-by-fold delta against REF, exactly as in
26b -- the fold-to-fold spread (sd ~0.023) is ten times any effect we are looking
for, so unpaired means are useless here.

    VCELL_WORKERS=16 VCELL_LGB_THREADS=8 python scripts/33_booster_knobs.py

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
N_WORKERS = int(os.environ.get("VCELL_WORKERS", 16))
LGB_THREADS = int(os.environ.get("VCELL_LGB_THREADS", 8))

# identical to 26b so the two runs are comparable
FOLDS = [(0, "CEK"), (1, "CGD"), (2, "DHY210"), (3, "BAH"), (4, "CEK"), (5, "CGD")]
TUNED = {"lam_plate": 0.3, "lam_plate_x_strain": 2.0}
BASE_BOOSTER = {"n_comp": 96, "n_estimators": 800, "learning_rate": 0.03}
REF = "REF (26b pick)"

# the two knobs 26b found, combined
WINNERS = {"n_comp": 160, "n_estimators": 1600, "learning_rate": 0.015}

CONFIGS = {
    REF:                        dict(TUNED),
    "scale 0.85":               {**TUNED, "booster": {"scale": 0.85}},
    "scale 0.925":              {**TUNED, "booster": {"scale": 0.925}},
    "scale 1.075":              {**TUNED, "booster": {"scale": 1.075}},
    "scale 1.15":               {**TUNED, "booster": {"scale": 1.15}},
    "leaves 127":               {**TUNED, "booster": {"num_leaves": 127}},
    "winners combined":         {**TUNED, "booster": dict(WINNERS)},
    "winners + leaves 127":     {**TUNED, "booster": {**WINNERS, "num_leaves": 127}},
    "seedbag x3":               {**TUNED, "seeds": [0, 1, 2]},
    "seedbag x3 + winners":     {**TUNED, "seeds": [0, 1, 2], "booster": dict(WINNERS)},
}


def fit_booster(cfg, fo, P):
    """Booster prediction, averaged over ``seeds`` if more than one is given."""
    bcfg = {**BASE_BOOSTER, **cfg.get("booster", {})}
    seeds = cfg.get("seeds", [bcfg.pop("seed", 0)])
    bcfg.pop("seed", None)
    acc = None
    for s in seeds:
        rb = ResidualBooster(n_jobs=LGB_THREADS, seed=s, **bcfg)
        rb.fit(fo.meta, fo.Y_obs, fo.obs_mask, P)
        q = rb.predict()
        acc = q if acc is None else acc + q
    return acc / len(seeds)


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
    if not cfg.get("no_booster"):
        P = P + fit_booster(cfg, fo, P)
    r = summary_row(name, evaluate(fo, P, INNER))
    r.update({"config": name, "seed": seed, "strain": strain,
              "secs": round(time.time() - t0, 1)})
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
                pd.DataFrame(done).to_csv(os.path.join(OUT, "booster_knobs_raw.csv"),
                                          index=False)
    raw = pd.DataFrame(done)
    raw.to_csv(os.path.join(OUT, "booster_knobs_raw.csv"), index=False)

    g = raw.groupby("config")["TOTAL"]
    summ = pd.DataFrame({"mean": g.mean(), "sem": g.sem(), "n": g.size()})
    piv = raw.pivot_table(index=["seed", "strain"], columns="config", values="TOTAL")
    cur = piv[REF]
    summ["delta_vs_ref"] = [(piv[c] - cur).mean() for c in summ.index]
    summ["delta_sem"] = [(piv[c] - cur).sem() for c in summ.index]
    summ["beats_ref"] = summ.delta_vs_ref > 2 * summ.delta_sem.replace(0, np.nan)
    summ = summ.sort_values("mean", ascending=False)
    summ.to_csv(os.path.join(OUT, "booster_knobs.csv"))
    pd.set_option("display.width", 240)
    print("\n=== mean over 6 inner folds, paired against REF ===")
    print(summ.round(4).to_string())
    print(f"\nfold-to-fold sd of REF = {cur.std():.4f}; unpaired differences below "
          f"~{2*cur.sem():.4f} are not real, which is why only the paired columns count.")
    print(f"total {time.time()-t0:.0f}s")
