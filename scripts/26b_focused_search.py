"""Wide search across many inner folds, using the whole machine.

Every configuration choice so far was made on 2-3 inner folds, where the
differences between candidates (0.001-0.002) are the same size as the fold-to-fold
spread.  Those choices are therefore not established.  This re-runs them across
SIX independent inner mirrors in parallel and reports mean +- standard error, so
a "winner" has to actually clear the noise.

Options previously rejected on thin evidence are re-tested here with the same
power as the ones that were accepted.
"""
import itertools
import json
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

# six inner mirrors: each visible strain held out once, plus a second compound draw
FOLDS = [(0, "CEK"), (1, "CGD"), (2, "DHY210"), (3, "BAH"), (4, "CEK"), (5, "CGD")]

CTX = ("compound", "Medium", "Temperature", "pert_time", "data_source")
CELL = ("compound", "Strains", "Medium", "Temperature", "pert_time")

# NOTE: an empty dict falls back to the BATCH_FACTORS defaults, where lam_plate
# is 1.0 -- NOT the tuned 0.3.  The reference must state its values explicitly.
TUNED = {"lam_plate": 0.3, "lam_plate_x_strain": 2.0}
CONFIGS = {
    "REF tuned (lam_plate .3, booster 96/800)": dict(TUNED),
    "lam_plate=0.15": {**TUNED, "lam_plate": 0.15},
    "lam_plate=0.45": {**TUNED, "lam_plate": 0.45},
    "lam_plate=0.6": {**TUNED, "lam_plate": 0.6},
    "lam_plate=1.0 (untuned default)": {**TUNED, "lam_plate": 1.0},
    "lam_plate_x_strain=1": {**TUNED, "lam_plate_x_strain": 1.0},
    "lam_plate_x_strain=6 (default)": {**TUNED, "lam_plate_x_strain": 6.0},
    "booster 160 comps": {**TUNED, "booster": {"n_comp": 160}},
    "booster lr .015 / 1600 trees": {**TUNED, "booster": {"n_estimators": 1600,
                                                          "learning_rate": 0.015}},
    "additive only (no booster)": {**TUNED, "no_booster": True},
}
BASE_BOOSTER = {"n_comp": 96, "n_estimators": 800, "learning_rate": 0.03}


def one_job(arg):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "2"          # stop BLAS from oversubscribing every worker
    name, cfg, seed, strain = arg
    t0 = time.time()
    base_meta = build_fold().meta
    fo = build_fold(splits=make_inner_splits(base_meta, hold_strain=strain, seed=seed))
    pert = list(PERT_FACTORS) + list(cfg.get("extra_pert", []))
    um = UnifiedBackfit(
        batch_factors=[(a, c, cfg.get(f"lam_{a}", l)) for a, c, l in BATCH_FACTORS],
        pert_factors=[(a, c, cfg.get(f"lam_{a}", l)) for a, c, l in pert],
        n_pass=cfg.get("n_pass", 6), lowrank=cfg.get("lowrank", {}),
    ).fit(fo.meta, fo.Y_obs, fo.obs_mask)
    P = um.predict()
    if not cfg.get("no_booster"):
        bcfg = {**BASE_BOOSTER, **cfg.get("booster", {})}
        rb = ResidualBooster(n_jobs=LGB_THREADS, **bcfg)
        rb.fit(fo.meta, fo.Y_obs, fo.obs_mask, P)
        P = P + rb.predict()
    r = summary_row(name, evaluate(fo, P, INNER))
    r.update({"config": name, "seed": seed, "strain": strain,
              "secs": round(time.time() - t0, 1)})
    return r


if __name__ == "__main__":
    jobs = [(n, c, s, st) for (n, c), (s, st)
            in itertools.product(CONFIGS.items(), FOLDS)]
    print(f"{len(CONFIGS)} configs x {len(FOLDS)} folds = {len(jobs)} jobs, "
          f"{N_WORKERS} workers x {LGB_THREADS} LightGBM threads")
    t0 = time.time()
    done = []
    with Pool(N_WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(one_job, jobs), 1):
            done.append(r)
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done ({time.time()-t0:.0f}s)", flush=True)
                pd.DataFrame(done).to_csv(os.path.join(OUT, "parallel_search_raw.csv"),
                                          index=False)
    raw = pd.DataFrame(done)
    raw.to_csv(os.path.join(OUT, "parallel_search_raw.csv"), index=False)

    g = raw.groupby("config")["TOTAL"]
    summ = pd.DataFrame({"mean": g.mean(), "sem": g.sem(), "n": g.size()})
    ref = summ.loc["REF tuned (lam_plate .3, booster 96/800)", "mean"]
    # paired comparison against the current pick, fold by fold
    piv = raw.pivot_table(index=["seed", "strain"], columns="config", values="TOTAL")
    cur = piv["REF tuned (lam_plate .3, booster 96/800)"]
    summ["delta_vs_current"] = [(piv[c] - cur).mean() for c in summ.index]
    summ["delta_sem"] = [(piv[c] - cur).sem() for c in summ.index]
    summ["beats_current"] = (summ.delta_vs_current
                             > 2 * summ.delta_sem.replace(0, np.nan))
    summ = summ.sort_values("mean", ascending=False)
    summ.to_csv(os.path.join(OUT, "parallel_search.csv"))
    pd.set_option("display.width", 240)
    print("\n=== mean over 6 inner folds, paired against the current pick ===")
    print(summ.round(4).to_string())
    print(f"\nfold-to-fold spread of the current pick: sd = {cur.std():.4f} "
          f"(so anything under ~{2*cur.sem():.4f} is not a real difference)")
    print(f"total {time.time()-t0:.0f}s")
