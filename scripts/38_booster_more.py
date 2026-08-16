"""Push the two axes that actually worked, one notch further.

33_booster_knobs.py established (six inner mirrors, paired):

    seedbag x3 + winners   +0.0014 +- 0.0002   <- adopted
    winners combined       +0.0010 +- 0.0001
    seedbag x3             +0.0008 +- 0.0001
    everything else        null

Both winners are variance-reduction mechanisms, and neither is obviously at its
limit:

  * **more seeds.**  Bagging removes a fraction (1 - 1/n) of the fit-to-fit
    variance, so three seeds captures 67% of what is available.  If +0.0008 is
    67% of the total, five seeds should give +0.0010 and ten +0.0011 -- an extra
    +0.0003 or +0.0004, which is above the paired sem of 0.0002 and therefore
    measurable.  If the observed curve does *not* follow (1 - 1/n), the gain was
    not variance reduction and the story needs revisiting.
  * **finer steps / more components.**  The 'winners' gain came from halving the
    learning rate while doubling the trees, and from 96 -> 160 components.
    Neither was pushed again.

Cost is why this is a second round: seeds=10 with the winners config is ~33x the
original booster.  That is affordable once, on six folds, on an idle machine.

    VCELL_WORKERS=12 VCELL_LGB_THREADS=8 python scripts/38_booster_more.py

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
N_WORKERS = int(os.environ.get("VCELL_WORKERS", 12))
LGB_THREADS = int(os.environ.get("VCELL_LGB_THREADS", 8))

FOLDS = [(0, "CEK"), (1, "CGD"), (2, "DHY210"), (3, "BAH"), (4, "CEK"), (5, "CGD")]
TUNED = {"lam_plate": 0.3, "lam_plate_x_strain": 2.0}
WIN = {"n_comp": 160, "n_estimators": 1600, "learning_rate": 0.015}
REF = "ADOPTED (winners + seeds x3)"

CONFIGS = {
    REF:                    {**TUNED, "booster": {**WIN, "seeds": [0, 1, 2]}},
    "seeds x6":             {**TUNED, "booster": {**WIN, "seeds": [0, 1, 2, 3, 4, 5]}},
    "seeds x10":            {**TUNED, "booster": {**WIN, "seeds": list(range(10))}},
    "240 comp, seeds x3":   {**TUNED, "booster": {**WIN, "n_comp": 240, "seeds": [0, 1, 2]}},
    "lr .0075 / 3200, x3":  {**TUNED, "booster": {**WIN, "n_estimators": 3200,
                                                  "learning_rate": 0.0075,
                                                  "seeds": [0, 1, 2]}},
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
        n_pass=6,
    ).fit(fo.meta, fo.Y_obs, fo.obs_mask)
    P = um.predict()
    rb = ResidualBooster(n_jobs=LGB_THREADS, **cfg["booster"])
    rb.fit(fo.meta, fo.Y_obs, fo.obs_mask, P)
    r = summary_row(name, evaluate(fo, P + rb.predict(), INNER))
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
            if i % 3 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done ({time.time()-t0:.0f}s)", flush=True)
                pd.DataFrame(done).to_csv(os.path.join(OUT, "booster_more_raw.csv"),
                                          index=False)
    raw = pd.DataFrame(done)
    raw.to_csv(os.path.join(OUT, "booster_more_raw.csv"), index=False)

    g = raw.groupby("config")["TOTAL"]
    summ = pd.DataFrame({"mean": g.mean(), "sem": g.sem(), "n": g.size()})
    piv = raw.pivot_table(index=["seed", "strain"], columns="config", values="TOTAL")
    cur = piv[REF]
    summ["delta_vs_adopted"] = [(piv[c] - cur).mean() for c in summ.index]
    summ["delta_sem"] = [(piv[c] - cur).sem() for c in summ.index]
    summ["beats_adopted"] = summ.delta_vs_adopted > 2 * summ.delta_sem.replace(0, np.nan)
    summ = summ.sort_values("mean", ascending=False)
    summ.to_csv(os.path.join(OUT, "booster_more.csv"))
    pd.set_option("display.width", 240)
    print("\n=== mean over 6 inner folds, paired against the adopted config ===")
    print(summ.round(5).to_string())
    print("\nseed-count check: if bagging is variance reduction, the gain over a single "
          "seed should track (1 - 1/n).  x3 = 0.667, x6 = 0.833, x10 = 0.900 of the limit.")
    print(f"total {time.time()-t0:.0f}s")
