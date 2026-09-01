"""Average predictions across near-optimal configurations.

Variance reduction is the one thing that has reliably paid on this data: seed
bagging the booster is worth +0.0008 for nothing but averaging three draws of the
same estimator.  Averaging across *configurations* is the stronger form of the
same move -- it averages estimators whose biases point in different directions,
not just different noise draws of one estimator.

It is also cheap insurance against a specific risk in what was adopted tonight.
The lambda surface is flat near its optimum (44/45: plate 1 / 2 / 4 all within
0.002 of each other, and the whole grid was measured on three folds), so the exact
point chosen carries real selection noise.  Averaging over a neighbourhood of the
optimum is the standard answer to that, and it cannot be worse than the average of
its members.

Members are drawn along the two axes that actually moved this session -- plate
shrinkage and perturbation shrinkage -- kept inside the region where every point
was individually at or near the top:

    A  plate 2, pert x4    <- adopted
    B  plate 1, pert x4
    C  plate 4, pert x4
    D  plate 2, pert x2
    E  plate 2, pert x8

Screened on the cheap booster, six orphan-free folds.

    VCELL_WORKERS=8 python scripts/53_config_ensemble.py

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
N_WORKERS = int(os.environ.get("VCELL_WORKERS", 8))
LGB_THREADS = int(os.environ.get("VCELL_LGB_THREADS", 8))

FOLDS = [(1, "CGD"), (5, "CGD"), (6, "CGD"), (3, "BAH"), (7, "BAH"), (8, "BAH")]
CHEAP = {"n_comp": 96, "n_estimators": 800, "learning_rate": 0.03}


def lam_of(plate, pert_mult):
    d = {"lam_plate": plate, "lam_plate_x_strain": 6.0}
    d.update({f"lam_{a}": l * pert_mult for a, _, l in PERT_FACTORS})
    return d


MEMBERS = {
    "A": lam_of(2.0, 4.0),      # adopted
    "B": lam_of(1.0, 4.0),
    "C": lam_of(4.0, 4.0),
    "D": lam_of(2.0, 2.0),
    "E": lam_of(2.0, 8.0),
}
REF = "A alone (adopted)"
CONFIGS = {
    REF:              ["A"],
    "A+B":            ["A", "B"],
    "A+D":            ["A", "D"],
    "A+B+C":          ["A", "B", "C"],
    "A+D+E":          ["A", "D", "E"],
    "A+B+C+D+E":      ["A", "B", "C", "D", "E"],
}


def one_job(arg):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "2"
    name, members, seed, strain = arg
    t0 = time.time()
    base_meta = build_fold().meta
    fo = build_fold(splits=make_inner_splits(base_meta, hold_strain=strain, seed=seed))
    acc = None
    for key in members:
        lam = MEMBERS[key]
        um = UnifiedBackfit(
            batch_factors=[(a, c, lam.get(f"lam_{a}", l)) for a, c, l in BATCH_FACTORS],
            pert_factors=[(a, c, lam.get(f"lam_{a}", l)) for a, c, l in PERT_FACTORS],
            n_pass=6,
        ).fit(fo.meta, fo.Y_obs, fo.obs_mask)
        P = um.predict()
        rb = ResidualBooster(n_jobs=LGB_THREADS, **CHEAP)
        rb.fit(fo.meta, fo.Y_obs, fo.obs_mask, P)
        P = P + rb.predict()
        acc = P if acc is None else acc + P
    r = summary_row(name, evaluate(fo, acc / len(members), INNER))
    r.update({"config": name, "seed": seed, "strain": strain, "n_members": len(members),
              "secs": round(time.time() - t0, 1)})
    return r


if __name__ == "__main__":
    jobs = [(n, c, s, st) for (n, c), (s, st) in itertools.product(CONFIGS.items(), FOLDS)]
    print(f"{len(CONFIGS)} configs x {len(FOLDS)} orphan-free folds = {len(jobs)} jobs",
          flush=True)
    t0 = time.time()
    done = []
    with Pool(N_WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(one_job, jobs), 1):
            done.append(r)
            if i % 4 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done ({time.time()-t0:.0f}s)", flush=True)
                pd.DataFrame(done).to_csv(os.path.join(OUT, "ensemble_raw.csv"), index=False)
    raw = pd.DataFrame(done)
    raw.to_csv(os.path.join(OUT, "ensemble_raw.csv"), index=False)
    piv = raw.pivot_table(index=["seed", "strain"], columns="config", values="TOTAL")
    cur = piv[REF]
    summ = pd.DataFrame({"mean": piv.mean(), "sem": piv.sem()})
    summ["delta"] = [(piv[c] - cur).mean() for c in summ.index]
    summ["delta_sem"] = [(piv[c] - cur).sem() for c in summ.index]
    summ["folds_up"] = [int(((piv[c] - cur) > 0).sum()) for c in summ.index]
    summ["beats"] = summ.delta > 2 * summ.delta_sem.replace(0, np.nan)
    pd.set_option("display.width", 240)
    print("\n=== six orphan-free folds, paired against the single adopted config ===")
    print(summ.sort_values("delta", ascending=False).round(5).to_string())
    summ.to_csv(os.path.join(OUT, "ensemble.csv"))
    print(f"\ntotal {time.time()-t0:.0f}s")
