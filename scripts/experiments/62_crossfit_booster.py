"""Does the residual booster need out-of-fold residuals?  (HANDOFF section 9-4)

The booster is fitted on the residual the additive model leaves on rows it was
itself fitted on -- an in-sample residual, systematically smaller and differently
shaped than the residual it meets on a held-out strain.  Every other explanation
for that mismatch has been closed off: the ``scale`` scan (33) found nothing at
0.85 / 0.925 / 1.075 / 1.15, so the magnitude is not wrong; what is left is the
possibility that the *direction* is biased, which only cross-fitting can fix.

Cross-fitting: partition the trainable rows into K parts, refit the additive model
K times each leaving one part out, and give the booster the out-of-fold residual on
every trainable row.  Prediction still uses the full-data additive fit, so only the
booster's training target changes.  Cost is K extra additive fits per fold.

Reference is the shipped configuration (source+instrument, plate 2 / pxs 6, pert x4,
n_pass 6), cheap booster for screening, the six orphan-free folds, paired.

    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    VCELL_WORKERS=6 VCELL_LGB_THREADS=6 \
      python scripts/62_crossfit_booster.py

2026-08-27
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
N_WORKERS = int(os.environ.get("VCELL_WORKERS", 6))
LGB_THREADS = int(os.environ.get("VCELL_LGB_THREADS", 6))

FOLDS = [(1, "CGD"), (5, "CGD"), (6, "CGD"), (3, "BAH"), (7, "BAH"), (8, "BAH")]
CHEAP = {"n_comp": 96, "n_estimators": 800, "learning_rate": 0.03}
PLATE = {"lam_plate": 2.0, "lam_plate_x_strain": 6.0}
PERT4 = {f"lam_{a}": l * 4.0 for a, _, l in PERT_FACTORS}
BASE_LAM = {**PLATE, **PERT4}

REF = "in-sample (shipped)"
CONFIGS = {REF: 0, "crossfit K=3": 3, "crossfit K=5": 5}


def _additive(meta, Y_obs, use):
    return UnifiedBackfit(
        batch_factors=[(a, c, BASE_LAM.get(f"lam_{a}", l)) for a, c, l in BATCH_FACTORS],
        pert_factors=[(a, c, BASE_LAM.get(f"lam_{a}", l)) for a, c, l in PERT_FACTORS],
        n_pass=6,
    ).fit(meta, Y_obs, use).predict()


def one_job(arg):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "2")
    name, K, seed, strain = arg
    t0 = time.time()
    base_meta = build_fold().meta
    fo = build_fold(splits=make_inner_splits(base_meta, hold_strain=strain, seed=seed))
    obs = np.asarray(fo.obs_mask, dtype=bool)

    P = _additive(fo.meta, fo.Y_obs, obs)          # prediction base, always full-data
    if K == 0:
        base_for_booster = P
    else:
        idx = np.where(obs)[0]
        grp = np.full(obs.shape[0], -1, dtype=int)
        grp[idx] = np.random.RandomState(0).permutation(idx.size) % K
        base_for_booster = P.copy()
        for k in range(K):
            m = obs.copy()
            m[grp == k] = False
            Pk = _additive(fo.meta, fo.Y_obs, m)
            rows = grp == k
            base_for_booster[rows] = Pk[rows]

    rb = ResidualBooster(n_jobs=LGB_THREADS, **CHEAP)
    rb.fit(fo.meta, fo.Y_obs, obs, base_for_booster)
    r = summary_row(name, evaluate(fo, P + rb.predict(), INNER))
    resid = fo.Y_obs[obs] - base_for_booster[obs]
    r.update({"config": name, "K": K, "seed": seed, "strain": strain,
              "rms_resid_train": float(np.sqrt(np.nanmean(resid ** 2))),
              "secs": round(time.time() - t0, 1)})
    return r


if __name__ == "__main__":
    jobs = [(n, K, s, st) for (n, K), (s, st) in itertools.product(CONFIGS.items(), FOLDS)]
    print(f"{len(CONFIGS)} configs x {len(FOLDS)} orphan-free folds = {len(jobs)} jobs",
          flush=True)
    t0 = time.time()
    done = []
    raw_path = os.path.join(OUT, "62_crossfit_raw.csv")
    with Pool(N_WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(one_job, jobs), 1):
            done.append(r)
            print("  {}/{} {} seed{} {} TOTAL={:.4f} resid={:.4f} ({:.0f}s)".format(
                i, len(jobs), r["config"], r["seed"], r["strain"], r["TOTAL"],
                r["rms_resid_train"], r["secs"]), flush=True)
            pd.DataFrame(done).to_csv(raw_path, index=False)
    raw = pd.DataFrame(done)
    raw.to_csv(raw_path, index=False)

    piv = raw.pivot_table(index=["seed", "strain"], columns="config", values="TOTAL")
    cur = piv[REF]
    summ = pd.DataFrame({"mean": piv.mean(), "sem": piv.sem()})
    summ["delta"] = [(piv[c] - cur).mean() for c in summ.index]
    summ["delta_sem"] = [(piv[c] - cur).sem() for c in summ.index]
    summ["folds_up"] = [int(((piv[c] - cur) > 0).sum()) for c in summ.index]
    summ["beats"] = summ.delta > 2 * summ.delta_sem.replace(0, np.nan)
    pd.set_option("display.width", 240)
    print("\n=== six orphan-free folds, paired against the shipped in-sample booster ===")
    print(summ.sort_values("delta", ascending=False).round(5).to_string())
    print("\n=== training-residual rms actually shown to the booster ===")
    print(raw.pivot_table(index="config", values="rms_resid_train").round(4).to_string())
    summ.to_csv(os.path.join(OUT, "62_crossfit.csv"))
    print(f"\ntotal {time.time()-t0:.0f}s")
