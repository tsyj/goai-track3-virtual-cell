"""The booster's regularisation parameters -- an entire family never swept.

Everything tuned on the booster so far has been capacity: n_comp, n_estimators,
learning_rate, num_leaves.  Its four regularisation knobs still carry the values
they were first written with:

    min_child_samples 30, subsample 0.8, colsample_bytree 0.9, reg_lambda 1.0

The prior that they are mis-set is high.  This data is noise-dominated -- the true
perturbation effect is rms 0.146 log2 against measurement noise of 0.26 -- and the
two largest gains of the session both came from shrinking harder (plate lambda
0.3 -> 2, perturbation family lambda x4).  A residual model fitted on 5,920 rows
with ~10 features has every opportunity to overfit, and nothing has ever asked it
not to.

Screened on the cheap booster (96 comps / 800 trees), which is the pattern that
has worked all session: settle the structural question cheaply, then re-confirm
the winner on the booster that ships.  Six orphan-free folds throughout, so every
fold counts.

    VCELL_WORKERS=14 python scripts/52_booster_regularisation.py

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

FOLDS = [(1, "CGD"), (5, "CGD"), (6, "CGD"), (3, "BAH"), (7, "BAH"), (8, "BAH")]
CHEAP = {"n_comp": 96, "n_estimators": 800, "learning_rate": 0.03}
PLATE = {"lam_plate": 2.0, "lam_plate_x_strain": 6.0}
PERT4 = {f"lam_{a}": l * 4.0 for a, _, l in PERT_FACTORS}
LAM = {**PLATE, **PERT4}
REF = "defaults (mcs30 sub.8 col.9 rl1)"

CONFIGS = {
    REF:                    {},
    "min_child 60":         {"min_child_samples": 60},
    "min_child 120":        {"min_child_samples": 120},
    "min_child 240":        {"min_child_samples": 240},
    "reg_lambda 10":        {"reg_lambda": 10.0},
    "reg_lambda 100":       {"reg_lambda": 100.0},
    "subsample 0.5":        {"subsample": 0.5},
    "colsample 0.6":        {"colsample_bytree": 0.6},
    "mcs120 + rl10":        {"min_child_samples": 120, "reg_lambda": 10.0},
    "mcs120 + rl100 + col.6": {"min_child_samples": 120, "reg_lambda": 100.0,
                               "colsample_bytree": 0.6},
}


class TunedBooster(ResidualBooster):
    """ResidualBooster with the regularisation knobs exposed."""

    def __init__(self, reg=None, **kw):
        super().__init__(**kw)
        self.reg = reg or {}

    def fit(self, meta, Y_obs, use, base):
        import lightgbm as lgb
        R = np.where(np.isfinite(Y_obs) & use[:, None], Y_obs - base, np.nan)
        Rv = np.nan_to_num(R[use]).astype(np.float32)
        U, S, Vt = np.linalg.svd(Rv, full_matrices=False)
        k = min(self.n_comp, Vt.shape[0])
        self.V = Vt[:k]
        self.explained = float((S[:k] ** 2).sum() / (S ** 2).sum())
        Z = Rv @ self.V.T
        X = self.featurise(meta, fit=True)
        params = dict(min_child_samples=30, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.9, reg_lambda=1.0)
        params.update(self.reg)
        self.model_sets = []
        for s in self.seeds:
            models = []
            for j in range(k):
                g = lgb.LGBMRegressor(
                    n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                    num_leaves=self.num_leaves, random_state=s + j, verbose=-1,
                    n_jobs=self.n_jobs, **params)
                g.fit(X[use], Z[:, j], categorical_feature=self.CAT)
                models.append(g)
            self.model_sets.append(models)
        self.models = self.model_sets[0]
        self._X = X
        return self


def one_job(arg):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "2"
    name, reg, seed, strain = arg
    t0 = time.time()
    base_meta = build_fold().meta
    fo = build_fold(splits=make_inner_splits(base_meta, hold_strain=strain, seed=seed))
    um = UnifiedBackfit(
        batch_factors=[(a, c, LAM.get(f"lam_{a}", l)) for a, c, l in BATCH_FACTORS],
        pert_factors=[(a, c, LAM.get(f"lam_{a}", l)) for a, c, l in PERT_FACTORS],
        n_pass=6,
    ).fit(fo.meta, fo.Y_obs, fo.obs_mask)
    P = um.predict()
    rb = TunedBooster(reg=reg, n_jobs=LGB_THREADS, **CHEAP)
    rb.fit(fo.meta, fo.Y_obs, fo.obs_mask, P)
    r = summary_row(name, evaluate(fo, P + rb.predict(), INNER))
    r.update({"config": name, "seed": seed, "strain": strain,
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
            if i % 6 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done ({time.time()-t0:.0f}s)", flush=True)
                pd.DataFrame(done).to_csv(os.path.join(OUT, "booster_reg_raw.csv"), index=False)
    raw = pd.DataFrame(done)
    raw.to_csv(os.path.join(OUT, "booster_reg_raw.csv"), index=False)
    piv = raw.pivot_table(index=["seed", "strain"], columns="config", values="TOTAL")
    cur = piv[REF]
    summ = pd.DataFrame({"mean": piv.mean(), "sem": piv.sem()})
    summ["delta"] = [(piv[c] - cur).mean() for c in summ.index]
    summ["delta_sem"] = [(piv[c] - cur).sem() for c in summ.index]
    summ["folds_up"] = [int(((piv[c] - cur) > 0).sum()) for c in summ.index]
    summ["beats"] = summ.delta > 2 * summ.delta_sem.replace(0, np.nan)
    pd.set_option("display.width", 240)
    print("\n=== six orphan-free folds, paired ===")
    print(summ.sort_values("delta", ascending=False).round(5).to_string())
    summ.to_csv(os.path.join(OUT, "booster_reg.csv"))
    print(f"\ntotal {time.time()-t0:.0f}s")
