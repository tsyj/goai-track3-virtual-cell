"""Condition main effects -- the same hole the instrument level exposed, one level deeper.

41 showed that what the batch model was missing is not information but *levels*:
BATCH_FACTORS jumps straight from the global mean to a 144-level plate term, and
inserting coarse parents above plate is worth +0.007 (6/6 folds).  Crucially the
gain only appears when the parent is fitted *before* plate (+0.0056 vs +0.0013
after), which is the signature of hierarchical partial pooling rather than new
information.

The same hole exists on the biological-condition side, and it bites hardest
exactly where the score is weakest.  BATCH_FACTORS carries

    strain_x_medium, strain_x_temp, strain_x_time, strain_x_source

but **no main effect for Medium, Temperature or pert_time**.  For a held-out
strain every strain-indexed term evaluates to zero -- strain, plate_x_strain and
all four interactions above -- so those rows currently get *no condition-specific
structure at all*, only plate.  A medium/temperature/time main effect is estimable
from every other strain and would apply to the held-out one unchanged.

That is consistent with what 41 measured: the coarse-level gains landed almost
entirely on the unseen-strain axis (M4 +0.0092, FC[strain_only] +0.0088, versus
+0.0006 on chem_only).

Baseline here is the new best (source + instrument), not the old REF, so the
question asked is whether condition main effects add anything *on top*.

    VCELL_WORKERS=14 python scripts/43_condition_main_effects.py

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
CHEAP = {"n_comp": 96, "n_estimators": 800, "learning_rate": 0.03}

COARSE = [("source", ("data_source",), 3.0), ("instrument", ("instrument",), 3.0)]
MED = ("medium_main", ("Medium",), 3.0)
TEMP = ("temp_main", ("Temperature",), 3.0)
TIME = ("time_main", ("pert_time",), 3.0)
MEDxT = ("medium_x_time", ("Medium", "pert_time"), 6.0)
REF = "source + instrument (new best)"

CONFIGS = {
    REF:                 {"extra": []},
    "+ medium":          {"extra": [MED]},
    "+ temperature":     {"extra": [TEMP]},
    "+ time":            {"extra": [TIME]},
    "+ all three":       {"extra": [MED, TEMP, TIME]},
    "+ all three + medxtime": {"extra": [MED, TEMP, TIME, MEDxT]},
}


def one_job(arg):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "2"
    name, cfg, seed, strain = arg
    t0 = time.time()
    base_meta = build_fold().meta
    fo = build_fold(splits=make_inner_splits(base_meta, hold_strain=strain, seed=seed))
    # coarse parents first, then the condition main effects, then the original ladder
    batch = COARSE + list(cfg["extra"]) + list(BATCH_FACTORS)
    um = UnifiedBackfit(
        batch_factors=[(a, c, TUNED.get(f"lam_{a}", l)) for a, c, l in batch],
        pert_factors=[(a, c, TUNED.get(f"lam_{a}", l)) for a, c, l in PERT_FACTORS],
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
            if i % 6 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done ({time.time()-t0:.0f}s)", flush=True)
                pd.DataFrame(done).to_csv(os.path.join(OUT, "condition_main_raw.csv"), index=False)
    raw = pd.DataFrame(done)
    raw.to_csv(os.path.join(OUT, "condition_main_raw.csv"), index=False)

    pd.set_option("display.width", 240)
    for col, tag in [("TOTAL", "with cheap booster"), ("TOTAL_additive_only", "additive only")]:
        g = raw.groupby("config")[col]
        summ = pd.DataFrame({"mean": g.mean(), "sem": g.sem(), "n": g.size()})
        piv = raw.pivot_table(index=["seed", "strain"], columns="config", values=col)
        cur = piv[REF]
        summ["delta_vs_ref"] = [(piv[c] - cur).mean() for c in summ.index]
        summ["delta_sem"] = [(piv[c] - cur).sem() for c in summ.index]
        summ["n_folds_up"] = [int(((piv[c] - cur) > 0).sum()) for c in summ.index]
        summ["beats_ref"] = summ.delta_vs_ref > 2 * summ.delta_sem.replace(0, np.nan)
        print(f"\n=== {tag}: paired against '{REF}' ===")
        print(summ.sort_values("mean", ascending=False).round(5).to_string())
        summ.to_csv(os.path.join(OUT, f"condition_main_{col}.csv"))
    for col in ("M1_abs(20%)", "M4_drug(20%)", "FC[strain_only]", "FC[chem_only]"):
        if col in raw.columns:
            print(f"\n{col}:\n" + raw.groupby("config")[col].mean().round(4).to_string())
    print(f"\ntotal {time.time()-t0:.0f}s")
