"""Nail down the instrument level -- the biggest single gain found so far.

39_pert_variant.py tested `instrument` as an afterthought, with a low prior: it is
perfectly nested in plate (all 144 plates have exactly one instrument), so it
carries no grouping the plate term cannot already express.  It came back at
**+0.0056 +- 0.0022, 6/6 folds** -- four times the whole booster-tuning gain -- and
the improvement sits almost entirely on the unseen-strain axis:

    M4 +0.0092 | FC[strain_only] +0.0088 | M1 +0.0083 | M2 +0.0049
    FC[chem_only] +0.0006 | FC[time] +0.0004

The prior was wrong because it only weighed *information*, not *estimation
variance*.  Nesting does not make a level useless: global -> instrument -> plate is
partial pooling, and the coarse part is then estimated from 7-45 plates' worth of
wells instead of one plate's ~60.

Ruled out already: instrument is not a stand-in for data_source -- every
instrument spans 2-3 sources and every source uses 3-7 instruments.

Open questions this run settles:
  * how strong should the instrument shrinkage be (lam sweep);
  * does it matter that the term is fitted *before* plate (backfitting order sets
    which level absorbs the coarse mean first);
  * is `data_source` -- also missing as a batch main effect -- worth adding, on its
    own or alongside;
  * is the effect specific to instrument, or would any coarse parent do.

Cheap booster (96/800, one seed): the question is the factor structure.  The
winner gets re-confirmed on the adopted booster afterwards.

    VCELL_WORKERS=14 VCELL_LGB_THREADS=8 python scripts/41_instrument_level.py

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
REF = "REF (no instrument)"

INSTR = ("instrument", ("instrument",))
SOURCE = ("source", ("data_source",))

CONFIGS = {
    REF:                        {},
    "instr lam=1":              {"extra": [(*INSTR, 1.0)]},
    "instr lam=3":              {"extra": [(*INSTR, 3.0)]},
    "instr lam=8":              {"extra": [(*INSTR, 8.0)]},
    "instr lam=20":             {"extra": [(*INSTR, 20.0)]},
    "instr lam=3 after plate":  {"extra": [(*INSTR, 3.0)], "where": "after_plate"},
    "source lam=3":             {"extra": [(*SOURCE, 3.0)]},
    "source+instr lam=3":       {"extra": [(*SOURCE, 3.0), (*INSTR, 3.0)]},
}


def build_batch(cfg):
    extra = list(cfg.get("extra", []))
    base = list(BATCH_FACTORS)
    if not extra:
        return base
    if cfg.get("where") == "after_plate":
        return [base[0]] + extra + base[1:]
    return extra + base


def one_job(arg):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "2"
    name, cfg, seed, strain = arg
    t0 = time.time()
    base_meta = build_fold().meta
    fo = build_fold(splits=make_inner_splits(base_meta, hold_strain=strain, seed=seed))
    um = UnifiedBackfit(
        batch_factors=[(a, c, TUNED.get(f"lam_{a}", l)) for a, c, l in build_batch(cfg)],
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
                pd.DataFrame(done).to_csv(os.path.join(OUT, "instrument_level_raw.csv"), index=False)
    raw = pd.DataFrame(done)
    raw.to_csv(os.path.join(OUT, "instrument_level_raw.csv"), index=False)

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
        summ = summ.sort_values("mean", ascending=False)
        summ.to_csv(os.path.join(OUT, f"instrument_level_{col}.csv"))
        print(f"\n=== {tag}: mean over 6 inner folds, paired against REF ===")
        print(summ.round(5).to_string())
    for col in ("M1_abs(20%)", "M2_rawFC(25%)", "M4_drug(20%)", "FC[strain_only]", "FC[chem_only]"):
        if col in raw.columns:
            print(f"\n{col}:\n" + raw.groupby("config")[col].mean().round(4).to_string())
    print(f"\ntotal {time.time()-t0:.0f}s")
