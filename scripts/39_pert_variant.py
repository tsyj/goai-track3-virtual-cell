"""The perturbation identity is too coarse: `compound` merges distinct pert_ids.

`vcell` uses `perturbation_no_concentration` as the compound identity everywhere,
because `pert_id` is not comparable across data sources (P1-14 in OPEN_QUESTIONS:
`#2` is EDTA in WAYB but DMSO in WAYC).  That was the right call for cross-source
comparison, but it also merges perturbations that are distinct *within* one source.

Six (data_source, compound) pairs carry two pert_ids each -- EDTA in WAYB /
WAYB_rep1 / WAYB_rep2, and 1-10 Phenanthroline / Anisomycin / Brefeldin A in
WAYC.  Are the two ids really the same thing?  Their mean Delta vectors, each
averaged over ~90 samples, correlate only

    WAYB EDTA 0.85 | WAYB_rep1 EDTA 0.41 | WAYB_rep2 EDTA 0.57
    WAYC 1-10 Phen 0.49 | WAYC Anisomycin 0.36 | WAYC Brefeldin A 0.51

with rms 0.06-0.12 log2.  Noise on a 90-sample mean is only ~0.027, which by
itself would attenuate a true correlation of 1.0 to about 0.93.  So no: the two
ids are genuinely different perturbations (different concentrations, most likely),
and pooling them makes the compound term an average of two different things.

This touches **17.6% of treated rows**, including **288 rows of val_strain_only**
(the split M4 is scored on) and 26 of val_time.

`cmpd_x_source` already separates WAYB from WAYC; what is missing is pert_id
*within* a source.  So the fix is one more perturbation factor keyed on
(data_source, pert_id) -- 78 levels, each nested inside exactly one compound, so
backfitting treats it as the residual variant-specific deviation.  For an unseen
compound the level is unseen and the term is zero, as it should be.

Also tested here, at negligible extra cost: `instrument` as an intermediate batch
level.  It is perfectly nested in plate (all 144 plates have exactly one
instrument), so it adds no new grouping -- but it does give sparse plates a
better shrinkage target than zero.  Low prior, cheap to settle.

Run on the *cheap* booster (96/800, one seed): the question is about the factor
structure, and the winner gets re-confirmed on the adopted booster afterwards.
That keeps this round at ~1/10 the cost.

    VCELL_WORKERS=14 VCELL_LGB_THREADS=8 python scripts/39_pert_variant.py

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
REF = "REF (compound only)"

VAR = ("data_source", "pert_id")          # unambiguous perturbation identity
CONFIGS = {
    REF:                      dict(TUNED),
    "variant lam=4":          {**TUNED, "extra_pert": [("cmpd_variant", VAR, 4.0)]},
    "variant lam=8":          {**TUNED, "extra_pert": [("cmpd_variant", VAR, 8.0)]},
    "variant lam=16":         {**TUNED, "extra_pert": [("cmpd_variant", VAR, 16.0)]},
    "variant lam=32":         {**TUNED, "extra_pert": [("cmpd_variant", VAR, 32.0)]},
    "instrument level":       {**TUNED, "extra_batch": [("instrument", ("instrument",), 3.0)]},
    "variant lam=8 + instr":  {**TUNED, "extra_pert": [("cmpd_variant", VAR, 8.0)],
                               "extra_batch": [("instrument", ("instrument",), 3.0)]},
}


def one_job(arg):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "2"
    name, cfg, seed, strain = arg
    t0 = time.time()
    base_meta = build_fold().meta
    fo = build_fold(splits=make_inner_splits(base_meta, hold_strain=strain, seed=seed))
    # instrument sits between the global mean and the plate, so it goes first
    batch = list(cfg.get("extra_batch", [])) + list(BATCH_FACTORS)
    pert = list(PERT_FACTORS) + list(cfg.get("extra_pert", []))
    um = UnifiedBackfit(
        batch_factors=[(a, c, cfg.get(f"lam_{a}", l)) for a, c, l in batch],
        pert_factors=[(a, c, cfg.get(f"lam_{a}", l)) for a, c, l in pert],
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
    print(f"{len(CONFIGS)} configs x {len(FOLDS)} folds = {len(jobs)} jobs, "
          f"{N_WORKERS} workers x {LGB_THREADS} LightGBM threads", flush=True)
    t0 = time.time()
    done = []
    with Pool(N_WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(one_job, jobs), 1):
            done.append(r)
            if i % 5 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done ({time.time()-t0:.0f}s)", flush=True)
                pd.DataFrame(done).to_csv(os.path.join(OUT, "pert_variant_raw.csv"), index=False)
    raw = pd.DataFrame(done)
    raw.to_csv(os.path.join(OUT, "pert_variant_raw.csv"), index=False)

    pd.set_option("display.width", 240)
    for col, tag in [("TOTAL", "with cheap booster"), ("TOTAL_additive_only", "additive only")]:
        g = raw.groupby("config")[col]
        summ = pd.DataFrame({"mean": g.mean(), "sem": g.sem(), "n": g.size()})
        piv = raw.pivot_table(index=["seed", "strain"], columns="config", values=col)
        cur = piv[REF]
        summ["delta_vs_ref"] = [(piv[c] - cur).mean() for c in summ.index]
        summ["delta_sem"] = [(piv[c] - cur).sem() for c in summ.index]
        summ["beats_ref"] = summ.delta_vs_ref > 2 * summ.delta_sem.replace(0, np.nan)
        summ = summ.sort_values("mean", ascending=False)
        summ.to_csv(os.path.join(OUT, f"pert_variant_{col}.csv"))
        print(f"\n=== {tag}: mean over 6 inner folds, paired against REF ===")
        print(summ.round(5).to_string())
    for col in ("M2_rawFC(25%)", "M4_drug(20%)", "M3_ctx(20%)", "FC[strain_only]", "FC[time]"):
        if col in raw.columns:
            print(f"\n{col}:")
            print(raw.groupby("config")[col].mean().round(4).to_string())
    print(f"\ntotal {time.time()-t0:.0f}s")
