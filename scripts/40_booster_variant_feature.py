"""Give the booster the finer perturbation identity too.

39_pert_variant.py asks whether the *additive* model needs a (data_source, pert_id)
term.  This asks the same question of the booster, which is a separate matter: its
feature list has `compound` but nothing that distinguishes the two pert_ids inside
one source, and the 21_residual_sweep ablation showed `compound` is the single
biggest driver of what the booster contributes (dropping it costs -0.0119, more
than plate id and well position together).

So if the compound label is too coarse for the additive model, it is too coarse
here as well -- and the two fixes are not redundant: the additive term is a
per-protein shrunken mean, the booster feature lets the trees split on it in
combination with plate, well and context.

Implemented by subclassing rather than editing ResidualBooster, so the audited
class keeps exactly the behaviour that every earlier result was measured with.

Cheap booster (96/800, one seed) -- the question is about features.

    VCELL_WORKERS=14 VCELL_LGB_THREADS=8 python scripts/40_booster_variant_feature.py

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
VAR = ("data_source", "pert_id")
REF = "REF (compound only, both places)"


class VariantBooster(ResidualBooster):
    """ResidualBooster + a (data_source, pert_id) categorical."""

    CAT = ResidualBooster.CAT + ["pert_variant"]

    @classmethod
    def _raw_column(cls, meta, c):
        if c == "pert_variant":
            return meta["data_source"].astype(str) + "|" + meta["pert_id"].astype(str)
        return super()._raw_column(meta, c)


CONFIGS = {
    REF:                          {"booster_cls": ResidualBooster},
    "booster sees variant":       {"booster_cls": VariantBooster},
    "additive sees variant":      {"booster_cls": ResidualBooster,
                                   "extra_pert": [("cmpd_variant", VAR, 8.0)]},
    "both see variant":           {"booster_cls": VariantBooster,
                                   "extra_pert": [("cmpd_variant", VAR, 8.0)]},
}


def one_job(arg):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "2"
    name, cfg, seed, strain = arg
    t0 = time.time()
    base_meta = build_fold().meta
    fo = build_fold(splits=make_inner_splits(base_meta, hold_strain=strain, seed=seed))
    pert = list(PERT_FACTORS) + list(cfg.get("extra_pert", []))
    um = UnifiedBackfit(
        batch_factors=[(a, c, TUNED.get(f"lam_{a}", l)) for a, c, l in BATCH_FACTORS],
        pert_factors=[(a, c, TUNED.get(f"lam_{a}", l)) for a, c, l in pert],
        n_pass=6,
    ).fit(fo.meta, fo.Y_obs, fo.obs_mask)
    P = um.predict()
    rb = cfg["booster_cls"](n_jobs=LGB_THREADS, **CHEAP)
    rb.fit(fo.meta, fo.Y_obs, fo.obs_mask, P)
    r = summary_row(name, evaluate(fo, P + rb.predict(), INNER))
    r.update({"config": name, "seed": seed, "strain": strain,
              "secs": round(time.time() - t0, 1)})
    return r


if __name__ == "__main__":
    jobs = [(n, c, s, st) for (n, c), (s, st) in itertools.product(CONFIGS.items(), FOLDS)]
    print(f"{len(CONFIGS)} configs x {len(FOLDS)} folds = {len(jobs)} jobs", flush=True)
    t0 = time.time()
    done = []
    with Pool(N_WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(one_job, jobs), 1):
            done.append(r)
            if i % 4 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done ({time.time()-t0:.0f}s)", flush=True)
                pd.DataFrame(done).to_csv(os.path.join(OUT, "variant_feature_raw.csv"), index=False)
    raw = pd.DataFrame(done)
    raw.to_csv(os.path.join(OUT, "variant_feature_raw.csv"), index=False)

    g = raw.groupby("config")["TOTAL"]
    summ = pd.DataFrame({"mean": g.mean(), "sem": g.sem(), "n": g.size()})
    piv = raw.pivot_table(index=["seed", "strain"], columns="config", values="TOTAL")
    cur = piv[REF]
    summ["delta_vs_ref"] = [(piv[c] - cur).mean() for c in summ.index]
    summ["delta_sem"] = [(piv[c] - cur).sem() for c in summ.index]
    summ["beats_ref"] = summ.delta_vs_ref > 2 * summ.delta_sem.replace(0, np.nan)
    summ = summ.sort_values("mean", ascending=False)
    summ.to_csv(os.path.join(OUT, "variant_feature.csv"))
    pd.set_option("display.width", 240)
    print("\n=== mean over 6 inner folds, paired against REF ===")
    print(summ.round(5).to_string())
    for col in ("M2_rawFC(25%)", "M4_drug(20%)", "FC[strain_only]", "FC[time]"):
        if col in raw.columns:
            print(f"\n{col}:\n" + raw.groupby("config")[col].mean().round(4).to_string())
    print(f"\ntotal {time.time()-t0:.0f}s")
