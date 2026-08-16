"""Confirm the re-tuned plate shrinkage on the *adopted* booster, before it ships.

44/45 swept lam_plate x lam_plate_x_strain on the cheap booster and found a clean
interior optimum at (2, 6): +0.00516 +- 0.00139 over the incumbent (0.3, 2) on the
three orphan-free folds, 3/3, with every point beyond it monotonically worse out to
(32, 48) at -0.013.  Same discipline as 42: a factor that helps a weak residual model
does not automatically help the strong one, so it gets re-measured on the booster that
actually ships before BATCH_FACTORS changes.

Original 42 docstring follows.

Confirm the instrument level on the *adopted* booster, before it goes into the model.

39/41 measured the instrument level on the cheap booster (96/800, one seed),
because the question there was about factor structure and the cheap booster keeps
a round at 1/10 the cost.  Everything that actually ships runs the adopted booster
(160 comps, 1600 trees, lr .015, seeds x3), and a factor that helps a weak
residual model does not automatically help a stronger one -- the booster may
already be recovering part of what the instrument level supplies.

So: same six inner mirrors, same paired test, adopted booster, with and without.
Only if it survives here does it go into BATCH_FACTORS.

Also folds in the one other thing 38 found worth its cost (240 components,
+0.00023 +- 0.00005) so the final config is decided in one pass.

    VCELL_INSTR_LAM=3 VCELL_WORKERS=12 python scripts/42_confirm_instrument.py

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
LAM = float(os.environ.get("VCELL_INSTR_LAM", 3.0))
AFTER_PLATE = os.environ.get("VCELL_INSTR_AFTER_PLATE", "0") == "1"

FOLDS = [(0, "CEK"), (1, "CGD"), (2, "DHY210"), (3, "BAH"), (4, "CEK"), (5, "CGD")]
TUNED = {"lam_plate": 0.3, "lam_plate_x_strain": 2.0}
ADOPTED = {"n_comp": 240, "n_estimators": 1600, "learning_rate": 0.015, "seeds": [0, 1, 2]}
INSTR = ("instrument", ("instrument",), LAM)
SOURCE = ("source", ("data_source",), LAM)
# 41 号: source+instr 一起最好(+0.00707, 6/6); 顺序必须在 plate 之前(放后面只剩 +0.0013)
REF = "adopted (pert x1)"

PERT4 = {f"lam_{a}": l * 4.0 for a, _, l in PERT_FACTORS}
CONFIGS = {
    "adopted (pert x1)":        {"booster": dict(ADOPTED),
                                 "lam": {"lam_plate": 2.0, "lam_plate_x_strain": 6.0}},
    "adopted + pert x4":        {"booster": dict(ADOPTED),
                                 "lam": {"lam_plate": 2.0, "lam_plate_x_strain": 6.0, **PERT4}},
    "adopted + pert x8":        {"booster": dict(ADOPTED),
                                 "lam": {"lam_plate": 2.0, "lam_plate_x_strain": 6.0,
                                         **{k: v * 2 for k, v in PERT4.items()}}},
}


def one_job(arg):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "2"
    name, cfg, seed, strain = arg
    t0 = time.time()
    base_meta = build_fold().meta
    fo = build_fold(splits=make_inner_splits(base_meta, hold_strain=strain, seed=seed))
    base = list(BATCH_FACTORS)          # source + instrument 已在其中
    um = UnifiedBackfit(
        batch_factors=[(a, c, cfg["lam"].get(f"lam_{a}", l)) for a, c, l in base],
        pert_factors=[(a, c, cfg["lam"].get(f"lam_{a}", l)) for a, c, l in PERT_FACTORS],
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
    print(f"instrument lam={LAM}, after_plate={AFTER_PLATE}", flush=True)
    jobs = [(n, c, s, st) for (n, c), (s, st) in itertools.product(CONFIGS.items(), FOLDS)]
    print(f"{len(CONFIGS)} configs x {len(FOLDS)} folds = {len(jobs)} jobs", flush=True)
    t0 = time.time()
    done = []
    with Pool(N_WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(one_job, jobs), 1):
            done.append(r)
            if i % 3 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done ({time.time()-t0:.0f}s)", flush=True)
                pd.DataFrame(done).to_csv(os.path.join(OUT, "confirm_final_raw.csv"),
                                          index=False)
    raw = pd.DataFrame(done)
    raw.to_csv(os.path.join(OUT, "confirm_final_raw.csv"), index=False)

    g = raw.groupby("config")["TOTAL"]
    summ = pd.DataFrame({"mean": g.mean(), "sem": g.sem(), "n": g.size()})
    piv = raw.pivot_table(index=["seed", "strain"], columns="config", values="TOTAL")
    cur = piv[REF]
    summ["delta_vs_ref"] = [(piv[c] - cur).mean() for c in summ.index]
    summ["delta_sem"] = [(piv[c] - cur).sem() for c in summ.index]
    summ["n_folds_up"] = [int(((piv[c] - cur) > 0).sum()) for c in summ.index]
    summ["beats_ref"] = summ.delta_vs_ref > 2 * summ.delta_sem.replace(0, np.nan)
    summ = summ.sort_values("mean", ascending=False)
    summ.to_csv(os.path.join(OUT, "confirm_final.csv"))
    pd.set_option("display.width", 240)
    print("\n=== adopted booster, six inner folds, paired ===")
    print(summ.round(5).to_string())
    for col in ("M1_abs(20%)", "M2_rawFC(25%)", "M4_drug(20%)", "FC[strain_only]"):
        if col in raw.columns:
            print(f"\n{col}:\n" + raw.groupby("config")[col].mean().round(4).to_string())
    print(f"\ntotal {time.time()-t0:.0f}s")
