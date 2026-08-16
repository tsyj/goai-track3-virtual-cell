"""Amplify the perturbation term on rows whose strain was never labelled.

Motivation (scripts/34_error_budget.py, results/error_budget.csv)
----------------------------------------------------------------
On the unseen-strain splits the prediction error is dominated by a *constant*
per-protein term: rms 0.343 (strain_only) and 0.363 (both) log2, against only
0.057 on chem_only and 0.035 on time.  That constant is the held-out strain's own
baseline proteome, which is unidentifiable -- CRD contributes no training row at
all, so its strain term shrinks to the cross-strain centre.

Write b for that missing baseline.  The scorer subtracts the *measured* control of
the same strain from both sides, so

    dT_i = y_i - c_i = delta_i + noise_i - noise'_i        <- b cancels
    dP_i = y_hat_i - c_i = delta_hat_i - b - noise'_i      <- b does NOT cancel

b is the same vector for every sample of that strain, and it is large: rms 0.34
against a true delta of rms 0.39.  Measured dP rms is 0.513 vs dT rms 0.387 --
our predicted delta is *bigger* than the truth, and the excess is contamination.

Because b is fixed while delta_hat varies, scaling delta_hat by k > 1 is NOT a
scale transform of dP, so it does change every per-sample correlation: it raises
the signal-to-contamination ratio.  This is why the global pert_scale sweep
peaking at 1.0 (results/pert_scale_sweep.csv) does not settle the question -- that
sweep scaled the perturbation everywhere, including the seen-strain rows where
there is no b to dilute and where 1.0 is genuinely optimal.

Only rows of an unlabelled strain are touched; nothing else changes.  Note the
perturbation term is identically zero for an unseen *compound* (every PERT factor
is compound-indexed), so this can only move the strain_only split, not chem_only.

The model is fitted once per fold and then rescored for every k -- no refit.

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
N_WORKERS = int(os.environ.get("VCELL_WORKERS", 6))
LGB_THREADS = int(os.environ.get("VCELL_LGB_THREADS", 8))
FOLDS = [(0, "CEK"), (1, "CGD"), (2, "DHY210"), (3, "BAH"), (4, "CEK"), (5, "CGD")]
TUNED = {"lam_plate": 0.3, "lam_plate_x_strain": 2.0}
KS = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]


def one_fold(arg):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "2"
    seed, strain = arg
    t0 = time.time()
    base_meta = build_fold().meta
    fo = build_fold(splits=make_inner_splits(base_meta, hold_strain=strain, seed=seed))
    um = UnifiedBackfit(
        batch_factors=[(a, c, TUNED.get(f"lam_{a}", l)) for a, c, l in BATCH_FACTORS],
        pert_factors=[(a, c, TUNED.get(f"lam_{a}", l)) for a, c, l in PERT_FACTORS],
        n_pass=6,
    ).fit(fo.meta, fo.Y_obs, fo.obs_mask)
    P_batch = um.predict(pert_scale=0.0)
    P_add = um.predict()
    pert = P_add - P_batch
    rb = ResidualBooster(n_comp=96, n_estimators=800, learning_rate=0.03,
                         n_jobs=LGB_THREADS)
    rb.fit(fo.meta, fo.Y_obs, fo.obs_mask, P_add)
    boost = rb.predict()

    # strains with no labelled row in this fold -- exactly what CRD looks like at test
    lab = fo.meta.loc[fo.obs_mask, "Strains"].astype(str).unique()
    unseen = ~fo.meta["Strains"].astype(str).isin(lab).to_numpy()

    out = []
    for k in KS:
        s = np.where(unseen, k, 1.0).astype(np.float32)[:, None]
        P = P_batch + s * pert + boost
        r = summary_row(f"k={k}", evaluate(fo, P, INNER))
        r.update({"k": k, "seed": seed, "strain": strain,
                  "n_unseen_rows": int(unseen.sum()), "secs": round(time.time() - t0, 1)})
        out.append(r)
    # same amplification applied to the booster's contribution as well
    for k in KS[1:]:
        s = np.where(unseen, k, 1.0).astype(np.float32)[:, None]
        P = P_batch + s * (pert + boost)
        r = summary_row(f"k={k} (incl booster)", evaluate(fo, P, INNER))
        r.update({"k": k, "seed": seed, "strain": strain, "variant": "incl_booster",
                  "n_unseen_rows": int(unseen.sum()), "secs": round(time.time() - t0, 1)})
        out.append(r)
    return out


if __name__ == "__main__":
    print(f"{len(FOLDS)} folds x {len(KS)} k values, {N_WORKERS} workers", flush=True)
    t0 = time.time()
    done = []
    with Pool(N_WORKERS) as pool:
        for i, rs in enumerate(pool.imap_unordered(one_fold, FOLDS), 1):
            done.extend(rs)
            print(f"  fold {i}/{len(FOLDS)} done ({time.time()-t0:.0f}s)", flush=True)
            pd.DataFrame(done).to_csv(os.path.join(OUT, "strain_amplify_raw.csv"), index=False)
    raw = pd.DataFrame(done)
    raw["variant"] = raw.get("variant", pd.Series(index=raw.index, dtype=object)).fillna("pert_only")
    raw.to_csv(os.path.join(OUT, "strain_amplify_raw.csv"), index=False)

    pd.set_option("display.width", 250)
    for variant, sub in raw.groupby("variant"):
        piv = sub.pivot_table(index=["seed", "strain"], columns="k", values="TOTAL")
        ref = piv[1.0] if 1.0 in piv.columns else raw[raw.k == 1.0].pivot_table(
            index=["seed", "strain"], columns="k", values="TOTAL")[1.0]
        summ = pd.DataFrame({
            "mean": piv.mean(), "sem": piv.sem(),
            "delta_vs_k1": [(piv[c] - ref).mean() for c in piv.columns],
            "delta_sem": [(piv[c] - ref).sem() for c in piv.columns],
        })
        summ["beats_k1"] = summ.delta_vs_k1 > 2 * summ.delta_sem.replace(0, np.nan)
        print(f"\n=== {variant}: TOTAL, paired against k=1 ===")
        print(summ.round(4).to_string())
        for col in ("M2_rawFC(25%)", "M4_drug(20%)", "M1_abs(20%)", "FC[strain_only]"):
            if col in sub.columns:
                print(f"  {col:<18}" + "  ".join(
                    f"k={k}:{sub[sub.k == k][col].mean():.4f}" for k in sorted(sub.k.unique())))
    raw.to_csv(os.path.join(OUT, "strain_amplify_raw.csv"), index=False)
    print(f"\ntotal {time.time()-t0:.0f}s")
