"""Hyper-parameter selection on an INNER mirror carved out of the training rows.

The organisers' val_* mirror is deliberately not used here -- it is scored once,
at the end, by scripts/09_final_eval.py.  Tuning and reporting on the same
held-out set is how a leaderboard result stops being an estimate of anything.
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcell.harness import (INNER, build_fold, evaluate, make_inner_splits,   # noqa: E402
                           summary_row)
from vcell.models import BATCH_FACTORS, PERT_FACTORS, UnifiedBackfit         # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
pd.set_option("display.width", 260)

base = build_fold()
folds = []
for seed, strain in [(0, "CEK"), (1, "CGD"), (2, "DHY210")]:
    sp = make_inner_splits(base.meta, hold_strain=strain, seed=seed)
    folds.append((f"{strain}/s{seed}", build_fold(splits=sp)))
    print(f"inner fold {strain}/s{seed}: "
          + ", ".join(f"{k}={int(v)}" for k, v in
                      sp.value_counts().reindex(
                          ["train", "in_chem_only", "in_strain_only", "in_both",
                           "in_time"]).items()))


def score(cfg: dict) -> tuple:
    tot, det = [], []
    for _, fo in folds:
        um = UnifiedBackfit(
            batch_factors=[(n, c, cfg.get(f"lam_{n}", l)) for n, c, l in BATCH_FACTORS],
            pert_factors=[(n, c, cfg.get(f"lam_{n}", l)) for n, c, l in PERT_FACTORS],
            n_pass=cfg.get("n_pass", 6), lowrank=cfg.get("lowrank", {}),
        ).fit(fo.meta, fo.Y_obs, fo.obs_mask)
        r = summary_row("x", evaluate(fo, um.predict(cfg.get("pert_scale", 1.0)), INNER))
        tot.append(r["TOTAL"]); det.append(r)
    return float(np.mean(tot)), pd.DataFrame(det).mean(numeric_only=True)


t0 = time.time()
results = []
best_cfg, best = {}, -9
trials = [
    ("default", {}),
    ("lowrank cmpd=5", {"lowrank": {"compound": 5}}),
    ("lowrank cmpd=10", {"lowrank": {"compound": 10}}),
    ("lowrank cmpd=10 + inter=5", {"lowrank": {
        "compound": 10, "cmpd_x_time": 5, "cmpd_x_temp": 5,
        "cmpd_x_medium": 5, "cmpd_x_source": 5, "cmpd_x_strain": 5}}),
    ("lowrank cmpd=15 + inter=8", {"lowrank": {
        "compound": 15, "cmpd_x_time": 8, "cmpd_x_temp": 8,
        "cmpd_x_medium": 8, "cmpd_x_source": 8, "cmpd_x_strain": 8}}),
    ("lam_plate=0.3", {"lam_plate": 0.3}),
    ("lam_plate=3", {"lam_plate": 3.0}),
    ("lam_plate_x_strain=2", {"lam_plate_x_strain": 2.0}),
    ("lam_plate_x_strain=15", {"lam_plate_x_strain": 15.0}),
    ("lam_compound=3", {"lam_compound": 3.0}),
    ("lam_compound=20", {"lam_compound": 20.0}),
    ("n_pass=10", {"n_pass": 10}),
]
for name, cfg in trials:
    tot, det = score(cfg)
    results.append({"trial": name, "TOTAL": tot, **{k: det[k] for k in
                    ["M1_abs(20%)", "M2_rawFC(25%)", "M3_ctx(20%)", "M4_drug(20%)",
                     "M5_bt(10%)", "M6_DEP(5%)"]}})
    print(f"  {name:30s} inner TOTAL={tot:.4f}   ({time.time()-t0:.0f}s)")
    if tot > best:
        best, best_cfg = tot, cfg

# refine: combine the winning low-rank setting with the winning shrinkages
print("\nrefinement around the best single change ...")
for name, extra in [("+lowrank10/5", {"lowrank": {
                        "compound": 10, "cmpd_x_time": 5, "cmpd_x_temp": 5,
                        "cmpd_x_medium": 5, "cmpd_x_source": 5, "cmpd_x_strain": 5}}),
                    ("+lam_plate_x_strain=2", {"lam_plate_x_strain": 2.0}),
                    ("+lam_compound=3", {"lam_compound": 3.0}),
                    ("+n_pass=10", {"n_pass": 10})]:
    cfg = {**best_cfg, **extra}
    if cfg == best_cfg:
        continue
    tot, det = score(cfg)
    results.append({"trial": f"best {name}", "TOTAL": tot, **{k: det[k] for k in
                    ["M1_abs(20%)", "M2_rawFC(25%)", "M3_ctx(20%)", "M4_drug(20%)",
                     "M5_bt(10%)", "M6_DEP(5%)"]}})
    print(f"  best {name:26s} inner TOTAL={tot:.4f}")
    if tot > best:
        best, best_cfg = tot, cfg

df = pd.DataFrame(results).sort_values("TOTAL", ascending=False)
df.to_csv(os.path.join(OUT, "inner_tuning.csv"), index=False)
print("\n" + df.to_string(index=False))
print("\nselected config:", json.dumps(best_cfg), f"  inner TOTAL={best:.4f}")
with open(os.path.join(OUT, "best_config.json"), "w") as fh:
    json.dump({"config": best_cfg, "inner_total": best,
               "folds": [n for n, _ in folds]}, fh, indent=1)
print(f"total {time.time()-t0:.0f}s")
