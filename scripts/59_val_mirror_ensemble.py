"""Score the shipped ensemble ONCE on the organisers' val mirror (held-out strain BAI).

The inner six folds hold out CGD / BAH; the official val split holds out BAI and a
different compound set.  This is the report number for the write-up (single config
was 0.4992 here) and, incidentally, a third held-out strain for the strain-early
family -- it is *not* a tuning set: nothing is selected on it.

Fits every member of the shipped ensemble on the official mirror (train labels
only), saves results/pool_val/<member>.npy, then scores singles, the previous
5-member ensemble and the shipped 10-member ensemble with the same summary_row
as 14_final_eval.py.

    VCELL_WORKERS=10 VCELL_LGB_THREADS=12 python scripts/59_val_mirror_ensemble.py

Jiao Xinyuan 2026-08-16 (evening session)
"""
import importlib.util
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
from vcell.harness import VAL, build_fold, evaluate, summary_row       # noqa: E402
from vcell.models import PERT_FACTORS, ResidualBooster, UnifiedBackfit  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "pool55", os.path.join(ROOT, "scripts", "55_member_pool.py"))
pool55 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pool55)

OUT = os.path.join(ROOT, "results")
POOL_DIR = os.path.join(OUT, "pool_val")
N_WORKERS = int(os.environ.get("VCELL_WORKERS", 10))
LGB_THREADS = int(os.environ.get("VCELL_LGB_THREADS", 12))
REAL = {"n_comp": 240, "n_estimators": 1600, "learning_rate": 0.015, "seeds": [0, 1, 2]}

TEN = ["A", "B", "C", "D", "E", "F_strain_early", "FB_early_plate1",
       "FC_early_plate4", "FD_early_pert2", "FE_early_pert8"]
# 21:08: pert x16 added on both orders -> 2 x 6 grid, 12 members (sha b1b5493a)
SHIPPED = TEN + ["E16_pert16", "FE16_early_pert16"]
ENSEMBLES = {
    "A alone (v4 single, online until 20:16)": ["A"],
    "A-E 5-member (53/54 design)": ["A", "B", "C", "D", "E"],
    "F family 5-member": TEN[5:],
    "10-member (online 20:16-21:08)": TEN,
    "SHIPPED 12-member (2 orders x 6 lambda points)": SHIPPED,
}


def one_job(member):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "2")
    path = os.path.join(POOL_DIR, f"{member}.npy")
    t0 = time.time()
    fo = build_fold()
    if not os.path.exists(path):
        cfg = pool55.MEMBERS[member]
        lam = {**pool55.BASE_LAM, **cfg.get("lam", {})}
        batch = [pool55.BY_NAME[a] for a in cfg.get("order", pool55.CUR_ORDER)]
        um = UnifiedBackfit(
            batch_factors=[(a, c, lam.get(f"lam_{a}", l)) for a, c, l in batch],
            pert_factors=[(a, c, lam.get(f"lam_{a}", l)) for a, c, l in PERT_FACTORS],
            n_pass=cfg.get("n_pass", 6), fit_offset=cfg.get("fit_offset", True),
        ).fit(fo.meta, fo.Y_obs, fo.obs_mask)
        P = um.predict()
        rb = ResidualBooster(n_jobs=LGB_THREADS, **cfg.get("booster", REAL))
        rb.fit(fo.meta, fo.Y_obs, fo.obs_mask, P)
        P = (P + rb.predict()).astype(np.float32)
        np.save(path + ".tmp.npy", P)
        os.replace(path + ".tmp.npy", path)
    P = np.load(path)
    r = summary_row(member, evaluate(fo, P, VAL))
    r.update({"config": member, "secs": round(time.time() - t0)})
    return r


if __name__ == "__main__":
    os.makedirs(POOL_DIR, exist_ok=True)
    t0 = time.time()
    with Pool(N_WORKERS) as pool:
        rows = []
        for r in pool.imap_unordered(one_job, SHIPPED):
            rows.append(r)
            print(f"  {r['config']:18s} TOTAL={r['TOTAL']:.4f}  ({r['secs']}s, "
                  f"{time.time()-t0:.0f}s elapsed)", flush=True)
    singles = pd.DataFrame(rows).set_index("config").loc[SHIPPED]

    fo = build_fold()
    ens_rows = []
    for name, members in ENSEMBLES.items():
        acc = None
        for m in members:
            p = np.load(os.path.join(POOL_DIR, f"{m}.npy")).astype(np.float64)
            acc = p if acc is None else acc + p
        r = summary_row(name, evaluate(fo, (acc / len(members)).astype(np.float32), VAL))
        r["n_members"] = len(members)
        ens_rows.append(r)
    ens = pd.DataFrame(ens_rows).set_index("model")

    cols = ["TOTAL", "FC[chem_only]", "FC[strain_only]", "FC[both]", "FC[time]",
            "M1_abs(20%)", "M2_rawFC(25%)", "M3_ctx(20%)", "M4_drug(20%)", "M5_bt(10%)",
            "M6_DEP(5%)"]
    pd.set_option("display.width", 260)
    print("\n=== official val mirror (held-out strain BAI), single members ===")
    print(singles[cols].round(4).to_string())
    print("\n=== official val mirror, ensembles ===")
    print(ens[cols + ["n_members"]].round(4).to_string())
    singles.to_csv(os.path.join(OUT, "val_mirror_members.csv"))
    ens.to_csv(os.path.join(OUT, "val_mirror_ensembles.csv"))
    print(f"\ntotal {time.time()-t0:.0f}s")
