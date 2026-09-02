# -*- coding: utf-8 -*-
"""迭代 8：booster 成分按序号衰减。

高序号 SVD 成分方差小、信噪比低，每成分一棵树学到的多半是噪声；全局 booster_scale 扫过
（1.0 最优），但从未按成分序号分别缩放。s_j = 1/(1+(j/τ)^p)，τ∈{48,96,∞}，p∈{1,2}。
复用 69 的加性部件；便宜 booster 但保存逐成分预测，缩放后处理。
"""
import os, sys, time, warnings
from multiprocessing import Pool
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vcell.harness import INNER, build_fold, make_inner_splits, evaluate, summary_row
from vcell.models import ResidualBooster

OUT = os.path.join(ROOT, "results")
CACHE = os.path.join(OUT, "unseen_parts_cheap")
FOLDS = [(1, "CGD"), (5, "CGD"), (6, "CGD"), (3, "BAH"), (7, "BAH"), (8, "BAH")]
CHEAP = {"n_comp": 96, "n_estimators": 800, "learning_rate": 0.03}
SCHED = [("none", None, None), ("t96p2", 96, 2), ("t64p2", 64, 2), ("t48p2", 48, 2),
         ("t96p1", 96, 1), ("t48p1", 48, 1), ("lin", "lin", None)]


class CompBooster(ResidualBooster):
    def predict_components(self):
        X = self._X
        Z = np.zeros((len(X), len(self.models)), np.float32)
        for models in self.model_sets:
            for j, g in enumerate(models):
                Z[:, j] += g.predict(X)
        return Z / len(self.model_sets)


def one_fold(arg):
    seed, held = arg
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "2")
    t0 = time.time()
    base_meta = build_fold().meta
    fo = build_fold(splits=make_inner_splits(base_meta, hold_strain=held, seed=seed))
    P_add = np.load(os.path.join(CACHE, f"{seed}_{held}_add.npy"))
    rb = CompBooster(n_jobs=6, **CHEAP).fit(fo.meta, fo.Y_obs, fo.obs_mask, P_add)
    Z = rb.predict_components(); V = rb.V; k = Z.shape[1]
    j = np.arange(k)
    rows = []
    for name, tau, p in SCHED:
        if tau is None: s = np.ones(k)
        elif tau == "lin": s = 1.0 - 0.5 * j / max(k - 1, 1)
        else: s = 1.0 / (1.0 + (j / tau) ** p)
        P = (P_add + rb.scale * ((Z * s[None, :].astype(np.float32)) @ V)).astype(np.float32)
        r = summary_row(name, evaluate(fo, P, INNER)); r.update({"seed": seed, "strain": held, "sched": name})
        rows.append(r)
    print(f"  seed{seed} {held} done ({time.time()-t0:.0f}s)", flush=True)
    return rows


if __name__ == "__main__":
    with Pool(6) as pool:
        out = [r for rows in pool.imap_unordered(one_fold, FOLDS) for r in rows]
    d = pd.DataFrame(out); d.to_csv(os.path.join(OUT, "comp_decay.csv"), index=False)
    piv = d.pivot_table(index=["seed", "strain"], columns="sched", values="TOTAL")
    print("\n=== booster 成分衰减（六折配对，vs none）===")
    for name, _, _ in SCHED:
        dd = piv[name] - piv["none"]
        print(f"  {name:6s} delta={dd.mean():+.5f}  sem={dd.sem():.5f}  up={(dd>0).sum()}/6")
