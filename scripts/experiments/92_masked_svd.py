# -*- coding: utf-8 -*-
"""迭代 21：booster 的残差基改为掩码感知（迭代低秩填补代替 NaN→0）。
缺失（未检出）在残差矩阵里被当成 0，会把成分基往"缺失模式"上拉；用 5 轮秩-k 重构填补后再取基。
复用 69 的加性部件（A 配置），便宜 booster，六折配对。"""
import os, sys, time, warnings
from multiprocessing import Pool
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vcell.harness import INNER, build_fold, make_inner_splits, evaluate, summary_row
from vcell.models import ResidualBooster
OUT = os.path.join(ROOT, "results"); CACHE = os.path.join(OUT, "unseen_parts_cheap")
FOLDS = [(1,"CGD"),(5,"CGD"),(6,"CGD"),(3,"BAH"),(7,"BAH"),(8,"BAH")]
CHEAP = {"n_comp": 96, "n_estimators": 800, "learning_rate": 0.03}

class MaskedBooster(ResidualBooster):
    def __init__(self, iters=0, **kw):
        super().__init__(**kw); self.iters = iters
    def fit(self, meta, Y_obs, use, base):
        import lightgbm as lgb
        R = np.where(np.isfinite(Y_obs) & use[:, None], Y_obs - base, np.nan)
        Ru = R[use]; miss = ~np.isfinite(Ru); Rv = np.nan_to_num(Ru).astype(np.float32)
        k = self.n_comp
        for _ in range(self.iters):                       # 迭代低秩填补
            U, S, Vt = np.linalg.svd(Rv, full_matrices=False)
            rec = (U[:, :k] * S[:k]) @ Vt[:k]
            Rv = np.where(miss, rec, Rv).astype(np.float32)
        U, S, Vt = np.linalg.svd(Rv, full_matrices=False)
        self.V = Vt[:k]; Z = Rv @ self.V.T
        X = self.featurise(meta, fit=True); self.model_sets = []
        for s in self.seeds:
            models = []
            for j in range(k):
                g = lgb.LGBMRegressor(n_estimators=self.n_estimators, learning_rate=self.learning_rate, num_leaves=self.num_leaves,
                                      min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.9, reg_lambda=1.0,
                                      random_state=s + j, verbose=-1, n_jobs=self.n_jobs)
                g.fit(X[use], Z[:, j], categorical_feature=self.cat); models.append(g)
            self.model_sets.append(models)
        self.models = self.model_sets[0]; self._X = X
        return self

def one_fold(arg):
    seed, held = arg
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(v, "4")
    t0 = time.time(); base_meta = build_fold().meta
    fo = build_fold(splits=make_inner_splits(base_meta, hold_strain=held, seed=seed))
    P_add = np.load(os.path.join(CACHE, f"{seed}_{held}_add.npy")); rows = []
    miss_frac = float((~np.isfinite(fo.Y_obs[fo.obs_mask])).mean())
    for iters in [0, 3, 6]:
        rb = MaskedBooster(iters=iters, n_jobs=6, **CHEAP).fit(fo.meta, fo.Y_obs, fo.obs_mask, P_add)
        r = summary_row(f"it{iters}", evaluate(fo, (P_add + rb.predict()).astype(np.float32), INNER)); r.update({"seed": seed, "strain": held, "iters": iters}); rows.append(r)
    print(f"  seed{seed} {held} done ({time.time()-t0:.0f}s, 训练残差缺失率 {miss_frac:.3f})", flush=True); return rows

if __name__ == "__main__":
    with Pool(6) as pool: out = [r for rows in pool.imap_unordered(one_fold, FOLDS) for r in rows]
    d = pd.DataFrame(out); d.to_csv(os.path.join(OUT, "masked_svd.csv"), index=False)
    piv = d.pivot_table(index=["seed","strain"], columns="iters", values="TOTAL")
    print("\n=== 掩码感知残差基（六折配对，vs 0 填充）===")
    for it in [3, 6]:
        dd = piv[it] - piv[0]; print(f"  iters={it} delta={dd.mean():+.5f} sem={dd.sem():.5f} up={(dd>0).sum()}/6")
    for mod in ["M1_abs(20%)","M2_rawFC(25%)","M3_ctx(20%)","M4_drug(20%)"]:
        q = d.pivot_table(index=["seed","strain"], columns="iters", values=mod); print(f"  {mod:16s} " + "  ".join(f"it{it}:{(q[it]-q[0]).mean():+.5f}" for it in [3,6]))
