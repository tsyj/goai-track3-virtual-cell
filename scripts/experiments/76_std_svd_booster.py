# -*- coding: utf-8 -*-
"""迭代 6：booster 的残差 SVD 前按蛋白标准化。

原 SVD 在原始残差上做，高方差蛋白主导前几个成分；评分的 Δ 类模块是逐样本跨蛋白的相关，
每个蛋白权重相同。按蛋白 z 化后再取 240 成分，再乘回去——改变的是成分基，不是容量。
变体：none / std（除以训练残差的逐蛋白 sd）/ sqrt（除以 sd 的平方根，折中）。
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
CACHE = os.path.join(OUT, "unseen_parts_cheap")      # 复用 69 的加性部件（A 配置）
FOLDS = [(1, "CGD"), (5, "CGD"), (6, "CGD"), (3, "BAH"), (7, "BAH"), (8, "BAH")]
CHEAP = {"n_comp": 96, "n_estimators": 800, "learning_rate": 0.03}
MODES = ["none", "sqrt", "std"]


class StdBooster(ResidualBooster):
    def __init__(self, mode="none", **kw):
        super().__init__(**kw); self.mode = mode

    def fit(self, meta, Y_obs, use, base):
        import lightgbm as lgb
        R = np.where(np.isfinite(Y_obs) & use[:, None], Y_obs - base, np.nan)
        Rv = np.nan_to_num(R[use]).astype(np.float32)
        sd = np.nanstd(R[use], axis=0); sd = np.where(np.isfinite(sd) & (sd > 1e-3), sd, 1.0)
        self.w = {"none": np.ones_like(sd), "sqrt": np.sqrt(sd), "std": sd}[self.mode].astype(np.float32)
        Rs = Rv / self.w
        U, S, Vt = np.linalg.svd(Rs, full_matrices=False)
        k = min(self.n_comp, Vt.shape[0]); self.V = Vt[:k]
        Z = Rs @ self.V.T
        X = self.featurise(meta, fit=True)
        self.model_sets = []
        for s in self.seeds:
            models = []
            for j in range(k):
                g = lgb.LGBMRegressor(n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                                      num_leaves=self.num_leaves, min_child_samples=30, subsample=0.8,
                                      subsample_freq=1, colsample_bytree=0.9, reg_lambda=1.0,
                                      random_state=s + j, verbose=-1, n_jobs=self.n_jobs)
                g.fit(X[use], Z[:, j], categorical_feature=self.cat); models.append(g)
            self.model_sets.append(models)
        self.models = self.model_sets[0]; self._X = X
        return self

    def predict(self):
        return super().predict() * self.w[None, :]


def one_fold(arg):
    seed, held = arg
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "2")
    t0 = time.time()
    base_meta = build_fold().meta
    fo = build_fold(splits=make_inner_splits(base_meta, hold_strain=held, seed=seed))
    P_add = np.load(os.path.join(CACHE, f"{seed}_{held}_add.npy"))
    rows = []
    for mode in MODES:
        rb = StdBooster(mode=mode, n_jobs=6, **CHEAP).fit(fo.meta, fo.Y_obs, fo.obs_mask, P_add)
        r = summary_row(mode, evaluate(fo, (P_add + rb.predict()).astype(np.float32), INNER))
        r.update({"seed": seed, "strain": held, "mode": mode}); rows.append(r)
    print(f"  seed{seed} {held} done ({time.time()-t0:.0f}s)", flush=True)
    return rows


if __name__ == "__main__":
    with Pool(6) as pool:
        out = [r for rows in pool.imap_unordered(one_fold, FOLDS) for r in rows]
    d = pd.DataFrame(out); d.to_csv(os.path.join(OUT, "std_svd_booster.csv"), index=False)
    piv = d.pivot_table(index=["seed", "strain"], columns="mode", values="TOTAL")
    print("\n=== SVD 前按蛋白标准化（六折配对，vs none）===")
    for m in MODES:
        dd = piv[m] - piv["none"]
        print(f"  {m:6s} delta={dd.mean():+.5f}  sem={dd.sem():.5f}  up={(dd>0).sum()}/6")
    for mod in ["M1_abs(20%)", "M2_rawFC(25%)", "M3_ctx(20%)", "M4_drug(20%)", "M6_DEP(5%)"]:
        q = d.pivot_table(index=["seed", "strain"], columns="mode", values=mod)
        print(f"  {mod:16s} " + "  ".join(f"{m}:{(q[m]-q['none']).mean():+.5f}" for m in MODES if m != "none"))
