# -*- coding: utf-8 -*-
"""迭代 4：训练期修复 booster 对未见类别的塌缩——strain 特征 dropout 增强。

机制：预测期重定标（69/70/73）在部署口径失效，因为衰减幅度取决于可见菌株数。
换成训练期修复：把训练行复制一份、strain 特征置为 -1（LightGBM 的缺失类别），
让每棵树显式学出「strain 未知时」的回退路径。若有效，它不依赖口径。

    python scripts/74_strain_dropout.py
"""
import os, sys, time, warnings
from multiprocessing import Pool
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vcell.harness import INNER, build_fold, make_inner_splits, evaluate, summary_row
from vcell.models import BATCH_FACTORS, PERT_FACTORS, ResidualBooster, UnifiedBackfit

OUT = os.path.join(ROOT, "results")
CACHE = os.path.join(OUT, "unseen_parts_cheap")      # 复用 69 的加性部件
FOLDS = [(1, "CGD"), (5, "CGD"), (6, "CGD"), (3, "BAH"), (7, "BAH"), (8, "BAH")]
CHEAP = {"n_comp": 96, "n_estimators": 800, "learning_rate": 0.03}
FRACS = [0.0, 0.15, 0.30, 0.50]


class DropBooster(ResidualBooster):
    def __init__(self, drop_frac=0.0, drop_cols=("Strains",), **kw):
        super().__init__(**kw)
        self.drop_frac, self.drop_cols = float(drop_frac), list(drop_cols)

    def fit(self, meta, Y_obs, use, base):
        import lightgbm as lgb
        R = np.where(np.isfinite(Y_obs) & use[:, None], Y_obs - base, np.nan)
        Rv = np.nan_to_num(R[use]).astype(np.float32)
        U, S, Vt = np.linalg.svd(Rv, full_matrices=False)
        k = min(self.n_comp, Vt.shape[0])
        self.V = Vt[:k]
        Z = Rv @ self.V.T
        X = self.featurise(meta, fit=True)
        Xu = X[use].reset_index(drop=True)
        if self.drop_frac > 0:
            rng = np.random.default_rng(7)
            n_aug = int(len(Xu) * self.drop_frac)
            idx = rng.choice(len(Xu), n_aug, replace=False)
            Xa = Xu.iloc[idx].copy()
            for c in self.drop_cols:
                Xa[c] = -1                      # LightGBM: 负类别 = 缺失
            Xfit = pd.concat([Xu, Xa], ignore_index=True)
            Zfit = np.vstack([Z, Z[idx]])
        else:
            Xfit, Zfit = Xu, Z
        self.model_sets = []
        for s in self.seeds:
            models = []
            for j in range(k):
                g = lgb.LGBMRegressor(
                    n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                    num_leaves=self.num_leaves, min_child_samples=30, subsample=0.8,
                    subsample_freq=1, colsample_bytree=0.9, reg_lambda=1.0,
                    random_state=s + j, verbose=-1, n_jobs=self.n_jobs)
                g.fit(Xfit, Zfit[:, j], categorical_feature=self.cat)
                models.append(g)
            self.model_sets.append(models)
        self.models = self.model_sets[0]
        self._X = X
        return self


def one_fold(arg):
    seed, held = arg
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "2")
    t0 = time.time()
    base_meta = build_fold().meta
    fo = build_fold(splits=make_inner_splits(base_meta, hold_strain=held, seed=seed))
    P_add = np.load(os.path.join(CACHE, f"{seed}_{held}_add.npy"))
    rows = []
    for f in FRACS:
        rb = DropBooster(drop_frac=f, n_jobs=6, **CHEAP)
        rb.fit(fo.meta, fo.Y_obs, fo.obs_mask, P_add)
        r = summary_row(f"p={f}", evaluate(fo, (P_add + rb.predict()).astype(np.float32), INNER))
        r.update({"seed": seed, "strain": held, "p": f})
        rows.append(r)
    print(f"  seed{seed} {held} done ({time.time()-t0:.0f}s)", flush=True)
    return rows


if __name__ == "__main__":
    with Pool(6) as pool:
        out = [r for rows in pool.imap_unordered(one_fold, FOLDS) for r in rows]
    d = pd.DataFrame(out)
    d.to_csv(os.path.join(OUT, "strain_dropout.csv"), index=False)
    piv = d.pivot_table(index=["seed", "strain"], columns="p", values="TOTAL")
    print("\n=== strain 特征 dropout 增强（六折配对，vs p=0）===")
    for f in FRACS:
        dd = piv[f] - piv[0.0]
        print(f"  p={f:.2f}  delta={dd.mean():+.5f}  sem={dd.sem():.5f}  up={(dd>0).sum()}/6")
    for mod in ["M2_rawFC(25%)", "M4_drug(20%)", "M1_abs(20%)", "FC[strain_only]", "FC[chem_only]"]:
        q = d.pivot_table(index=["seed", "strain"], columns="p", values=mod)
        print(f"  {mod:16s} " + "  ".join(f"p={f}:{(q[f]-q[0.0]).mean():+.5f}" for f in FRACS))
