# -*- coding: utf-8 -*-
"""迭代 7：大效应的非线性扩张（打 M6 与 Δ 模块的尾部），纯缓存。

收缩估计把大效应压得偏小；M6 只看 |Δ_true|>1 的方向与相关。对模型隐含的
Δ̂ = P[row] − mean(P[同上下文对照孔]) 做 h(Δ̂)=Δ̂·(1+β·min(|Δ̂|/τ,1)^2)，
只放大大效应、不动小效应。用 12 成员真配置集成的缓存预测，六折配对。
"""
import os, sys, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vcell.harness import INNER, evaluate, summary_row

OUT = os.path.join(ROOT, "results")
FOLDS = [(1, "CGD"), (5, "CGD"), (6, "CGD"), (3, "BAH"), (7, "BAH"), (8, "BAH")]
MEMBERS = ["A","B","C","D","E","E16_pert16","F_strain_early","FB_early_plate1",
           "FC_early_plate4","FD_early_pert2","FE_early_pert8","FE16_early_pert16"]
GRID = [(0.0, 1.0), (0.1, 0.5), (0.2, 0.5), (0.1, 1.0), (0.2, 1.0), (0.3, 1.0), (0.2, 1.5), (0.4, 1.5)]

rows = []
for seed, st in FOLDS:
    fo = pickle.load(open(os.path.join(OUT, "folds", f"{seed}_{st}.pkl"), "rb"))
    P = np.mean([np.load(os.path.join(OUT, "pool_real", f"{seed}_{st}__{m}.npy")) for m in MEMBERS], 0).astype(np.float32)
    meta = fo.meta
    ctrl = meta["is_control"].to_numpy()
    ctx = meta["ctx_key"].astype(str).to_numpy()
    # 每个上下文的模型隐含对照 = 该上下文对照孔预测的均值
    C = np.zeros_like(P); has = np.zeros(len(P), bool)
    df = pd.DataFrame({"ctx": ctx, "i": np.arange(len(P))})
    for c, g in df[ctrl].groupby("ctx"):
        m = P[g.i.to_numpy()].mean(0)
        idx = df.index[df.ctx == c].to_numpy()
        C[idx] = m; has[idx] = True
    D = np.where(has[:, None], P - C, 0.0)
    for beta, tau in GRID:
        g = 1.0 + beta * np.minimum(np.abs(D) / tau, 1.0) ** 2
        Pn = np.where(has[:, None] & ~ctrl[:, None], C + D * g, P).astype(np.float32)
        r = summary_row(f"b{beta}_t{tau}", evaluate(fo, Pn, INNER))
        r.update({"seed": seed, "strain": st, "beta": beta, "tau": tau}); rows.append(r)
    print(f"  seed{seed} {st} done (ctx covered {has.mean():.2f})", flush=True)
d = pd.DataFrame(rows); d.to_csv(os.path.join(OUT, "delta_expand.csv"), index=False)
piv = d.pivot_table(index=["seed", "strain"], columns=["beta", "tau"], values="TOTAL")
base = piv[(0.0, 1.0)]
print("\n=== 大效应扩张 h(Δ̂)（六折配对，vs 不扩张）===")
for beta, tau in GRID:
    dd = piv[(beta, tau)] - base
    print(f"  β={beta:.1f} τ={tau:.1f}  delta={dd.mean():+.5f}  sem={dd.sem():.5f}  up={(dd>0).sum()}/6")
for mod in ["M1_abs(20%)", "M2_rawFC(25%)", "M4_drug(20%)", "M6_DEP(5%)"]:
    q = d.pivot_table(index=["seed", "strain"], columns=["beta", "tau"], values=mod)
    print(f"  {mod:14s} " + "  ".join(f"β{b}τ{t}:{(q[(b,t)]-q[(0.0,1.0)]).mean():+.4f}" for b, t in GRID[1:]))
