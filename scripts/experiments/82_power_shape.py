# -*- coding: utf-8 -*-
"""迭代 10：与 77 对照的另一族非线性——全局幂律 g(D)=sign(D)·|D|^γ·c（c 使中位 |D| 不变）。
77 只动尾部；这里全体都动。也测 77 的最优点与幂律的叠加。12 成员真配置缓存，六折配对。"""
import os, sys, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vcell.harness import INNER, evaluate, summary_row
OUT = os.path.join(ROOT, "results")
FOLDS = [(1,"CGD"),(5,"CGD"),(6,"CGD"),(3,"BAH"),(7,"BAH"),(8,"BAH")]
MEMBERS = ["A","B","C","D","E","E16_pert16","F_strain_early","FB_early_plate1","FC_early_plate4","FD_early_pert2","FE_early_pert8","FE16_early_pert16"]
VARS = [("base",1.0,0.0),("g1.05",1.05,0.0),("g1.10",1.10,0.0),("g1.15",1.15,0.0),("g1.20",1.20,0.0),("h0.3",1.0,0.3),("g1.05+h0.3",1.05,0.3),("g1.10+h0.3",1.10,0.3)]

def ctx_control(P, meta):
    ctrl = meta["is_control"].to_numpy(); ctx = meta["ctx_key"].astype(str).to_numpy()
    C = np.zeros_like(P); has = np.zeros(len(P), bool)
    df = pd.DataFrame({"ctx": ctx, "i": np.arange(len(P))})
    for c, g in df[ctrl].groupby("ctx"):
        idx = df.index[df.ctx == c].to_numpy(); C[idx] = P[g.i.to_numpy()].mean(0); has[idx] = True
    return C, has, ctrl

rows = []
for seed, st in FOLDS:
    fo = pickle.load(open(os.path.join(OUT, "folds", f"{seed}_{st}.pkl"), "rb"))
    P = np.mean([np.load(os.path.join(OUT, "pool_real", f"{seed}_{st}__{m}.npy")) for m in MEMBERS], 0).astype(np.float32)
    C, has, ctrl = ctx_control(P, fo.meta); D = np.where(has[:, None], P - C, 0.0)
    med = np.median(np.abs(D[has & ~ctrl])) + 1e-6
    for name, gam, beta in VARS:
        Dg = np.sign(D) * (np.abs(D) ** gam) * (med ** (1 - gam)) if gam != 1.0 else D
        if beta > 0:
            Dg = Dg * (1.0 + beta * np.minimum(np.abs(Dg) / 1.25, 1.0) ** 2)
        Pn = np.where(has[:, None] & ~ctrl[:, None], C + Dg, P).astype(np.float32)
        r = summary_row(name, evaluate(fo, Pn, INNER)); r.update({"seed": seed, "strain": st, "variant": name}); rows.append(r)
    print(f"  seed{seed} {st} done", flush=True)
d = pd.DataFrame(rows); d.to_csv(os.path.join(OUT, "power_shape.csv"), index=False)
piv = d.pivot_table(index=["seed","strain"], columns="variant", values="TOTAL")
print("\n=== 幂律 vs 尾部扩张（六折配对，vs base）===")
for name, _, _ in VARS[1:]:
    dd = piv[name] - piv["base"]; print(f"  {name:12s} delta={dd.mean():+.5f}  sem={dd.sem():.5f}  ratio={dd.mean()/max(dd.sem(),1e-9):4.1f}  up={(dd>0).sum()}/6")
