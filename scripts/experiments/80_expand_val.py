# -*- coding: utf-8 -*-
"""77 的终验：官方 val 镜像（留出 BAI）上，12 成员集成 + 大效应扩张 h(Δ̂)。
另在内层折上加密网格。用法：python scripts/80_expand_val.py
"""
import os, sys, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vcell.harness import INNER, VAL, build_fold, evaluate, summary_row
OUT = os.path.join(ROOT, "results")
MEMBERS = ["A","B","C","D","E","E16_pert16","F_strain_early","FB_early_plate1",
           "FC_early_plate4","FD_early_pert2","FE_early_pert8","FE16_early_pert16"]
GRID = [(0.0,1.0),(0.1,1.0),(0.15,1.0),(0.2,1.0),(0.25,1.0),(0.3,1.0),(0.2,1.25),(0.3,1.25),(0.2,1.5),(0.3,1.5),(0.4,1.5),(0.5,2.0)]

def expand(P, meta, beta, tau):
    ctrl = meta["is_control"].to_numpy(); ctx = meta["ctx_key"].astype(str).to_numpy()
    C = np.zeros_like(P); has = np.zeros(len(P), bool)
    df = pd.DataFrame({"ctx": ctx, "i": np.arange(len(P))})
    for c, g in df[ctrl].groupby("ctx"):
        idx = df.index[df.ctx == c].to_numpy(); C[idx] = P[g.i.to_numpy()].mean(0); has[idx] = True
    D = np.where(has[:, None], P - C, 0.0)
    g = 1.0 + beta * np.minimum(np.abs(D) / tau, 1.0) ** 2
    return np.where(has[:, None] & ~ctrl[:, None], C + D * g, P).astype(np.float32)

# ---- val 镜像
fo = build_fold()
P = (np.mean([np.load(os.path.join(OUT, "val_parts", f"{m}_add.npy")) for m in MEMBERS], 0)
     + np.mean([np.load(os.path.join(OUT, "val_parts", f"{m}_boost.npy")) for m in MEMBERS], 0)).astype(np.float32)
rows = []
for b, t in GRID:
    r = summary_row("x", evaluate(fo, expand(P, fo.meta, b, t), VAL)); r.update({"beta": b, "tau": t}); rows.append(r)
v = pd.DataFrame(rows); v.to_csv(os.path.join(OUT, "expand_val.csv"), index=False)
base = float(v.loc[(v.beta==0.0), "TOTAL"].iloc[0])
print(f"=== 官方 val 镜像（12 成员，基线 {base:.4f}）===")
for _, r in v.iterrows():
    print(f"  β={r.beta:.2f} τ={r.tau:.2f}  TOTAL {r.TOTAL:.4f} (Δ {r.TOTAL-base:+.4f})  M1 {r['M1_abs(20%)']:.4f} M2 {r['M2_rawFC(25%)']:.4f} M4 {r['M4_drug(20%)']:.4f} M6 {r['M6_DEP(5%)']:.4f}")

# ---- 内层加密网格
FOLDS = [(1, "CGD"), (5, "CGD"), (6, "CGD"), (3, "BAH"), (7, "BAH"), (8, "BAH")]
rows = []
for seed, st in FOLDS:
    fi = pickle.load(open(os.path.join(OUT, "folds", f"{seed}_{st}.pkl"), "rb"))
    Pi = np.mean([np.load(os.path.join(OUT, "pool_real", f"{seed}_{st}__{m}.npy")) for m in MEMBERS], 0).astype(np.float32)
    for b, t in GRID:
        r = summary_row("x", evaluate(fi, expand(Pi, fi.meta, b, t), INNER)); r.update({"seed": seed, "strain": st, "beta": b, "tau": t}); rows.append(r)
d = pd.DataFrame(rows); d.to_csv(os.path.join(OUT, "expand_inner_fine.csv"), index=False)
piv = d.pivot_table(index=["seed","strain"], columns=["beta","tau"], values="TOTAL"); b0 = piv[(0.0,1.0)]
print("\n=== 内层六折加密网格（vs 不扩张）===")
for b, t in GRID[1:]:
    dd = piv[(b,t)] - b0
    print(f"  β={b:.2f} τ={t:.2f}  delta={dd.mean():+.5f}  sem={dd.sem():.5f}  ratio={dd.mean()/max(dd.sem(),1e-9):4.1f}  up={(dd>0).sum()}/6")
