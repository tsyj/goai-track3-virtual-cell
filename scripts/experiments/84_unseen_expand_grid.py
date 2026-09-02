# -*- coding: utf-8 -*-
"""迭代 12：尾部扩张只作用于未见菌株行时的 (β, τ) 网格。内层六折 + val 镜像。"""
import os, sys, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vcell.harness import INNER, VAL, build_fold, evaluate, summary_row
OUT = os.path.join(ROOT, "results")
FOLDS = [(1,"CGD"),(5,"CGD"),(6,"CGD"),(3,"BAH"),(7,"BAH"),(8,"BAH")]
MEMBERS = ["A","B","C","D","E","E16_pert16","F_strain_early","FB_early_plate1","FC_early_plate4","FD_early_pert2","FE_early_pert8","FE16_early_pert16"]
GRID = [(0.0,1.25),(0.3,1.25),(0.5,1.25),(0.7,1.25),(1.0,1.25),(0.5,1.0),(0.7,1.0),(1.0,1.0),(0.5,1.5),(0.7,1.5),(1.0,1.5),(0.7,0.75),(1.0,0.75),(1.5,1.0)]
ALLROWS = (0.3, 1.25)   # 参照：当前 v2.1（全部行）

def ctx_ctrl(P, meta):
    ctrl = meta["is_control"].to_numpy(); ctx = meta["ctx_key"].astype(str).to_numpy()
    C = np.zeros_like(P); has = np.zeros(len(P), bool)
    df = pd.DataFrame({"ctx": ctx, "i": np.arange(len(P))})
    for c, g in df[ctrl].groupby("ctx"):
        idx = df.index[df.ctx == c].to_numpy(); C[idx] = P[g.i.to_numpy()].mean(0); has[idx] = True
    return C, has, ctrl

def expand(P, meta, mask, beta, tau):
    C, has, ctrl = ctx_ctrl(P, meta); D = np.where(has[:, None], P - C, 0.0)
    g = 1.0 + beta * np.minimum(np.abs(D) / tau, 1.0) ** 2
    apply = has & ~ctrl & mask
    return np.where(apply[:, None], C + D * g, P).astype(np.float32)

def run(fo, P, held, tag):
    meta = fo.meta; unseen = (meta["Strains"] == held).to_numpy(); allm = np.ones(len(meta), bool)
    out = {"v21_all": summary_row("x", evaluate(fo, expand(P, meta, allm, *ALLROWS), tag))["TOTAL"]}
    for b, t in GRID:
        out[f"u_b{b}_t{t}"] = summary_row("x", evaluate(fo, expand(P, meta, unseen, b, t), tag))["TOTAL"]
    return out

rows = []
for seed, st in FOLDS:
    fo = pickle.load(open(os.path.join(OUT, "folds", f"{seed}_{st}.pkl"), "rb"))
    P = np.mean([np.load(os.path.join(OUT, "pool_real", f"{seed}_{st}__{m}.npy")) for m in MEMBERS], 0).astype(np.float32)
    r = run(fo, P, st, INNER); r.update({"seed": seed, "strain": st}); rows.append(r); print(f"  seed{seed} {st}", flush=True)
d = pd.DataFrame(rows); d.to_csv(os.path.join(OUT, "unseen_expand_grid_inner.csv"), index=False)
fo = build_fold(); P = np.mean([np.load(os.path.join(OUT, "val_parts", f"{m}_add.npy")) + np.load(os.path.join(OUT, "val_parts", f"{m}_boost.npy")) for m in MEMBERS], 0).astype(np.float32)
v = run(fo, P, "BAI", VAL); pd.Series(v).to_csv(os.path.join(OUT, "unseen_expand_grid_val.csv"))
print(f"\n{'变体':16s} {'内层 delta vs v2.1':>20s} {'sem':>8s} {'up':>5s} {'ratio':>6s} {'val TOTAL':>10s} {'val vs v2.1':>11s}")
base_in = d["v21_all"]
for k in ["u_b0.0_t1.25"] + [f"u_b{b}_t{t}" for b, t in GRID[1:]]:
    dd = d[k] - base_in
    print(f"{k:16s} {dd.mean():+20.5f} {dd.sem():8.5f} {int((dd>0).sum()):>3d}/6 {dd.mean()/max(dd.sem(),1e-9):6.1f} {v[k]:10.4f} {v[k]-v['v21_all']:+11.4f}")
