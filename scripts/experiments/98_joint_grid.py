# -*- coding: utf-8 -*-
"""迭代 26：联合网格 (λ_strain, β, τ)。零标签菌株行先放大偏离 μ_drug 的分量，再做尾部扩张。
行的判定改为与划分名无关：菌株不在可见集合 且 化合物在可见集合。"""
import os, sys, pickle, warnings, itertools
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vcell.harness import INNER, VAL, build_fold, evaluate, summary_row
OUT = os.path.join(ROOT, "results")
MEM = ["A","B","C","D","E","E16_pert16","F_strain_early","FB_early_plate1","FC_early_plate4","FD_early_pert2","FE_early_pert8","FE16_early_pert16"]
FOLDS = [(1,"CGD"),(5,"CGD"),(6,"CGD"),(3,"BAH"),(7,"BAH"),(8,"BAH")]
LAM = [1.0, 1.4, 1.55, 1.7, 1.9]
EXP = [(0.25, 0.75), (0.4, 0.75), (0.55, 0.75), (0.7, 0.75)]

def ctx_ctrl(P, meta):
    ctrl = meta["is_control"].to_numpy(); ctx = meta["ctx_key"].astype(str).to_numpy()
    C = np.zeros_like(P); has = np.zeros(len(P), bool)
    df = pd.DataFrame({"ctx": ctx, "i": np.arange(len(P))})
    for c, g in df[ctrl].groupby("ctx"):
        idx = df.index[df.ctx == c].to_numpy(); C[idx] = P[g.i.to_numpy()].mean(0); has[idx] = True
    return C, has, ctrl

def prep(fo, P):
    meta = fo.meta; tr = fo.obs_mask; Dt = fo.Y - fo.C_true
    seen_s = set(meta.loc[tr, "Strains"].astype(str)); seen_c = set(meta.loc[tr, "compound"].astype(str))
    comp = meta["compound"].astype(str).to_numpy()
    mu = {}
    df = pd.DataFrame({"k": comp, "i": np.arange(len(meta))})
    for kk, g in df[tr].groupby("k"):
        v = np.nanmean(Dt[g.i.to_numpy()], 0)
        if np.isfinite(v).any(): mu[kk] = np.nan_to_num(v)
    C, has, ctrl = ctx_ctrl(P, meta)
    unseen_s = ~meta["Strains"].astype(str).isin(seen_s).to_numpy()
    tgt = has & ~ctrl & unseen_s & np.array([c in mu for c in comp])
    return C, has, ctrl, unseen_s, tgt, np.stack([mu[c] for c in comp[tgt]]) if tgt.any() else None

def transform(P, C, has, ctrl, unseen_s, tgt, M, lam, beta, tau):
    D = np.where(has[:, None], P - C, 0.0); Dn = D.copy()
    if lam != 1.0 and M is not None: Dn[tgt] = M + lam*(D[tgt] - M)
    if beta > 0:
        g = 1.0 + beta*np.minimum(np.abs(Dn)/tau, 1.0)**2
        Dn = np.where((unseen_s & has & ~ctrl)[:, None], Dn*g, Dn)
    return np.where((has & ~ctrl)[:, None], C + Dn, P).astype(np.float32)

def run(fo, P, which):
    C, has, ctrl, us, tgt, M = prep(fo, P); out = {}
    for lam, (b, t) in itertools.product(LAM, EXP):
        r = summary_row("x", evaluate(fo, transform(P, C, has, ctrl, us, tgt, M, lam, b, t), which))
        out[f"l{lam}_b{b}"] = r["TOTAL"]
        if lam == 1.0 and b == 0.7: out["ref_v22"] = r["TOTAL"]
    return out

fo = build_fold()
Pv = np.mean([np.load(f"{OUT}/val_parts/{m}_add.npy") + np.load(f"{OUT}/val_parts/{m}_boost.npy") for m in MEM], 0).astype(np.float32)
v = run(fo, Pv, VAL); print("val 完成", flush=True)
rows = []
for seed, st in FOLDS:
    fi = pickle.load(open(f"{OUT}/folds/{seed}_{st}.pkl", "rb"))
    Pi = np.mean([np.load(f"{OUT}/pool_real/{seed}_{st}__{m}.npy") for m in MEM], 0).astype(np.float32)
    r = run(fi, Pi, INNER); r.update({"seed": seed, "strain": st}); rows.append(r); print(f"  seed{seed} {st}", flush=True)
d = pd.DataFrame(rows); d.to_csv(f"{OUT}/joint_grid2_inner.csv", index=False); pd.Series(v).to_csv(f"{OUT}/joint_grid2_val.csv")
print(f"\n{'λ_strain':>9s} {'β':>5s} | {'内层 vs v2.2':>13s} {'sem':>8s} {'up':>4s} {'ratio':>6s} | {'val':>8s} {'val vs v2.2':>11s}")
for lam, (b, t) in itertools.product(LAM, EXP):
    k = f"l{lam}_b{b}"; dd = d[k] - d["ref_v22"]
    star = "  ←" if dd.mean() > 0 and (dd > 0).sum() == 6 else ""
    print(f"{lam:9.2f} {b:5.2f} | {dd.mean():+13.5f} {dd.sem():8.5f} {int((dd>0).sum()):>2d}/6 {dd.mean()/max(dd.sem(),1e-9):6.1f} | {v[k]:8.4f} {v[k]-v['ref_v22']:+11.4f}{star}")
