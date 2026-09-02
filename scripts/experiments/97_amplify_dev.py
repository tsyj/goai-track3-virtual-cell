# -*- coding: utf-8 -*-
"""迭代 25：放大"偏离冻结参照的分量"。D' = μ + λ(D − μ)，λ>1。
未见化合物行用 μ_ctx（打 M3，20%），未见菌株行用 μ_drug（打 M4，20%）。μ 只用训练行 Δ_true。
之后仍套 v2.2 的零标签菌株尾部扩张。内层六折 + val。"""
import os, sys, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vcell.harness import INNER, VAL, build_fold, evaluate, summary_row
OUT = os.path.join(ROOT, "results")
MEM = ["A","B","C","D","E","E16_pert16","F_strain_early","FB_early_plate1","FC_early_plate4","FD_early_pert2","FE_early_pert8","FE16_early_pert16"]
BETA, TAU = 0.7, 0.75
FOLDS = [(1,"CGD"),(5,"CGD"),(6,"CGD"),(3,"BAH"),(7,"BAH"),(8,"BAH")]
GRID = [1.15, 1.3, 1.5, 1.8, 2.2]

def ctx_ctrl(P, meta):
    ctrl = meta["is_control"].to_numpy(); ctx = meta["ctx_key"].astype(str).to_numpy()
    C = np.zeros_like(P); has = np.zeros(len(P), bool)
    df = pd.DataFrame({"ctx": ctx, "i": np.arange(len(P))})
    for c, g in df[ctrl].groupby("ctx"):
        idx = df.index[df.ctx == c].to_numpy(); C[idx] = P[g.i.to_numpy()].mean(0); has[idx] = True
    return C, has, ctrl

def mu_of(fo, key):
    meta = fo.meta; tr = fo.obs_mask; Dt = fo.Y - fo.C_true
    k = meta[key].astype(str).to_numpy(); tab = {}
    df = pd.DataFrame({"k": k, "i": np.arange(len(meta))})
    for kk, g in df[tr].groupby("k"):
        v = np.nanmean(Dt[g.i.to_numpy()], 0)
        if np.isfinite(v).any(): tab[kk] = np.nan_to_num(v)
    return k, tab

def transform(P, fo, held, lc, ls):
    meta = fo.meta; C, has, ctrl = ctx_ctrl(P, meta); D = np.where(has[:, None], P - C, 0.0)
    sp = meta["split_final"].astype(str).to_numpy()
    chem = np.isin(sp, ["in_chem_only", "val_chem_only"]); strn = np.isin(sp, ["in_strain_only", "val_strain_only"])
    Dn = D.copy()
    if lc != 1.0:
        kc, mc = mu_of(fo, "ctx_key")
        rows = [i for i in np.where(has & ~ctrl & chem)[0] if kc[i] in mc]
        if rows:
            M = np.stack([mc[kc[i]] for i in rows]); Dn[rows] = M + lc*(D[rows] - M)
    if ls != 1.0:
        kd, md = mu_of(fo, "compound")
        rows = [i for i in np.where(has & ~ctrl & strn)[0] if kd[i] in md]
        if rows:
            M = np.stack([md[kd[i]] for i in rows]); Dn[rows] = M + ls*(D[rows] - M)
    unseen = (meta["Strains"] == held).to_numpy()
    g = 1.0 + BETA*np.minimum(np.abs(Dn)/TAU, 1.0)**2
    Dn = np.where(unseen[:, None], Dn*g, Dn)
    return np.where((has & ~ctrl)[:, None], C + Dn, P).astype(np.float32)

def run(fo, P, held, which):
    out = {}
    b = summary_row("x", evaluate(fo, transform(P, fo, held, 1.0, 1.0), which))
    out["v22"] = b["TOTAL"]; out["v22|M3"] = b["M3_ctx(20%)"]; out["v22|M4"] = b["M4_drug(20%)"]; out["v22|M2"] = b["M2_rawFC(25%)"]
    for l in GRID:
        r = summary_row("x", evaluate(fo, transform(P, fo, held, l, 1.0), which))
        out[f"chem{l}"] = r["TOTAL"]; out[f"chem{l}|M3"] = r["M3_ctx(20%)"]; out[f"chem{l}|M2"] = r["M2_rawFC(25%)"]
        r = summary_row("x", evaluate(fo, transform(P, fo, held, 1.0, l), which))
        out[f"strain{l}"] = r["TOTAL"]; out[f"strain{l}|M4"] = r["M4_drug(20%)"]; out[f"strain{l}|M2"] = r["M2_rawFC(25%)"]
    return out

fo = build_fold()
Pv = np.mean([np.load(f"{OUT}/val_parts/{m}_add.npy") + np.load(f"{OUT}/val_parts/{m}_boost.npy") for m in MEM], 0).astype(np.float32)
v = run(fo, Pv, "BAI", VAL)
rows = []
for seed, st in FOLDS:
    fi = pickle.load(open(f"{OUT}/folds/{seed}_{st}.pkl", "rb"))
    Pi = np.mean([np.load(f"{OUT}/pool_real/{seed}_{st}__{m}.npy") for m in MEM], 0).astype(np.float32)
    r = run(fi, Pi, st, INNER); r.update({"seed": seed, "strain": st}); rows.append(r); print(f"  seed{seed} {st}", flush=True)
d = pd.DataFrame(rows); d.to_csv(f"{OUT}/amplify_dev_inner.csv", index=False); pd.Series(v).to_csv(f"{OUT}/amplify_dev_val.csv")
print(f"\n{'变体':10s} {'内层 vs v2.2':>13s} {'sem':>8s} {'up':>4s} {'ratio':>6s} {'val':>8s} {'val vs v2.2':>11s} {'val 目标模块':>18s}")
for l in GRID:
    for tag, mod in [("chem", "M3"), ("strain", "M4")]:
        k = f"{tag}{l}"; dd = d[k] - d["v22"]
        print(f"{k:10s} {dd.mean():+13.5f} {dd.sem():8.5f} {int((dd>0).sum()):>2d}/6 {dd.mean()/max(dd.sem(),1e-9):6.1f} {v[k]:8.4f} {v[k]-v['v22']:+11.4f}   {mod} {v[k+'|'+mod]:.4f} ({v[k+'|'+mod]-v['v22|'+mod]:+.4f})")
