# -*- coding: utf-8 -*-
"""诊断：(a) 各划分上我们相对"纯参照"（μ_drug / μ_ctx）的增量；(b) 蛋白轴 R² 的再校准空间。
纯缓存，val 镜像 + 内层六折。"""
import os, sys, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vcell.harness import INNER, VAL, build_fold, evaluate, summary_row
from vcell.metrics import _pcc_rows, _r2_rows
OUT = os.path.join(ROOT, "results")
MEM = ["A","B","C","D","E","E16_pert16","F_strain_early","FB_early_plate1","FC_early_plate4","FD_early_pert2","FE_early_pert8","FE16_early_pert16"]
BETA, TAU = 0.7, 0.75
FOLDS = [(1,"CGD"),(5,"CGD"),(6,"CGD"),(3,"BAH"),(7,"BAH"),(8,"BAH")]

def ctx_ctrl(P, meta):
    ctrl = meta["is_control"].to_numpy(); ctx = meta["ctx_key"].astype(str).to_numpy()
    C = np.zeros_like(P); has = np.zeros(len(P), bool)
    df = pd.DataFrame({"ctx": ctx, "i": np.arange(len(P))})
    for c, g in df[ctrl].groupby("ctx"):
        idx = df.index[df.ctx == c].to_numpy(); C[idx] = P[g.i.to_numpy()].mean(0); has[idx] = True
    return C, has, ctrl

def expand(P, meta, unseen):
    C, has, ctrl = ctx_ctrl(P, meta); D = np.where(has[:, None], P - C, 0.0)
    g = 1.0 + BETA * np.minimum(np.abs(D)/TAU, 1.0)**2
    return np.where((has & ~ctrl & unseen)[:, None], C + D*g, P).astype(np.float32)

fo = build_fold()
P = np.mean([np.load(f"{OUT}/val_parts/{m}_add.npy") + np.load(f"{OUT}/val_parts/{m}_boost.npy") for m in MEM], 0).astype(np.float32)
meta = fo.meta; unseen = (meta["Strains"] == "BAI").to_numpy()
P22 = expand(P, meta, unseen)
tr = fo.obs_mask; Dt = fo.Y - fo.C_true; Dp = P22 - fo.C_true
def mu_table(key):
    k = meta[key].astype(str).to_numpy(); out = {}
    df = pd.DataFrame({"k": k, "i": np.arange(len(P))})
    for kk, g in df[tr].groupby("k"): out[kk] = np.nanmean(Dt[g.i.to_numpy()], 0)
    return k, out
print("=== (a) 各划分：我们 vs 纯参照 ===")
for split, key in [("val_strain_only","compound"),("val_both","compound"),("val_chem_only","ctx_key"),("val_time","ctx_key")]:
    rows = (meta.split_final == split).to_numpy() & ~meta.is_control.to_numpy()
    if rows.sum() == 0: continue
    k, mu = mu_table(key)
    M = np.stack([np.nan_to_num(mu.get(x, np.zeros(Dt.shape[1]))) for x in k[rows]])
    ours = np.nanmean(_pcc_rows(Dp[rows], Dt[rows])); ref = np.nanmean(_pcc_rows(M, Dt[rows]))
    resid = np.nanmean(_pcc_rows(Dp[rows]-M, Dt[rows]-M))
    print(f"  {split:16s} n={int(rows.sum()):5d}  我们 Δ-PCC={ours:.4f}  纯参照({key})={ref:.4f}  残差 PCC={resid:.4f}")
print("\n=== (b) 蛋白轴 R² 再校准（γ 缩放每蛋白跨样本离散度；m_j 取训练行预测均值）===")
mj = P22[tr].mean(0)
base = summary_row("x", evaluate(fo, P22, VAL))
pr2 = 4*base["M1_abs(20%)"] - base["sampPCC"] - base["sampR2"] - base["protPCC"]
print(f"  当前 protR2={pr2:.4f}  protPCC={base['protPCC']:.4f}（上限 protPCC²={base['protPCC']**2:.4f}）  M1={base['M1_abs(20%)']:.4f}  TOTAL={base['TOTAL']:.4f}")
for g in [1.0, 1.02, 1.05, 1.08, 1.12, 1.18, 1.25]:
    Q = (mj[None,:] + g*(P22 - mj[None,:])).astype(np.float32)
    r = summary_row("x", evaluate(fo, Q, VAL))
    p2 = 4*r["M1_abs(20%)"] - r["sampPCC"] - r["sampR2"] - r["protPCC"]
    print(f"  γ={g:.2f}  TOTAL {r['TOTAL']:.4f} ({r['TOTAL']-base['TOTAL']:+.4f})  M1 {r['M1_abs(20%)']:.4f}  protR2 {p2:.4f}  protPCC {r['protPCC']:.4f}  M2 {r['M2_rawFC(25%)']:.4f}")
