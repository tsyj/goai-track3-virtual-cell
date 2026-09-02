# -*- coding: utf-8 -*-
"""辩护性诊断：预测效应相对真实效应是否欠离散？（收缩估计的经典症状）
对 val 镜像，按"零标签菌株行 / 可见菌株行"分别报告：
  rms(D_pred), rms(D_true), 回归斜率 slope = <D_true,D_pred>/<D_pred,D_pred>（>1 即欠离散）
  以及按 |D_pred| 分桶的斜率——扩张只放大大效应，应看到大桶斜率更大。
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vcell.harness import build_fold
OUT = os.path.join(ROOT, "results")
MEM = ["A","B","C","D","E","E16_pert16","F_strain_early","FB_early_plate1","FC_early_plate4","FD_early_pert2","FE_early_pert8","FE16_early_pert16"]

fo = build_fold(); meta = fo.meta
P = np.mean([np.load(f"{OUT}/val_parts/{m}_add.npy") + np.load(f"{OUT}/val_parts/{m}_boost.npy") for m in MEM], 0).astype(np.float32)
ctrl = meta["is_control"].to_numpy(); ctx = meta["ctx_key"].astype(str).to_numpy()
C = np.zeros_like(P); has = np.zeros(len(P), bool)
df = pd.DataFrame({"ctx": ctx, "i": np.arange(len(P))})
for c, g in df[ctrl].groupby("ctx"):
    idx = df.index[df.ctx == c].to_numpy(); C[idx] = P[g.i.to_numpy()].mean(0); has[idx] = True
Dp = P - C                      # 模型隐含效应（未扩张）
Dt = fo.Y - fo.C_true           # 真实效应（评分口径）
val = meta["split_final"].astype(str).str.startswith("val").to_numpy()
unseen = (meta["Strains"] == "BAI").to_numpy()
print(f"{'行集':22s} {'n':>7s} {'rms(D_pred)':>12s} {'rms(D_true)':>12s} {'斜率':>8s}")
sets = [("零标签菌株 BAI", val & has & ~ctrl & unseen), ("可见菌株（val）", val & has & ~ctrl & ~unseen)]
for name, m in sets:
    a = Dp[m].ravel(); b = Dt[m].ravel(); k = np.isfinite(a) & np.isfinite(b); a, b = a[k], b[k]
    print(f"{name:22s} {int(m.sum()):7d} {np.sqrt((a**2).mean()):12.4f} {np.sqrt((b**2).mean()):12.4f} {float((a*b).sum()/(a*a).sum()):8.3f}")
print(f"\n按 |D_pred| 分桶的回归斜率（零标签菌株行；斜率>1 = 该桶预测偏小）")
m = val & has & ~ctrl & unseen
a = Dp[m].ravel(); b = Dt[m].ravel(); k = np.isfinite(a) & np.isfinite(b); a, b = a[k], b[k]
edges = [0, 0.2, 0.4, 0.75, 1.25, 2.0, 99]
print(f"  {'|D_pred| 区间':>16s} {'n':>10s} {'斜率':>8s} {'rms(D_true)/rms(D_pred)':>24s}")
for lo, hi in zip(edges[:-1], edges[1:]):
    s = (np.abs(a) >= lo) & (np.abs(a) < hi)
    if s.sum() < 100: continue
    sl = float((a[s]*b[s]).sum()/(a[s]*a[s]).sum()); rr = float(np.sqrt((b[s]**2).mean())/np.sqrt((a[s]**2).mean()))
    print(f"  {f'[{lo}, {hi})':>16s} {int(s.sum()):10d} {sl:8.3f} {rr:24.3f}")
m2 = val & has & ~ctrl & ~unseen
a2 = Dp[m2].ravel(); b2 = Dt[m2].ravel(); k2 = np.isfinite(a2) & np.isfinite(b2); a2, b2 = a2[k2], b2[k2]
print(f"\n对照：可见菌株行同样分桶")
for lo, hi in zip(edges[:-1], edges[1:]):
    s = (np.abs(a2) >= lo) & (np.abs(a2) < hi)
    if s.sum() < 100: continue
    print(f"  {f'[{lo}, {hi})':>16s} {int(s.sum()):10d} {float((a2[s]*b2[s]).sum()/(a2[s]*a2[s]).sum()):8.3f}")
