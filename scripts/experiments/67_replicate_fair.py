"""Apples-to-apples: replicate-as-prediction vs our model, on the SAME rows.

results/ceiling.csv's 'ORACLE replicate-measurement' (0.276) mixes replicate rows
(WAYB only) with a control-baseline fallback everywhere else, so quoting it against
our 0.503 invites a fair objection.  This script restricts to val rows that HAVE a
replicate partner and compares, on exactly those rows:

  * a真实重复测量 as the prediction
  * our shipped 12-member ensemble prediction (results/pool_val average)

Metrics: per-sample PCC / R^2 vs truth, and Delta-PCC vs matched control.

    python scripts/67_replicate_fair.py

前置条件：需要 results/pool_val/*.npy（12 个成员在官方 val 镜像上的预测），
由 scripts/evaluate_val_mirror.py 先行生成（约 35 分钟，或并行更快）。
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vcell.harness import build_fold
from vcell.metrics import _pcc_rows, _r2_rows

f = build_fold(vehicle="both")
meta, Y, n = f.meta, f.Y, len(f.meta)

rep_key = ["Strains", "compound", "Medium", "Temperature", "pert_time"]
wayb = meta["data_source"].str.startswith("WAYB").to_numpy()
sub = meta[wayb].copy(); sub["row"] = np.where(wayb)[0]
groups = [g["row"].tolist() for _, g in sub.groupby(rep_key) if len(g) > 1]
partner = np.full(n, -1)
for rows in groups:
    for i, a in enumerate(rows):
        partner[a] = rows[(i + 1) % len(rows)]

val = meta["split_final"].astype(str).str.startswith("val").to_numpy()
sel = val & (partner >= 0)
print(f"val 行总数 {val.sum()}，其中有真实重复伙伴的 {sel.sum()} 行"
      f"（按划分: {meta.loc[sel,'split_final'].value_counts().to_dict()}）")

members = ["A","B","C","D","E","E16_pert16","F_strain_early","FB_early_plate1",
           "FC_early_plate4","FD_early_pert2","FE_early_pert8","FE16_early_pert16"]
acc = None
for m in members:
    p = np.load(os.path.join(ROOT, "results", "pool_val", f"{m}.npy")).astype(np.float64)
    acc = p if acc is None else acc + p
P_model = (acc / len(members)).astype(np.float32)
P_rep = Y[np.maximum(partner, 0)]

rows = np.where(sel)[0]
D_true = (Y - f.C_true)[rows]
for name, P in [("真实重复测量", P_rep[rows]), ("我们的 12 成员模型", P_model[rows])]:
    pcc = np.nanmean(_pcc_rows(P, Y[rows]))
    r2 = np.nanmean(_r2_rows(P, Y[rows]))
    dp = P - f.C_true[rows]
    dpcc = np.nanmean(_pcc_rows(dp, D_true))
    err = np.nanstd((P - Y[rows]).astype(np.float64))
    print(f"  {name:14s} 逐样本PCC={pcc:.4f}  逐样本R²={r2:.4f}  Δ-PCC={dpcc:.4f}  误差sd={err:.3f}")
