# -*- coding: utf-8 -*-
"""终验：官方 val 镜像（留出 BAI）上，12 成员集成 + 未见菌株 booster 重定标 k。

只评一次性的报告数；k 的选择已在内层折完成（k=1.6），此处是独立验证而非调参。

    python scripts/73_val_scale_eval.py
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vcell.harness import VAL, build_fold, evaluate, summary_row

RES = os.path.join(ROOT, "results")
PARTS = os.path.join(RES, "val_parts")
MEMBERS = ["A","B","C","D","E","E16_pert16","F_strain_early","FB_early_plate1",
           "FC_early_plate4","FD_early_pert2","FE_early_pert8","FE16_early_pert16"]
KS = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

fo = build_fold()
unseen = (fo.meta["Strains"] == "BAI").to_numpy()          # val 镜像的零标签菌株
print(f"BAI 行 {unseen.sum()} / {len(fo.meta)}")
A = np.mean([np.load(os.path.join(PARTS, f"{m}_add.npy")) for m in MEMBERS], 0)
B = np.mean([np.load(os.path.join(PARTS, f"{m}_boost.npy")) for m in MEMBERS], 0)
rows = []
for k in KS:
    mult = np.where(unseen[:, None], k, 1.0).astype(np.float32)
    r = summary_row(f"k={k}", evaluate(fo, (A + mult * B).astype(np.float32), VAL))
    r["k"] = k
    rows.append(r)
d = pd.DataFrame(rows)
d.to_csv(os.path.join(RES, "val_scale_eval.csv"), index=False)
cols = ["k","TOTAL","FC[chem_only]","FC[strain_only]","FC[both]","FC[time]",
        "M1_abs(20%)","M2_rawFC(25%)","M3_ctx(20%)","M4_drug(20%)","M6_DEP(5%)"]
print(d[cols].round(4).to_string(index=False))
base = float(d.loc[d.k==1.0,"TOTAL"].iloc[0])
for k in KS[1:]:
    print(f"  k={k}: TOTAL {float(d.loc[d.k==k,'TOTAL'].iloc[0]):.4f}  (Δ {float(d.loc[d.k==k,'TOTAL'].iloc[0])-base:+.4f})")
