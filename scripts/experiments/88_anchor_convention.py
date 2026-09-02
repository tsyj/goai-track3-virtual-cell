# -*- coding: utf-8 -*-
"""迭代 16：扩张锚点 C 对"对照约定"的敏感性（DMSO-only / Water-only / 两者全部 / 三者平均）。
97.3% 的测试行同上下文里两种载体并存，官方不公布匹配规则；看 v2.2 的扩张对此是否稳健。"""
import os, sys, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from vcell.harness import INNER, VAL, build_fold, evaluate, summary_row
OUT = os.path.join(ROOT, "results")
FOLDS = [(1,"CGD"),(5,"CGD"),(6,"CGD"),(3,"BAH"),(7,"BAH"),(8,"BAH")]
MEMBERS = ["A","B","C","D","E","E16_pert16","F_strain_early","FB_early_plate1","FC_early_plate4","FD_early_pert2","FE_early_pert8","FE16_early_pert16"]
BETA, TAU = 0.7, 0.75

def anchor(P, meta, vehicle):
    ctrl = meta["is_control"].to_numpy(); ctx = meta["ctx_key"].astype(str).to_numpy()
    comp = meta["compound"].astype(str).to_numpy()
    sel = ctrl if vehicle == "both" else (ctrl & (comp == vehicle))
    C = np.full_like(P, np.nan); has = np.zeros(len(P), bool)
    df = pd.DataFrame({"ctx": ctx, "i": np.arange(len(P))})
    for c, g in df[sel].groupby("ctx"):
        idx = df.index[df.ctx == c].to_numpy(); C[idx] = P[g.i.to_numpy()].mean(0); has[idx] = True
    return C, has, ctrl

def expand(P, meta, unseen, conv):
    if conv == "avg3":
        Cs = [anchor(P, meta, v) for v in ["both", "DMSO", "Water"]]
        has = np.zeros(len(P), bool); C = np.zeros_like(P); n = np.zeros(len(P))
        for Cv, hv, _ in Cs:
            C[hv] += Cv[hv]; n[hv] += 1; has |= hv
        C[has] /= n[has][:, None]; ctrl = Cs[0][2]
    else:
        C, has, ctrl = anchor(P, meta, conv)
    D = np.where(has[:, None], P - np.nan_to_num(C), 0.0)
    g = 1.0 + BETA * np.minimum(np.abs(D) / TAU, 1.0) ** 2
    return np.where((has & ~ctrl & unseen)[:, None], np.nan_to_num(C) + D * g, P).astype(np.float32), int((has & ~ctrl & unseen).sum())

rows = []
for seed, st in FOLDS:
    fo = pickle.load(open(os.path.join(OUT, "folds", f"{seed}_{st}.pkl"), "rb"))
    P = np.mean([np.load(os.path.join(OUT, "pool_real", f"{seed}_{st}__{m}.npy")) for m in MEMBERS], 0).astype(np.float32)
    unseen = (fo.meta["Strains"] == st).to_numpy()
    r = {"seed": seed, "strain": st}
    for conv in ["both", "DMSO", "Water", "avg3"]:
        Q, n = expand(P, fo.meta, unseen, conv); r[conv] = summary_row("x", evaluate(fo, Q, INNER))["TOTAL"]; r[conv + "_n"] = n
    rows.append(r); print(f"  seed{seed} {st}", flush=True)
d = pd.DataFrame(rows); d.to_csv(os.path.join(OUT, "anchor_convention_inner.csv"), index=False)
fo = build_fold(); P = np.mean([np.load(os.path.join(OUT, "val_parts", f"{m}_add.npy")) + np.load(os.path.join(OUT, "val_parts", f"{m}_boost.npy")) for m in MEMBERS], 0).astype(np.float32)
unseen = (fo.meta["Strains"] == "BAI").to_numpy(); v = {}
for conv in ["both", "DMSO", "Water", "avg3"]:
    Q, n = expand(P, fo.meta, unseen, conv); v[conv] = summary_row("x", evaluate(fo, Q, VAL))["TOTAL"]; v[conv + "_n"] = n
print(f"\n{'约定':8s} {'内层 vs both':>14s} {'sem':>8s} {'up':>4s} {'作用行(折1)':>10s} {'val':>8s} {'val vs both':>11s}")
for conv in ["both", "DMSO", "Water", "avg3"]:
    dd = d[conv] - d["both"]; print(f"{conv:8s} {dd.mean():+14.5f} {dd.sem():8.5f} {int((dd>0).sum()):>2d}/6 {int(d[conv+'_n'].iloc[0]):>10d} {v[conv]:8.4f} {v[conv]-v['both']:+11.4f}")
